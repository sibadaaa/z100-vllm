# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hygon DCU DP4X-mode Triton launch guard.

Z100 boards (PCI device id ``0x54B7``) boot with the firmware in *DP4X* mode;
Z100L boards (``0x55B7``) run in DP2X mode. The mode is visible per board in the
KFD topology (``dp4x_status`` in
``/sys/class/kfd/kfd/topology/nodes/*/properties``) and it changes the VGPR
granularity / context-save geometry the userspace has to use.

The vendor DTK userspace knows about DP4X (its HSA loader has an
``EF_GPU_DP4X`` path and a "vgpr granularity" query). An upstream-derived
HIP/ROCr runtime (HIP >= 7.x, e.g. the self-built ROCm 10.1 + torch 2.16
stack) does not: on DP4X boards a Triton kernel whose gfx906 kernel descriptor
requests the architectural VGPR maximum (256 VGPR/thread, ``VGRPBlocks 63``)
silently mis-executes -- only the first partial vector store commits, the rest
of the output is left untouched, and neither HIP nor the kernel log reports an
error. Kernels whose VGPR allocation stays below that envelope are correct.

The guard below is a runtime workaround, not a fix: on DP4X devices it forces
every Triton kernel to be compiled with at least
``VLLM_GFX906_DP4X_MIN_WARPS`` warps (default 8). More warps halve the
per-thread register demand, which moves the compiled kernel below the failing
VGPR envelope. See ``docs/result/F24-investigation-20261018.md``.

Environment variables:
    VLLM_GFX906_DP4X_GUARD: ``auto`` (default) enables the guard when Hygon
        DCU hardware is present, at least one board reports DP4X, and the
        loaded HIP runtime is an upstream-derived one (``torch.version.hip``
        >= 7.0; the vendor DTK stack reports 6.3). ``1``/``force`` applies the
        override to every device (benchmarking only), ``0`` disables the guard.
    VLLM_GFX906_DP4X_MIN_WARPS: minimum number of warps forced on DP4X devices
        (default 8).
    VLLM_GFX906_DP4X_WATCH: ``1`` (default) logs a warning if a kernel still
        compiles to VGPR=256 on a DP4X device after the override.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_KFD_TOPOLOGY_GLOB = "/sys/class/kfd/kfd/topology/nodes/*/properties"
_HYGON_KFD_VENDOR_ID = 7572  # 0x1d94, Hygon DCU boards
_GFX906_TARGET_VERSION = 90106
_DP4X_VGPR_GRANULE = 4
# gfx9 encodes VGPRs in granules of 4: 253/254/255/256 all round up to 256,
# which is the architectural maximum and the failing envelope (F24 probe:
# 253->256 FAIL, 254->256 writes 136).
_DP4X_MAX_VGPR = 253

_DP4X_GUARD_ENV = "VLLM_GFX906_DP4X_GUARD"
_DP4X_MIN_WARPS_ENV = "VLLM_GFX906_DP4X_MIN_WARPS"
_DP4X_WATCH_ENV = "VLLM_GFX906_DP4X_WATCH"
_DEFAULT_MIN_WARPS = 8

# State filled by install_dp4x_triton_guard().
_guard_installed = False
_original_jit_run = None
_logged_active = False
_warned_kernels: set[str] = set()
_debugged_launches: set[tuple[str, int]] = set()
_logged_first_guarded_launch = False


@dataclass(frozen=True)
class KfdGpuNode:
    """One GPU node of the KFD topology (subset of `properties`)."""

    node_id: int
    vendor_id: int
    gfx_target_version: int
    location_id: int
    dp4x_status: int | None

    @property
    def pci_bus(self) -> int:
        # KFD location_id packs (bus << 8) | (device << 3) | function.
        return (self.location_id >> 8) & 0xFF


def _read_kfd_gpu_nodes() -> list[KfdGpuNode]:
    """KFD topology GPU nodes (CPU nodes are skipped), in node-id order."""
    nodes: list[KfdGpuNode] = []
    paths = glob.glob(_KFD_TOPOLOGY_GLOB)
    def _node_order(path: str) -> int:
        name = Path(path).parent.name
        return int(name) if name.isdigit() else 1 << 30

    paths.sort(key=_node_order)
    for path in paths:
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        props = dict(re.findall(r"^([a-z0-9_]+)\s+(\S+)$", text, re.MULTILINE))
        if int(props.get("simd_count", "0")) == 0:
            continue
        node_dir = Path(path).parent.name
        nodes.append(
            KfdGpuNode(
                node_id=int(node_dir) if node_dir.isdigit() else -1,
                vendor_id=int(props.get("vendor_id", "0")),
                gfx_target_version=int(props.get("gfx_target_version", "0")),
                location_id=int(props.get("location_id", "0")),
                dp4x_status=(
                    int(props["dp4x_status"]) if "dp4x_status" in props else None
                ),
            )
        )
    return nodes


@lru_cache(maxsize=1)
def kfd_gpu_nodes() -> tuple[KfdGpuNode, ...]:
    return tuple(_read_kfd_gpu_nodes())


@lru_cache(maxsize=1)
def dp4x_status_by_pci_bus() -> dict[int, bool]:
    """``PCI bus -> DP4X`` for every board that reports ``dp4x_status``."""
    return {
        node.pci_bus: bool(node.dp4x_status)
        for node in kfd_gpu_nodes()
        if node.dp4x_status is not None
    }


@lru_cache(maxsize=1)
def is_hygon_dcu_host() -> bool:
    """Whether the host has Hygon DCU boards (no GPU runtime call needed)."""
    for node in kfd_gpu_nodes():
        if node.vendor_id == _HYGON_KFD_VENDOR_ID:
            return True
        if node.gfx_target_version == _GFX906_TARGET_VERSION:
            return True
    return False


@lru_cache(maxsize=1)
def dp4x_board_count() -> int:
    return sum(1 for dp4x in dp4x_status_by_pci_bus().values() if dp4x)


def _hip_version_tuple() -> tuple[int, ...]:
    hip = torch.version.hip
    if not hip:
        return ()
    parts: list[int] = []
    for piece in re.split(r"[.\-+]", hip):
        match = re.match(r"^(\d+)", piece)
        if match is None:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


@lru_cache(maxsize=1)
def runtime_needs_dp4x_guard() -> bool:
    """Whether the loaded HIP runtime is DP4X-unaware.

    The vendor DTK stack reports HIP 6.3 (and ships a DP4X-aware HSA loader);
    the upstream-derived stacks built for this board report HIP >= 7.0 and
    have no DP4X support. Treat every HIP >= 7 runtime as DP4X-unaware until
    a vendor build proves otherwise.
    """
    version = _hip_version_tuple()
    if not version:
        return False
    return version[0] >= 7


def _guard_mode() -> str:
    raw = os.environ.get(_DP4X_GUARD_ENV, "auto").strip().lower()
    if raw in {"0", "false", "off", "no", "disable", "disabled"}:
        return "off"
    if raw in {"1", "true", "on", "yes", "force", "all"}:
        return "all"
    return "auto"


@lru_cache(maxsize=1)
def dp4x_guard_enabled() -> bool:
    """Whether the Triton DP4X guard should be installed in this process."""
    mode = _guard_mode()
    if mode == "off":
        return False
    if mode == "all":
        return True
    return (
        is_hygon_dcu_host()
        and dp4x_board_count() > 0
        and runtime_needs_dp4x_guard()
    )


@lru_cache(maxsize=1)
def dp4x_min_warps() -> int:
    raw = os.environ.get(_DP4X_MIN_WARPS_ENV, str(_DEFAULT_MIN_WARPS)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_MIN_WARPS
    return max(1, min(value, 32))


@lru_cache(maxsize=1)
def _watch_enabled() -> bool:
    raw = os.environ.get(_DP4X_WATCH_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _device_pci_bus(device_index: int) -> int | None:
    """PCI bus number of a torch/HIP device ordinal."""
    try:
        props = torch.cuda.get_device_properties(device_index)
    except Exception:
        return None
    bus = getattr(props, "pci_bus_id", None)
    if bus is not None:
        try:
            return int(bus)
        except (TypeError, ValueError):
            pass
    # Fallback: map the ordinal through HIP_VISIBLE_DEVICES onto the KFD node
    # order (node order matches the PCI-sorted HIP enumeration on these boards).
    visible = (
        os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("ROCR_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    buses = sorted(dp4x_status_by_pci_bus())
    physical = list(range(len(buses)))
    if visible and visible.strip():
        try:
            physical = [int(part) for part in visible.split(",")]
        except ValueError:
            physical = list(range(len(buses)))
    if device_index >= len(physical):
        return None
    physical_index = physical[device_index]
    if physical_index >= len(buses):
        return None
    return buses[physical_index]


@lru_cache(maxsize=16)
def device_is_dp4x(device_index: int) -> bool:
    """Whether a torch device ordinal is a DP4X (Z100) board."""
    if not dp4x_guard_enabled():
        return False
    if _guard_mode() == "all":
        return True
    bus = _device_pci_bus(device_index)
    if bus is None:
        return False
    return dp4x_status_by_pci_bus().get(bus, False)


def _current_device_is_dp4x() -> bool:
    if not dp4x_guard_enabled():
        return False
    try:
        return device_is_dp4x(torch.cuda.current_device())
    except Exception:
        return False


def _kernel_name(jit_fn) -> str:
    for attr in ("kernel_name", "fn"):
        value = getattr(jit_fn, attr, None)
        if value is not None:
            name = getattr(value, "__name__", None)
            if name:
                return name
    return repr(jit_fn)


def _watch_compiled_kernel(jit_fn, kernel) -> None:
    if not _watch_enabled():
        return
    try:
        n_regs = getattr(kernel, "n_regs", None)
        if not isinstance(n_regs, int) or n_regs < _DP4X_MAX_VGPR:
            return
        # Fail-closed: a granulated 256 (n_regs 253..256) mis-executes.
        name = _kernel_name(jit_fn)
        if name in _warned_kernels:
            return
        _warned_kernels.add(name)
        logger.warning(
            "DP4X guard: Triton kernel %r still compiled to VGPR=%d on a DP4X "
            "device despite num_warps>=%d; it may silently mis-execute (F24).",
            name,
            n_regs,
            dp4x_min_warps(),
        )
    except Exception:  # pragma: no cover - diagnostics must never break a run
        pass


@lru_cache(maxsize=1)
def _debug_enabled() -> bool:
    raw = os.environ.get("VLLM_GFX906_DP4X_DEBUG", "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _debug_log_launch(jit_fn, kernel, device_index: int, on_dp4x: bool) -> None:
    """Log every distinct (kernel, device) launch in debug mode.

    ``on_dp4x=False`` lines are equally important: in a process whose current
    device *is* a DP4X board, they mean a launch slipped past the guard (wrong
    current-device state, different thread, ...).
    """
    if not _debug_enabled():
        return
    name = _kernel_name(jit_fn)
    key = (name, device_index, on_dp4x)
    if key in _debugged_launches:
        return
    _debugged_launches.add(key)
    logger.info(
        "DP4X guard debug: kernel=%r device=%d overridden=%s n_regs=%s n_spills=%s",
        name,
        device_index,
        on_dp4x,
        getattr(kernel, "n_regs", None),
        getattr(kernel, "n_spills", None),
    )


def install_dp4x_triton_guard() -> bool:
    """Install the process-wide Triton launch guard (idempotent).

    Returns whether the guard is active. Cheapest to call from the ROCm
    platform module import so every process that can launch Triton kernels is
    covered before the model starts.
    """
    global _guard_installed, _original_jit_run, _logged_active
    if _guard_installed:
        return True
    if not dp4x_guard_enabled():
        return False

    try:
        from triton.runtime.jit import JITFunction
    except Exception as exc:  # pragma: no cover - triton is a hard dependency
        logger.debug("DP4X guard: triton is not importable: %s", exc)
        return False

    if getattr(JITFunction.run, "_dp4x_guarded", False):
        _guard_installed = True
        return True

    _original_jit_run = JITFunction.run
    original = _original_jit_run
    min_warps = dp4x_min_warps()

    def _dp4x_guarded_run(self, *args, grid, warmup, **kwargs):
        global _logged_first_guarded_launch
        on_dp4x = _current_device_is_dp4x()
        if on_dp4x:
            requested = kwargs.get("num_warps")
            if requested is None or requested < min_warps:
                kwargs["num_warps"] = min_warps
                if not _logged_first_guarded_launch:
                    _logged_first_guarded_launch = True
                    logger.info(
                        "Hygon DCU DP4X guard: forcing num_warps=%d (was %s) "
                        "on device %d (HIP %s). F24 runtime workaround.",
                        min_warps,
                        requested,
                        torch.cuda.current_device(),
                        torch.version.hip,
                    )
        device_index = -1
        if _debug_enabled():
            try:
                device_index = torch.cuda.current_device()
            except Exception:
                device_index = -1
        kernel = original(self, *args, grid=grid, warmup=warmup, **kwargs)
        if not warmup and kernel is not None:
            if on_dp4x:
                _watch_compiled_kernel(self, kernel)
            if _debug_enabled() and device_index >= 0:
                try:
                    _debug_log_launch(self, kernel, device_index, on_dp4x)
                except Exception:
                    pass
        return kernel

    _dp4x_guarded_run._dp4x_guarded = True  # type: ignore[attr-defined]
    JITFunction.run = _dp4x_guarded_run
    _guard_installed = True
    if not _logged_active:
        _logged_active = True
        logger.info(
            "Hygon DCU DP4X guard active: min num_warps=%d for DP4X boards "
            "(HIP %s, DP4X boards: %s). This is the F24 runtime workaround.",
            min_warps,
            torch.version.hip,
            [bus for bus, dp4x in sorted(dp4x_status_by_pci_bus().items()) if dp4x],
        )
    return True


def dp4x_guard_status() -> dict:
    """Human-readable guard state (used by tools/dcu self-test and logs)."""
    nodes = [
        {
            "node_id": node.node_id,
            "pci_bus": node.pci_bus,
            "dp4x": bool(node.dp4x_status),
            "gfx_target_version": node.gfx_target_version,
        }
        for node in kfd_gpu_nodes()
        if node.dp4x_status is not None
    ]
    return {
        "hip_version": torch.version.hip,
        "hygon_dcu_host": is_hygon_dcu_host(),
        "runtime_needs_guard": runtime_needs_dp4x_guard(),
        "guard_mode": _guard_mode(),
        "guard_enabled": dp4x_guard_enabled(),
        "guard_installed": _guard_installed,
        "min_warps": dp4x_min_warps(),
        "boards": nodes,
    }
