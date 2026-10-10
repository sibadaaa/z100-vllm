# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime compatibility helpers for Hygon DCU boards running DTK.

Hygon's DCU Toolkit (DTK) exposes a ROCm-compatible stack: the same HIP
runtime API, rocBLAS/RCCL-style libraries and a populated
``torch.version.hip``. The differences that matter to vLLM are confined to
the runtime layer:

* the install prefix is ``/opt/dtk`` rather than ``/opt/rocm``;
* ``amdsmi`` is either missing or reports a bogus GCN arch such as
  ``gfx18446744073709551615``;
* GCN arch, VRAM size and device UUID are readable from sysfs without
  initializing a HIP context;
* the DCU ISA used by Z100 is gfx906 (Vega20), which has *no* bfloat16
  instructions, so bf16 is emulated in software.

Every helper returns a neutral value (``None``/``False``) on a community
ROCm installation, so callers keep their existing code path.

Related environment variables:
    VLLM_DCU_GCN_ARCH: explicit GCN arch override, bypasses all probing.
    VLLM_DCU_BF16: ``auto`` (default, probe the device), ``1`` to force
        bf16 on, ``0`` to force it off.
"""

from __future__ import annotations

import glob
import os
import re
from functools import lru_cache
from pathlib import Path

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_DTK_INFO_VERSION = Path("/opt/dtk/.info/rocm_version")
_KFD_TOPOLOGY_GLOB = "/sys/class/kfd/kfd/topology/nodes/*/properties"
_DRM_CARD_GLOB = "/sys/class/drm/card[0-9]*"

# PCI vendor id of Hygon DCU boards, also reported as ``vendor_id 7572`` by
# the KFD topology. A DTK host can have other DRM cards (for example an
# NVIDIA card as card0), so DRM cards are filtered by vendor and ordered by
# PCI bus/device to line up with HIP device ordinals.
_HYGON_PCI_VENDOR = "0x1d94"

# A GCN arch name is "gfx" followed by 3 or 4 hex digits, optionally with
# target-feature suffixes such as "gfx906:sramecc+:xnack-".
_GCN_ARCH_RE = re.compile(r"^gfx[0-9a-f]{3,4}$")

# Hygon's SMI/driver reports the Z100 as ``gfx916`` while the driver, torch
# and every compiled kernel use ``gfx906``. This must be an alias, not an
# allow-list entry: an allow-list would accept ``gfx916`` and leave
# ``_capability_from_gcn_arch('gfx916')`` at (9, 1), which silently disables
# every ``capability.minor == 0`` gate. Only ISA names verified on real
# hardware belong here.
_HYGON_DCU_ARCH_ALIAS = {"gfx916": "gfx906"}

# ``gfx_target_version`` as packed by the KFD driver. The packing is
# major * 10000 + minor * 100 + stepping, but GFX9 parts do not follow the
# LLVM minor numbering, so only values observed on real hardware are
# mapped. Unknown values yield None on purpose: the arch reported by torch
# stays authoritative for every capability gate in vLLM, and this table is
# only a context-free fallback.
_GFX_TARGET_VERSION_TO_ARCH = {
    # 4x Hygon DCU, device ids 0x54B7 / 0x55B7, 240 SIMD, hy-smi reports
    # Z100 / ZIFANG.
    90106: "gfx906",
}

_BF16_MAX_REL_ERR = 0.05


def is_hygon_dtk() -> bool:
    """Whether vLLM is running on a Hygon DCU stack (DTK).

    Cheap, torch-free and free of GPU runtime calls, so it is safe to use
    on import.
    """
    return _is_hygon_dtk_cached()


@lru_cache(maxsize=1)
def _is_hygon_dtk_cached() -> bool:
    if os.environ.get("DTKROOT") or os.environ.get("HYHAL_PATH"):
        return True
    if _DTK_INFO_VERSION.is_file():
        return True
    if glob.glob("/opt/dtk*/bin/hy-smi"):
        return True
    return False


def dtk_version() -> str | None:
    """DTK version string from ``/opt/dtk/.info/rocm_version``."""
    try:
        return _DTK_INFO_VERSION.read_text().strip() or None
    except OSError:
        return None


def _base_gcn_arch(arch: str) -> str:
    """Strip HIP target-feature suffixes (``gfx906:xnack-`` -> ``gfx906``)."""
    return arch.strip().split(":", 1)[0].lower()


def is_valid_gcn_arch(arch: str | None) -> bool:
    """Whether ``arch`` looks like a real GCN arch name.

    Rejects the bogus values a DTK ``amdsmi`` shim can return, e.g.
    ``gfx18446744073709551615``.
    """
    if not arch:
        return False
    return _GCN_ARCH_RE.match(_base_gcn_arch(arch)) is not None


def normalize_gcn_arch(arch: str | None) -> str | None:
    """Canonical GCN arch name for ``arch``, or None if unusable.

    Hygon SMI reports the Z100 as ``gfx916``; the driver, torch and all
    compiled kernels use ``gfx906``. Validating such a name is not enough
    (the capability would stay at (9, 1)), so the alias table above remaps
    it. The input is normalized to lowercase and the HIP target-feature
    suffix is stripped, because callers use the result for substring
    gating. Non-Hygon architectures are returned unchanged.

    Args:
        arch: Raw arch name, possibly with a ``:feature`` suffix.

    Returns:
        The canonical arch name, or None when the input is empty,
        None or syntactically invalid.
    """
    if not arch:
        return None
    base = _base_gcn_arch(arch)
    normalized = _HYGON_DCU_ARCH_ALIAS.get(base, base)
    return normalized if is_valid_gcn_arch(normalized) else None


def arch_from_gfx_target_version(value: int) -> str | None:
    """Map a KFD ``gfx_target_version`` to a GCN arch name."""
    return normalize_gcn_arch(_GFX_TARGET_VERSION_TO_ARCH.get(value))


def _read_kfd_topology() -> list[int]:
    """``gfx_target_version`` for each KFD GPU node, in node order.

    CPU nodes (no ``gfx_target_version`` or zero ``simd_count``) are skipped,
    so index 0 is the first GPU, matching HIP device ordinals.
    """
    versions: list[int] = []
    for path in sorted(glob.glob(_KFD_TOPOLOGY_GLOB)):
        try:
            props = Path(path).read_text()
        except OSError:
            continue
        if re.search(r"^simd_count\s+0$", props, re.MULTILINE):
            continue
        match = re.search(r"^gfx_target_version\s+(\d+)$", props, re.MULTILINE)
        if match is None or int(match.group(1)) == 0:
            # CPU nodes report gfx_target_version 0 as well.
            continue
        versions.append(int(match.group(1)))
    return versions


def gcn_arch_from_kfd(device_index: int = 0) -> str | None:
    """GCN arch for ``device_index`` from the KFD topology, or None.

    Reads sysfs only: no HIP context, no SMI library, no subprocess.
    """
    versions = _read_kfd_topology()
    if device_index >= len(versions):
        return None
    return arch_from_gfx_target_version(versions[device_index])


def _pci_sort_key(card: Path) -> tuple[int, int, int, int]:
    try:
        slot = os.path.basename(os.path.realpath(card / "device"))
        domain, bus, dev_fn = slot.split(":")
        dev, func = dev_fn.split(".")
        return (int(domain, 16), int(bus, 16), int(dev, 16), int(func, 16))
    except (OSError, ValueError):
        return (1 << 16, 1 << 16, 1 << 16, 1 << 16)


@lru_cache(maxsize=1)
def _dcu_drm_cards() -> tuple[str, ...]:
    """DRM card sysfs paths for DCU boards, in HIP ordinal order.

    Matches only ``cardN`` nodes (not connectors such as ``card0-DP-1``),
    keeps Hygon-vendor boards, and orders them by PCI bus/device. When no
    Hygon vendor id is found (newer DTK may change it), any DRM card that
    exposes a VRAM size is accepted instead.
    """
    exact_cards: list[Path] = []
    for path in sorted(glob.glob(_DRM_CARD_GLOB)):
        card = Path(path)
        if not re.fullmatch(r"card\d+", card.name):
            continue
        if (card / "device").exists():
            exact_cards.append(card)

    def vendor_of(card: Path) -> str:
        try:
            return (card / "device" / "vendor").read_text().strip().lower()
        except OSError:
            return ""

    def has_vram(card: Path) -> bool:
        return (card / "device" / "mem_info_vram_total").exists()

    selected = [c for c in exact_cards if vendor_of(c) == _HYGON_PCI_VENDOR]
    if not selected:
        selected = [c for c in exact_cards if has_vram(c)]
    selected.sort(key=_pci_sort_key)
    return tuple(str(card / "device") for card in selected)


def dcu_device_memory_total_bytes(device_index: int = 0) -> int | None:
    """Total VRAM in bytes from sysfs, without initializing a context."""
    return _read_dcu_sysfs(device_index, "mem_info_vram_total", int)


def dcu_device_unique_id(device_index: int = 0) -> str | None:
    """Device UUID from sysfs, without initializing a context."""
    return _read_dcu_sysfs(device_index, "unique_id", str)


def _read_dcu_sysfs(device_index: int, attr: str, transform):
    cards = _dcu_drm_cards()
    if device_index >= len(cards):
        return None
    try:
        raw = (Path(cards[device_index]) / attr).read_text().strip()
    except OSError:
        return None
    try:
        return transform(raw)
    except ValueError:
        return None


def rocminfo_candidates() -> list[str]:
    """Candidate ``rocminfo`` binaries, community ROCm first, DTK second."""
    candidates = ["/opt/rocm/bin/rocminfo"]
    dtk_root = os.environ.get("DTKROOT")
    if dtk_root:
        candidates.append(str(Path(dtk_root) / "bin" / "rocminfo"))
    candidates.append("/opt/dtk/bin/rocminfo")
    candidates.extend(sorted(glob.glob("/opt/dtk*/bin/rocminfo")))
    seen: set[str] = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def gcn_arch_from_torch(device_index: int = 0) -> str | None:
    """GCN arch reported by torch for ``device_index``, or None.

    Uses an integer device index on purpose: DTK's torch deadlocks when
    ``get_device_properties`` is called with the string ``"cuda"``.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        prop = torch.cuda.get_device_properties(device_index)
    except Exception as exc:  # pragma: no cover - depends on runtime state
        logger.debug("Failed to query gcnArchName via torch: %s", exc)
        return None
    arch = getattr(prop, "gcnArchName", None)
    normalized = normalize_gcn_arch(arch)
    if normalized is not None:
        return normalized
    if arch:
        logger.warning_once(
            "torch reported GCN arch %r, which is not a valid arch name. "
            "Set VLLM_DCU_GCN_ARCH to override.",
            arch,
        )
    return None


def gcn_arch(device_index: int = 0) -> str | None:
    """Resolve the current GCN arch, preferring explicit, validated sources.

    Order: ``VLLM_DCU_GCN_ARCH``, torch, KFD topology. The KFD topology is
    last because its packing is not fully documented across architectures.
    """
    override = envs.VLLM_DCU_GCN_ARCH
    if override:
        arch = normalize_gcn_arch(override)
        if arch is None:
            raise ValueError(
                f"VLLM_DCU_GCN_ARCH={override!r} is not a valid GCN arch name."
            )
        return arch

    arch = gcn_arch_from_torch(device_index)
    if arch is not None:
        return arch

    arch = gcn_arch_from_kfd(device_index)
    if arch is not None:
        logger.info_once("Resolved GCN arch %s from the KFD topology.", arch)
    return arch


def bf16_policy() -> str:
    """``VLLM_DCU_BF16`` as one of ``auto``/``on``/``off``."""
    raw = envs.VLLM_DCU_BF16.strip().lower()
    if raw in ("1", "true", "on", "yes", "force"):
        return "on"
    if raw in ("0", "false", "off", "no"):
        return "off"
    return "auto"


@lru_cache(maxsize=1)
def dcu_bf16_probe() -> bool:
    """Numerically probe whether bf16 is usable on this DCU.

    gfx906 has no bf16 instructions: DTK's assembler rejects both
    ``v_cvt_f32_bf16`` and ``v_dot2_f32_bf16`` for gfx906, while accepting
    ``v_pk_fma_f16``/``v_dot2_f32_f16``. Any bf16 math is therefore
    emulated, which can still be numerically correct. Rather than assuming
    either answer, run a small deterministic check of the operations vLLM
    relies on and report the observed result.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return _run_bf16_probe(torch)
    except Exception as exc:  # pragma: no cover - depends on runtime state
        logger.warning(
            "bf16 probe failed on this device (%s: %s); treating bf16 as "
            "unsupported. Set VLLM_DCU_BF16=1 to force it on.",
            type(exc).__name__,
            exc,
        )
        return False


def _run_bf16_probe(torch) -> bool:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(128, 256, generator=generator, dtype=torch.float32).to(device)
    b = torch.randn(256, 128, generator=generator, dtype=torch.float32).to(device)

    a_bf = a.to(torch.bfloat16)
    b_bf = b.to(torch.bfloat16)

    # 1. Storage round-trip: bf16 keeps 8 mantissa bits.
    round_trip_err = (a_bf.to(torch.float32) - a).abs().max().item()
    if round_trip_err > _BF16_MAX_REL_ERR * a.abs().max().item():
        logger.warning_once(
            "bf16 storage round-trip error %.5f is too large; treating bf16 "
            "as unsupported.",
            round_trip_err,
        )
        return False

    # 2. GEMM: bf16 inputs, compare against fp32 math on the same inputs.
    reference = a_bf.to(torch.float32) @ b_bf.to(torch.float32)
    observed = (a_bf @ b_bf).to(torch.float32)
    if not bool(torch.isfinite(observed).all()):
        logger.warning_once("bf16 GEMM produced non-finite values.")
        return False
    scale = reference.abs().max().item()
    gemm_err = (observed - reference).abs().max().item()
    if gemm_err > _BF16_MAX_REL_ERR * scale:
        logger.warning_once(
            "bf16 GEMM deviates from the fp32 reference by %.5f (%.1f%%); "
            "treating bf16 as unsupported.",
            gemm_err,
            100.0 * gemm_err / max(scale, 1e-6),
        )
        return False

    # 3. Reduction-heavy op: softmax.
    reference_sm = torch.softmax(a_bf.to(torch.float32), dim=-1)
    observed_sm = torch.softmax(a_bf, dim=-1).to(torch.float32)
    sm_err = (observed_sm - reference_sm).abs().max().item()
    if sm_err > _BF16_MAX_REL_ERR:
        logger.warning_once("bf16 softmax deviates from fp32 by %.5f.", sm_err)
        return False

    logger.info_once(
        "bf16 probe passed on this device (round-trip %.5f, gemm %.5f, "
        "softmax %.5f).",
        round_trip_err,
        gemm_err,
        sm_err,
    )
    return True


def gfx906_bf16_enabled() -> bool:
    """Whether bf16 should be offered as a supported dtype on gfx906.

    ``VLLM_DCU_BF16=1`` forces it on (useful to experiment), ``=0`` forces
    it off, and the default probes the device.
    """
    policy = bf16_policy()
    if policy == "on":
        return True
    if policy == "off":
        return False
    return dcu_bf16_probe()
