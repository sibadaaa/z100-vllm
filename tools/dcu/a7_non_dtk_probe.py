#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A7 (S11a) probe: vLLM's ROCm platform behaviour when DTK is not present.

The A7 gate ("behaviour on machines without DTK is unchanged") is measured by
running the *same* probe against two source trees in the same container:

* the canonical fork copy with all of our changes, and
* a pristine ``HEAD`` worktree,

and diffing the JSON lines printed here.  Every probe result is wrapped so a
raising call is reported as ``{"ok": false, "exc": "..."}`` instead of killing
the process: "throws the same exception as HEAD" is a valid outcome.

Run it with the tree under test first on ``PYTHONPATH`` (or as ``cwd``)::

    python3 tools/dcu/a7_non_dtk_probe.py env
    python3 tools/dcu/a7_non_dtk_probe.py plugin
    python3 tools/dcu/a7_non_dtk_probe.py rocm --torch-arch gfx942:sramecc+:xnack-

The script is GPU-free.  It never initialises a HIP context unless the caller
passes ``--torch-arch``, in which case ``torch.cuda`` is stubbed *before*
``vllm`` is imported so the module-level ``_GCN_ARCH`` resolution can be driven
deterministically.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import types

# Hard rule 8: import torch before vllm, otherwise importing vllm in a DTK
# container aborts with std::bad_alloc / SIGABRT.
import torch  # noqa: F401  (import order matters)

_ENV_KEYS = (
    "DTKROOT",
    "HYHAL_PATH",
    "ROCM_PATH",
    "VLLM_DCU_GCN_ARCH",
    "VLLM_DCU_BF16",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "TRITON_CACHE_DIR",
    "LD_LIBRARY_PATH",
)


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)


def probe(fn):
    """Run ``fn`` and report its value or the exception it raised."""
    try:
        return {"ok": True, "value": fn()}
    except BaseException as exc:  # noqa: BLE001 - the exception is the datum
        return {"ok": False, "exc": f"{type(exc).__name__}: {exc}"}


def _env_snapshot() -> dict:
    snap = {key: os.environ.get(key) for key in _ENV_KEYS}
    # LD_LIBRARY_PATH is long; keep only whether the DTK lib dirs are present.
    ld = snap.pop("LD_LIBRARY_PATH") or ""
    snap["LD_LIBRARY_PATH_has_dtk"] = "/opt/dtk" in ld
    snap["LD_LIBRARY_PATH_has_hyhal"] = "/opt/hyhal" in ld
    return snap


def _install_torch_stub(arch: str, name: str) -> None:
    """Make ``torch.cuda`` report ``arch`` without a HIP context."""
    props = types.SimpleNamespace(
        gcnArchName=arch,
        name=name,
        total_memory=192 * 1024**3,
        multi_processor_count=304,
    )
    torch.cuda.get_device_properties = lambda index=0: props
    torch.cuda.get_device_name = lambda index=0: name


def _install_fake_amdsmi(arch: str, fail: str = "") -> None:
    """Synthesise a community-ROCm-style amdsmi with ``arch`` as the GPU.

    ``fail`` is a comma-separated list of queries that raise
    ``AmdSmiException`` instead of answering: ``uuid``, ``mem``, ``topo``,
    ``handles``, ``init``.  It drives the error paths of
    ``RocmPlatform.get_device_uuid()`` / ``get_device_total_memory()`` /
    ``is_fully_connected()``.
    """
    failing = {part for part in fail.split(",") if part}
    mod = types.ModuleType("amdsmi")

    class AmdSmiException(Exception):
        pass

    class AmdSmiMemoryType:
        VRAM = 0

    # Four handles so a logical device remapped by HIP_VISIBLE_DEVICES still
    # lands inside the list.
    handles = [object() for _ in range(4)]

    def boom(what):
        def raiser(*a, **k):
            raise AmdSmiException(f"fake amdsmi failure: {what}")

        return raiser

    mod.AmdSmiException = AmdSmiException
    mod.AmdSmiMemoryType = AmdSmiMemoryType
    mod.amdsmi_init = boom("init") if "init" in failing else (lambda *a, **k: None)
    mod.amdsmi_shut_down = lambda *a, **k: None
    mod.amdsmi_get_processor_handles = (
        boom("handles") if "handles" in failing else (lambda: handles)
    )
    mod.amdsmi_get_gpu_asic_info = lambda h: {
        "target_graphics_version": arch,
        "device_id": "0x74a1",
        "market_name": "Fake Radeon/Instinct",
    }
    mod.amdsmi_get_gpu_device_uuid = (
        boom("uuid") if "uuid" in failing else (lambda h: "fake-amdsmi-uuid")
    )
    mod.amdsmi_get_gpu_memory_total = (
        boom("mem") if "mem" in failing else (lambda h, t: 192 * 1024**3)
    )
    mod.amdsmi_topo_get_link_type = (
        boom("topo") if "topo" in failing else (lambda a, b: {"hops": 1, "type": 2})
    )
    mod.amdsmi_topo_get_numa_node_number = lambda h: 0
    sys.modules["amdsmi"] = mod


def mode_env(args) -> None:
    try:
        from vllm.platforms import hygon_dtk as h
    except BaseException as exc:  # noqa: BLE001
        # HEAD has no such module; that is itself a (documented) datum.
        emit(
            {
                "probe": "env",
                "env": _env_snapshot(),
                "hygon_dtk_module": None,
                "import_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return

    emit(
        {
            "probe": "env",
            "env": _env_snapshot(),
            "hygon_dtk_module": h.__file__,
            "is_hygon_dtk": h.is_hygon_dtk(),
            "dtk_version": h.dtk_version(),
            "dtk_marker_is_file": h._DTK_INFO_VERSION.is_file(),
            "dtk_marker_exists": h._DTK_INFO_VERSION.exists(),
            "hy_smi_glob": glob.glob("/opt/dtk*/bin/hy-smi"),
            "rocminfo_candidates": h.rocminfo_candidates(),
            "gcn_arch": probe(lambda: h.gcn_arch()),
            "bf16_policy": h.bf16_policy(),
            "gfx906_bf16_enabled": probe(h.gfx906_bf16_enabled),
        }
    )


def mode_plugin(args) -> None:
    import vllm.platforms as platforms

    if args.drop_visible_devices:
        # The import above runs _sync_hip_cuda_env_vars(), which may populate
        # these from each other; drop them so the amdsmi/rocminfo detection
        # path is the one under test.
        for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            os.environ.pop(var, None)

    selected = None
    try:
        from vllm.platforms import current_platform

        selected = type(current_platform).__name__
    except BaseException as exc:  # noqa: BLE001
        selected = f"{type(exc).__name__}: {exc}"

    emit(
        {
            "probe": "plugin",
            "env": _env_snapshot(),
            "rocm_platform_plugin": probe(platforms.rocm_platform_plugin),
            "current_platform": selected,
        }
    )


def mode_rocm(args) -> None:
    if args.block_amdsmi:
        # Any subsequent ``import amdsmi`` raises ImportError.
        sys.modules["amdsmi"] = None
    if args.fake_amdsmi_arch:
        _install_fake_amdsmi(args.fake_amdsmi_arch, args.fake_amdsmi_fail)
    if args.torch_arch:
        _install_torch_stub(args.torch_arch, args.torch_name)

    payload: dict = {"probe": "rocm", "env": _env_snapshot()}

    try:
        import vllm.platforms.rocm as rocm
    except BaseException as exc:  # noqa: BLE001 - the exception is the datum
        # e.g. the torch fallback in _get_gcn_arch() with no GPU present.
        payload["import_error"] = f"{type(exc).__name__}: {exc}"
        emit(payload)
        return

    payload["tree"] = rocm.__file__
    payload["has_amdsmi"] = getattr(rocm, "_HAS_AMDSMI", "<absent>")
    payload["gcn_arch_module_level"] = probe(lambda: rocm._GCN_ARCH)
    payload["get_gcn_arch"] = probe(rocm._get_gcn_arch)
    payload["on_gfx906"] = probe(rocm.on_gfx906)
    payload["on_cdna"] = probe(rocm.on_cdna)
    payload["supported_dtypes"] = probe(
        lambda: [str(d) for d in rocm._rocm_supported_dtypes()]
    )
    payload["device_capability"] = probe(
        lambda: str(rocm.RocmPlatform.get_device_capability(0))
    )
    payload["device_name"] = probe(lambda: rocm.RocmPlatform.get_device_name(0))
    payload["device_uuid"] = probe(lambda: rocm.RocmPlatform.get_device_uuid(0))
    payload["device_total_memory"] = probe(
        lambda: rocm.RocmPlatform.get_device_total_memory(0)
    )
    payload["is_fully_connected"] = probe(
        lambda: rocm.RocmPlatform.is_fully_connected([0, 1])
    )
    payload["use_rocm_custom_paged_attention"] = probe(
        lambda: rocm.use_rocm_custom_paged_attention(
            torch.float16, 128, 16, 8, 4096, 0, "auto"
        )
    )

    if args.stub_arch:
        rocm._GCN_ARCH = args.stub_arch
        rocm._ON_GFX906 = "gfx906" in args.stub_arch
        rocm._ON_CDNA = any(a in args.stub_arch for a in ["gfx9", "gfx1250"])
        rocm.use_rocm_custom_paged_attention.cache_clear()
        # get_device_capability() is lru_cached and was already called above
        # with the module-level arch; drop it so the stub is what is measured.
        rocm.RocmPlatform.get_device_capability.cache_clear()
        payload["stub_arch"] = args.stub_arch
        payload["stub_on_gfx906"] = rocm.on_gfx906()
        payload["stub_on_cdna"] = rocm.on_cdna()
        payload["stub_supported_dtypes"] = probe(
            lambda: [str(d) for d in rocm._rocm_supported_dtypes()]
        )
        payload["stub_custom_paged_attention"] = probe(
            lambda: rocm.use_rocm_custom_paged_attention(
                torch.float16, 128, 16, 8, 4096, 0, "auto"
            )
        )
        payload["stub_device_capability"] = probe(
            lambda: str(rocm.RocmPlatform.get_device_capability(0))
        )

    emit(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_env = sub.add_parser("env", help="DTK detection / arch sources")
    p_env.set_defaults(func=mode_env)

    p_plugin = sub.add_parser("plugin", help="rocm_platform_plugin()")
    p_plugin.add_argument("--drop-visible-devices", action="store_true")
    p_plugin.set_defaults(func=mode_plugin)

    p_rocm = sub.add_parser("rocm", help="RocmPlatform surface")
    p_rocm.add_argument("--torch-arch", default=None)
    p_rocm.add_argument("--torch-name", default="Fake GPU")
    p_rocm.add_argument("--block-amdsmi", action="store_true")
    p_rocm.add_argument("--fake-amdsmi-arch", default=None)
    p_rocm.add_argument("--fake-amdsmi-fail", default="")
    p_rocm.add_argument("--stub-arch", default=None)
    p_rocm.set_defaults(func=mode_rocm)

    parser.add_argument(
        "--stub-device-name",
        default=None,
        help=(
            "Patch torch.cuda.get_device_name only. The DTK triton build queries "
            "the device name while being imported, so a GPU-less container needs "
            "this to import vllm at all. It does not touch arch resolution."
        ),
    )

    args = parser.parse_args()
    if args.stub_device_name:
        torch.cuda.get_device_name = lambda index=0: args.stub_device_name
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
