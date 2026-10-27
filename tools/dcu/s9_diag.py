#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S9 (A8) DTK / gfx906 runtime diagnostic.

Run *inside* a DTK container with the target stack on PYTHONPATH:

    source /opt/dtk/env.sh
    source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
    python /mnt/game/port/gfx906-vllm-211/tools/dcu/s9_diag.py

It answers the A8 checklist from docs/design/z100-09-acceptance-and-bench.md:
DTK version, torch.version.hip, per-card arch from *both* torch and the KFD
topology, the normalize_gcn_arch() result, the `_rocm_C` operator surface,
rocBLAS / hipBLASLt gfx906 coverage, LD_LIBRARY_PATH order, rsmi_init
resolvability and the VLLM_DCU_* switches.

Exit code 0 iff every hard assertion passes. Hard rule 8: `import torch` must
come before `import vllm`.
"""

import glob
import json
import os
import re
import subprocess
import sys

import torch  # noqa: F401  (must precede `import vllm`, hard rule 8)

SECTION = []


def say(line: str = "") -> None:
    print(line, flush=True)
    SECTION.append(line)


def header(title: str) -> None:
    say()
    say(f"----- {title} -----")


def read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def main() -> int:
    failures = []
    report = {}

    header("1) DTK / runtime versions")
    dtk_root = os.environ.get("DTK_ROOT", "/opt/dtk")
    dtk_version = read_text(f"{dtk_root}/.info/rocm_version") or "<missing>"
    say(f"DTK_ROOT            : {dtk_root}")
    say(f"DTK version         : {dtk_version}")
    say(f"torch.__version__   : {torch.__version__}")
    say(f"torch.version.hip   : {torch.version.hip}")
    say(f"torch.version.cuda  : {torch.version.cuda}")
    try:
        import importlib.metadata as md

        for pkg in ("torch", "triton", "vllm", "compressed-tensors"):
            try:
                say(f"{pkg:20s}: {md.version(pkg)}")
            except md.PackageNotFoundError:
                say(f"{pkg:20s}: <not installed>")
    except Exception as exc:  # pragma: no cover - diagnostics only
        say(f"importlib.metadata    : {exc!r}")
    report["dtk_version"] = dtk_version
    report["torch"] = torch.__version__
    report["hip"] = torch.version.hip

    header("2) per-card arch: torch vs KFD topology")
    cards = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cards.append({"index": i, "name": props.name, "gcnArchName": props.gcnArchName})
        say(f"  card {i}: {props.name}  {props.gcnArchName}")
    kfd = []
    for path in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        value = None
        for line in (read_text(path) or "").splitlines():
            if line.startswith("gfx_target_version "):
                value = line.split(None, 1)[1].strip()
        if value and value != "0":
            kfd.append({"node": path.split("/")[-2], "gfx_target_version": value})
    for entry in kfd:
        say(f"  KFD node {entry['node']}: gfx_target_version="
            f"{entry['gfx_target_version']}")
    report["cards"] = cards
    report["kfd"] = kfd

    if torch.cuda.device_count() < 1:
        failures.append("torch sees no GPU")
    for card in cards:
        if not card["gcnArchName"].startswith("gfx906"):
            failures.append(f"card {card['index']} arch {card['gcnArchName']} != gfx906")
    # KFD topology is host-wide: 4 nodes with gfx_target_version 90106 on Z100.
    if kfd and (
        len(kfd) != 4 or any(e["gfx_target_version"] != "90106" for e in kfd)
    ):
        failures.append(f"KFD topology not 4x 90106: {kfd}")

    header("3) normalize_gcn_arch() / platform predicates")
    from vllm.platforms.hygon_dtk import normalize_gcn_arch  # noqa: E402
    from vllm.platforms.rocm import (  # noqa: E402
        _GCN_ARCH,
        _capability_from_gcn_arch,
        get_cdna_version,
        on_cdna,
        on_gfx906,
        on_gfx9,
    )

    cases = {
        "gfx906": "gfx906",
        "gfx916": "gfx906",
        "gfx906:xnack-": "gfx906",
        "gfx18446744073709551615": None,
        None: None,
    }
    for raw, expect in cases.items():
        got = normalize_gcn_arch(raw)
        say(f"  normalize_gcn_arch({raw!r:28}) = {got!r} (expect {expect!r})")
        if got != expect:
            failures.append(f"normalize_gcn_arch({raw!r}) = {got!r}, expected {expect!r}")
    from vllm.platforms.hygon_dtk import arch_from_gfx_target_version  # noqa: E402

    _from_kfd = arch_from_gfx_target_version(90106)
    say(f"  arch_from_gfx_target_version(90106)  = {_from_kfd!r} (expect 'gfx906')")
    if _from_kfd != "gfx906":
        failures.append(f"arch_from_gfx_target_version(90106) = {_from_kfd!r}")
    say(f"  _GCN_ARCH                  = {_GCN_ARCH!r}")
    say(f"  capability                 = {_capability_from_gcn_arch(_GCN_ARCH)!r}")
    say(f"  on_gfx906/on_cdna/on_gfx9  = "
        f"{on_gfx906()}/{on_cdna()}/{on_gfx9()}")
    say(f"  get_cdna_version()         = {get_cdna_version()}")
    say("  note: on_cdna() is upstream vLLM's substring test "
        "('gfx9' in 'gfx906') -> True on gfx906; get_cdna_version()==0 is the "
        "value the version-gated paths use.")
    report["gcn_arch"] = _GCN_ARCH
    report["on_gfx906"] = on_gfx906()
    report["on_cdna"] = on_cdna()
    report["on_gfx9"] = on_gfx9()
    if _GCN_ARCH != "gfx906" or not on_gfx906() or on_gfx9():
        failures.append("platform predicates wrong")

    header("4) `_rocm_C` operator surface")
    schemas = sorted(
        s.name for s in torch._C._jit_get_all_schemas() if s.name.startswith("_rocm_C::")
    )
    say(f"  registered: {schemas}")
    report["rocm_c_ops"] = schemas
    if schemas != ["_rocm_C::paged_attention"]:
        failures.append(f"unexpected _rocm_C surface: {schemas}")

    header("5) rocBLAS / hipBLASLt coverage")
    rocblas = sorted(
        os.path.basename(p)
        for p in glob.glob(f"{dtk_root}/lib/rocblas/library/*gfx*.dat")
        + glob.glob(f"{dtk_root}/lib/rocblas/library/*gfx*.co")
    )
    hipblaslt = sorted(
        os.path.basename(p) for p in glob.glob(f"{dtk_root}/lib/hipblaslt/library/*.co")
    )
    say(f"  rocBLAS library    : {rocblas}")
    say(f"  hipBLASLt library  : {hipblaslt}")
    say(f"  rocBLAS gfx906     : {[p for p in rocblas if 'gfx906' in p]}")
    say(f"  hipBLASLt gfx906   : {[p for p in hipblaslt if 'gfx906' in p]}")
    report["rocblas"] = rocblas
    report["hipblaslt"] = hipblaslt
    if not any("gfx906" in p for p in rocblas):
        failures.append("rocBLAS has no gfx906 entry at all")
    if any("gfx906" in p for p in hipblaslt):
        failures.append("hipBLASLt unexpectedly ships gfx906 kernels")

    header("6) LD_LIBRARY_PATH order")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for i, entry in enumerate(ld.split(":")):
        say(f"  [{i:2d}] {entry}")
    report["ld_library_path"] = ld.split(":")
    say(f"  ROCBLAS_TENSILE_LIBPATH = "
        f"{os.environ.get('ROCBLAS_TENSILE_LIBPATH', '<unset>')}")

    header("7) rsmi_init resolvability")
    rsmi_report = []
    for lib in sorted(
        glob.glob(f"{dtk_root}/.hyhal/rocm_smi/lib/librocm_smi64.so*")
        + glob.glob(f"{dtk_root}/lib/librocm_smi64.so*")
        + glob.glob(f"{dtk_root}/lib64/librocm_smi64.so*")
    ):
        out = subprocess.run(
            ["nm", "-D", lib], capture_output=True, text=True, check=False
        ).stdout
        has = re.search(r"\bT rsmi_init\b", out) is not None
        rsmi_report.append({"lib": lib, "rsmi_init": has})
        say(f"  {'EXPORTS' if has else 'stub   '} rsmi_init  {lib}")
    report["rsmi"] = rsmi_report

    header("8) VLLM_DCU_* / vLLM environment")
    dcu_env = {k: v for k, v in os.environ.items() if k.startswith(("VLLM_DCU_",
                                                                    "VLLM_ROCM_",
                                                                    "VLLM_CACHE_ROOT",
                                                                    "TRITON_",
                                                                    "HIP_VISIBLE"))}
    for key in sorted(dcu_env):
        say(f"  {key}={dcu_env[key]}")
    report["env"] = dcu_env
    try:
        import vllm.envs as envs

        say(f"  VLLM_DCU_GCN_ARCH (envs)     : {envs.VLLM_DCU_GCN_ARCH!r}")
        say(f"  VLLM_DCU_BF16 (envs)         : {envs.VLLM_DCU_BF16!r}")
        say(f"  VLLM_ROCM_USE_SKINNY_GEMM    : {envs.VLLM_ROCM_USE_SKINNY_GEMM!r}")
        say(f"  VLLM_CACHE_ROOT (envs)       : {envs.VLLM_CACHE_ROOT!r}")
    except Exception as exc:  # pragma: no cover
        say(f"  import vllm.envs failed: {exc!r}")

    header("9) verdict")
    if failures:
        for item in failures:
            say(f"  FAIL {item}")
        say("S9_DIAG rc=1")
    else:
        say("  all A8 checks passed")
        say("S9_DIAG rc=0")
    report["failures"] = failures
    if os.environ.get("S9_DIAG_JSON"):
        with open(os.environ["S9_DIAG_JSON"], "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
