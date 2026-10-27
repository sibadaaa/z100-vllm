#!/usr/bin/env python3
"""task-5 evidence: tie the running _rocm_C.abi3.so to the workspace sources.

    docker exec z100-vllm bash -c 'cd /src && python3 tools/dcu/perf1_verify_build.py \
        --out /mnt/game/perf1/verify_build_<change>.json [--label <change>]'

Records, for the artifact the service would load:
  * md5 of /src/vllm/_rocm_C.abi3.so (the symlink the container resolves),
  * md5 of every csrc/rocm source in the workspace AND in the staged build tree
    (/mnt/game/port/gfx906-vllm-211/csrc/rocm) plus whether they differ,
  * python-side hotspot sources (dispatch, GDN op, envs) whose md5 pins the
    python half of the change,
  * the newest build log tail and its mtime (what produced the .so),
  * git HEAD / status when git exists.

Exit status: 0 when the staged tree matches the workspace for the recorded files
(i.e. the built .so can be attributed to the workspace source), 1 otherwise.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

WS = "/src"
STAGE = "/mnt/game/port/gfx906-vllm-211"

C_HEADERS = [
    "csrc/rocm/fp8_emu_gemm.cu",
    "csrc/rocm/fp8_emu_gemm.h",
    "csrc/rocm/gfx906_emu_common.cuh",
    "csrc/rocm/bf16_emu_gemv.cu",
]
PY_SOURCES = [
    "vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py",
    "vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py",
    "vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py",
    "vllm/envs.py",
    "tools/dcu/s11f_e2e_fp8.sh",
]


def md5(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/game/perf1/verify_build.json")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    so_link = os.path.join(WS, "vllm/_rocm_C.abi3.so")
    so_real = os.path.realpath(so_link)
    report = {
        "label": args.label,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "so": {"link": so_link, "realpath": so_real, "md5": md5(so_link),
               "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                      time.localtime(os.path.getmtime(so_real)))
               if os.path.exists(so_real) else None},
        "sources": {}, "staged_mismatch": [], "git": {},
        "build_log": {},
    }
    for rel in C_HEADERS:
        ws_p = os.path.join(WS, rel)
        st_p = os.path.join(STAGE, rel)
        entry = {"ws_md5": md5(ws_p),
                 "staged_md5": md5(st_p) if os.path.exists(st_p) else None}
        entry["identical"] = (entry["staged_md5"] is not None and
                              entry["staged_md5"] == entry["ws_md5"])
        report["sources"][rel] = entry
        if not entry["identical"]:
            report["staged_mismatch"].append(rel)
    # python sources run straight from /src (never staged): pin their md5 only
    for rel in PY_SOURCES:
        report["sources"][rel] = {"ws_md5": md5(os.path.join(WS, rel)),
                                  "staged_md5": None, "identical": None}

    report["git"]["head"] = run(["git", "-C", WS, "rev-parse", "HEAD"])
    report["git"]["status"] = run(["git", "-C", WS, "status", "--porcelain"]).splitlines()

    logs = []
    try:
        for name in os.listdir("/mnt/game/port"):
            if name.startswith("build-") and name.endswith(".log"):
                p = os.path.join("/mnt/game/port", name)
                logs.append((os.path.getmtime(p), p))
    except OSError:
        pass
    if logs:
        _, newest = max(logs)
        report["build_log"] = {
            "path": newest,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(newest))),
            "tail": run(["tail", "-5", newest]).splitlines(),
        }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[verify_build] so md5={report['so']['md5']} realpath={so_real}")
    print(f"[verify_build] staged mismatches: {report['staged_mismatch'] or 'none'}")
    print(f"[verify_build] build log: {report['build_log'].get('path')} "
          f"@{report['build_log'].get('mtime')}")
    print(f"[saved] {args.out}")
    return 1 if report["staged_mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
