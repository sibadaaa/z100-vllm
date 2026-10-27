#!/usr/bin/env python3
"""Kernel-name census for one arm: which fp8 emu kernel did the env select?

The O4 gate is a *host* selection rule, so its only observable proof is the
kernel name that runs at M=8 for the 5120:1536 shape.  kernel-dev's falsifier,
used verbatim:

  with ``VLLM_DCU_FP8_STREAM_MAX_M=8`` the M=8 census must show
  ``fp8_emu_gemm_block_wide2_kernel<64,8,32,1,2,Half>`` for all five production
  shapes including ``5120:1536``, and **no**
  ``fp8_emu_gemv_stream_block_kernel<8,3,Half>``.  If the stream kernel still
  appears at M=8 the env did not reach the process and the arm is INVALID.

This reuses ``.s23/probes/o3_mprobe.py``'s memory-safe allocation recipe (the
four Z100s are held by production; only ~64 MiB allocations succeed) instead of
re-deriving it, and only adds a torch.profiler view of the kernel names.

  docker exec z100-regress ... .venv/bin/python tools/dcu/s23_kernel_census.py \
      --out <arm>/census-kernels.json
"""

import argparse
import collections
import importlib.util
import json
import os
import sys

PROBE = os.environ.get("S23_MPROBE", "/src/.s23/probes/o3_mprobe.py")
SHAPES = [(8704, 5120), (5120, 4352), (4096, 5120), (3584, 5120), (5120, 1536)]
STREAM_RX = "fp8_emu_gemv_stream_block_kernel"
WIDE_RX = "wide2"


def load_probe(path):
    spec = importlib.util.spec_from_file_location("s23_o3_mprobe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--expect-gate", choices=("on", "off", "any"), default="any")
    args = ap.parse_args()

    import torch

    m = load_probe(PROBE)
    tdir = getattr(torch, "profiler", None)
    if tdir is None:
        print("[census] torch.profiler unavailable", file=sys.stderr)
        return 2

    lut = m.lut_for()
    buf = m.make_wbuf()
    stream_max = os.environ.get("VLLM_DCU_FP8_STREAM_MAX_M", "<unset>")
    decode_cfg = os.environ.get("VLLM_DCU_FP8_DECODE_CFG", "<unset>")
    per_shape = {}
    total = collections.Counter()

    for n, k in SHAPES:
        w = m.w_view(buf, n, k)
        w_s = m.make_w_s(w)
        a_q, a_s = m.make_a(args.m, k)
        fn = m.make_call("prod", a_q, w, a_s, w_s, lut)
        fn()
        torch.cuda.synchronize()
        with tdir.profile(activities=[tdir.ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        names = collections.Counter()
        for evt in prof.key_averages():
            if getattr(evt, "self_device_time_total", 0):
                names[evt.key] += int(evt.count)
        per_shape[f"{n}:{k}"] = dict(names)
        total.update(names)
        del a_q, a_s, w_s, w

    keys = " ".join(sorted(total))
    has_stream = STREAM_RX in keys
    has_wide = WIDE_RX in keys
    verdict = "unknown"
    if args.expect_gate == "on":
        verdict = "gate-on-ok" if (has_wide and not has_stream) else "gate-on-FAILED"
    elif args.expect_gate == "off":
        verdict = "gate-off-ok" if has_stream else "gate-off-FAILED"
    else:
        verdict = "recorded"

    payload = {
        "decode_cfg": decode_cfg,
        "stream_max_m": stream_max,
        "m": args.m,
        "per_shape": per_shape,
        "total_counts": dict(total),
        "has_stream_kernel": has_stream,
        "has_wide2_kernel": has_wide,
        "expect_gate": args.expect_gate,
        "verdict": verdict,
    }
    print(f"[census] decode_cfg={decode_cfg} stream_max_m={stream_max} "
          f"m={args.m} wide2={has_wide} stream={has_stream} verdict={verdict}")
    for name, count in sorted(total.items()):
        print(f"[census]   {count:4d} x {name}")
    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[census] saved -> {args.out}")
    return 0 if verdict != "gate-on-FAILED" and not verdict.endswith("FAILED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
