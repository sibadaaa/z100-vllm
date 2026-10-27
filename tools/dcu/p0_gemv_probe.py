#!/usr/bin/env python3
"""Isolated P0 stream-kernel correctness probe: op output vs fp32 dequant ref."""
import os
import sys

sys.path.insert(0, "/home/lyh/z100-vllm")
import vllm._rocm_C  # noqa: F401,E402

from tools.dcu.p0_gemv_bench import build_case, run_op  # noqa: E402


def dequant_ref(a, w, a_s, w_s, route, k):
    if route == "w8a16":
        a_f = a.float()
    else:
        a_f = a.float().reshape(a.shape[0], k // 128, 128)
        a_f = a_f * a_s[:, :, None]
        a_f = a_f.reshape(a.shape[0], k)
    wf = w.float().reshape(w.shape[0] // 128, 128, k // 128, 128)
    wf = wf * w_s[:, None, :, None]
    wf = wf.reshape(w.shape[0], k)
    return a_f @ wf.t()


def main():
    k, n, m = (int(x) for x in sys.argv[1:4])
    route = sys.argv[4] if len(sys.argv) > 4 else "w8a16"
    a, w, a_s, w_s, lut = build_case(k, n, m, route, "cuda")
    out = run_op(route, a, w, a_s, w_s, lut)
    ref = dequant_ref(a, w, a_s, w_s, route, k)
    d = (out.float() - ref).abs()
    print(f"route={route} K={k} N={n} M={m} STREAM="
          f"{os.environ.get('VLLM_DCU_FP8_GEMV_STREAM', 'unset')}")
    print(f"  max|delta|={d.max().item():.6f}  mean|delta|={d.mean().item():.6f}")
    print(f"  out[0,:4] ={out.float()[0, :4].tolist()}")
    print(f"  ref[0,:4] ={ref[0, :4].tolist()}")
    # per-row max delta: which rows are wrong?
    if m > 1:
        prow = d.max(dim=1).values
        print(f"  per-row max: {[round(x, 4) for x in prow.tolist()]}")


if __name__ == "__main__":
    main()
