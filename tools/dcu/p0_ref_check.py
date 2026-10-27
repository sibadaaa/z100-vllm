#!/usr/bin/env python3
"""Check saved P0 ref outputs against a fresh fp32 dequant reference."""
import sys

import torch

sys.path.insert(0, "/home/lyh/z100-vllm")
import vllm._rocm_C  # noqa: F401,E402

from tools.dcu.p0_gemv_bench import build_case  # noqa: E402
from tools.dcu.p0_gemv_probe import dequant_ref  # noqa: E402


def main():
    name, k, n, m, route = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                            int(sys.argv[4]), sys.argv[5])
    a, w, a_s, w_s, lut = build_case(k, n, m, route, "cuda")
    ref = dequant_ref(a, w, a_s, w_s, route, k)
    saved = torch.load(f".s27/p0-ref/{route}_{name}_{k}x{n}_M{m}.pt",
                       weights_only=True).cuda()
    d = (saved.float() - ref).abs()
    print(f"{route}_{name}_{k}x{n}_M{m}: saved-vs-dequant max={d.max().item():.6f}")
    print(f"  saved[0,:3]={saved.float()[0,:3].tolist()}")
    print(f"  dequant[0,:3]={ref[0,:3].tolist()}")


if __name__ == "__main__":
    main()
