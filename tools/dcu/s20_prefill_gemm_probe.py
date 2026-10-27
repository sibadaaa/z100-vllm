#!/usr/bin/env python3
"""S20: is the FP8 *emulation* the right kernel for prefill-sized M?

The c=4 profile shows the prefill step is 3091 ms of which 1272 ms is
`fp8_emu_gemm_block_kernel` at ~7 ms/call -- 640 GFLOP/s/card, versus the
20 TFLOPS/card the same kernel family reaches in the standalone probe.  The
question this answers: how much of that gap is the *emulation*, i.e. what would
the same GEMM cost if the weights were a native datatype?

Compares, per production shape and per M in {8, 32, 128, 512, 2048}:
  * the production fp8-emu kernel (block-scaled, block_n=block_k=128)
  * a native bf16 `linear` (Tensile/rocBLAS) at the same shape

Usage:
  HIP_VISIBLE_DEVICES=0 python tools/dcu/s20_prefill_gemm_probe.py
"""

import torch  # noqa: F401  (must precede vllm imports)


def bench(fn, iters=15):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> int:
    # importing the wrapper registers torch.ops._rocm_C.fp8_emu_*
    from vllm.model_executor.kernels.linear.scaled_mm import gfx906_fp8  # noqa: F401
    import torch.nn.functional as F

    if not hasattr(torch.ops._rocm_C, "fp8_emu_gemm_block"):
        print("fp8_emu_gemm_block not registered -> cannot probe")
        return 2

    shapes = [(8704, 5120), (5120, 4352), (4096, 5120), (3584, 5120), (5120, 1536)]
    ms = [8, 32, 128, 512, 2048]
    print(f"{'shape':>13}{'M':>6}{'fp8-emu ms':>12}{'bf16 ms':>10}{'ratio':>8}"
          f"{'fp8 TF/s':>11}{'bf16 TF/s':>11}")
    tot = {m: [0.0, 0.0] for m in ms}
    for n, k in shapes:
        w8 = torch.randint(0, 127, (n, k), device="cuda", dtype=torch.uint8)
        w_s = torch.rand((n // 128, k // 128), device="cuda") * 0.01
        lut = torch.randn(256, device="cuda").half()
        wb = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        for m in ms:
            a8 = torch.randint(0, 127, (m, k), device="cuda", dtype=torch.uint8)
            a_s = torch.rand((m, k // 128), device="cuda") * 0.01
            ab = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1

            def f8():
                return torch.ops._rocm_C.fp8_emu_gemm_block(
                    a8, w8, a_s, w_s, lut, None, torch.float16, 0, 128, 128
                )

            def fb():
                return F.linear(ab, wb)

            t8 = bench(f8)
            tb = bench(fb)
            fl = 2.0 * m * n * k
            tot[m][0] += t8
            tot[m][1] += tb
            print(f"{n:>6}x{k:<6}{m:>6}{t8:>12.3f}{tb:>10.3f}{t8 / tb:>7.2f}x"
                  f"{fl / t8 / 1e9:>11.0f}{fl / tb / 1e9:>11.0f}")
    print("\n-- sum over the 5 production shapes (one forward pass each) --")
    print(f"{'M':>6}{'fp8-emu ms':>12}{'bf16 ms':>10}{'ratio':>8}")
    for m in ms:
        print(f"{m:>6}{tot[m][0]:>12.3f}{tot[m][1]:>10.3f}{tot[m][0] / tot[m][1]:>7.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
