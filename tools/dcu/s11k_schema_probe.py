#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11k: does `torch.library.custom_op` accept the sdot4 signature
(`Tensor?, ScalarType`) and stay usable under FakeTensor + eager?

Also checks that a *second* registration in the same process is refused, which
is why the kernel module must guard its registration.
"""

import torch


def main() -> None:
    print("torch", torch.__version__)

    @torch.library.custom_op("s11kprobe::sdot4", mutates_args=())
    def sdot4(
        x_q: torch.Tensor,
        w: torch.Tensor,
        x_s: torch.Tensor,
        w_s: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        gemv_max_m: int,
    ) -> torch.Tensor:
        out = torch.empty((x_q.shape[0], w.shape[0]), dtype=out_dtype, device=x_q.device)
        return out

    @sdot4.register_fake
    def _(x_q, w, x_s, w_s, bias, out_dtype, gemv_max_m):
        return x_q.new_empty((x_q.shape[0], w.shape[0]), dtype=out_dtype)

    print("schema:", sdot4._schema)
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode()
    with mode:
        xq = mode.from_tensor(torch.empty(7, 64, dtype=torch.int8, device="cuda"))
        w = mode.from_tensor(torch.empty(32, 64, dtype=torch.int8, device="cuda"))
        xs = mode.from_tensor(torch.empty(7, dtype=torch.float32, device="cuda"))
        ws = mode.from_tensor(torch.empty(32, dtype=torch.float32, device="cuda"))
        o = sdot4(xq, w, xs, ws, None, torch.float16, 8)
        print("fake out:", tuple(o.shape), o.dtype)
    # eager call on the real device
    xq = torch.zeros(7, 64, dtype=torch.int8, device="cuda")
    w = torch.zeros(32, 64, dtype=torch.int8, device="cuda")
    xs = torch.zeros(7, dtype=torch.float32, device="cuda")
    ws = torch.zeros(32, dtype=torch.float32, device="cuda")
    bias = torch.zeros(32, dtype=torch.float16, device="cuda")
    o = sdot4(xq, w, xs, ws, bias, torch.float16, 8)
    print("eager out:", tuple(o.shape), o.dtype, float(o.abs().sum()))
    o2 = sdot4(xq, w, xs, ws, None, torch.float16, 8)
    print("eager out (bias=None):", tuple(o2.shape))

    # torch.compile through it, dynamic dims, two different M
    torch._dynamo.mark_dynamic(xq, 0)
    torch._dynamo.mark_dynamic(xs, 0)

    def f(a, b, c, d, pos):
        o = sdot4(a, b, c, d, None, torch.float16, 8)
        nt = pos.shape[1]
        return o, pos.reshape(-1).view(nt, -1)

    pos = torch.arange(3 * 7, device="cuda").reshape(3, 7)
    torch._dynamo.mark_dynamic(pos, 1)
    cf = torch.compile(f, dynamic=True)
    r1 = cf(xq, w, xs, ws, pos)
    xq2 = torch.zeros(2048, 64, dtype=torch.int8, device="cuda")
    xs2 = torch.zeros(2048, dtype=torch.float32, device="cuda")
    pos2 = torch.arange(3 * 2048, device="cuda").reshape(3, 2048)
    r2 = cf(xq2, w, xs2, ws, pos2)
    print("compiled M=7  ->", tuple(r1[0].shape), tuple(r1[1].shape))
    print("compiled M=2048 ->", tuple(r2[0].shape), tuple(r2[1].shape))


if __name__ == "__main__":
    main()
