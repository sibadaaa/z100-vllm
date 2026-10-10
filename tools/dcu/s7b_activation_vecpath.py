# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: does the corruption come from the vectorized (``__half2``) path?

``csrc/libtorch_stable/activation_kernels.cu:250-256`` picks the implementation
with ``use_vec = (d % vec_size == 0)`` where ``vec_size = ARCH_MAX_VEC_SIZE /
element_size`` (``VecTraits<false>`` -> 16 / 2 = 8 for fp16 on gfx906).  The
vectorized branch calls ``packed_silu_kernel<__half2>`` (``:171``); the scalar
branch calls ``silu_kernel<half>`` (``:165``).

Sweeping ``d`` therefore selects the two implementations for the *same* data,
which isolates the defect and shows whether a non-vectorizable shape is safe.

    python tools/dcu/s7b_activation_vecpath.py
"""

import torch  # noqa: F401

import vllm._C  # noqa: F401

torch.manual_seed(0)
rows = 4
print("d     d%8  path   nonfinite  maxerr      nbad>1e-2")
for d in (3064, 3068, 3070, 3072, 3076, 3080, 1024, 1020, 512, 510):
    x = ((torch.rand(rows, 2 * d, device="cuda") * 2 - 1) * 5.9).half()
    out = torch.empty((rows, d), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    torch.cuda.synchronize()
    ref = torch.nn.functional.silu(x[:, :d].float()) * x[:, d:].float()
    o = out.float()
    e = (o - ref).abs()
    e = torch.where(torch.isfinite(e), e, torch.full_like(e, float("inf")))
    path = "vec  " if d % 8 == 0 else "scalar"
    print(
        f"{d:<5} {d % 8:<4} {path} {int((~torch.isfinite(o)).sum()):<10} "
        f"{e.max().item():<11.6g} {int((e > 1e-2).sum())}"
    )
print("[s7b-vec] DONE")
