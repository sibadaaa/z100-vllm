# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b minimal repro: fp16 ``_C.silu_and_mul`` is numerically corrupt on gfx906.

    python tools/dcu/s7b_min_repro.py

``d % 8 == 0`` selects the vectorized ``__half2`` branch of
``csrc/libtorch_stable/activation_kernels.cu`` (``use_vec``), which is the one
the eager (``CompilationMode.NONE``) path uses because ``custom_ops`` defaults
to ``all`` there.
"""

import torch  # noqa: F401  (must come first on DTK, todolist hard rule 8)

import vllm._C  # noqa: F401  (loads the kernels behind torch.ops._C)

d = 3072
x = ((torch.rand(8, 2 * d, device="cuda") * 2 - 1) * 5.9).half()
out = torch.empty(8, d, dtype=torch.float16, device="cuda")
torch.ops._C.silu_and_mul(out, x)
torch.cuda.synchronize()

o = out.float()
ref = torch.nn.functional.silu(x[:, :d].float()) * x[:, d:].float()
safe = torch.nan_to_num(o, posinf=1e9, neginf=-1e9)
native = torch.nn.functional.silu(x[:, :d]) * x[:, d:]
print("[repro] _C.silu_and_mul  nonfinite =", int((~torch.isfinite(o)).sum()), "/", o.numel())
print("[repro] _C.silu_and_mul  max_abs_err =", (safe - ref).abs().max().item())
print("[repro] F.silu(gate)*up  max_abs_err =", (native.float() - ref).abs().max().item())
