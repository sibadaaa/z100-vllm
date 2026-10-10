# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: the MoE activation entry point calls the broken kernel directly.

``vllm/model_executor/layers/fused_moe/activation.py:356`` calls
``torch.ops._C.silu_and_mul`` from a plain function (``apply_moe_activation``),
not from a ``CustomOp``, so the ``CustomOp``-level gfx906 fallback in
``vllm/model_executor/layers/activation.py`` cannot intercept it.  This script
proves that entry point is still corrupt on gfx906 and shows the exact call to
patch.

    python tools/dcu/s7b_moe_activation_check.py
"""

import torch  # noqa: F401

import vllm._C  # noqa: F401
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)

torch.manual_seed(3)
rows, d = 32, 3072
x = ((torch.rand(rows, 2 * d, device="cuda") * 2 - 1) * 5.9).half()
out = torch.empty((rows, d), dtype=torch.float16, device="cuda")
apply_moe_activation(MoEActivation.SILU, out, x)
torch.cuda.synchronize()

o = out.float()
ref = torch.nn.functional.silu(x[:, :d].float()) * x[:, d:].float()
safe = torch.nan_to_num(o, posinf=1e9, neginf=-1e9)
print(
    f"[s7b-moe] apply_moe_activation(SILU) fp16 gfx906: "
    f"nonfinite={int((~torch.isfinite(o)).sum())}/{o.numel()} "
    f"max_abs_err={(safe - ref).abs().max().item():.6g}"
)
