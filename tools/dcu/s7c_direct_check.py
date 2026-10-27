# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7c: direct ``_C`` activation checks for the gfx906 fp16 packed-SiLU fix.

Three probes, all bypassing the Python-side fallback in
``vllm/model_executor/layers/activation.py`` so that they measure the compiled
extension itself (``docs/result/z100-07c-packed-silu-csrc-fix-result.md``):

1. ``_C.silu_and_mul`` on fp16: 20 identical launches, non-finite count and
   max abs error against ``silu(gate) * up`` for every launch;
2. ``_C.silu_and_mul_with_clamp``: one launch, same metrics;
3. ``apply_moe_activation(MoEActivation.SILU)``: the plain-function call site in
   ``vllm/model_executor/layers/fused_moe/activation.py`` that the ``CustomOp``
   fallback cannot reach.

    python tools/dcu/s7c_direct_check.py
"""

import hashlib
import importlib

import torch  # noqa: F401  (must come first on DTK, see todolist hard rule 8)

import vllm._C  # noqa: F401
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)


def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


so = importlib.import_module("vllm._C").__file__
stable_so = importlib.import_module("vllm._C_stable_libtorch").__file__
props = torch.cuda.get_device_properties(0)
print(f"[s7c] device={props.name} gcnArchName={getattr(props, 'gcnArchName', '?')}")
print(f"[s7c] _C={so} md5={_md5(so)} size={__import__('os').path.getsize(so)}")
print(
    f"[s7c] _C_stable_libtorch={stable_so} md5={_md5(stable_so)} "
    f"size={__import__('os').path.getsize(stable_so)}"
)

torch.manual_seed(1234)
rows, cols = 8, 6144
x = ((torch.rand(rows, cols, device="cuda") * 2 - 1) * 5.9).half()
d = cols // 2
exact_gate = torch.nn.functional.silu(x[:, :d].float()) * x[:, d:].float()


def metrics(out: torch.Tensor, ref: torch.Tensor) -> tuple[int, int, float]:
    o = out.float()
    nonfinite = int((~torch.isfinite(o)).sum())
    err = (o - ref).abs()
    err = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))
    return nonfinite, int((err > 1e-2).sum()), float(err.max())


print(f"=== _C.silu_and_mul fp16 {rows}x{cols} (d={d}), 20 launches ===")
masks = []
nonfinites = []
for trial in range(20):
    out = torch.empty((rows, d), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    torch.cuda.synchronize()
    nonfinite, nbad, maxerr = metrics(out, exact_gate)
    nonfinites.append(nonfinite)
    err = (out.float() - exact_gate).abs()
    err = torch.where(
        torch.isfinite(err), err, torch.full_like(err, float("inf"))
    )
    masks.append(err > 1e-2)
    print(
        f"[s7c-rep] trial={trial:>2} nonfinite={nonfinite:<6} "
        f"nbad>1e-2={nbad:<6} maxerr={maxerr:.6g}"
    )
same = sum(1 for m in masks[1:] if bool(torch.equal(m, masks[0])))
print(
    f"[s7c-rep] masks identical to trial 0: {same}/19; nonfinite across 20 "
    f"runs: min={min(nonfinites)} max={max(nonfinites)} sum={sum(nonfinites)}"
)

print("=== _C.silu_and_mul_with_clamp fp16 (limit=5.9), 1 launch ===")
gate_c = torch.clamp(x[:, :d], max=5.9)
up_c = torch.clamp(x[:, d:], min=-5.9, max=5.9)
ref_c = torch.nn.functional.silu(gate_c.float()) * up_c.float()
out_c = torch.empty((rows, d), dtype=x.dtype, device=x.device)
# Schema: silu_and_mul_with_clamp(result, input, limit, alpha=1.0, beta=0.0).
torch.ops._C.silu_and_mul_with_clamp(out_c, x, 5.9, 1.0, 0.0)
torch.cuda.synchronize()
nonfinite, nbad, maxerr = metrics(out_c, ref_c)
print(
    f"[s7c-clamp] nonfinite={nonfinite}/{out_c.numel()} nbad>1e-2={nbad} "
    f"maxerr={maxerr:.6g}"
)

print("=== apply_moe_activation(SILU) fp16, 1 launch ===")
torch.manual_seed(3)
mrows, md = 32, 3072
xm = ((torch.rand(mrows, 2 * md, device="cuda") * 2 - 1) * 5.9).half()
outm = torch.empty((mrows, md), dtype=torch.float16, device="cuda")
apply_moe_activation(MoEActivation.SILU, outm, xm)
torch.cuda.synchronize()
refm = torch.nn.functional.silu(xm[:, :md].float()) * xm[:, md:].float()
nonfinite, nbad, maxerr = metrics(outm, refm)
print(
    f"[s7c-moe] nonfinite={nonfinite}/{outm.numel()} nbad>1e-2={nbad} "
    f"maxerr={maxerr:.6g}"
)

print("=== control: bf16 / fp32 / mul_and_silu must stay exact ===")
for dtype in (torch.bfloat16, torch.float32):
    xd = x.to(dtype)
    refd = torch.nn.functional.silu(xd[:, :d].float()) * xd[:, d:].float()
    outd = torch.empty((rows, d), dtype=dtype, device=x.device)
    torch.ops._C.silu_and_mul(outd, xd)
    torch.cuda.synchronize()
    nonfinite, nbad, maxerr = metrics(outd, refd)
    print(
        f"[s7c-ctl] silu_and_mul {dtype} nonfinite={nonfinite} "
        f"nbad>1e-2={nbad} maxerr={maxerr:.6g}"
    )
ref_swap = x[:, d:].float() * torch.nn.functional.silu(x[:, :d].float())
out_swap = torch.empty((rows, d), dtype=x.dtype, device=x.device)
torch.ops._C.mul_and_silu(out_swap, torch.cat([x[:, d:], x[:, :d]], dim=-1).contiguous())
torch.cuda.synchronize()
nonfinite, nbad, maxerr = metrics(out_swap, ref_swap)
print(
    f"[s7c-ctl] mul_and_silu fp16 (half-swapped) nonfinite={nonfinite} "
    f"nbad>1e-2={nbad} maxerr={maxerr:.6g}"
)
print("[s7c] DONE")
