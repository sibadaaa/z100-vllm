# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: is the gfx906 fp16 ``_C.silu_and_mul`` corruption deterministic?

A deterministic codegen error would corrupt the same elements on every launch
with the same input; the gfx906 SFU/VALU hazard described in
``docs/result/z100-06-triton-gfx906-result.md`` is scheduling dependent and
therefore expected to move around.  Also sweeps a dense grid of ``gate``
values (with ``up == 1``) to show the error as a function of the input.

    python tools/dcu/s7b_activation_repeat.py
"""

import torch  # noqa: F401

import vllm._C  # noqa: F401

torch.manual_seed(1234)
rows, cols = 8, 6144
x = ((torch.rand(rows, cols, device="cuda") * 2 - 1) * 5.9).half()
d = cols // 2
gate, up = x[:, :d], x[:, d:]
exact = torch.nn.functional.silu(gate.float()) * up.float()

masks = []
print("=== 20 identical launches, same input ===")
for trial in range(20):
    out = torch.empty((rows, d), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    torch.cuda.synchronize()
    o = out.float()
    err = (o - exact).abs()
    err = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))
    mask = err > 1e-2
    masks.append(mask)
    print(
        f"[s7b-rep] trial={trial:>2} nonfinite={int((~torch.isfinite(o)).sum()):<5} "
        f"nbad>1e-2={int(mask.sum()):<6} maxerr={err.max().item():.6g}"
    )

m0 = masks[0]
same = sum(1 for m in masks[1:] if bool(torch.equal(m, m0)))
union = torch.zeros_like(m0)
for m in masks:
    union |= m
inter = masks[0].clone()
for m in masks[1:]:
    inter &= m
print(
    f"[s7b-rep] masks identical to trial 0: {same}/19   union_bad={int(union.sum())} "
    f"intersection_bad={int(inter.sum())} trial0_bad={int(m0.sum())}"
)

print("=== gate grid sweep with up == 1 (out must be silu(gate)) ===")
g = torch.linspace(-6, 6, 3072, device="cuda").half().unsqueeze(0)
one = torch.ones_like(g)
xo = torch.cat([g, one], dim=-1).contiguous()
o = torch.empty_like(g)
torch.ops._C.silu_and_mul(o, xo)
torch.cuda.synchronize()
ex = torch.nn.functional.silu(g.float())
err = (o.float() - ex).abs()
bad = err > 1e-2
print(f"[s7b-rep] grid bad={int(bad.sum())}/{bad.numel()} maxerr={err.max().item():.6g}")
idx = bad.flatten().nonzero().flatten().tolist()
print(f"[s7b-rep] bad gate range: {g.flatten()[idx].min().item():.4g}..{g.flatten()[idx].max().item():.4g}")
for i in idx[:10]:
    print(f"[s7b-rep] gate={g.flatten()[i].item():>10.5g} kernel={o.flatten()[i].item():>12.6g} exact={ex.flatten()[i].item():>12.6g}")
print("[s7b-rep] DONE")
