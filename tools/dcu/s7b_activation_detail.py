# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: per-element view of the gfx906 fp16 ``_C.silu_and_mul`` corruption.

Compares the kernel output against the exact fp32 expression
``gate / (1 + exp(-gate)) * up`` element by element and prints the inputs,
the kernel result, the exact result and the intermediate ``exp(-gate)`` so the
arithmetic that goes wrong can be identified.

    python tools/dcu/s7b_activation_detail.py
"""

import torch  # noqa: F401

import vllm._C  # noqa: F401

torch.manual_seed(0)
rows, cols = 1, 6144
x = ((torch.rand(rows, cols, device="cuda") * 2 - 1) * 5.9).half()
d = cols // 2
gate = x[:, :d]
up = x[:, d:]

out = torch.empty((rows, d), dtype=x.dtype, device=x.device)
torch.ops._C.silu_and_mul(out, x)
torch.cuda.synchronize()

gf, uf = gate.float(), up.float()
sig = 1.0 / (1.0 + torch.exp(-gf))
ref = sig * gf * uf
of = out.float()
err = (of - ref).abs()
errf = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))
bad = errf > 1e-2
print(f"[s7b-det] bad={int(bad.sum())}/{bad.numel()} nonfinite={int((~torch.isfinite(of)).sum())}")

idx = bad.nonzero().flatten()[:12]
print(f"{'col':>6} {'gate':>12} {'up':>12} {'kernel_out':>14} {'exact':>14} {'exp(-gate)':>12} {'1+exp':>12}")
for i in idx.tolist():
    g, u, o = gf[0, i].item(), uf[0, i].item(), of[0, i].item()
    e = float(torch.exp(-gf[0, i]).item())
    print(f"{i:>6} {g:>12.6g} {u:>12.6g} {o:>14.6g} {ref[0, i].item():>14.6g} {e:>12.6g} {1 + e:>12.6g}")

print("=== same elements recomputed through a triton/eager equivalent ===")
d_ = d


def torch_silu_mul(g8, u8):
    return torch.nn.functional.silu(g8) * u8


print(
    "[s7b-det] eager fp16 F.silu(gate)*up equals exact within",
    (torch_silu_mul(gate, up).float() - ref).abs().max().item(),
)
print(
    "[s7b-det] kernel_out/up for bad cols:",
    (of[0, idx] / uf[0, idx]).tolist()[:12],
    " exact silu:",
    (sig[0, idx] * gf[0, idx]).tolist()[:12],
)
print("[s7b-det] DONE")
