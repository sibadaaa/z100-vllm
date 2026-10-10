# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: localize the gfx906 fp16 activation corruption inside the kernel.

``_C.silu_and_mul`` and ``_C.mul_and_silu`` instantiate the *same*
``act_and_mul_kernel`` with the *same* ``silu_kernel`` and only differ in the
``act_first`` template parameter::

    act_first = true   -> out = silu_kernel(x) * y     (csrc/activation_kernels.cu:22)
    act_first = false  -> out = x * silu_kernel(y)

Feeding ``mul_and_silu`` the swapped input ``cat([up, gate])`` therefore
computes exactly the same mathematical expression as ``silu_and_mul`` with
``cat([gate, up])``.  If one is exact and the other corrupt, the defect is in
the code generated for one template instantiation, not in ``silu_kernel``.

Also checks whether the kernel writes *permuted* values (a read/write indexing
problem) rather than wrong arithmetic.

    python tools/dcu/s7b_activation_localize.py
"""

import torch  # noqa: F401

import vllm._C  # noqa: F401

torch.manual_seed(0)
rows, cols = 8, 6144
x = ((torch.rand(rows, cols, device="cuda") * 2 - 1) * 5.9).half()
d = cols // 2
gate, up = x[:, :d], x[:, d:]
exact = (torch.nn.functional.silu(gate.float()) * up.float())


def report(name, out):
    o = out.float()
    err = (o - exact).abs()
    err = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))
    print(
        f"[s7b-loc] {name:<34} nonfinite={int((~torch.isfinite(o)).sum())}/{o.numel()} "
        f"maxerr={err.max().item():.6g} nbad>1e-2={int((err > 1e-2).sum())}"
    )
    return out


a = torch.empty((rows, d), dtype=x.dtype, device=x.device)
torch.ops._C.silu_and_mul(a, x)
torch.cuda.synchronize()
report("silu_and_mul(gate,up)", a)

b = torch.empty((rows, d), dtype=x.dtype, device=x.device)
torch.ops._C.mul_and_silu(b, torch.cat([up, gate], dim=-1).contiguous())
torch.cuda.synchronize()
report("mul_and_silu(up,gate) [same math]", b)

print(f"[s7b-loc] a==b exactly: {bool(torch.equal(a, b))}  max|a-b|={float((a.float() - b.float()).abs().max()):.6g}")

# Are the wrong values *permuted* copies of the correct ones?
af, bf = a.float(), b.float()
bad = (af - exact).abs()
bad = torch.where(torch.isfinite(bad), bad, torch.full_like(bad, float("inf"))) > 1e-2
perm = 0
tot = int(bad.sum())
for r in range(rows):
    cols_bad = bad[r].nonzero().flatten().tolist()
    for i in cols_bad[:400]:
        hit = (exact[r] == af[r, i]).nonzero().flatten()
        if hit.numel():
            perm += 1
print(f"[s7b-loc] bad elements whose value equals some *other* column's exact value: {perm}/{tot}")

# Same check against the mul_and_silu (correct) output, i.e. is it a shuffle?
perm2 = 0
for r in range(rows):
    for i in bad[r].nonzero().flatten().tolist()[:400]:
        if (bf[r] == af[r, i]).nonzero().numel():
            perm2 += 1
print(f"[s7b-loc] bad elements matching some column of the correct output: {perm2}/{tot}")

# Does the corruption depend on the *values* of up?  silu_kernel(x) alone, via
# mul_and_silu with up == 1, isolates the activation result.
one = torch.ones_like(up)
probe = torch.cat([one, gate], dim=-1).contiguous()  # out = 1 * silu(gate)
s = torch.empty((rows, d), dtype=x.dtype, device=x.device)
torch.ops._C.mul_and_silu(s, probe)
torch.cuda.synchronize()
err = (s.float() - torch.nn.functional.silu(gate.float())).abs()
print(f"[s7b-loc] mul_and_silu(1,gate) -> silu(gate) maxerr={err.max().item():.6g}")

# and the exact template the broken one uses: silu_kernel(x) * up, computed by
# mul_and_silu with x == 1 is not possible, so emulate with act_first=true via
# silu_and_mul on a tensor whose *second half* is all ones.
probe2 = torch.cat([gate, one], dim=-1).contiguous()
s2 = torch.empty((rows, d), dtype=x.dtype, device=x.device)
torch.ops._C.silu_and_mul(s2, probe2)
torch.cuda.synchronize()
err2 = (s2.float() - torch.nn.functional.silu(gate.float())).abs()
print(f"[s7b-loc] silu_and_mul(gate,1) -> silu(gate) maxerr={err2.max().item():.6g}")

# up == 0 must give exactly 0 for both instantiations.
zero = torch.zeros_like(up)
z1 = torch.empty((rows, d), dtype=x.dtype, device=x.device)
torch.ops._C.silu_and_mul(z1, torch.cat([gate, zero], dim=-1).contiguous())
torch.cuda.synchronize()
print(f"[s7b-loc] silu_and_mul(gate,0) all-zero: {bool((z1 == 0).all())} nonfinite={int((~torch.isfinite(z1.float())).sum())}")
print("[s7b-loc] DONE")
