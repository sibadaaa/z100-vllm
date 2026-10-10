# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: which device kernel does ``_C.silu_and_mul`` actually launch?

Names the kernel with the torch profiler and (for reference) checks whether the
corruption follows the input *values* or fixed output *lanes*: it re-runs the
op on the same input, then on a column-shuffled copy of that input.

    python tools/dcu/s7b_kernel_identity.py
"""

import torch  # noqa: F401
from torch.profiler import ProfilerActivity, profile

import vllm._C  # noqa: F401

torch.manual_seed(7)
rows, cols = 4, 6144
x = ((torch.rand(rows, cols, device="cuda") * 2 - 1) * 5.9).half()
d = cols // 2
out = torch.empty((rows, d), dtype=x.dtype, device=x.device)
torch.ops._C.silu_and_mul(out, x)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    torch.ops._C.silu_and_mul(out, x)
    torch.cuda.synchronize()
for evt in prof.key_averages():
    if evt.device_type is not None and "act" in evt.key or "kernel" in evt.key.lower():
        print(f"[s7b-id] kernel={evt.key!r} count={evt.count} cuda_time_us={evt.device_time}")
print("[s7b-id] all kernels:", [e.key for e in prof.key_averages() if e.count])

gate, up = x[:, :d], x[:, d:]
exact = torch.nn.functional.silu(gate.float()) * up.float()


def badmask(t):
    o = t.float()
    e = (o - exact).abs()
    e = torch.where(torch.isfinite(e), e, torch.full_like(e, float("inf")))
    return e > 1e-2, int((~torch.isfinite(o)).sum())


m1, nf1 = badmask(out)
o2 = torch.empty_like(out)
torch.ops._C.silu_and_mul(o2, x)
torch.cuda.synchronize()
m2, nf2 = badmask(o2)
print(f"[s7b-id] same input twice: identical bad mask={bool(torch.equal(m1, m2))} nonfinite {nf1} vs {nf2}")

# column-shuffled input: are the same *columns* bad, or do bad columns follow values?
perm = torch.randperm(d, device="cuda")
xs = torch.cat([gate[:, perm], up[:, perm]], dim=-1).contiguous()
os_ = torch.empty_like(out)
torch.ops._C.silu_and_mul(os_, xs)
torch.cuda.synchronize()
exact_s = torch.nn.functional.silu(xs[:, :d].float()) * xs[:, d:].float()
e = (os_.float() - exact_s).abs()
e = torch.where(torch.isfinite(e), e, torch.full_like(e, float("inf")))
ms = e > 1e-2
print(
    f"[s7b-id] shuffled input: nonfinite={int((~torch.isfinite(os_.float())).sum())} "
    f"bad={int(ms.sum())} (baseline {int(m1.sum())})"
)
print(
    f"[s7b-id] columns bad in both: {int((m1.any(0) & ms.any(0)).sum())} / "
    f"baseline_cols={int(m1.any(0).sum())} shuffled_cols={int(ms.any(0).sum())}"
)
print("[s7b-id] DONE")
