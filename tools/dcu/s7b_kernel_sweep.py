# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: fp16 correctness sweep of the ``vllm._C`` kernels the eager path uses.

The eager (``CompilationMode.NONE``) path dispatches ``CustomOp`` layers to
their HIP kernels in ``vllm/_C.abi3.so`` instead of the PyTorch-native
implementations that ``VLLM_COMPILE`` selects.  ``_C.silu_and_mul`` is known to
be corrupt on gfx906 (see ``s7b_activation_probe.py``); this script checks the
other fp16 kernels Qwen3-0.6B reaches in that mode against fp32 references so
the blast radius of the defect is known.

    python tools/dcu/s7b_kernel_sweep.py
"""

import torch  # noqa: F401

import vllm._C  # noqa: F401

torch.manual_seed(0)
DEV = "cuda"


def report(name, out, ref):
    o, r = out.float(), ref.float()
    err = (o - r).abs()
    err = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))
    print(
        f"[s7b-sweep] {name:<34} nonfinite={int((~torch.isfinite(o)).sum()):<6}/{o.numel():<8} "
        f"maxerr={err.max().item():.6g} nbad>1e-2={int((err > 1e-2).sum())}"
    )


def rms_ref(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * w.float()


def fused_ref(x, res, w, eps):
    s = x.float() + res.float()
    var = s.pow(2).mean(-1, keepdim=True)
    return s * torch.rsqrt(var + eps) * w.float(), s


for dtype in (torch.float16, torch.bfloat16, torch.float32):
    print(f"--- dtype={dtype} ---")
    x = (torch.randn(64, 1024, device=DEV) * 2).to(dtype)
    w = (torch.rand(1024, device=DEV) + 0.5).to(dtype)
    eps = 1e-6

    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, w, eps)
    torch.cuda.synchronize()
    report("_C.rms_norm", out, rms_ref(x, w, eps))

    x2, res = x.clone(), (torch.randn(64, 1024, device=DEV) * 2).to(dtype)
    torch.ops._C.fused_add_rms_norm(x2, res, w, eps)
    torch.cuda.synchronize()
    ref, ref_res = fused_ref(x, res, w, eps)
    report("_C.fused_add_rms_norm out", x2, ref)
    report("_C.fused_add_rms_norm res", res, ref_res)

    g = (torch.randn(64, 2048, device=DEV) * 2).to(dtype)
    o1 = torch.empty(64, 1024, dtype=dtype, device=DEV)
    torch.ops._C.silu_and_mul(o1, g)
    torch.cuda.synchronize()
    report("_C.silu_and_mul (known bad)", o1, torch.nn.functional.silu(g[:, :1024].float()) * g[:, 1024:].float())

print("[s7b-sweep] DONE")
