# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: is ``_C.silu_and_mul`` correct on gfx906 / DTK 26.04?

The eager (``CompilationMode.NONE``) path of Qwen3-0.6B dispatches
``SiluAndMul`` to ``torch.ops._C.silu_and_mul``
(``vllm/model_executor/layers/activation.py:138`` -> ``forward_cuda``), while the
``VLLM_COMPILE`` path uses ``forward_native`` (``F.silu(gate) * up``).  This
script feeds both implementations the same fp16 tensor and reports the
mismatch, the non-finite count and the reproducibility over repeated calls.

    python tools/dcu/s7b_activation_probe.py [--dump <saved_input.pt>]
"""

import argparse

import torch  # noqa: F401

import vllm._C  # noqa: F401  (registers torch.ops._C.*)

silu_and_mul = torch.ops._C.silu_and_mul

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default=None, help="saved (num_tokens, 2*d) fp16 input")
ap.add_argument("--rows", type=int, default=5)
ap.add_argument("--cols", type=int, default=6144)
args = ap.parse_args()

torch.manual_seed(0)
if args.dump:
    x = torch.load(args.dump, map_location="cuda", weights_only=True).cuda().half()
    print(f"[s7b-act] input from {args.dump} shape={tuple(x.shape)}")
else:
    x = ((torch.rand(args.rows, args.cols, device="cuda") * 2 - 1) * 5.922).half()
    print(f"[s7b-act] synthetic input shape={tuple(x.shape)}")

d = x.shape[-1] // 2
xf = x.float()
print(f"[s7b-act] input absmax={xf.abs().max().item():.6g} finite={torch.isfinite(xf).all().item()}")

ref = torch.nn.functional.silu(xf[:, :d]) * xf[:, d:]

for trial in range(4):
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    silu_and_mul(out, x)
    torch.cuda.synchronize()
    o = out.float()
    nbad = int((~torch.isfinite(o)).sum())
    oo = torch.where(torch.isfinite(o), o, torch.zeros_like(o))
    diff = (oo - ref).abs()
    print(
        f"[s7b-act] trial={trial} _C.silu_and_mul  nonfinite={nbad}/{o.numel()} "
        f"absmax={oo.abs().max().item():.6g} max_abs_err_vs_native={diff.max().item():.6g}"
    )

native = torch.nn.functional.silu(x[:, :d]) * x[:, d:]
print(
    f"[s7b-act] native fp16 absmax={native.float().abs().max().item():.6g} "
    f"nonfinite={int((~torch.isfinite(native.float())).sum())}"
)

# Same math in pure eager torch, in fp32, to show what the value must be.
for name, fn in (
    ("torch.sigmoid*fp32", lambda t: (torch.sigmoid(t[:, :d].float()) * t[:, d:].float())),
    ("torch.sigmoid*fp16", lambda t: (torch.sigmoid(t[:, :d]) * t[:, d:])),
):
    r = fn(x)
    rf = r.float()
    print(f"[s7b-act] {name}: absmax={rf.abs().max().item():.6g} nonfinite={int((~torch.isfinite(rf)).sum())}")

print("[s7b-act] DONE")
