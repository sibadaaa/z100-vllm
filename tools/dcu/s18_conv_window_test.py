#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18: does the gfx906 MTP conv kernel pick the CORRECT history window when the
# conv-state slot is wider than width-1 (the speculative layout)?
#
#   HIP_VISIBLE_DEVICES=0 python /src/tools/dcu/s18_conv_window_test.py
#
# engine evidence (appendix K/L): with width=4 the non-spec conv state holds 3
# positions, the spec one holds 4, and the spec slot looks like
#   [extra_older, ctl_pos0, ctl_pos1, ctl_pos2]
# The kernel reads `history = state[acc-1 : acc+width-2]`, i.e. positions
# [0,1,2] for acc=1.  Ground truth for "predict the token after the newest
# history token" must use the *most recent* width-1 positions = [1,2,3].
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops

DIM, WIDTH, STATE_LEN = 16, 4, 4   # state_len = 4 > width-1 = 3 (speculative)
torch.manual_seed(3)
dev = "cuda"
weight = torch.randn(DIM, WIDTH, device=dev, dtype=torch.float16) * 0.3
bias = torch.randn(DIM, device=dev, dtype=torch.float16) * 0.1
state = torch.randn(1, DIM, STATE_LEN, device=dev, dtype=torch.float16) * 0.5
x = torch.randn(1, DIM, device=dev, dtype=torch.float16) * 0.5


def conv_out(hist):  # hist: [dim, width-1]
    acc = (hist.float() * weight.float()[:, :-1]).sum(-1)
    acc = acc + x[0].float() * weight.float()[:, -1] + bias.float()
    return F.silu(acc).to(torch.float16)


last3 = state[0][:, 1:4]   # most recent width-1 positions (expected correct)
first3 = state[0][:, 0:3]  # what `acc-1 = 0` would select
out_last, out_first = conv_out(last3), conv_out(first3)

def run(mode):
    import os
    if mode == "triton":
        os.environ["VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE"] = "1"
    else:
        os.environ.pop("VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE", None)
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update as pub,
    )
    return pub(
        x=x, conv_state=state.clone(), weight=weight, bias=bias,
        activation="silu",
        conv_state_indices=torch.tensor([0], device=dev, dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1], device=dev, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], device=dev, dtype=torch.int32),
        max_query_len=1,
    )


out_raw = ops.causal_conv1d_gfx906_mtp_update(
    x=x, conv_state=state.clone(), weight=weight, bias=bias,
    state_indices=torch.tensor([0], device=dev, dtype=torch.int32),
    cu_seqlens=torch.tensor([0, 1], device=dev, dtype=torch.int32),
    num_accepted_tokens=torch.tensor([1], device=dev, dtype=torch.int32),
    pad_slot_id=-1, silu_activation=True,
)
# post-fix layout: the widened slot carries the newest width-1 tokens at the FRONT
fixed_state = state.clone()
fixed_state[0][:, : WIDTH - 1] = state[0][:, 1:4]
fixed_state[0][:, WIDTH - 1 :] = 0.0
out_fixed = ops.causal_conv1d_gfx906_mtp_update(
    x=x, conv_state=fixed_state, weight=weight, bias=bias,
    state_indices=torch.tensor([0], device=dev, dtype=torch.int32),
    cu_seqlens=torch.tensor([0, 1], device=dev, dtype=torch.int32),
    num_accepted_tokens=torch.tensor([1], device=dev, dtype=torch.int32),
    pad_slot_id=-1, silu_activation=True,
)
d_fixed = (out_fixed[0].float() - out_last.float()).abs().max().item()

print(f"state_len={STATE_LEN} width={WIDTH} (width-1={WIDTH-1}) acc=1")
for label, o in (("custom_op(raw _C)", out_raw[0]), ("public_api(gfx906)", run("gfx906")[0]),
                 ("public_api(triton)", run("triton")[0])):
    dl = (o.float() - out_last.float()).abs().max().item()
    df = (o.float() - out_first.float()).abs().max().item()
    print(f"  {label:<22} |out-conv(last3)|={dl:.3e}  |out-conv(first3)|={df:.3e}  -> "
          + ("LAST3 (correct window)" if dl < df else "FIRST3 (shifted)"))
print(f"  FIXED front-aligned slot  |out - conv(last3)| = {d_fixed:.3e}  -> "
      + ("PASS" if d_fixed < 1e-3 else "FAIL"))
