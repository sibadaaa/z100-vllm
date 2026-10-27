#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18: multi-step evolution test for the gfx906 speculative causal-conv update.
#
#   HIP_VISIBLE_DEVICES=0 python /src/tools/dcu/s18_conv_mtp_sequence.py
#
# The existing single-call test (tests/kernels/mamba/test_causal_conv1d.py
# ::test_gfx906_causal_conv1d_mtp_update, 48 passed) only checks ONE call. Real
# speculative decoding calls this op once per step with a *sliding* conv state
# and a varying (query_len, num_accepted), so a state-write-back / offset bug
# can survive that test. This probe runs a multi-step sequence through:
#   A) the gfx906 MTP custom op (default)
#   B) the fallback path (VLLM_QWEN35_MTP_CONV_CUSTOM=0)
#   C) a torch reference implementing the documented semantics
# and prints, per step, the max |out - ref| and max |state - ref| for A and B.
import os

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

DIM, WIDTH, STATE_LEN, SLOTS = 37, 4, 12, 2
STEPS = [(2, 1), (1, 1), (1, 2), (2, 2), (3, 1)]  # (query_len, num_accepted)
SEED = 7


def build():
    torch.manual_seed(SEED)
    weight = (torch.randn(DIM, WIDTH, device="cuda", dtype=torch.float16) * 0.2)
    bias = (torch.randn(DIM, device="cuda", dtype=torch.float16) * 0.1)
    state = (torch.randn(SLOTS, DIM, STATE_LEN, device="cuda",
                         dtype=torch.float32) * 0.2)
    torch.manual_seed(SEED + 1)
    xs = [torch.randn(q, DIM, device="cuda", dtype=torch.float16) * 0.5
          for q, _ in STEPS]
    idx = torch.tensor([1], device="cuda", dtype=torch.int32)
    return weight, bias, state, xs, idx


def reference(weight, bias, state, xs, idx, steps):
    """Documented semantics of the MTP path (mirrors the single-call test)."""
    s = int(idx[0])
    st = state.clone()
    outs = []
    for (q, acc), x in zip(steps, xs):
        history = st[s, :, acc - 1: acc + WIDTH - 2].float()
        out = []
        for tok in x.float():
            o = (history * weight.float()[:, :-1]).sum(-1)
            o = o + tok * weight.float()[:, -1] + bias.float()
            out.append(F.silu(o))
            history = torch.cat((history[:, 1:], tok[:, None]), dim=1)
        outs.append(torch.stack(out).to(torch.float16))
        new = st.clone()
        new[s, :, : WIDTH - 2] = st[s, :, acc: acc + WIDTH - 2]
        new[s, :, WIDTH - 2: WIDTH - 2 + q] = x.transpose(0, 1)
        st = new
    return outs, st


def run(mode):
    weight, bias, state, xs, idx = build()
    st = state.clone()
    outs = []
    for (q, acc), x in zip(STEPS, xs):
        if mode == "fallback":
            os.environ["VLLM_QWEN35_MTP_CONV_CUSTOM"] = "0"
        else:
            os.environ.pop("VLLM_QWEN35_MTP_CONV_CUSTOM", None)
        if mode == "triton":
            # Bypass the gfx906 fallback so the upstream triton
            # causal_conv1d_update implementation runs (mainline semantics).
            os.environ["VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE"] = "1"
        else:
            os.environ.pop("VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE", None)
        qsl = torch.tensor([0, q], device="cuda", dtype=torch.int32)
        accepted = torch.tensor([acc], device="cuda", dtype=torch.int32)
        out = causal_conv1d_update(
            x=x, conv_state=st, weight=weight, bias=bias, activation="silu",
            conv_state_indices=idx, num_accepted_tokens=accepted,
            query_start_loc=qsl, max_query_len=q,
        )
        outs.append(out)
    return outs, st


def main() -> int:
    if not torch.cuda.is_available():
        print("needs a HIP device")
        return 2
    print(f"steps (query_len, num_accepted) = {STEPS}  dim={DIM} width={WIDTH} "
          f"state_len={STATE_LEN}")
    ref_outs, ref_state = reference(*build(), STEPS)
    for mode in ("custom", "triton", "fallback"):
        try:
            outs, st = run(mode)
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            print(f"[{mode}] EXCEPTION {type(exc).__name__}: {exc}")
            continue
        print(f"--- {mode}")
        for i, (o, r) in enumerate(zip(outs, ref_outs)):
            do = (o.float() - r.float()).abs().max().item()
            print(f"    step {i} (q={STEPS[i][0]}, acc={STEPS[i][1]}): "
                  f"max|out-ref|={do:.4e}")
        ds = (st.float() - ref_state.float()).abs().max().item()
        print(f"    final state max|state-ref|={ds:.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
