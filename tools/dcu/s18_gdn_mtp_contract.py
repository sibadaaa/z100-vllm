#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18: apples-to-apples contract check for the gfx906 GDN MTP kernel.
#
#   HIP_VISIBLE_DEVICES=0 python /src/tools/dcu/s18_gdn_mtp_contract.py
#
# Semantics (read from csrc/mamba/gfx906_decode_kernels.cu:544-563): the kernel
# walks the WHOLE speculative token chain; `num_accepted_tokens - 1` only picks
# the column of the state-index table that token 0 reads its starting state
# from, and token t>0 reads the slot token t-1 just wrote.  The torch eager
# reference (vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:47-147)
# implements the same thing with the state in [slots, kv_heads, V, K] layout.
#
# So the meaningful comparison is: identical inputs (same acc, same per-token
# slot table, same starting state) through the HIP kernel vs the torch eager
# reference -> compare all outputs and every written state slot.
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    _fused_sigmoid_gating_delta_rule_update_gfx906_eager as eager_ref,
)

T, H, HV, K, V, SLOTS = 4, 2, 4, 16, 16, 6
SEED = 11


def build():
    torch.manual_seed(SEED)
    dev = "cuda"

    def half(*shape, scale=0.2):
        return torch.randn(*shape, device=dev, dtype=torch.float16) * scale

    A_log = half(HV)
    a, b = half(1, T, HV), half(1, T, HV)
    dt_bias = half(HV, scale=0.1)
    q, k = half(1, T, H, K, scale=2.5), half(1, T, H, K, scale=2.5)
    v = half(1, T, HV, V, scale=2.5)
    # persistent cache is KxV; both callers hand over the [slots, HV, V, K] view
    state = torch.randn(SLOTS, HV, K, V, device=dev, dtype=torch.float32) * 0.2
    return A_log, a, b, dt_bias, q, k, v, state


def main() -> int:
    if not torch.cuda.is_available():
        print("needs a HIP device")
        return 2
    # distinct slot per token position: exercises the per-token write-back
    idx = torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.int32)
    A_log, a, b, dt_bias, q, k, v, state = build()
    cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    print(f"T={T} heads={H} v_heads={HV} K={K} V={V} slots={SLOTS} "
          f"state_indices={idx.tolist()}")
    touched = [int(x) for x in idx[0]]
    for acc_len in (1, 2, 3):
        acc = torch.tensor([acc_len], device="cuda", dtype=torch.int32)
        st_op = state.clone()
        out_op = ops.fused_sigmoid_gating_delta_rule_gfx906_mtp_update(
            A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
            state=st_op.transpose(-1, -2), state_indices=idx, cu_seqlens=cu,
            num_accepted_tokens=acc, use_qk_l2norm_in_kernel=True,
        )
        st_ref = state.clone()
        out_ref, _ = eager_ref(
            A_log, a, b, dt_bias, q, k, v, 1.0, 20.0, K ** -0.5,
            st_ref.transpose(-1, -2), True, cu, idx, acc, True,
        )
        do = (out_op.float() - out_ref.float()).abs().max().item()
        slot_ds = {
            s: round((st_op.transpose(-1, -2)[s].float()
                      - st_ref.transpose(-1, -2)[s].float()).abs().max().item(), 5)
            for s in touched
        }
        print(f"acc={acc_len}: max|out_op-out_ref|={do:.4e}  "
              f"touched-slot |dstate|={slot_ds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
