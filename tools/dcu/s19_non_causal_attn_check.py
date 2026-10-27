# SPDX-License-Identifier: Apache-2.0
# S19: correctness check for the gfx906 unified-attention non-causal path.
#
#   docker exec z100-vllm bash -lc '. /opt/dtk/env.sh; . /src/tools/dcu/env.sh;
#     PYTHONPATH=/src:/mnt/game/triton-gfx906/install python /src/tools/dcu/s19_non_causal_attn_check.py'
#
# Builds a paged KV cache, runs vllm.v1.attention.ops.triton_unified_attention
# with causal=True/False, and compares against a PyTorch reference that applies
# the same mask (causal or bidirectional, optionally windowed).
import math

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention

HEAD = 128
NQ = 32
NKV = 8
BLOCK = 16
Q_LEN = 8
CTX_LENS = [7, 20, 33]


def build(device="cuda", dtype=torch.float16):
    torch.manual_seed(0)
    num_seqs = len(CTX_LENS)
    seq_lens = [c + Q_LEN for c in CTX_LENS]
    max_seq = max(seq_lens)
    total_q = num_seqs * Q_LEN

    q = (torch.randn(total_q, NQ, HEAD, device=device, dtype=torch.float32) * 0.5).to(
        dtype
    )
    kv = (
        torch.randn(num_seqs, max_seq, NKV, HEAD, device=device, dtype=torch.float32)
        * 0.5
    ).to(dtype)

    blocks_per_seq = [math.ceil(s / BLOCK) for s in seq_lens]
    total_blocks = sum(blocks_per_seq)
    k_cache = torch.zeros(total_blocks, BLOCK, NKV, HEAD, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(num_seqs, max(blocks_per_seq), dtype=torch.int32, device=device)

    nxt = 0
    for s, s_len in enumerate(seq_lens):
        nb = blocks_per_seq[s]
        block_table[s, :nb] = torch.arange(nxt, nxt + nb, dtype=torch.int32, device=device)
        for t in range(s_len):
            b = int(block_table[s, t // BLOCK])
            k_cache[b, t % BLOCK] = kv[s, t]
            v_cache[b, t % BLOCK] = kv[s, t]
        nxt += nb

    cu_seqlens_q = torch.arange(0, total_q + 1, Q_LEN, dtype=torch.int32, device=device)
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, kv, block_table, cu_seqlens_q, seqused_k, seq_lens


def run_op(
    q, k, v, block_table, cu_seqlens_q, seqused_k, causal, window, scale, three_d=False
):
    out = torch.empty(q.shape[0], NQ, HEAD, device=q.device, dtype=q.dtype)
    kwargs = {}
    if three_d:
        nseg = 2
        total_q = q.shape[0]
        kwargs = dict(
            seq_threshold_3D=total_q,
            num_par_softmax_segments=nseg,
            softmax_segm_output=torch.empty(
                total_q, NQ, nseg, 128, device=q.device, dtype=torch.float32
            ),
            softmax_segm_max=torch.empty(
                total_q, NQ, nseg, device=q.device, dtype=torch.float32
            ),
            softmax_segm_expsum=torch.empty(
                total_q, NQ, nseg, device=q.device, dtype=torch.float32
            ),
        )
    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=Q_LEN,
        seqused_k=seqused_k,
        max_seqlen_k=int(seqused_k.max().item()),
        softmax_scale=scale,
        causal=causal,
        window_size=window,
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        **kwargs,
    )
    return out


def reference(q, kv, seq_lens, causal, window, scale):
    """PyTorch reference for the same mask semantics."""
    win = 1 + window[0] if window[0] >= 0 else 0
    outs = []
    grp = NQ // NKV
    for s, seq_len in enumerate(seq_lens):
        ctx = seq_len - Q_LEN
        qs = q[s * Q_LEN : (s + 1) * Q_LEN].float()  # [Q, NQ, HEAD]
        ks = kv[s, :seq_len].float()  # [S, NKV, HEAD]
        vs = kv[s, :seq_len].float()
        k_rep = ks.repeat_interleave(grp, dim=1)  # [S, NQ, HEAD]
        v_rep = vs.repeat_interleave(grp, dim=1)
        logits = torch.einsum("qhd,shd->qhs", qs, k_rep) * scale  # [Q, NQ, S]
        q_abs = torch.arange(ctx, seq_len, device=q.device)[:, None]
        k_abs = torch.arange(seq_len, device=q.device)[None, :]
        mask = torch.ones_like(q_abs, dtype=torch.bool)
        if causal:
            mask = mask & (k_abs <= q_abs)
        if win > 0:
            mask = mask & ((q_abs - k_abs).abs() < win)
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        outs.append(torch.einsum("qhs,shd->qhd", probs, v_rep))
    return torch.cat(outs, dim=0).to(q.dtype)


def check(name, causal, window, three_d=False):
    q, k, v, kv, bt, cuq, seqk, seq_lens = build()
    scale = 1.0 / math.sqrt(HEAD)
    got = run_op(q, k, v, bt, cuq, seqk, causal, window, scale, three_d=three_d)
    ref = reference(q, kv, seq_lens, causal, window, scale)
    d = (got.float() - ref.float()).abs()  # [T, NQ, HEAD]
    diff = d.max().item()
    per_seq = []
    for s_idx, s_len in enumerate(seq_lens):
        per_seq.append(round(d[s_idx * Q_LEN:(s_idx + 1) * Q_LEN].max().item(), 4))
    per_row = [round(d[s_idx * Q_LEN + r].max().item(), 4) for s_idx in range(len(seq_lens)) for r in range(Q_LEN)]
    status = "PASS" if diff < 2e-2 else "FAIL"
    print(f"{status} {name}: causal={causal} window={window} max_abs_diff={diff:.5f} per_seq={per_seq} per_row={per_row}")
    return status == "PASS"


if __name__ == "__main__":
    ok = True
    ok &= check("causal nowindow", True, (-1, -1))
    ok &= check("noncausal nowindow", False, (-1, -1))
    ok &= check("causal window32", True, (31, 31))
    ok &= check("noncausal window32", False, (31, 31))
    ok &= check("noncausal window8", False, (7, 7))
    ok &= check("causal window8", True, (7, 7))
    # The decode-shaped 3D (split-KV softmax) kernel takes the same mask path.
    ok &= check("3d causal nowindow", True, (-1, -1), three_d=True)
    ok &= check("3d noncausal nowindow", False, (-1, -1), three_d=True)
    ok &= check("3d noncausal window8", False, (7, 7), three_d=True)
    print("ALL PASS" if ok else "SOME FAILED")
