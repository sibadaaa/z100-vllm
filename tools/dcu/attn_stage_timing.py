#!/usr/bin/env python3
"""Stage-level timing for the gfx906 int8-KV attention inner loop.

Plan of record: docs/perf/PLAN-20261113-attn-latency-vs-bandwidth.md T1/T2.

Drives ``gfx906_attnd_decode_inner_kvint8`` (the production decode/verify and
chunked-prefill inner loop for the shipped ``int8_per_token_head`` KV cache)
and its ``*_stage`` sibling, on the production per-rank geometry:

    6 q heads / 1 kv head / head_dim 256 / block_size 1616
    KV page = (blocks, 1, 1616, 520) int8: [K 256][K fp32 scale][V 256][V scale]

The packed 520-byte page is the real thing, not the probe-only split layout, so
the load stage sees the production address pattern.

Two modes:

  prod   production op.  Wall time + effective KV GB/s for the re-read model.
  stage  instrumented op.  Per-CTA clock64() deltas -> per-stage share of the
         kernel wall time, and the KV-load stage's own GB/s.

Correctness is checked before any timing is trusted: the stage op must produce
an out_md5 identical to the production op on every shape, and the production
op's output is compared against an fp32 reference built from the *dequantized*
cache (the int8 quantization is part of the format, not a defect).

Usage::

    # decode shapes, production op
    python tools/dcu/attn_stage_timing.py --mode prod --out runs/.../t2.json
    # decode shapes, stage-attributed
    python tools/dcu/attn_stage_timing.py --mode stage --out runs/.../t1.json
    # prefill chunk shapes
    python tools/dcu/attn_stage_timing.py --mode stage --kind prefill
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time

import torch

os.environ.setdefault("VLLM_GFX906_ATTN_DECODE_HIP_INNER", "1")
os.environ.setdefault("VLLM_GFX906_ATTN_PREFILL_HIP_INNER", "1")

# Load the extension explicitly: the ops live in vllm/_rocm_C.abi3.so and the
# probe must not depend on which import path happens to touch it first.
import vllm._rocm_C  # noqa: E402,F401

NUM_Q_HEADS = 6
NUM_KV_HEADS = 1
HEAD_DIM = 256
PAGE_BYTES = 2 * (HEAD_DIM + 4)  # K data + K scale + V data + V scale
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)
STAGES = ("load", "qk", "sm", "pv", "wait")


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def md5t(t: torch.Tensor) -> str:
    return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()[:8]


def quant_per_token_head(x: torch.Tensor):
    """x: [T, HEAD] fp32 -> (int8 codes, [T] fp32 scale), symmetric."""
    scale = x.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    t = x / scale.unsqueeze(-1)
    q = torch.where(t >= 0, torch.floor(t + 0.5), -torch.floor(-t + 0.5))
    return q.clamp(-127, 127).to(torch.int8), scale


def build_cache(ctx: int, block_size: int, seed: int, device: str) -> dict:
    """Packed int8_per_token_head page: [K 256][K scale][V 256][V scale]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = cdiv(ctx, block_size)
    big = g  # one generator, deterministic order
    k = torch.randn(nb * block_size, HEAD_DIM, generator=big).to(torch.float32)
    v = torch.randn(nb * block_size, HEAD_DIM, generator=big).to(torch.float32)
    kq, ks = quant_per_token_head(k)
    vq, vs = quant_per_token_head(v)

    # Logical layout is (blocks, nkv, block, PAGE_BYTES); the backend's
    # per-token-head views are kv.transpose(1, 2).split(260, dim=-1), so the
    # per-token slot stride the kernel sees is 520 B (K and V interleaved).
    kv = torch.zeros((nb, NUM_KV_HEADS, block_size, PAGE_BYTES),
                     dtype=torch.int8)
    kq_v = kq.view(nb, block_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    vq_v = vq.view(nb, block_size, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    kv[..., :HEAD_DIM] = kq_v
    kv[..., HEAD_DIM + 4:2 * HEAD_DIM + 4] = vq_v
    f32 = kv.view(torch.float32)  # (nb, nkv, block, 130)
    f32[..., 0, :, HEAD_DIM // 4] = ks.view(nb, block_size)
    f32[..., 0, :, (PAGE_BYTES // 2 + HEAD_DIM) // 4] = vs.view(nb, block_size)
    kv = kv.to(device)

    kv_t = kv.transpose(1, 2)                       # (nb, block, 1, 520)
    key_cache, value_cache = kv_t.split(PAGE_BYTES // 2, dim=-1)
    # Scale views: fp32 as_strided over the same storage, 4 B after each half.
    f32d = kv.view(torch.float32)
    slot_f32 = PAGE_BYTES // 4
    block_f32 = block_size * slot_f32
    k_scale = torch.as_strided(
        f32d, (nb, block_size, NUM_KV_HEADS),
        (block_f32, slot_f32, block_f32),
        storage_offset=HEAD_DIM // 4)
    v_scale = torch.as_strided(
        f32d, (nb, block_size, NUM_KV_HEADS),
        (block_f32, slot_f32, block_f32),
        storage_offset=(PAGE_BYTES // 2 + HEAD_DIM) // 4)

    # Dequantized fp16 K/V for the reference (format is part of the data).
    kdq = (kq.float() * ks.unsqueeze(-1)).to(torch.float16)
    vdq = (vq.float() * vs.unsqueeze(-1)).to(torch.float16)
    return dict(kv=kv, key=key_cache, value=value_cache,
                k_scale=k_scale, v_scale=v_scale,
                kdq=kdq.to(device), vdq=vdq.to(device),
                nb=nb, block_size=block_size)


def build_q(q_len: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(q_len, NUM_Q_HEADS, HEAD_DIM, generator=g) \
        .to(torch.float16).to(device)


def segment_buffers(q_len: int, segments: int, device: str,
                    fp16: bool = False):
    rows = next_pow2(q_len)
    dtype = torch.float16 if fp16 else torch.float32
    segm = torch.empty((rows, NUM_Q_HEADS, segments, HEAD_DIM), dtype=dtype,
                       device=device)
    smax = torch.empty((rows, NUM_Q_HEADS, segments), dtype=torch.float32,
                       device=device)
    sexp = torch.empty_like(smax)
    return segm, smax, sexp


def launch_prod(cache, q, bt, seq_len, cu, segm, smax, sexp, segments,
                tps_hint):
    torch.ops._rocm_C.gfx906_attnd_decode_inner_kvint8(
        segm, smax, sexp, q, cache["key"], cache["value"],
        cache["k_scale"], cache["v_scale"], bt, seq_len, cu,
        SOFTMAX_SCALE, segments, True,
        tps_hint if tps_hint >= 0 else 0, segm.shape[2])
    return segm


def launch_stage(cache, q, bt, seq_len, cu, segm, smax, sexp, segments,
                 tps_hint, cycles):
    torch.ops._rocm_C.gfx906_attnd_decode_inner_kvint8_stage(
        segm, smax, sexp, q, cache["key"], cache["value"],
        cache["k_scale"], cache["v_scale"], bt, seq_len, cu,
        SOFTMAX_SCALE, segments, True,
        tps_hint if tps_hint >= 0 else 0, segm.shape[2], cycles)
    return segm


def launch_dispatch(cache, q, bt, seq_len, cu, segm, smax, sexp, segments,
                    max_seqlen_k: int):
    """Production entry point: dispatch + split-KV + reduce_segments."""
    import vllm.v1.attention.ops.triton_unified_attention as mod
    from vllm.v1.kv_cache_interface import KVQuantMode

    out = torch.empty_like(q)
    mod.unified_attention(
        q, cache["key"], cache["value"], out, cu, q.shape[0], seq_len,
        max_seqlen_k, SOFTMAX_SCALE, True, (-1, -1), bt, 0.0, None, None, None,
        seq_threshold_3D=32, num_par_softmax_segments=segments,
        softmax_segm_output=segm, softmax_segm_max=smax,
        softmax_segm_expsum=sexp,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        k_scale_cache=cache["k_scale"], v_scale_cache=cache["v_scale"],
    )
    return out


def reference(cache, q, ctx: int, block_size: int, q_chunk: int = 64):
    """fp32 causal reference over the dequantized paged KV (one sequence)."""
    total = cache["nb"] * block_size
    idx = torch.arange(ctx, device=q.device)
    sel = torch.div(idx, block_size, rounding_mode="floor")
    off = idx % block_size
    # (nb, block, dim) -> gather the first ctx tokens
    kd = cache["kdq"].view(-1, HEAD_DIM)[:total].float()
    vd = cache["vdq"].view(-1, HEAD_DIM)[:total].float()
    flat = sel * block_size + off
    kmat = kd[flat]
    vmat = vd[flat]
    m = q.shape[0]
    out = torch.empty_like(q, dtype=torch.float32)
    ctx_len = ctx - m
    for s in range(0, m, q_chunk):
        e = min(s + q_chunk, m)
        pos = ctx_len + torch.arange(s, e, device=q.device)
        mask = idx[None, :] <= pos[:, None]
        for h in range(NUM_Q_HEADS):
            sc = (q[s:e, h, :].float() @ kmat.T) * SOFTMAX_SCALE
            sc = sc.masked_fill(~mask, float("-inf"))
            out[s:e, h, :] = torch.softmax(sc, dim=-1) @ vmat
    return out


def time_launch(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2], times


def run_shape(kind: str, m: int, ctx: int, segments: int, block_size: int,
              args, seed: int) -> dict:
    q_len = m if kind == "decode" else m  # same field: rows of Q
    seq_len_k = ctx if kind == "decode" else ctx + q_len
    cache = build_cache(seq_len_k, block_size, seed, args.device)
    q = build_q(q_len, seed + 1, args.device)
    nb = cache["nb"]
    bt = torch.arange(nb, dtype=torch.int32, device=args.device).view(1, nb)
    seq_len = torch.tensor([seq_len_k], dtype=torch.int32, device=args.device)
    cu = torch.tensor([0, q_len], dtype=torch.int32, device=args.device)
    segm, smax, sexp = segment_buffers(q_len, segments, args.device)
    tps_hint = max(1, cdiv(seq_len_k, segments * 16)) if args.tps_auto else -1

    n_qblocks = cdiv(q_len * NUM_Q_HEADS, 16)
    kv_bytes = n_qblocks * seq_len_k * PAGE_BYTES
    macs = NUM_Q_HEADS * HEAD_DIM * (
        q_len * (seq_len_k - q_len) + q_len * (q_len - 1) // 2)

    row = {"kind": kind, "m": q_len, "ctx": ctx, "seq_k": seq_len_k,
           "segments": segments, "block_size": block_size,
           "n_qblocks": n_qblocks, "kv_bytes": kv_bytes, "macs": macs,
           "tps_hint": tps_hint}

    # ---- correctness -------------------------------------------------------
    # (a) stage op segm must be bit-identical to the production op's segm;
    # (b) the production entry point (unified_attention -> reduce_segments)
    #     must match an fp32 reference over the dequantized cache.
    if args.check and (m, ctx) in args.check_shapes:
        prod_out = launch_prod(cache, q, bt, seq_len, cu, segm, smax, sexp,
                               segments, tps_hint).clone()
        torch.cuda.synchronize()
        row["segm_md5"] = md5t(prod_out)
        row["check_segments"] = segments
        if args.mode in ("stage", "both"):
            cycles = torch.zeros(7, dtype=torch.int64, device=args.device)
            stg = launch_stage(cache, q, bt, seq_len, cu, segm, smax, sexp,
                               segments, tps_hint, cycles).clone()
            torch.cuda.synchronize()
            row["stage_segm_md5"] = md5t(stg)
            row["stage_bitexact"] = row["stage_segm_md5"] == row["segm_md5"]
        if args.reference:
            disp = launch_dispatch(cache, q, bt, seq_len, cu, segm, smax, sexp,
                                   segments, seq_len_k).clone()
            torch.cuda.synchronize()
            ref = reference(cache, q, seq_len_k, block_size)
            d = disp.float() - ref
            row["ref_max"] = float(d.abs().max())
            row["ref_rel"] = float((d.norm(dim=-1) /
                                    ref.norm(dim=-1).clamp_min(1e-6)).mean())
            row["dispatch_out_md5"] = md5t(disp)

    # ---- timing -----------------------------------------------------------
    if args.mode == "prod":
        med, alls = time_launch(
            lambda: launch_prod(cache, q, bt, seq_len, cu, segm, smax, sexp,
                                segments, tps_hint),
            args.warmup, args.iters)
        row.update(ms=med, all_ms=alls)
        row["kv_gbs_all"] = kv_bytes / (med * 1e-3) / 1e9
        row["tf_all"] = 4 * macs / (med * 1e-3) / 1e12
    else:
        cycles = torch.zeros(7, dtype=torch.int64, device=args.device)

        def once():
            launch_stage(cache, q, bt, seq_len, cu, segm, smax, sexp,
                         segments, tps_hint, cycles)

        med, alls = time_launch(once, args.warmup, args.iters)
        cycles.zero_()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        once()
        e.record()
        torch.cuda.synchronize()
        cyc = cycles.cpu().tolist()
        row.update(ms=med, all_ms=alls)
        row["kv_gbs_all"] = kv_bytes / (med * 1e-3) / 1e9
        row["tf_all"] = 4 * macs / (med * 1e-3) / 1e12
        load, qk, sm, pv, wait, ncta, total = cyc[:7]
        ssum = max(1, load + qk + sm + pv + wait)
        row["cycles"] = {"load": load, "qk": qk, "sm": sm, "pv": pv,
                         "wait": wait, "ctas": ncta, "sum": ssum,
                         "wall_cycles": total}
        share = {k: v / ssum for k, v in
                 (("load", load), ("qk", qk), ("sm", sm), ("pv", pv),
                  ("wait", wait))}
        row["share"] = share
        row["stage_ms"] = {k: share[k] * med for k in STAGES}
        load_ms = row["stage_ms"]["load"]
        row["load_gbs"] = (kv_bytes / (load_ms * 1e-3) / 1e9
                           if load_ms > 0 else None)
        row["qk_tf"] = (4 * NUM_Q_HEADS * HEAD_DIM * q_len *
                        (seq_len_k - q_len) /
                        (row["stage_ms"]["qk"] * 1e-3) / 1e12
                        if row["stage_ms"]["qk"] > 0 else None)
        row["pv_tf"] = (4 * NUM_Q_HEADS * HEAD_DIM * q_len * (seq_len_k - q_len)
                        / (row["stage_ms"]["pv"] * 1e-3) / 1e12
                        if row["stage_ms"]["pv"] > 0 else None)
    del cache, q, segm, smax, sexp
    torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prod", "stage", "both"],
                    default="stage")
    ap.add_argument("--kind", choices=["decode", "prefill"], default="decode")
    ap.add_argument("--ms", default="1,3,8")
    ap.add_argument("--ctxs", default="8192,16384,32768,65536")
    ap.add_argument("--segments", default="256")
    ap.add_argument("--block-size", type=int, default=1616)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--adaptive-iters", type=int, default=1)
    ap.add_argument("--tps-auto", type=int, default=1)
    ap.add_argument("--check", type=int, default=1)
    ap.add_argument("--check-shapes", default="")
    ap.add_argument("--reference", type=int, default=0,
                    help="also compare the production dispatch to an fp32 "
                         "reference at the checked shapes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.cuda.set_device(0)

    if a.check_shapes:
        pairs = []
        for item in a.check_shapes.split(","):
            mm, cc = item.split(":")
            pairs.append((int(mm), int(cc)))
        a.check_shapes = set(pairs)
    else:
        # cheapest shape of the sweep that still exercises the mask/odd edges
        ms = [int(x) for x in a.ms.split(",")]
        ctxs = [int(x) for x in a.ctxs.split(",")]
        a.check_shapes = {(ms[0], ctxs[0])}

    rows = []
    so = os.path.realpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "vllm", "_rocm_C.abi3.so"))
    import vllm._rocm_C as _rc
    print(f"# attn_stage_timing mode={a.mode} kind={a.kind} "
          f"so={so} loaded={_rc.__file__} block={a.block_size}", flush=True)
    for seg in [int(x) for x in a.segments.split(",")]:
        for ctx in [int(x) for x in a.ctxs.split(",")]:
            for m in [int(x) for x in a.ms.split(",")]:
                row = run_shape(a.kind, m, ctx, seg, a.block_size, a, a.seed)
                row["tag"] = a.tag
                row["so_md5"] = hashlib.md5(open(so, "rb").read()).hexdigest()[:8]
                rows.append(row)
                print(json.dumps(row), flush=True)
    payload = {"tag": a.tag, "mode": a.mode, "kind": a.kind,
               "block_size": a.block_size, "so": so,
               "ts": time.time(), "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[done] {len(rows)} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
