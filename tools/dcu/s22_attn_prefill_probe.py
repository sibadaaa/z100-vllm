#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone gfx906 (Hygon Z100) prefill-attention microbenchmark.

Measures the *prefill* path of
``vllm/v1/attention/ops/triton_unified_attention.py`` with no engine, no model
weights and no server.  ``/src`` is never modified: every swept knob is applied
by monkeypatching module globals of the imported module and by wrapping the two
Triton kernel launch sites (``num_warps`` / ``num_stages``) from this file.

Geometry defaults are the production 27B TP=4 per-rank shape:
``num_query_heads=6``, ``num_kv_heads=1``, ``head_dim=256``, fp16 Q, fp16 KV.

Sequence model: one sequence, ``seq_len == ctx`` (total KV length), and each
``M`` in ``--m-list`` is the number of *new* query tokens in this forward pass,
so ``context_len = ctx - M`` (chunked prefill).  The production trace
(``ctx=49152``, ``M=7920``, ``kernel_unified_attention_2d``) is exactly the
``--ctx 49152 --m-list 7920`` point.

Config spec string::

    BLOCK_M/TILE/WARPS/STAGES/2d           # 2D kernel (unsplit)
    BLOCK_M/TILE/WARPS/STAGES/3d:SEGMENTS  # split-KV 3D kernel + reduce_segments

Usage (inside the z100-vllm container)::

    HIP_VISIBLE_DEVICES=1 python /src/tools/dcu/s22_attn_prefill_probe.py \
        --ctx 49152 --m-list 7920 --preset ofat --out /mnt/game/s22/x.json

Notes for the S22 arms actually run (raw data in /mnt/game/s22/*.json):
  * the production KV cache uses block_size=800 (hybrid Mamba/attention
    pagination), so pass ``--block-size 800`` for production-representative
    numbers; the 3D segment optimum is block_size-dependent;
  * ``--snapshot`` pins the module under test against concurrent edits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import sys
import time

import torch

MOD_NAME = "vllm.v1.attention.ops.triton_unified_attention"
NUM_QUERY_HEADS = 6
NUM_KV_HEADS = 1
HEAD_DIM = 256
QK_HEADS = NUM_QUERY_HEADS  # GQA: 6 query heads share 1 KV head

BASELINE = {"path": "2d", "block_m": 32, "tile": 32, "warps": 4, "stages": 1, "segments": 0}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def spec_to_cfg(spec: str) -> dict:
    """'32/64/4/1/2d' or '32/64/4/1/3d:2' -> config dict."""
    parts = spec.split("/")
    if len(parts) == 5:
        bm, tile, warps, stages, tail = parts
    elif len(parts) == 4:  # path-less form defaults to 2d
        bm, tile, warps, stages = parts
        tail = "2d"
    else:
        raise ValueError(f"bad config spec: {spec!r}")
    segments = 0
    if ":" in tail:
        tail, _, seg = tail.partition(":")
        segments = int(seg)
    path = tail.strip().lower()
    if path not in ("2d", "3d"):
        raise ValueError(f"bad path in spec {spec!r}")
    cfg = {
        "path": path,
        "block_m": int(bm),
        "tile": int(tile),
        "warps": int(warps),
        "stages": int(stages),
        "segments": segments,
    }
    if path == "3d" and segments <= 0:
        cfg["segments"] = 2
    return cfg


def cfg_to_spec(cfg: dict) -> str:
    s = f"{cfg['block_m']}/{cfg['tile']}/{cfg['warps']}/{cfg['stages']}/{cfg['path']}"
    if cfg["path"] == "3d":
        s += f":{cfg['segments']}"
    return s


def cfg_key(cfg: dict) -> tuple:
    return (cfg["path"], cfg["block_m"], cfg["tile"], cfg["warps"], cfg["stages"], cfg["segments"])


def build_configs(args) -> list[dict]:
    """Expand the requested sweep into an ordered, de-duplicated config list."""
    bms = [int(v) for v in args.block_m_list.split(",") if v]
    tiles = [int(v) for v in args.tile_list.split(",") if v]
    warpss = [int(v) for v in args.warps_list.split(",") if v]
    stagess = [int(v) for v in args.stages_list.split(",") if v]
    segss = [int(v) for v in args.segments_list.split(",") if v]

    cfgs: list[dict] = []
    if args.preset == "baseline":
        cfgs = [dict(BASELINE)]
    elif args.preset == "full":
        for bm, tl, w, st in itertools.product(bms, tiles, warpss, stagess):
            cfgs.append({"path": "2d", "block_m": bm, "tile": tl, "warps": w, "stages": st, "segments": 0})
        for bm, tl, w, st, sg in itertools.product(bms, tiles, warpss, stagess, segss):
            cfgs.append({"path": "3d", "block_m": bm, "tile": tl, "warps": w, "stages": st, "segments": sg})
    elif args.preset == "quick":
        specs = [
            "32/32/4/1/2d",  # baseline
            "16/32/4/1/2d",
            "64/32/4/1/2d",
            "32/64/4/1/2d",
            "32/128/4/1/2d",
            "32/64/8/1/2d",
            "64/64/8/1/2d",
            "32/64/4/2/2d",
            "32/32/4/1/3d:2",
            "32/64/4/1/3d:2",
            "32/64/8/1/3d:2",
            "64/64/8/1/3d:2",
        ]
        cfgs = [spec_to_cfg(s) for s in specs]
    else:  # ofat: baseline + one-factor-at-a-time + a few long-context combos
        cfgs = [dict(BASELINE)]
        for bm in bms:
            cfgs.append({"path": "2d", "block_m": bm, "tile": 32, "warps": 4, "stages": 1, "segments": 0})
        for tl in tiles:
            cfgs.append({"path": "2d", "block_m": 32, "tile": tl, "warps": 4, "stages": 1, "segments": 0})
        for w in warpss:
            cfgs.append({"path": "2d", "block_m": 32, "tile": 32, "warps": w, "stages": 1, "segments": 0})
        for st in stagess:
            cfgs.append({"path": "2d", "block_m": 32, "tile": 32, "warps": 4, "stages": st, "segments": 0})
        for sg in segss:
            cfgs.append({"path": "3d", "block_m": 32, "tile": 32, "warps": 4, "stages": 1, "segments": sg})
        # two-knob combos that plausibly matter at >=32k context
        combos = [
            (32, 64, 4, 1, "2d", 0),
            (32, 64, 8, 1, "2d", 0),
            (32, 64, 8, 2, "2d", 0),
            (32, 128, 8, 1, "2d", 0),
            (32, 128, 8, 2, "2d", 0),
            (16, 64, 4, 1, "2d", 0),
            (16, 64, 8, 1, "2d", 0),
            (64, 32, 8, 1, "2d", 0),
            (64, 64, 8, 1, "2d", 0),
            (64, 64, 4, 1, "2d", 0),
            (64, 128, 8, 1, "2d", 0),
            (64, 64, 8, 2, "2d", 0),
            (32, 32, 8, 1, "2d", 0),
            (32, 32, 2, 1, "2d", 0),
            (32, 32, 4, 2, "3d", 2),
            (32, 32, 4, 1, "3d", 4),
            (32, 32, 4, 1, "3d", 8),
            (32, 64, 4, 1, "3d", 2),
            (32, 64, 8, 1, "3d", 2),
            (32, 64, 8, 1, "3d", 4),
            (64, 64, 8, 1, "3d", 2),
            (16, 64, 4, 1, "3d", 2),
            (32, 128, 8, 1, "3d", 2),
        ]
        for bm, tl, w, st, p, sg in combos:
            cfgs.append({"path": p, "block_m": bm, "tile": tl, "warps": w, "stages": st, "segments": sg})
    for s in [v for v in (args.extra_configs or "").split(",") if v.strip()]:
        cfgs.append(spec_to_cfg(s.strip()))

    seen, out = set(), []
    for c in cfgs:
        k = cfg_key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    if args.order == "shuffle":
        base = [c for c in out if c["path"] == "2d" and c["block_m"] == 32 and c["tile"] == 32
                and c["warps"] == 4 and c["stages"] == 1]
        rest = [c for c in out if c not in base]
        out = base + rest
    return out


# --------------------------------------------------------------------------- #
# module patching
# --------------------------------------------------------------------------- #
class KernelLaunchProxy:
    """Force num_warps/num_stages on a Triton JITFunction launch site.

    The call sites in triton_unified_attention.py pass every kernel argument as
    a keyword, so overriding the launch options here is enough.
    """

    def __init__(self, fn, num_warps, num_stages):
        self._fn = fn
        self._num_warps = num_warps
        self._num_stages = num_stages
        self.last_kernel = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs["num_warps"] = self._num_warps
            if self._num_stages is not None:
                kwargs["num_stages"] = self._num_stages
            k = self._fn[grid](*args, **kwargs)
            self.last_kernel = k
            return k

        return launch


def kernel_stats(proxy):
    """Register/spill/LDS usage of the last compiled kernel, if available."""
    k = getattr(proxy, "last_kernel", None)
    md = getattr(k, "metadata", None)
    if md is None:
        return {}
    out = {}
    for name in ("n_regs", "n_spills", "shared", "num_warps", "num_stages"):
        v = getattr(md, name, None)
        if v is not None:
            out[name] = int(v)
    return out


def patch_module(mod, cfg, orig):
    """Point every swept knob at cfg. ``orig`` holds the unpatched objects."""
    mod.ENABLE_GFX906_ATTN_PREFILL_3D = cfg["path"] == "3d"
    mod.GFX906_PREFILL_SEGMENTS = cfg["segments"] if cfg["segments"] > 0 else 2
    mod._prefill_block_m = lambda head_size, num_queries_per_kv: cfg["block_m"]
    mod._get_tile_size = lambda *a, **k: cfg["tile"]
    mod.kernel_unified_attention_2d = KernelLaunchProxy(
        orig["k2d"], cfg["warps"], cfg["stages"]
    )
    mod.kernel_unified_attention_3d = KernelLaunchProxy(
        orig["k3d"], cfg["warps"], cfg["stages"]
    )


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def build_inputs(ctx, m, block_size, dtype, block_map, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_blocks = cdiv(ctx, block_size)
    q = torch.randn(m, NUM_QUERY_HEADS, HEAD_DIM, generator=g, dtype=torch.float32).to(dtype).to(device)
    k = torch.randn(num_blocks, block_size, NUM_KV_HEADS, HEAD_DIM, generator=g, dtype=torch.float32).to(dtype).to(device)
    v = torch.randn(num_blocks, block_size, NUM_KV_HEADS, HEAD_DIM, generator=g, dtype=torch.float32).to(dtype).to(device)
    if block_map == "shuffle":
        perm = torch.randperm(num_blocks, generator=g).to(torch.int32)
    else:
        perm = torch.arange(num_blocks, dtype=torch.int32)
    block_table = perm.view(1, num_blocks).to(device)
    cu = torch.tensor([0, m], dtype=torch.int32, device=device)
    seqused = torch.tensor([ctx], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    scal = torch.ones(1, 1, dtype=torch.float32, device=device)
    return {
        "q": q, "k": k, "v": v, "out": out, "cu": cu, "seqused": seqused,
        "block_table": block_table, "scal": scal, "block_size": block_size,
    }


def call_unified(mod, inp, ctx, m, scale):
    mod.unified_attention(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        out=inp["out"],
        cu_seqlens_q=inp["cu"],
        max_seqlen_q=m,
        seqused_k=inp["seqused"],
        max_seqlen_k=ctx,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=inp["block_table"],
        softcap=0.0,
        q_descale=None,
        k_descale=inp["scal"],
        v_descale=inp["scal"],
    )


# --------------------------------------------------------------------------- #
# reference + work accounting
# --------------------------------------------------------------------------- #
def reference_attention(q, k, v, block_table, block_size, ctx, m, scale, q_chunk=32):
    """Chunked fp32 torch reference for one sequence, causal, GQA broadcast.

    Logical position i lives in physical block block_table[0, i // block_size]
    at slot i % block_size.
    """
    ctx_len = ctx - m
    out = torch.empty_like(q, dtype=torch.float32)
    idx = torch.arange(ctx, device=q.device)
    blk = block_table[0, idx // block_size].long()
    off = idx % block_size
    kmat = k[blk, off, 0, :].float()  # [ctx, D]
    vmat = v[blk, off, 0, :].float()
    keys = torch.arange(ctx, device=q.device)
    for s in range(0, m, q_chunk):
        e = min(s + q_chunk, m)
        pos = ctx_len + torch.arange(s, e, device=q.device)
        mask = keys[None, :] <= pos[:, None]
        for h in range(NUM_QUERY_HEADS):
            sc = (q[s:e, h, :].float() @ kmat.T) * scale
            sc = sc.masked_fill(~mask, float("-inf"))
            p = torch.softmax(sc, dim=-1)
            out[s:e, h, :] = p @ vmat
    return out.to(q.dtype)


def issued_tiles(cfg, ctx, m):
    """KV tiles actually iterated by the launched grid (matches kernel masks)."""
    ctx_len = ctx - m
    bq = cfg["block_m"] // NUM_QUERY_HEADS
    nblk = cdiv(m, bq)
    per_blk = (cfg["block_m"] - 1) // NUM_QUERY_HEADS + 1
    nseg = cfg["segments"] if cfg["path"] == "3d" else 1
    tps = cdiv(ctx, nseg * cfg["tile"])
    total = 0
    for b in range(nblk):
        msp = min(ctx_len + b * bq + per_blk, ctx)
        ntiles = cdiv(msp, cfg["tile"])
        if cfg["path"] == "2d":
            total += ntiles
        else:
            for s in range(nseg):
                lo = max(s * tps, 0)
                hi = min((s + 1) * tps, ntiles)
                total += max(0, hi - lo)
    return total, nblk


def metrics(cfg, ctx, m, ms, dtype_bytes):
    ctx_len = ctx - m
    tiles, nblk = issued_tiles(cfg, ctx, m)
    # useful (causally required) FLOPs: every query head sees exactly the keys
    # up to its own absolute position.
    useful = 4 * QK_HEADS * HEAD_DIM * (m * ctx_len + m * (m + 1) // 2)
    issued = 4 * cfg["block_m"] * cfg["tile"] * HEAD_DIM * tiles
    kv_bytes = tiles * cfg["tile"] * HEAD_DIM * dtype_bytes * 2  # K + V
    io_bytes = kv_bytes + 2 * m * QK_HEADS * HEAD_DIM * dtype_bytes
    reduce_bytes = 0
    if cfg["path"] == "3d":
        nseg = cfg["segments"]
        reduce_bytes = m * QK_HEADS * nseg * (HEAD_DIM * 4 + 4 + 4) + m * QK_HEADS * HEAD_DIM * dtype_bytes
    sec = ms / 1e3
    return {
        "issued_kv_tiles": tiles,
        "grid_q_blocks": nblk,
        "tflops_useful": useful / sec / 1e12,
        "tflops_issued": issued / sec / 1e12,
        "gbs_kv": (io_bytes + reduce_bytes) / sec / 1e9,
        "kv_bytes": kv_bytes + reduce_bytes,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ctx", default="49152",
                   help="comma list of total KV lengths (seq_len); default 49152")
    p.add_argument("--m-list", default="1760,3520,7920", help="comma list of new query-token counts")
    p.add_argument("--block-m-list", default="8,16,32,64,128")
    p.add_argument("--tile-list", default="16,32,64,128")
    p.add_argument("--warps-list", default="2,4,8")
    p.add_argument("--stages-list", default="1,2,3")
    p.add_argument("--segments-list", default="2,4")
    p.add_argument("--preset", default="ofat", choices=["baseline", "quick", "ofat", "full"])
    p.add_argument("--extra-configs", default="", help="comma list of explicit specs appended to the preset")
    p.add_argument("--order", default="baseline-first", choices=["baseline-first", "shuffle"])
    p.add_argument("--iters", type=int, default=1, help="kernel calls per timed sample")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=3, help="timed samples; min is reported")
    p.add_argument("--block-size", type=int, default=16, help="KV cache block_size")
    p.add_argument("--block-map", default="shuffle", choices=["shuffle", "identity"])
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--snapshot", default="",
                   help="import the module from this pinned copy instead of /src "
                        "(the tracked file may be edited concurrently)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ref-check", default="8192:1760,49152:1760",
                   help="'ctx:M' pairs verified against the torch reference ('' disables)")
    p.add_argument("--no-consistency", action="store_true", help="skip per-config output diff vs baseline")
    p.add_argument("--out", default="", help="JSON output path")
    p.add_argument("--tag", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ctxs = [int(v) for v in args.ctx.split(",") if v]
    ms = [int(v) for v in args.m_list.split(",") if v]
    shapes = [(c, m) for c in ctxs for m in ms if m <= c]
    cfgs = build_configs(args)
    base = dict(BASELINE)

    print(f"# shapes: {shapes}")
    print(f"# configs: {len(cfgs)}")
    for c in cfgs:
        print(f"#   {cfg_to_spec(c)}")
    if args.dry_run:
        return 0

    t0 = time.time()
    if args.snapshot:
        # Pin the code under test: the tracked file may be edited concurrently.
        import importlib.util

        spec = importlib.util.spec_from_file_location("s22_tua_snapshot", args.snapshot)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["s22_tua_snapshot"] = mod
        spec.loader.exec_module(mod)
    else:
        import vllm.v1.attention.ops.triton_unified_attention as mod

    import hashlib

    with open(mod.__file__, "rb") as fh:
        module_md5 = hashlib.md5(fh.read()).hexdigest()

    orig = {
        "k2d": mod.kernel_unified_attention_2d,
        "k3d": mod.kernel_unified_attention_3d,
        "prefill_block_m": mod._prefill_block_m,
        "get_tile_size": mod._get_tile_size,
        "ENABLE_GFX906_ATTN_PREFILL_3D": mod.ENABLE_GFX906_ATTN_PREFILL_3D,
        "GFX906_PREFILL_SEGMENTS": mod.GFX906_PREFILL_SEGMENTS,
    }
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    dtype_bytes = 2
    scale = HEAD_DIM ** -0.5
    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    meta = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "multi_processor_count": prop.multi_processor_count,
        "l2_cache_bytes": prop.L2_cache_size,
        "total_mem_gb": round(prop.total_memory / 2**30, 1),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "python": platform.python_version(),
        "dtype": args.dtype,
        "block_size": args.block_size,
        "block_map": args.block_map,
        "seed": args.seed,
        "iters": args.iters,
        "warmup": args.warmup,
        "samples": args.samples,
        "gfx906_sched_tuning": mod._gfx906_sched_tuning_enabled(),
        "prefill_block_m_stock": mod._prefill_block_m(HEAD_DIM, NUM_QUERY_HEADS),
        "tile_size_stock": mod._get_tile_size(HEAD_DIM, 0, 2, True),
        "env_VLLM_GFX906_TRITON_ATTN_TUNING": os.getenv("VLLM_GFX906_TRITON_ATTN_TUNING", ""),
        "env_VLLM_GFX906_TRITON_ATTN_PREFILL_3D": os.getenv("VLLM_GFX906_TRITON_ATTN_PREFILL_3D", ""),
        "hip_visible_devices": os.getenv("HIP_VISIBLE_DEVICES", ""),
        "triton_cache_dir": os.getenv("TRITON_CACHE_DIR", ""),
        "module_file": mod.__file__,
        "module_md5": module_md5,
        "repo_head": json.loads(os.popen("git -C /src rev-parse HEAD 2>/dev/null || echo '\"\"'").read() or '""'),
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }

    ref_specs = [s for s in (args.ref_check or "").split(",") if s]
    ref_shapes = {tuple(int(x) for x in s.split(":")) for s in ref_specs}
    rows, ref_rows = [], []
    base_out = {}

    for (ctx, m) in shapes:
        print(f"\n=== shape ctx={ctx} M={m} (context_len={ctx - m}) ===", flush=True)
        inp = build_inputs(ctx, m, args.block_size, dtype, args.block_map, args.seed, device)
        torch.cuda.synchronize()

        if (ctx, m) in ref_shapes:
            t = time.time()
            r = reference_attention(inp["q"], inp["k"], inp["v"], inp["block_table"],
                                    args.block_size, ctx, m, scale)
            torch.cuda.synchronize()
            ref_rows.append({"ctx": ctx, "M": m, "reference_seconds": round(time.time() - t, 2)})
            inp["ref"] = r
            print(f"  [ref] torch reference computed in {time.time() - t:.1f}s", flush=True)

        order = [base] + [c for c in cfgs if cfg_key(c) != cfg_key(base)]
        for cfg in order:
            label = cfg_to_spec(cfg)
            patch_module(mod, cfg, orig)
            # force compilation / first-call allocation before timing
            tc = time.time()
            inp["out"].zero_()
            try:
                call_unified(mod, inp, ctx, m, scale)
                torch.cuda.synchronize()
            except Exception as exc:  # e.g. Triton OutOfResources on LDS
                torch.cuda.synchronize()
                msg = f"{type(exc).__name__}: {str(exc)[:200]}"
                print(f"  {label:>18}  FAILED {msg}", flush=True)
                rows.append({
                    "config": label, "path": cfg["path"], "block_m": cfg["block_m"],
                    "tile": cfg["tile"], "warps": cfg["warps"], "stages": cfg["stages"],
                    "segments": cfg["segments"], "ctx": ctx, "M": m, "context_len": ctx - m,
                    "failed": True, "error": msg, "finite": False,
                    "is_baseline": cfg_key(cfg) == cfg_key(base),
                })
                continue
            compile_s = time.time() - tc
            kstats = kernel_stats(
                mod.kernel_unified_attention_3d if cfg["path"] == "3d"
                else mod.kernel_unified_attention_2d
            )

            out_ok, finite, maxdiff_base, maxdiff_ref = True, True, float("nan"), float("nan")
            flag = ""
            if not args.no_consistency:
                finite = bool(torch.isfinite(inp["out"]).all().item())
                if not finite:
                    out_ok, flag = False, " NONFINITE"
                if cfg_key(cfg) == cfg_key(base):
                    base_out[(ctx, m)] = inp["out"].clone()
                elif (ctx, m) in base_out:
                    maxdiff_base = float((inp["out"].float() - base_out[(ctx, m)].float()).abs().max().item())
                    if maxdiff_base > 0.05:
                        flag += " DIFFvsBASE"
            if "ref" in inp:
                maxdiff_ref = float((inp["out"].float() - inp["ref"].float()).abs().max().item())
                if maxdiff_ref > 0.05:
                    flag += " DIFFvsREF"
            if not finite:
                continue

            for _ in range(max(0, args.warmup)):
                call_unified(mod, inp, ctx, m, scale)
            torch.cuda.synchronize()
            samples = []
            for _ in range(max(1, args.samples)):
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record()
                for _ in range(max(1, args.iters)):
                    call_unified(mod, inp, ctx, m, scale)
                en.record()
                torch.cuda.synchronize()
                samples.append(st.elapsed_time(en) / max(1, args.iters))
            ms_min, ms_med = min(samples), sorted(samples)[len(samples) // 2]
            mt = metrics(cfg, ctx, m, ms_min, dtype_bytes)
            row = {
                "config": label,
                "path": cfg["path"],
                "block_m": cfg["block_m"],
                "tile": cfg["tile"],
                "warps": cfg["warps"],
                "stages": cfg["stages"],
                "segments": cfg["segments"],
                "ctx": ctx,
                "M": m,
                "context_len": ctx - m,
                "ms_call": round(ms_min, 4),
                "ms_median": round(ms_med, 4),
                "ms_samples": [round(s, 4) for s in samples],
                "is_baseline": cfg_key(cfg) == cfg_key(base),
                "compile_s": round(compile_s, 2),
                "kern": kstats,
                "maxdiff_vs_baseline": maxdiff_base,
                "maxdiff_vs_reference": maxdiff_ref,
                "finite": finite,
                "flag": flag.strip(),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in mt.items()},
            }
            rows.append(row)
            print(
                f"  {label:>18}  {row['ms_call']:9.3f} ms  "
                f"useful {row['tflops_useful']:6.2f} TF  issued {row['tflops_issued']:6.2f} TF  "
                f"{row['gbs_kv']:8.1f} GB/s  tiles {row['issued_kv_tiles']:>7}  "
                f"regs {kstats.get('n_regs', -1)}/{kstats.get('n_spills', -1)}sp  "
                f"lds {kstats.get('shared', -1)}  "
                f"compile {compile_s:6.1f}s{('  [' + flag.strip() + ']') if flag else ''}",
                flush=True,
            )
        del inp
        torch.cuda.empty_cache()

    # restore module state
    mod.kernel_unified_attention_2d = orig["k2d"]
    mod.kernel_unified_attention_3d = orig["k3d"]
    mod._prefill_block_m = orig["prefill_block_m"]
    mod._get_tile_size = orig["get_tile_size"]
    mod.ENABLE_GFX906_ATTN_PREFILL_3D = orig["ENABLE_GFX906_ATTN_PREFILL_3D"]
    mod.GFX906_PREFILL_SEGMENTS = orig["GFX906_PREFILL_SEGMENTS"]

    base_by_shape = {}
    for r in rows:
        if r.get("is_baseline") and "ms_call" in r:
            base_by_shape[(r["ctx"], r["M"])] = r["ms_call"]
    for r in rows:
        if "ms_call" not in r:
            r["speedup_vs_baseline"] = None
            continue
        b = base_by_shape.get((r["ctx"], r["M"]))
        r["speedup_vs_baseline"] = round(b / r["ms_call"], 4) if b else None

    print_table(rows)
    payload = {
        "meta": meta,
        "args": vars(args),
        "elapsed_s": round(time.time() - t0, 1),
        "reference": ref_rows,
        "rows": rows,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\n[probe] wrote {args.out}")
    return 0


def print_table(rows):
    hdr = (f"{'config':>18} {'ctx':>6} {'M':>6} {'ms/call':>9} {'xbase':>6} "
           f"{'TF-useful':>9} {'TF-issued':>9} {'GB/s':>8} {'tiles':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["ctx"], r["M"], r.get("ms_call", 1e9))):
        if "ms_call" not in r:
            print(f"{r['config']:>18} {r['ctx']:>6} {r['M']:>6} {'FAILED':>9} "
                  f"{'-':>6} {'-':>9} {'-':>9} {'-':>8} {'-':>7}  {r.get('error', '')[:40]}")
            continue
        xb = f"{r['speedup_vs_baseline']:.2f}" if r.get("speedup_vs_baseline") else "-"
        mark = "*" if r["is_baseline"] else " "
        print(f"{r['config']:>18} {r['ctx']:>6} {r['M']:>6} {r['ms_call']:>9.3f} {xb:>6} "
              f"{r['tflops_useful']:>9.2f} {r['tflops_issued']:>9.2f} {r['gbs_kv']:>8.1f} "
              f"{r['issued_kv_tiles']:>7}{mark}")


if __name__ == "__main__":
    sys.exit(main())
