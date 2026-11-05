#!/usr/bin/env python3
"""Parse a rank0 torch-profiler chrome trace from the z100 no-MTP decode window.

v2: excludes the prefill annotation from per-step stats, splits kv-cache ops
out of the attention bucket, counts aten::item syncs per step, attributes
in-step GPU idle gaps to the enclosing CPU op, and reports per-stream busy.

Usage: parse_trace.py <trace.json.gz> [--out-dir DIR] [--tag TAG]
"""
import argparse
import gzip
import json
import os
import re
import statistics
from collections import defaultdict

BUCKETS = [
    ("fp8_gemm", r"fp8_emu_gemv"),
    ("bf16_gemm", r"bf16_emu_gemv|Cijk_|hipblaslt|gemm|gemv"),
    ("attn_core", r"kernel_unified_attention"),
    ("kv_cache", r"unified_kv_cache|reshape_and_cache"),
    ("gdn", r"gated_delta|causal_conv1d|fused_recurrent|selector_walk|conv1d"),
    ("allreduce", r"cross_device_reduce|ncclDevKernel|all_reduce|all_gather"),
    ("norm_elemwise", r"rms_norm|rope|triton_red_fused|triton_poi_fused|triton_per_fused"
                      r"|silu|elementwise|reduce_kernel|CATOMIC|atomic"),
    ("sampling", r"sampling|argmax|multinomial|softmax|topk|penalt|sort"),
    ("memcpy", r"Memcpy|memcpy|copy_"),
    ("other", r"."),
]


def bucket_of(name: str) -> str:
    for b, pat in BUCKETS:
        if re.search(pat, name):
            return b
    return "other"


def union(intervals):
    if not intervals:
        return []
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def busy(intervals):
    return sum(e - s for s, e in union(intervals))


def gaps(intervals, min_gap_us=50.0):
    u = union(intervals)
    return [(u[i][1], u[i + 1][0], u[i + 1][0] - u[i][1])
            for i in range(len(u) - 1) if u[i + 1][0] - u[i][1] >= min_gap_us]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tag", default="s34nospec")
    args = ap.parse_args()

    with gzip.open(args.trace, "rt") as fh:
        doc = json.load(fh)
    ev = doc["traceEvents"] if isinstance(doc, dict) else doc

    steps, kernels, memcpy, cpu_ops = [], [], [], []
    for e in ev:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        cat = e.get("cat", "")
        ts, dur = e.get("ts", 0), e.get("dur", 0)
        if cat == "kernel":
            kernels.append((ts, ts + dur, name, e.get("args", {}).get("stream")))
        elif cat in ("gpu_memcpy", "gpu_memset"):
            memcpy.append((ts, ts + dur, name, e.get("args", {}).get("stream")))
        elif name.startswith("execute_context") and cat in ("cpu_op", "user_annotation"):
            steps.append((ts, ts + dur, name))
        elif cat == "cpu_op":
            cpu_ops.append((ts, ts + dur, name, dur))

    steps.sort()
    kernels.sort()
    all_gpu = [(s, e) for s, e, _, _ in kernels] + [(s, e) for s, e, _, _ in memcpy]
    t0 = min([s[0] for s in steps] + [k[0] for k in kernels])
    t1 = max([s[1] for s in steps] + [k[1] for k in kernels])
    span = t1 - t0
    print(f"trace: {args.trace}")
    print(f"span={span/1000:.1f}ms steps={len(steps)} kernels={len(kernels)}")

    # prefill = first (or any) annotation with generation_0(0) / huge wall
    decode = [s for s in steps if (s[1] - s[0]) < 100_000]
    prefill = [s for s in steps if (s[1] - s[0]) >= 100_000]
    print(f"decode steps={len(decode)} prefill annotations={len(prefill)} "
          f"(walls {[round((p[1]-p[0])/1000) for p in prefill]}ms)")

    # smallest-enclosing-cpu-op lookup for gap attribution
    big_ops = [c for c in cpu_ops if c[3] >= 100 and "execute_context" not in c[2]]

    def enclosing(ts, te):
        best = None
        for s, e, nm, d in big_ops:
            if s <= ts and e >= te:
                if best is None or d < best[3]:
                    best = (s, e, nm, d)
        return best[2] if best else "<python>"

    rows = []
    item_per_step, item_ms_per_step = [], []
    for i, (ts, te, nm) in enumerate(decode):
        win_k = [k for k in kernels if ts <= k[0] < te]
        win_m = [m for m in memcpy if ts <= m[0] < te]
        win_gpu = [(s, e) for s, e, _, _ in win_k] + [(s, e) for s, e, _, _ in win_m]
        cat_busy = defaultdict(float)
        for b in {bucket_of(n) for _, _, n, _ in win_k}:
            cat_busy[b] = busy([(s, e) for s, e, n, _ in win_k if bucket_of(n) == b]) / 1000
        streams = defaultdict(lambda: [0.0])
        for s, e, n, st in win_k:
            streams[st][0] += e - s
        g = gaps(win_gpu)
        gap_attr = defaultdict(float)
        for a, b, d in g:
            gap_attr[enclosing(a, b)] += d
        items = [c for c in cpu_ops if c[2] == "aten::item" and ts <= c[0] < te]
        item_per_step.append(len(items))
        item_ms_per_step.append(sum(c[3] for c in items) / 1000)
        inter = (decode[i + 1][0] - te) / 1000 if i + 1 < len(decode) else 0.0
        rows.append(dict(
            i=i, cpu_wall_ms=round((te - ts) / 1000, 2),
            gpu_busy_ms=round(busy(win_gpu) / 1000, 2),
            gpu_span_ms=round((max(e for _, e, _, _ in win_k) -
                               min(s for s, _, _, _ in win_k)) / 1000, 2),
            inter_cpu_gap_ms=round(inter, 2),
            cats={k: round(v, 3) for k, v in cat_busy.items()},
            streams={str(k): round(v[0] / 1000, 2) for k, v in streams.items()},
            item_calls=len(items), item_ms=round(item_ms_per_step[-1], 2),
            gap_attr_ms={k: round(v / 1000, 2) for k, v in
                         sorted(gap_attr.items(), key=lambda kv: -kv[1])[:4]}))

    med = lambda v: statistics.median(v) if v else 0
    n = len(rows)
    print(f"\n== per-step medians over {n} decode steps ==")
    print(f"cpu_wall={med([r['cpu_wall_ms'] for r in rows]):.2f}ms  "
          f"gpu_busy={med([r['gpu_busy_ms'] for r in rows]):.2f}ms  "
          f"gpu_span={med([r['gpu_span_ms'] for r in rows]):.2f}ms  "
          f"inter_cpu_gap={med([r['inter_cpu_gap_ms'] for r in rows]):.2f}ms")
    print(f"aten::item: {med(item_per_step):.0f} calls/step, "
          f"{med(item_ms_per_step):.2f}ms/step CPU (self=sync wait)")

    cat_tot = defaultdict(float)
    for r in rows:
        for k, v in r["cats"].items():
            cat_tot[k] += v
    tot = sum(cat_tot.values()) or 1
    print("\n== GPU kernel time by category (union-based busy, ms/step) ==")
    for k, v in sorted(cat_tot.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v/n:7.3f}  ({100*v/tot:4.1f}%)")

    st_tot = defaultdict(float)
    for r in rows:
        for k, v in r["streams"].items():
            st_tot[k] += v
    print("\n== GPU kernel time by stream (sum of durations, ms/step) ==")
    for k, v in sorted(st_tot.items(), key=lambda kv: -kv[1]):
        print(f"  stream {k:<4} {v/n:7.3f}")

    gap_tot = defaultdict(float)
    for r in rows:
        for k, v in r["gap_attr_ms"].items():
            gap_tot[k] += v
    print("\n== in-step GPU idle gaps >=50us, attributed to enclosing CPU op (ms/step) ==")
    for k, v in sorted(gap_tot.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {v/n:7.3f}  {k[:90]}")

    ktot = defaultdict(lambda: [0.0, 0])
    for s, e, nm, _ in kernels:
        ktot[nm][0] += e - s
        ktot[nm][1] += 1
    print("\n== top 14 kernels (whole window incl. prefill) ==")
    for nm, (tot_k, cnt) in sorted(ktot.items(), key=lambda kv: -kv[1][0])[:14]:
        print(f"  {tot_k/1000:9.2f}ms {cnt:6d}x {tot_k/max(cnt,1):8.1f}us  {nm[:76]}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"steps-{args.tag}.json"), "w") as fh:
        json.dump(dict(tag=args.tag, rows=rows,
                       medians=dict(cpu_wall_ms=med([r['cpu_wall_ms'] for r in rows]),
                                    gpu_busy_ms=med([r['gpu_busy_ms'] for r in rows]),
                                    gpu_span_ms=med([r['gpu_span_ms'] for r in rows]),
                                    inter_cpu_gap_ms=med([r['inter_cpu_gap_ms'] for r in rows]),
                                    item_calls=med(item_per_step),
                                    item_ms=med(item_ms_per_step)),
                       cat_ms_per_step={k: round(v / n, 3) for k, v in cat_tot.items()},
                       gap_attr_ms_per_step={k: round(v / n, 3) for k, v in gap_tot.items()}),
                  fh, indent=1)
    print(f"\nwrote {args.out_dir}/steps-{args.tag}.json")


if __name__ == "__main__":
    main()
