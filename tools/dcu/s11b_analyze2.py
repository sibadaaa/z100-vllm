#!/usr/bin/env python3
"""S11b: independent re-analysis of torch-profiler chrome traces (stdlib only).

Re-derives the category table, top-20 kernels, per-step GPU/CPU split and the
GEMM shape inventory from the raw .pt.trace.json files, so the S11b archive does
not depend on any single (possibly lost) analysis script.

Usage:
  python3 tools/dcu/s11b_analyze2.py <trace.json> [<trace.json> ...]
"""
import collections
import glob
import json
import re
import sys

CATS = [
    ("gemm/int8", re.compile(
        r"scaled_mm|triton_matmul|cutlass|wgemm|shapes_.*matmul|dynamic_scaled_int8_quant|"
        r"int8_quant|quant_kernel|per_token_group_quant|skinn", re.I)),
    ("gdn+mamba", re.compile(
        r"sigmoid_gating|delta_rule|causal_conv1d|mamba|gdn|linear_attention|qwen_gdn", re.I)),
    ("attention", re.compile(
        r"unified_attention|reshape_and_cache|paged_attention|flash_attn|_attn|rope|rotary|"
        r"block_table|slot_mapping|kv_cache", re.I)),
    ("norm+activation", re.compile(
        r"rms_norm|layernorm|layer_norm|_norm|silu|gelu|act_and_mul|swiglu", re.I)),
    ("sampling", re.compile(
        r"gumbel|sampled|topk|top_k|argmax|reduce_kernel|_topp|penalt|logprob", re.I)),
]


def categorize(name):
    for cat, rx in CATS:
        if rx.search(name):
            return cat
    return "其它"


def union_len(intervals):
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    total = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    return total + (ce - cs)


def load(path):
    d = json.load(open(path))
    return d


def analyze(path, steps_expected=32):
    d = load(path)
    ev = d["traceEvents"]
    name = d.get("traceName") or path.rsplit("/", 1)[-1]

    kernels = [e for e in ev if e.get("cat") == "kernel"]
    cpu = [e for e in ev if e.get("cat") in ("cpu_op", "cuda_runtime", "user_annotation")
           and e.get("pid") not in (1, 2)]
    ann = sorted([e for e in ev if e.get("cat") == "user_annotation"
                  and e["name"].startswith("execute_context")], key=lambda x: x["ts"])

    print(f"### {name}")
    print(f"events={len(ev)} kernels={len(kernels)} ({len(set(k['name'] for k in kernels))} unique) "
          f"cpu_ops={len(cpu)} step_annotations={len(ann)}")

    # ---- per-step windows from execute_context starts -------------------------
    bounds = [a["ts"] for a in ann]
    bounds.append(max(k["ts"] + k["dur"] for k in kernels))
    nwin = len(bounds) - 1
    print(f"\nper-step windows (n={nwin}) from execute_context annotations")
    print(f"{'win':>4} {'gpu_span':>9} {'gpu_busy':>9} {'cpu_union':>9} {'kernels':>8}")
    rows = []
    for i in range(nwin):
        lo, hi = bounds[i], bounds[i + 1]
        ks = [k for k in kernels if lo <= k["ts"] < hi]
        busy = sum(k["dur"] for k in ks) / 1000.0
        span = ((max(k["ts"] + k["dur"] for k in ks) - min(k["ts"] for k in ks)) / 1000.0
                if ks else 0.0)
        civ = [(c["ts"], c["ts"] + c["dur"]) for c in cpu if lo <= c["ts"] < hi]
        cu = union_len(civ) / 1000.0
        rows.append((busy, span, cu, len(ks)))
        if i < 3 or i >= nwin - 2:
            print(f"{i:>4} {span:>9.2f} {busy:>9.2f} {cu:>9.2f} {len(ks):>8}")

    # drop the first window: it contains prefill/compile residue
    steady = rows[1:] if len(rows) > 1 else rows
    n = len(steady)
    gpu_busy = sum(r[0] for r in steady) / n
    gpu_span = sum(r[1] for r in steady) / n
    cpu_union = sum(r[2] for r in steady) / n
    kper = sum(r[3] for r in steady) / n
    print(f"\nsteady-state per-step (n={n}): gpu_span={gpu_span:.2f} ms  gpu_busy={gpu_busy:.2f} ms  "
          f"cpu_union={cpu_union:.2f} ms  kernels/step={kper:.1f}")
    print(f"cpu_union as %% of gpu_span = {100.0*cpu_union/gpu_span:.2f}%")

    # ---- category table over steady windows ---------------------------------
    lo = bounds[1]
    hi = bounds[-1]
    ks = [k for k in kernels if lo <= k["ts"] < hi]
    agg = collections.defaultdict(lambda: [0.0, 0, collections.defaultdict(lambda: [0.0, 0])])
    for k in ks:
        c = categorize(k["name"])
        a = agg[c]
        a[0] += k["dur"] / 1000.0 / n
        a[1] += 1.0 / n
        a[2][k["name"]][0] += k["dur"] / 1000.0 / n
        a[2][k["name"]][1] += 1.0 / n
    tot = sum(a[0] for a in agg.values())
    print(f"\n#### category table (GPU kernel busy, % of {tot:.1f} ms/step)")
    print(f"{'category':<17}{'ms/step':>9}{'% gpu':>8}{'calls/step':>12}")
    for c, a in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"{c:<17}{a[0]:>9.2f}{100*a[0]/tot:>7.1f}%{a[1]:>12.1f}")
        top = sorted(a[2].items(), key=lambda kv: -kv[1][0])[:6]
        print("      " + ", ".join(f"{nm[:56]} ({ms:.3f} ms, {cn:.0f}/step)" for nm, (ms, cn) in top))
    print(f"{'TOTAL(gpu busy)':<17}{tot:>9.2f}{100.0:>7.1f}%{sum(a[1] for a in agg.values()):>12.1f}")

    # ---- top-20 kernels ------------------------------------------------------
    print("\n#### top-20 kernels by GPU self time")
    print(f"{'#':<4}{'kernel':<100}{'calls/step':>12}{'ms/step':>9}{'% gpu':>8}")
    per_kernel = collections.defaultdict(lambda: [0.0, 0.0])
    for k in ks:
        per_kernel[k["name"]][0] += k["dur"] / 1000.0 / n
        per_kernel[k["name"]][1] += 1.0 / n
    for i, (nm, (ms, cn)) in enumerate(sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:20], 1):
        print(f"{i:<4}{nm[:100]:<100}{cn:>12.1f}{ms:>9.3f}{100*ms/tot:>7.1f}%")

    # ---- GEMM shape inventory from kernel launch geometry -------------------
    print("\n#### scaled_mm_kernel launch geometry (grid/block -> shape proxy)")
    geo = collections.Counter()
    geo_ms = collections.defaultdict(float)
    for k in kernels:
        if "scaled_mm" in k["name"]:
            a = k.get("args", {})
            key = (tuple(a.get("grid") or ()), tuple(a.get("block") or ()))
            geo[key] += 1
            geo_ms[key] += k["dur"] / 1000.0
    for (g, b), c in geo.most_common(20):
        print(f"  grid={str(g):<22} block={str(b):<22} calls={c:<6} total_ms={geo_ms[(g,b)]:.1f}")

    print("\n#### other GEMM-ish kernels launch geometry")
    geo2 = collections.Counter()
    geo2_ms = collections.defaultdict(float)
    for k in kernels:
        if re.search(r"triton_matmul|wgemm|cutlass", k["name"], re.I):
            a = k.get("args", {})
            key = (k["name"][:60], tuple(a.get("grid") or ()), tuple(a.get("block") or ()))
            geo2[key] += 1
            geo2_ms[key] += k["dur"] / 1000.0
    for key, c in geo2.most_common(10):
        print(f"  {key[0]:<60} grid={str(key[1]):<20} block={str(key[2]):<18} calls={c:<6} total_ms={geo2_ms[key]:.1f}")

    # ---- CPU side ------------------------------------------------------------
    print("\n#### CPU side (steady windows, per step)")
    cpu_only = [c for c in cpu if c.get("cat") in ("cpu_op", "cuda_runtime", "user_annotation")]
    hi2 = bounds[-1]
    cks = [c for c in cpu_only if lo <= c["ts"] < hi2]
    per_cpu = collections.defaultdict(lambda: [0.0, 0.0])
    for c in cks:
        per_cpu[(c["cat"], c["name"])][0] += c["dur"] / 1000.0 / n
        per_cpu[(c["cat"], c["name"])][1] += 1.0 / n
    cpu_total = sum(v[0] for v in per_cpu.values())
    print(f"  CPU op self-time sum  = {cpu_total:.2f} ms/step")
    print(f"  CPU union (busy wall) = {cpu_union:.2f} ms/step  = {100*cpu_union/gpu_span:.2f}% of step")
    print(f"  {'cat':<14}{'op':<52}{'ms/step':>9}{'calls/step':>12}")
    for (cat, nm), (ms, cn) in sorted(per_cpu.items(), key=lambda kv: -kv[1][0])[:20]:
        print(f"  {cat:<14}{nm[:52]:<52}{ms:>9.3f}{cn:>12.1f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("/mnt/game/port/s11b/run-default-z23b/*.trace.json"))
    for p in args:
        analyze(p)
        print("\n" + "=" * 110 + "\n")
