#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11b (P0): classify the profiled 27B decode kernels into categories.

Reads a run directory produced by ``s11b_profile_27b.py``:

* ``s11b-<mode>_*.pt.trace.json`` - per-worker chrome traces (rank 0 + rank 1)
* ``profiler_out_<rank>.txt``     - per-worker key_averages table (cross-check)
* ``result.json``                 - driver-side step timings and config

and writes ``analysis-rank0.txt`` / ``analysis-rank1.txt`` plus prints rank 0.

    python s11b_analyze.py /mnt/game/port/s11b/run-default 32
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

# Ordered: the first matching rule wins, so the more specific paths come first.
RULES: list[tuple[str, str]] = [
    (
        "attention",
        r"attention|attn|flash|paged|kv_cache|reshape_and_cache|rotary|rope|"
        r"unified_attention|mla",
    ),
    (
        "gdn+mamba",
        r"gdn|mamba|ssm|selective_scan|causal_conv|conv1d|chunk|recurrent|"
        r"gated_delta|linear_attention|l2norm|cumsum|gating|delta_rule",
    ),
    (
        "gemm/int8",
        r"gemm|matmul|scaled_mm|sdot|int8|\bmm\b|addmm|\bbmm\b|LLMM1|wvSplitK|"
        r"Tensile|rocblas|Cijk|dot|GEMV|hipblas|Cijk_",
    ),
    (
        "norm+activation",
        r"norm|silu|gelu|swiglu|act_and_mul|activation|sigmoid|relu",
    ),
    (
        "sampling",
        r"sampl|argmax|topk|topp|gumbel|penalt|logprob|softmax|logits|"
        r"multinomial|temperature",
    ),
]

CPU_CATS = {"cpu_op", "user_annotation", "python_function"}


def classify(name: str) -> str:
    for label, pattern in RULES:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return "其它"


def fmt_table(rows: list[tuple], header: tuple) -> str:
    if not rows:
        return "(empty)"
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    lines = [
        "  ".join(str(h).ljust(w) for h, w in zip(header, widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows]
    return "\n".join(lines)


def union_ms(intervals: list[tuple[float, float]]) -> float:
    """Merged length (us -> ms) of possibly overlapping intervals."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    total += cur_hi - cur_lo
    return total / 1e3


def analyze_trace(path: str, steps: int, label: str) -> tuple[str, dict]:
    with open(path) as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", [])

    kernels = [e for e in events if e.get("cat") == "kernel"]
    by_name: dict[str, dict] = defaultdict(lambda: {"count": 0, "dur": 0.0})
    for e in kernels:
        rec = by_name[e["name"]]
        rec["count"] += 1
        rec["dur"] += e.get("dur", 0.0)

    ranked = sorted(by_name.items(), key=lambda kv: -kv[1]["dur"])

    # Graph launches give the exact per-step boundaries when graphs are on.
    launches = sorted(
        (
            e
            for e in events
            if re.search(r"GraphLaunch|graph_launch", str(e.get("name", "")))
        ),
        key=lambda e: e["ts"],
    )
    # Fall back to the periodic kernel stream if no launch events are visible.
    per_step = []
    if launches:
        lo = launches[0]["ts"]
        hi = launches[-1]["ts"] + launches[-1].get("dur", 0.0)
        step_windows: list[tuple[float, float]] = []
        for i, e in enumerate(launches[:-1]):
            step_windows.append((e["ts"], launches[i + 1]["ts"]))
        for idx, (a, b) in enumerate(step_windows):
            kdur = sum(e["dur"] for e in kernels if a <= e["ts"] < b)
            cpu_iv = [
                (e["ts"], e["ts"] + e.get("dur", 0.0))
                for e in events
                if e.get("cat") in CPU_CATS and a <= e["ts"] < b
            ]
            per_step.append(
                {
                    "window": idx,
                    "gpu_ms": kdur / 1e3,
                    "cpu_ms": union_ms(cpu_iv),
                    "n_kernels": sum(1 for e in kernels if a <= e["ts"] < b),
                }
            )
        mode_note = (
            f"per-step windows from {len(launches)} graph launches "
            f"(ts {lo:.0f} .. {hi:.0f})"
        )
    else:
        mode_note = "no graph launch events: per-step values are window totals / steps"

    total_kernel_ms = sum(e.get("dur", 0.0) for e in kernels) / 1e3
    cpu_iv_all = [
        (e["ts"], e["ts"] + e.get("dur", 0.0))
        for e in events
        if e.get("cat") in CPU_CATS
    ]
    cpu_ms_total = union_ms(cpu_iv_all)

    out = [f"### trace {os.path.basename(path)} ({label})"]
    out.append(
        f"events={len(events)} kernels={len(kernels)} ({len(by_name)} unique)  "
        f"kernel_busy={total_kernel_ms:.1f} ms  cpu_union={cpu_ms_total:.1f} ms"
    )
    out.append(mode_note)
    if per_step:
        g = [p["gpu_ms"] for p in per_step]
        c = [p["cpu_ms"] for p in per_step]
        k = [p["n_kernels"] for p in per_step]
        out.append(
            f"per-step (n={len(per_step)}): gpu {sum(g) / len(g):.1f} ms "
            f"[{min(g):.1f}..{max(g):.1f}], cpu_union {sum(c) / len(c):.1f} ms, "
            f"kernels/step {sum(k) / len(k):.1f}"
        )
        out.append("")
        out.append(
            fmt_table(
                [
                    (
                        p["window"],
                        f"{p['gpu_ms']:.1f}",
                        f"{p['cpu_ms']:.1f}",
                        p["n_kernels"],
                    )
                    for p in per_step
                ],
                ("window", "gpu ms", "cpu union ms", "kernels"),
            )
        )
    else:
        out.append(
            f"window totals / {steps} steps: gpu {total_kernel_ms / steps:.1f} ms/step, "
            f"cpu_union {cpu_ms_total / steps:.1f} ms/step, "
            f"kernels/step {len(kernels) / steps:.1f}"
        )
        g = [total_kernel_ms / steps]
        c = [cpu_ms_total / steps]

    # Category table (GPU kernel busy only; the CPU side is measured separately).
    gpu_per_step = total_kernel_ms / steps
    per_cat: dict[str, dict] = defaultdict(lambda: {"ms": 0.0, "count": 0, "names": []})
    for name, rec in by_name.items():
        cat = classify(name)
        per_cat[cat]["ms"] += rec["dur"] / 1e3
        per_cat[cat]["count"] += rec["count"]
        per_cat[cat]["names"].append((name, rec))
    rows = []
    for cat in sorted(per_cat, key=lambda c_: -per_cat[c_]["ms"]):
        ms = per_cat[cat]["ms"]
        rows.append(
            (
                cat,
                f"{ms / steps:.1f}",
                f"{ms:.0f}",
                f"{100 * ms / total_kernel_ms:.1f}%",
                f"{per_cat[cat]['count'] / steps:.1f}",
            )
        )
    rows.append(
        (
            "TOTAL(gpu busy)",
            f"{gpu_per_step:.1f}",
            f"{total_kernel_ms:.0f}",
            "100.0%",
            f"{len(kernels) / steps:.1f}",
        )
    )
    out.append("")
    out.append("#### category table (GPU kernel busy; % of GPU busy)")
    out.append(
        fmt_table(rows, ("category", "ms/step", "ms/window", "% gpu", "calls/step"))
    )
    out.append("")
    for cat in sorted(per_cat, key=lambda c_: -per_cat[c_]["ms"]):
        top = sorted(per_cat[cat]["names"], key=lambda kv: -kv[1]["dur"])[:5]
        out.append(
            f"  {cat}: "
            + ", ".join(
                f"{n[:58]} ({r['dur'] / 1e3 / steps:.3f} ms/step, {r['count'] / steps:.1f}/step)"
                for n, r in top
            )
        )
    out.append("")
    out.append("#### top-20 kernels by GPU self time")
    out.append(
        fmt_table(
            [
                (
                    i + 1,
                    name[:110],
                    f"{rec['count'] / steps:.1f}",
                    f"{rec['dur'] / 1e3 / steps:.3f}",
                    f"{100 * rec['dur'] / 1e3 / total_kernel_ms:.1f}%",
                )
                for i, (name, rec) in enumerate(ranked[:20])
            ],
            ("#", "kernel", "calls/step", "ms/step", "% gpu"),
        )
    )
    out.append("")
    out.append("#### unmatched -> 其它, top-25")
    out.append(
        fmt_table(
            [
                (
                    name[:110],
                    f"{rec['count'] / steps:.1f}",
                    f"{rec['dur'] / 1e3 / steps:.3f}",
                )
                for name, rec in ranked
                if classify(name) == "其它"
            ][:25],
            ("kernel", "calls/step", "ms/step"),
        )
    )
    out.append("")
    out.append("#### op-name scan (gfx906-specific / quantized paths)")
    for pattern in (
        "gfx906",
        "packed",
        "sigmoid_gating",
        "sdot",
        "int8",
        "fp8",
        "gptq",
        "marlin",
        "moe",
        "sglang",
    ):
        hits = [(n, r) for n, r in ranked if re.search(pattern, n, re.IGNORECASE)]
        total = sum(r["dur"] for _, r in hits) / 1e3
        out.append(
            f"  {pattern:15s} hits={len(hits):4d} total_ms={total:9.1f} "
            f"names={[n[:56] for n, _ in hits[:3]]}"
        )

    stats = {
        "kernel_busy_ms": total_kernel_ms,
        "gpu_ms_per_step": gpu_per_step,
        "cpu_union_ms": cpu_ms_total,
        "cpu_ms_per_step": cpu_ms_total / steps,
        "kernels_per_step": len(kernels) / steps,
        "per_step": per_step,
        "n_launches": len(launches),
    }
    return "\n".join(out), stats


def gemm_shapes_from_cpu_ops(path: str, steps: int) -> str:
    """aten/int8 GEMM shapes, taken from CPU-side op args (record_shapes=True)."""
    with open(path) as fh:
        trace = json.load(fh)
    groups: dict[tuple[str, str], dict] = defaultdict(lambda: {"count": 0, "dur": 0.0})
    for e in trace.get("traceEvents", []):
        if e.get("cat") not in CPU_CATS:
            continue
        name = str(e.get("name", ""))
        if not re.search(
            r"scaled_mm|aten::mm|addmm|matmul|gemm|int8|reshape_and_cache",
            name,
            re.IGNORECASE,
        ):
            continue
        args = e.get("args", {})
        dims = args.get("Input Dims") or args.get("Input Dims (self)")
        key = (name, str(dims))
        groups[key]["count"] += 1
        groups[key]["dur"] += e.get("dur", 0.0)
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["dur"])
    if not ranked:
        return "(no CPU-side GEMM ops in the window - expected with FULL cudagraph)"
    return fmt_table(
        [
            (n[:70], str(d)[:60], f"{r['count'] / steps:.2f}", f"{r['dur'] / 1e3 / steps:.3f}")
            for (n, d), r in ranked[:40]
        ],
        ("cpu op", "input dims", "calls/step", "cpu ms/step"),
    )


def shape_inventory(run_dir: str, steps: int) -> str:
    """GEMM shape inventory from the wrappers installed by the profile script.

    Only `triton_matmul` (the unquantized fp16 path, a real custom-op
    boundary) is recorded per call; the int8 `scaled_mm_kernel` is traced into
    the compiled graph, so its shapes come from the inductor-generated code
    instead (see the run log with TORCH_LOGS=output_code).
    """
    files = sorted(glob.glob(os.path.join(run_dir, "shapes_*.tsv")))
    if not files:
        return "(no shapes_*.tsv found)"
    per: dict[tuple, int] = defaultdict(int)
    for path in files:
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                label = parts[3] if len(parts) > 3 else ""
                per[(os.path.basename(path).split("_pid")[0], parts[0], parts[1], parts[2], label)] += 1
    rows = []
    for (kind, m, k, n, label), count in sorted(per.items(), key=lambda kv: -kv[1]):
        rows.append((kind, m, k, n, label[:44], count, f"{count / steps:.2f}"))
    return fmt_table(
        rows,
        ("kind", "M", "K", "N", "label", "calls", "calls/step"),
    )


def main() -> None:
    run_dir = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    result = {}
    rpath = os.path.join(run_dir, "result.json")
    if os.path.exists(rpath):
        with open(rpath) as fh:
            result = json.load(fh)

    traces = sorted(glob.glob(os.path.join(run_dir, "*.pt.trace.json")))
    profiles = sorted(glob.glob(os.path.join(run_dir, "*.pt.trace.json.gz")))
    traces += profiles

    summary = [f"# S11b analysis: {run_dir}"]
    summary.append(f"mode={result.get('mode')} torch={result.get('torch')}")
    summary.append(f"resolved={json.dumps(result.get('resolved'))}")
    sync = result.get("sync_run", {})
    prof_run = result.get("profiled_run", {})
    summary.append("")
    summary.append("## 0. driver-side timings (engine.step)")
    if sync:
        summary.append(
            f"sync run: decode wall = {sync.get('mean_wall_ms_per_step'):.2f} ms/step, "
            f"driver cpu span = {sync.get('mean_cpu_ms_per_step'):.2f} ms/step, "
            f"prefill = {sync.get('prefill_ms'):.1f} ms, tail = {json.dumps(sync.get('tail'))}"
        )
    if prof_run:
        summary.append(
            f"profiled run: driver cpu span = {prof_run.get('driver_cpu_span_mean_ms'):.2f} "
            f"ms/step (no per-step sync), wall = "
            f"{prof_run.get('driver_wall_ms_per_step_synced'):.2f} ms/step, "
            f"tail = {json.dumps(prof_run.get('tail'))}"
        )
    summary.append(f"warmup={json.dumps(result.get('warmup'))}")
    summary.append(f"traces={[os.path.basename(t) for t in traces]}")
    summary.append("")
    summary.append("## GEMM shape inventory (runtime-recorded custom ops)")
    summary.append(shape_inventory(run_dir, steps))

    per_rank_stats = {}
    for path in traces:
        base = os.path.basename(path)
        label = base.split(".")[0]
        text, stats = analyze_trace(path, steps, label)
        per_rank_stats[base] = stats
        out_path = os.path.join(run_dir, f"analysis-{label}.txt")
        with open(out_path, "w") as fh:
            fh.write(text + "\n")
        summary.append("")
        summary.append(text.replace("### trace", "### trace"))
        if "rank" in label.lower() or "tp0" in label.lower():
            summary.append("")
            summary.append("## GEMM / attention op shapes (CPU-side args, rank 0)")
            summary.append(gemm_shapes_from_cpu_ops(path, steps))

    with open(os.path.join(run_dir, "analysis-summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    with open(os.path.join(run_dir, "analysis-stats.json"), "w") as fh:
        json.dump(per_rank_stats, fh, indent=1)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
