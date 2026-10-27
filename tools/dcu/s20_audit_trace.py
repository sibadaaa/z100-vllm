#!/usr/bin/env python3
"""S20 fixed-cost audit: streaming, low-memory census of a vLLM/kineto chrome trace.

All reported quantities are PER ENGINE STEP (mean over the selected ProfilerStep
windows).  The trace nests three ProfilerStep#N ranges per engine step, so the
outermost (max-duration) range per name is used.

Usage
  python3 s20_audit_trace.py steps  TRACE
  python3 s20_audit_trace.py census TRACE --kind decode [--top 25] [--json OUT]
  python3 s20_audit_trace.py detail TRACE --kind decode
  python3 s20_audit_trace.py window TRACE --ts0 A --ts1 B
"""
import argparse
import bisect
import collections
import gzip
import json
import re
import sys

CHUNK = 1 << 20
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel"}
RUNTIME_CAT = "cuda_runtime"

# exact-name blocking APIs (substring matching wrongly catches hipMemcpyAsync)
SYNC_APIS = {
    "hipStreamSynchronize", "hipDeviceSynchronize", "cudaStreamSynchronize",
    "cudaDeviceSynchronize", "hipEventSynchronize", "cudaEventSynchronize",
    "hipStreamQuery", "hipMemcpyDtoH", "cudaMemcpyDtoH",
    "hipMemcpy", "cudaMemcpy", "hipMemcpyWithStream", "cudaMemcpyWithStream",
    "hipStreamWaitValue32", "hipStreamWaitValue64",
}
ITEM_RX = re.compile(r"aten::item|_local_scalar_dense|aten::is_nonzero|"
                     r"aten::_assert_async|aten::nonzero|aten::equal|"
                     r"aten::_local_scalar|to_bool|aten::item")
NCCL_RX = re.compile(r"nccl|rccl|allreduce|all_reduce|allgather|all_gather|"
                     r"reduce_scatter|reducescatter|broadcast|sendrecv|"
                     r"c10d::|ProcessGroup", re.I)


def iter_events(path):
    """Yield one traceEvent dict at a time from the huge traceEvents array."""
    dec = json.JSONDecoder()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="ignore") as f:
        buf = ""
        while '"traceEvents"' not in buf:
            c = f.read(CHUNK)
            if not c:
                raise RuntimeError("no traceEvents key found")
            buf += c
        i = buf.index('"traceEvents"')
        j = buf.index("[", i)
        buf = buf[j + 1:]
        while True:
            k = 0
            n = len(buf)
            while k < n and buf[k] in " \t\r\n,":
                k += 1
            if k >= n:
                c = f.read(CHUNK)
                if not c:
                    return
                buf = buf[k:] + c
                continue
            if buf[k] == "]":
                return
            buf = buf[k:]
            try:
                obj, end = dec.raw_decode(buf)
            except ValueError:
                c = f.read(CHUNK)
                if not c:
                    return
                buf += c
                if len(buf) > 256 * CHUNK:
                    raise RuntimeError("single event larger than 256MB?")
                continue
            buf = buf[end:]
            yield obj


class Step:
    __slots__ = ("name", "ts", "dur", "kind", "idx")

    def __init__(self, name, ts, dur, idx=0):
        self.name, self.ts, self.dur, self.idx = name, ts, dur, idx
        self.kind = "decode" if dur < 600000 else "prefill"


def collect_steps(path):
    best = {}
    for e in iter_events(path):
        nm = e.get("name", "")
        if not nm.startswith("ProfilerStep#"):
            continue
        dur = e.get("dur") or 0
        if nm not in best or dur > best[nm].dur:
            best[nm] = Step(nm, e["ts"], dur)
    steps = sorted(best.values(), key=lambda s: s.ts)
    for i, s in enumerate(steps):
        s.idx = i
    return steps


def _new_acc():
    return {
        "kernels": collections.defaultdict(lambda: [0, 0.0]),
        "api": collections.defaultdict(lambda: [0, 0.0]),
        "cpuop": collections.defaultdict(lambda: [0, 0.0]),
        "ann": collections.defaultdict(lambda: [0, 0.0]),
        "api_by_tid": collections.defaultdict(lambda: [0, 0.0]),
        "cpu_by_tid": collections.defaultdict(lambda: [0, 0.0]),
        "nccl_kern": collections.defaultdict(lambda: [0, 0.0]),
        "nccl_cpu": collections.defaultdict(lambda: [0, 0.0]),
        "nccl_ann": collections.defaultdict(lambda: [0, 0.0]),
        "sync_api": collections.defaultdict(lambda: [0, 0.0]),
        "item_op": collections.defaultdict(lambda: [0, 0.0]),
        "gpu_iv": [], "sync_iv": [], "item_iv": [],
        "dtoh": [0, 0.0],
        "nccl_args": collections.Counter(),
    }


def _classify(e, a):
    ts = e.get("ts")
    dur = e.get("dur") or 0
    nm = e.get("name", "")
    cat = e.get("cat", "")
    if cat in GPU_CATS:
        g = a["kernels"][nm]
        g[0] += 1
        g[1] += dur
        a["gpu_iv"].append((ts, ts + dur))
        if NCCL_RX.search(nm):
            k = a["nccl_kern"][nm]
            k[0] += 1
            k[1] += dur
        if cat == "gpu_memcpy" and "DtoH" in nm:
            a["dtoh"][0] += 1
            a["dtoh"][1] += dur
    elif cat == RUNTIME_CAT:
        x = a["api"][nm]
        x[0] += 1
        x[1] += dur
        a["api_by_tid"][e.get("tid")][0] += 1
        a["api_by_tid"][e.get("tid")][1] += dur
        if nm in SYNC_APIS:
            y = a["sync_api"][nm]
            y[0] += 1
            y[1] += dur
            a["sync_iv"].append((ts, ts + dur))
    elif cat == "cpu_op":
        x = a["cpuop"][nm]
        x[0] += 1
        x[1] += dur
        a["cpu_by_tid"][e.get("tid")][0] += 1
        a["cpu_by_tid"][e.get("tid")][1] += dur
        if ITEM_RX.search(nm):
            y = a["item_op"][nm]
            y[0] += 1
            y[1] += dur
            a["item_iv"].append((ts, ts + dur))
        if NCCL_RX.search(nm):
            k = a["nccl_cpu"][nm]
            k[0] += 1
            k[1] += dur
            args = e.get("args") or {}
            for key in ("Input sizes", "Input type", "Input dims", "input_sizes",
                        "Process Group Description"):
                if key in args:
                    a["nccl_args"][f"{nm}|{key}={str(args[key])[:60]}"] += 1
    elif cat in ("user_annotation", "gpu_user_annotation"):
        x = a["ann"][nm]
        x[0] += 1
        x[1] += dur
        if NCCL_RX.search(nm):
            k = a["nccl_ann"][nm]
            k[0] += 1
            k[1] += dur


def union_ms(iv):
    if not iv:
        return 0.0
    iv = sorted(iv)
    tot, cs, ce = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    return (tot + ce - cs) / 1000.0


def complement_ms(iv, lo, hi):
    """Total ms inside [lo,hi) not covered by iv."""
    if not iv:
        return (hi - lo) / 1000.0
    iv = sorted(iv)
    tot, cur = 0.0, lo
    for s, e in iv:
        if s > cur:
            tot += s - cur
        cur = max(cur, e)
    if hi > cur:
        tot += hi - cur
    return tot / 1000.0


def overlap_ms(iv_a, iv_b, lo, hi):
    """Total ms where a nonempty part of iv_b lies inside union(iv_a)."""
    ga = sorted(iv_a)
    if not ga:
        return 0.0
    tot = 0.0
    for s, e in iv_b:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        # walk union segments overlapping [s,e)
        i = bisect.bisect_left(ga, (s,))
        if i > 0:
            i -= 1
        while i < len(ga) and ga[i][1] <= s:
            i += 1
        while i < len(ga) and ga[i][0] < e:
            a0, a1 = max(ga[i][0], s), min(ga[i][1], e)
            if a1 > a0:
                tot += a1 - a0
            if ga[i][1] >= e:
                break
            i += 1
    return tot / 1000.0


def windows_census(path, windows):
    """One pass; events bucketed into disjoint windows sorted by start."""
    windows = sorted(windows)
    starts = [w[0] for w in windows]
    accs = [_new_acc() for _ in windows]
    lo, hi = windows[0][0], windows[-1][1]
    for e in iter_events(path):
        ts = e.get("ts")
        if ts is None or ts < lo or ts >= hi:
            continue
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0 or ts >= windows[i][1]:
            continue
        _classify(e, accs[i])
    return [(a, w) for a, w in zip(accs, windows)]


def step_stats(a, w):
    ts0, ts1 = w
    span = (ts1 - ts0) / 1000.0
    busy = union_ms(a["gpu_iv"])
    idle_iv = []  # not materialised; computed as span - busy
    s = {
        "period_ms": span,
        "gpu_busy_ms": busy,
        "gpu_idle_ms": span - busy,
        "gpu_util_pct": 100.0 * busy / span if span else 0.0,
        "gpu_events": sum(v[0] for v in a["kernels"].values()),
        "gpu_kernel_ms": sum(v[1] for v in a["kernels"].values()) / 1000.0,
        "api_calls": sum(v[0] for v in a["api"].values()),
        "api_cpu_ms": sum(v[1] for v in a["api"].values()) / 1000.0,
        "cpuop_calls": sum(v[0] for v in a["cpuop"].values()),
        "cpuop_cpu_ms": sum(v[1] for v in a["cpuop"].values()) / 1000.0,
        "nccl_kern_calls": sum(v[0] for v in a["nccl_kern"].values()),
        "nccl_kern_ms": sum(v[1] for v in a["nccl_kern"].values()) / 1000.0,
        "ar_cpuop_calls": sum(v[0] for v in a["nccl_cpu"].values()),
        "ar_cpuop_ms": sum(v[1] for v in a["nccl_cpu"].values()) / 1000.0,
        "dtoh_calls": a["dtoh"][0],
        "dtoh_ms": a["dtoh"][1] / 1000.0,
        "item_calls": sum(v[0] for v in a["item_op"].values()),
        "item_cpu_ms": sum(v[1] for v in a["item_op"].values()) / 1000.0,
        "item_cpu_ms_in_gpu_idle": overlap_ms(_idle_iv(a["gpu_iv"], ts0, ts1),
                                              a["item_iv"], ts0, ts1),
        "syncapi_calls": sum(v[0] for v in a["sync_api"].values()),
        "syncapi_cpu_ms": sum(v[1] for v in a["sync_api"].values()) / 1000.0,
        "syncapi_cpu_ms_in_gpu_idle": overlap_ms(_idle_iv(a["gpu_iv"], ts0, ts1),
                                                 a["sync_iv"], ts0, ts1),
    }
    return s


def _idle_iv(gpu_iv, ts0, ts1):
    """Complement of the GPU-busy union inside [ts0,ts1), as intervals."""
    if not gpu_iv:
        return [(ts0, ts1)]
    iv = sorted(gpu_iv)
    out, cur = [], ts0
    for s, e in iv:
        if s > cur:
            out.append((cur, min(s, ts1)))
        cur = max(cur, e)
        if cur >= ts1:
            break
    if cur < ts1:
        out.append((cur, ts1))
    return [(s, e) for s, e in out if e > s]


def agg_steps(path, steps):
    """Per-step means over the given engine steps."""
    parts = windows_census(path, [(s.ts, s.ts + s.dur) for s in steps])
    stats = [step_stats(a, w) for a, w in parts]
    n = len(stats)
    keys = ("kernels", "api", "cpuop", "ann", "api_by_tid", "cpu_by_tid",
            "nccl_kern", "nccl_cpu", "nccl_ann", "sync_api", "item_op")
    mean_tables = {k: collections.defaultdict(lambda: [0.0, 0.0]) for k in keys}
    for a, _ in parts:
        for k in keys:
            for nm, (c, d) in a[k].items():
                t = mean_tables[k][nm]
                t[0] += c
                t[1] += d
    for k in keys:
        for nm in mean_tables[k]:
            t = mean_tables[k][nm]
            t[0] /= n
            t[1] = t[1] / n / 1000.0   # ms/step
    scal = {}
    for key in stats[0]:
        scal[key] = sum(s[key] for s in stats) / n
    nccl_args = collections.Counter()
    for a, _ in parts:
        nccl_args += a["nccl_args"]
    return {"n_steps": n, "scal": scal, "steps": [s for s in stats],
            "nccl_args": nccl_args, "names": [s.name for s in steps], **mean_tables}


def _sel_steps(a):
    steps = collect_steps(a.trace)
    if a.kind == "decode":
        sel = [s for s in steps if s.kind == "decode"]
    elif a.kind == "prefill":
        sel = [s for s in steps if s.kind == "prefill"]
    else:
        sel = steps
    if getattr(a, "min_ms", 0):
        sel = [s for s in sel if s.dur / 1000.0 >= a.min_ms]
    if a.steps:
        want = set(a.steps.split(","))
        sel = [s for s in sel if s.name in want or str(s.idx) in want]
    return sel


def cmd_steps(a):
    steps = collect_steps(a.trace)
    print(f"{'idx':>4} {'name':>16} {'kind':>8} {'period_ms':>10} {'ts_us':>18}")
    for s in steps:
        print(f"{s.idx:4d} {s.name:>16} {s.kind:>8} {s.dur/1000:10.2f} {s.ts:18.0f}")
    print(f"total={len(steps)} decode={sum(1 for s in steps if s.kind=='decode')} "
          f"prefill={sum(1 for s in steps if s.kind=='prefill')}")


def cmd_detail(a):
    steps = _sel_steps(a)
    R = agg_steps(a.trace, steps)
    hdr = ("period_ms", "gpu_busy_ms", "gpu_idle_ms", "gpu_events",
           "gpu_kernel_ms", "nccl_kern_calls", "nccl_kern_ms", "ar_cpuop_calls",
           "item_calls", "syncapi_calls", "syncapi_cpu_ms", "api_calls", "api_cpu_ms")
    print("  ".join(f"{h:>15}" for h in hdr))
    for nm, s in zip(R["names"], R["steps"]):
        print(f"{nm:>16} " + "  ".join(f"{s[h]:15.1f}" for h in hdr))
    print("\nMEAN per step:")
    for h in hdr:
        print(f"  {h:24s} {R['scal'][h]:12.2f}")
    return 0


def cmd_census(a):
    steps = _sel_steps(a)
    if not steps:
        print("no steps selected", file=sys.stderr)
        return 1
    R = agg_steps(a.trace, steps)
    S = R["scal"]
    n = R["n_steps"]
    print(f"== {a.kind} steps n={n}: {R['names']} ==")
    print(f"period/step        {S['period_ms']:9.2f} ms")
    print(f"GPU busy/step      {S['gpu_busy_ms']:9.2f} ms  ({S['gpu_util_pct']:.1f}%)"
          f"   GPU idle {S['gpu_idle_ms']:.2f} ms")
    print(f"GPU events/step    {S['gpu_events']:9.1f}   sum(kernel dur) {S['gpu_kernel_ms']:.1f} ms")
    print(f"HIP API calls/step {S['api_calls']:9.1f}   cpu_ms/step {S['api_cpu_ms']:.2f}")
    print(f"cpu_op calls/step  {S['cpuop_calls']:9.1f}   cpu_ms/step {S['cpuop_cpu_ms']:.2f}")
    print(f"NCCL kernels/step  {S['nccl_kern_calls']:9.1f}   ms/step {S['nccl_kern_ms']:.2f}")
    print(f"NCCL cpuops/step   {S['ar_cpuop_calls']:9.1f}   cpu_ms/step {S['ar_cpuop_ms']:.2f}")
    print(f"DtoH memcpy/step   {S['dtoh_calls']:9.1f}   ms/step {S['dtoh_ms']:.2f}")
    print(f"blocking APIs/step {S['syncapi_calls']:9.1f}   cpu_ms/step {S['syncapi_cpu_ms']:.2f}"
          f"   of which inside GPU-idle {S['syncapi_cpu_ms_in_gpu_idle']:.2f} ms")
    print(f"aten::item/step    {S['item_calls']:9.1f}   cpu_ms/step {S['item_cpu_ms']:.2f}"
          f"   of which inside GPU-idle {S['item_cpu_ms_in_gpu_idle']:.2f} ms")
    print(f"GPU idle/step      {S['gpu_idle_ms']:9.2f} ms; item+sync APIs cover "
          f"{min(100.0, 100.0*(S['item_cpu_ms_in_gpu_idle']+S['syncapi_cpu_ms_in_gpu_idle'])/S['gpu_idle_ms']) if S['gpu_idle_ms'] else 0:.0f}% of it")

    def table(title, d, key, k):
        print(f"\n-- {title} " + "-" * max(0, 62 - len(title)))
        print(f"{'n/step':>9} {'ms/step':>8} {'us/call':>8}  name")
        for nm, (c, ms) in sorted(d.items(), key=lambda kv: -kv[1][key])[:k]:
            us = ms * 1000.0 / c if c else 0.0
            print(f"{c:9.1f} {ms:8.3f} {us:8.2f}  {nm[:92]}")

    table("top kernels by COUNT/step", R["kernels"], 0, a.top)
    table("top kernels by MS/step", R["kernels"], 1, a.top)
    print("\n-- kernels with >500 launches/step --")
    for nm, (c, ms) in sorted(R["kernels"].items(), key=lambda kv: -kv[1][0]):
        if c > 500:
            print(f"{c:9.1f} n/step {ms:8.3f} ms/step  {nm[:92]}")
    table("NCCL kernels", R["nccl_kern"], 0, 12)
    table("collective cpu_op/annotation", {**R["nccl_cpu"], **R["nccl_ann"]}, 0, 12)
    table("blocking HIP APIs", R["sync_api"], 0, 12)
    table("item-ish cpu_op", R["item_op"], 0, 12)
    table("HIP API", R["api"], 0, a.top)
    table("cpu_op by count", R["cpuop"], 0, a.top)
    print("\n-- HIP API by tid (calls/step, cpu ms/step) " + "-" * 26)
    for t, (c, ms) in sorted(R["api_by_tid"].items(), key=lambda kv: -kv[1][1])[:6]:
        print(f"tid {t!s:>10}: {c:9.1f} calls  {ms:8.2f} ms")
    print("-- cpu_op by tid " + "-" * 43)
    for t, (c, ms) in sorted(R["cpu_by_tid"].items(), key=lambda kv: -kv[1][1])[:6]:
        print(f"tid {t!s:>10}: {c:9.1f} ops    {ms:8.2f} ms")
    table("user_annotation by ms", R["ann"], 1, 15)
    if R["nccl_args"]:
        print("\n-- collective args seen " + "-" * 42)
        for k, v in R["nccl_args"].most_common(10):
            print(f"{v:8d}  {k[:110]}")
    if a.json:
        ser = {"n_steps": n, "steps": R["names"], "scal": S}
        for k in ("kernels", "api", "cpuop", "nccl_kern", "nccl_cpu", "nccl_ann",
                  "sync_api", "item_op"):
            ser[k] = {nm: {"n_per_step": v[0], "ms_per_step": v[1]}
                      for nm, v in R[k].items()}
        with open(a.json, "w") as f:
            json.dump(ser, f, indent=1)
        print(f"\n[json] {a.json}")
    return 0


def cmd_window(a):
    a0 = _new_acc()
    for e in iter_events(a.trace):
        ts = e.get("ts")
        if ts is None or ts < a.ts0 or ts >= a.ts1:
            continue
        _classify(e, a0)
    S = step_stats(a0, (a.ts0, a.ts1))
    for k in sorted(S):
        print(f"{k:34s} {S[k]:12.2f}")
    print("\ntop kernels by ms:")
    for nm, (c, d) in sorted(a0["kernels"].items(), key=lambda kv: -kv[1][1])[:a.top]:
        print(f"  {d/1000:8.3f} ms {c:7d}x {d/c:8.2f} us  {nm[:88]}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("steps"); p.add_argument("trace"); p.set_defaults(fn=cmd_steps)
    for name, fn in (("census", cmd_census), ("detail", cmd_detail)):
        p = sub.add_parser(name); p.add_argument("trace")
        p.add_argument("--kind", default="decode", choices=["decode", "prefill", "all"])
        p.add_argument("--steps", default=None, help="comma list of ProfilerStep names/indices")
        p.add_argument("--top", type=int, default=25)
        p.add_argument("--min-ms", type=float, default=0.0,
                       help="ignore steps shorter than this (drops partial sampler steps)")
        p.add_argument("--json", default=None)
        p.set_defaults(fn=fn)
    p = sub.add_parser("window"); p.add_argument("trace")
    p.add_argument("--ts0", type=float, required=True)
    p.add_argument("--ts1", type=float, required=True)
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(fn=cmd_window)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
