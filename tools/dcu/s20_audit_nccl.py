#!/usr/bin/env python3
"""S20: cross-rank decomposition of NCCL all-reduce latency in a 4-rank trace set.

For each collective index i (calls are index-aligned across ranks because every
rank executes the same TP-symmetric program):
    floor_i  = min_r d(r,i)      -> skew-free intrinsic cost of that collective
    wait_r,i = d(r,i) - floor_i  -> arrival wait on rank r (symptom of rank skew)
    crit_i   = max_r d(r,i)      -> when the barrier actually releases

Usage:
  python3 s20_audit_nccl.py --rank0 TRACE0 [--rank1 T1 --rank2 T2 --rank3 T3]
                            [--kind decode|prefill|all] [--min-ms 150] [--json OUT]
One rank trace is streamed at a time and only small call lists are retained.
"""
import argparse
import bisect
import collections
import gzip
import json
import re
import statistics as st
import sys

sys.path.insert(0, "/home/lyh/z100-vllm/tools/dcu")
from s20_audit_trace import collect_steps, iter_events  # noqa: E402

AR_RX = re.compile(r"AllReduce|allreduce|all_reduce")
AG_RX = re.compile(r"AllGather|allgather|all_gather")


def proto(name):
    for p in ("Ring_LL", "Ring_Simple", "Tree", "Ring"):
        if p in name:
            return p
    return "?"


def dtype(name):
    for t in ("f16", "b16", "f32", "i8", "bf16", "f64"):
        if f"_{t}" in name:
            return t
    return "?"


def collect_calls(path, windows):
    """windows: sorted disjoint [(ts0,ts1)]; only kernels inside a step are kept."""
    wins = sorted(windows)
    starts = [w[0] for w in wins]
    lo, hi = wins[0][0], wins[-1][1]
    ar, ag = [], []
    for e in iter_events(path):
        if e.get("cat") not in ("kernel", "Kernel"):
            continue
        ts = e.get("ts")
        if ts is None or ts < lo or ts >= hi:
            continue
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0 or ts >= wins[i][1]:
            continue
        nm = e.get("name", "")
        if AR_RX.search(nm):
            ar.append((ts, e.get("dur") or 0, nm))
        elif AG_RX.search(nm):
            ag.append((ts, e.get("dur") or 0, nm))
    ar.sort()
    ag.sort()
    return ar, ag


def decompose(calls_per_rank, label, n_steps):
    n = min(len(c) for c in calls_per_rank)
    if n == 0:
        return None
    r = len(calls_per_rank)
    floors, waits = [], [[] for _ in range(r)]
    maxs = []
    for i in range(n):
        ds = [calls_per_rank[k][i][1] for k in range(r)]
        f = min(ds)
        floors.append(f)
        maxs.append(max(ds))
        for k in range(r):
            waits[k].append(ds[k] - f)
    out = {
        "label": label, "n_calls": n, "n_steps": n_steps,
        "calls_per_step": n / n_steps,
        "floor_ms_per_step": sum(floors) / 1000.0 / n_steps,
        "crit_ms_per_step": sum(maxs) / 1000.0 / n_steps,
        "per_rank": {},
    }
    for k in range(r):
        tot = sum(calls_per_rank[k][i][1] for i in range(n))
        wsum = sum(waits[k])
        out["per_rank"][k] = {
            "ms_per_step": tot / 1000.0 / n_steps,
            "wait_ms_per_step": wsum / 1000.0 / n_steps,
            "wait_frac": wsum / tot if tot else 0.0,
            "mean_us": tot / n,
            "p50_us": st.median([calls_per_rank[k][i][1] for i in range(n)]),
        }
    out["floor_us"] = {
        "mean": sum(floors) / n, "p50": st.median(floors),
        "p10": sorted(floors)[int(0.1 * n)], "p90": sorted(floors)[min(n - 1, int(0.9 * n))],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank0", required=True)
    for k in (1, 2, 3):
        ap.add_argument(f"--rank{k}", default=None)
    ap.add_argument("--kind", default="decode", choices=["decode", "prefill", "all"])
    ap.add_argument("--min-ms", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    paths = [a.rank0] + [getattr(a, f"rank{k}") for k in (1, 2, 3)]

    steps = collect_steps(paths[0])
    sel = [s for s in steps if (a.kind == "all" or s.kind == a.kind)
           and s.dur / 1000.0 >= a.min_ms]
    steps = sel
    print(f"steps used (from rank0): {[s.name for s in steps]}")
    windows = [(s.ts, s.ts + s.dur) for s in steps]
    lo = min(s.ts for s in steps)
    hi = max(s.ts + s.dur for s in steps)
    print(f"window {lo:.0f}..{hi:.0f}  ({(hi-lo)/1e6:.2f} s)  n_steps={len(steps)}")

    allar, allag = [], []
    for p in paths:
        if not p:
            continue
        ar, ag = collect_calls(p, windows)
        allar.append(ar)
        allag.append(ag)
        print(f"{p.split('/')[-1][:22]}: {len(ar)} AllReduce, {len(ag)} AllGather kernels")
    res = {}
    for label, per_rank in (("AllReduce", allar), ("AllGather", allag)):
        out = decompose(per_rank, label, len(steps))
        if out:
            res[label] = out
    # per-step counts and protocol mix, from rank0
    cnt = collections.Counter(proto(c[2]) for c in allar[0])
    dt = collections.Counter(dtype(c[2]) for c in allar[0])
    res["rank0_mix"] = {"proto": dict(cnt), "dtype": dict(dt)}
    print(json.dumps(res, indent=1, default=str))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
