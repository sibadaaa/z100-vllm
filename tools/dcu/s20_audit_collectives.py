#!/usr/bin/env python3
"""S20: link every collective (cpu_op) to its payload shape and its NCCL kernel time.

Pass 1 streams the trace and records, for each collective-ish cpu_op, the
External id -> (op name, dtype, dims).  Pass 2 streams again and attributes each
NCCL kernel (matched by "External id") to that payload class, restricted to the
selected engine-step windows.  Output is per-step counts and ms, per payload
class, which answers "which collectives carry 8-row payloads and how big are they".

Usage:
  python3 s20_audit_collectives.py TRACE [--kind decode|prefill|all] [--min-ms 150]
                                         [--json OUT]
"""
import argparse
import bisect
import collections
import json
import re
import sys

sys.path.insert(0, "/home/lyh/z100-vllm/tools/dcu")
from s20_audit_trace import collect_steps, iter_events  # noqa: E402

COLL_RX = re.compile(r"all_reduce|allreduce|all_gather|allgather|reduce_scatter|"
                     r"broadcast|sendrecv|c10d::", re.I)
NCCL_KERN_RX = re.compile(r"ncclDevKernel|ncclAllReduce|ncclKernel", re.I)


def shape_of(e):
    a = e.get("args") or {}
    dims = a.get("Input Dims")
    dts = a.get("Input type")
    if dims is None:
        return None
    try:
        return str(dims), str(dts)
    except Exception:
        return str(dims), str(dts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--kind", default="decode", choices=["decode", "prefill", "all"])
    ap.add_argument("--min-ms", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    steps = collect_steps(a.trace)
    sel = [s for s in steps if (a.kind == "all" or s.kind == a.kind)
           and s.dur / 1000.0 >= a.min_ms]
    n_steps = len(sel)
    wins = sorted((s.ts, s.ts + s.dur) for s in sel)
    starts = [w[0] for w in wins]
    lo, hi = wins[0][0], wins[-1][1]
    print(f"steps n={n_steps} kind={a.kind}: {[s.name for s in sel]}")

    def in_window(ts):
        i = bisect.bisect_right(starts, ts) - 1
        return i >= 0 and ts < wins[i][1]

    # pass 1: External id -> payload class for every collective cpu_op
    ids = {}
    ops = collections.Counter()
    for e in iter_events(a.trace):
        if e.get("cat") != "cpu_op":
            continue
        nm = e.get("name", "")
        if not COLL_RX.search(nm):
            continue
        xid = (e.get("args") or {}).get("External id")
        if xid is None:
            continue
        sh = shape_of(e) or ("<no shape>", "<no dtype>")
        ids[xid] = (nm, sh[0], sh[1])
    print(f"collective cpu_ops with External id: {len(ids)}")

    # pass 2: attribute NCCL kernels + cpu_ops inside step windows
    kern = collections.defaultdict(lambda: [0, 0.0])   # class -> [n, us]
    cpu = collections.defaultdict(lambda: [0, 0.0])
    cpu_us = collections.defaultdict(lambda: [0, 0.0])
    other_kern = collections.Counter()
    for e in iter_events(a.trace):
        cat = e.get("cat", "")
        ts = e.get("ts")
        if ts is None or ts < lo or ts >= hi or not in_window(ts):
            continue
        nm = e.get("name", "")
        if cat == "kernel":
            if not NCCL_KERN_RX.search(nm):
                continue
            xid = (e.get("args") or {}).get("External id")
            cls = ids.get(xid)
            if cls is None:
                other_kern[nm[:70]] += 1
                cls = ("<unlinked kernel>", nm[:60], "")
            kern[cls][0] += 1
            kern[cls][1] += e.get("dur") or 0
        elif cat == "cpu_op" and COLL_RX.search(nm):
            xid = (e.get("args") or {}).get("External id")
            cls = ids.get(xid, (nm, "<no shape>", "<no dtype>"))
            cpu[cls][0] += 1
            cpu_us[cls][1] += e.get("dur") or 0

    print(f"\n{'n/step':>8} {'ms/step':>8} {'us/call':>9}  payload (op | dims | dtype)")
    rows = []
    for cls, (c, d) in sorted(kern.items(), key=lambda kv: -kv[1][0]):
        op, dims, dt = cls
        ms = d / 1000.0 / n_steps
        rows.append({"op": op, "dims": dims, "dtype": dt,
                     "kern_n_per_step": c / n_steps, "kern_ms_per_step": ms,
                     "kern_us_per_call": d / c if c else 0.0})
        print(f"{c/n_steps:8.2f} {ms:8.3f} {d/c:9.1f}  {op[:34]} | {dims[:40]} | {dt[:24]}")
    print(f"\n-- cpu_op side (dispatch, incl. launch + python) --")
    print(f"{'n/step':>8} {'ms/step':>8} {'us/call':>9}  payload")
    for cls, (c, d) in sorted(cpu.items(), key=lambda kv: -kv[1][0]):
        op, dims, dt = cls
        print(f"{c/n_steps:8.2f} {d/1000.0/n_steps:8.3f} {d/c:9.1f}  {op[:34]} | {dims[:40]} | {dt[:24]}")
    if other_kern:
        print("\n-- NCCL kernels with no matching collective cpu_op --")
        for k, v in other_kern.most_common(8):
            print(f"{v/n_steps:8.2f} n/step  {k}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"n_steps": n_steps, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
