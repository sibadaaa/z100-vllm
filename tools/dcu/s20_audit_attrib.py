#!/usr/bin/env python3
"""S20: attribute each GPU kernel to the ATen op that launched it, and locate
host-sync (.item()/nonzero) call sites by their CPU-side ancestor chain.

Kineto links a kernel to its launcher through args["External id"], and CPU ops
also carry that id.  Ambiguous ids (same id reused) are reported as such.

Usage: python3 s20_audit_attrib.py TRACE [--kind decode|prefill|all] [--min-ms 150]
                                        [--top 25] [--json OUT]
"""
import argparse
import bisect
import collections
import json
import sys

sys.path.insert(0, "/home/lyh/z100-vllm/tools/dcu")
from s20_audit_trace import collect_steps, iter_events  # noqa: E402
from s20_audit_collectives import NCCL_KERN_RX  # noqa: E402

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset", "Kernel")
SYNC_RX = ("aten::item", "aten::_local_scalar_dense", "aten::nonzero",
           "aten::is_nonzero", "aten::_assert_async")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--kind", default="decode")
    ap.add_argument("--min-ms", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    steps = [s for s in collect_steps(a.trace)
             if (a.kind == "all" or s.kind == a.kind) and s.dur / 1000.0 >= a.min_ms]
    n = len(steps)
    wins = sorted((s.ts, s.ts + s.dur) for s in steps)
    starts = [w[0] for w in wins]
    lo, hi = wins[0][0], wins[-1][1]

    def inw(ts):
        i = bisect.bisect_right(starts, ts) - 1
        return i >= 0 and ts < wins[i][1]

    id2op = {}
    for e in iter_events(a.trace):
        if e.get("cat") not in ("cpu_op", "user_annotation"):
            continue
        ts = e.get("ts")
        if ts is None or ts < lo or ts >= hi or not inw(ts):
            continue
        xid = (e.get("args") or {}).get("External id")
        if xid is not None:
            id2op.setdefault(xid, e.get("name", "?"))

    kern = collections.defaultdict(lambda: [0, 0.0])
    by_parent = collections.defaultdict(lambda: [0, 0.0])
    unlinked = collections.defaultdict(lambda: [0, 0.0])
    for e in iter_events(a.trace):
        if e.get("cat") not in GPU_CATS:
            continue
        ts = e.get("ts")
        if ts is None or ts < lo or ts >= hi or not inw(ts):
            continue
        nm = e.get("name", "")
        d = e.get("dur") or 0
        xid = (e.get("args") or {}).get("External id")
        par = id2op.get(xid, "<no link>")
        kern[(nm, par)][0] += 1
        kern[(nm, par)][1] += d
        by_parent[par][0] += 1
        by_parent[par][1] += d
        if par == "<no link>":
            unlinked[nm][0] += 1
            unlinked[nm][1] += d

    print(f"== {a.kind} steps n={n} ==")
    print(f"\n-- top kernels, by count, with launching ATen op --")
    print(f"{'n/step':>8} {'ms/step':>8} {'us':>7}  kernel <- parent op")
    for (nm, par), (c, d) in sorted(kern.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f"{c/n:8.1f} {d/1000/n:8.3f} {d/c:7.2f}  {nm[:58]} <- {par[:40]}")
    print(f"\n-- top kernels, by ms --")
    for (nm, par), (c, d) in sorted(kern.items(), key=lambda kv: -kv[1][1])[:a.top]:
        print(f"{d/1000/n:8.3f} ms {c/n:8.1f}x {d/c:7.2f}  {nm[:58]} <- {par[:40]}")
    print(f"\n-- GPU ms/step grouped by launching ATen op --")
    for par, (c, d) in sorted(by_parent.items(), key=lambda kv: -kv[1][1])[:a.top]:
        print(f"{d/1000/n:8.3f} ms/step {c/n:8.1f} kernels  {par[:70]}")
    print(f"\n-- kernels with no External-id link (top 10) --")
    for nm, (c, d) in sorted(unlinked.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"{d/1000/n:8.3f} ms/step {c/n:8.1f}x  {nm[:70]}")

    # host-sync call sites: ancestor chain of sync cpu_ops
    print(f"\n-- host-sync cpu_ops and their ancestor chain --")
    stack = []
    chains = collections.Counter()
    for e in iter_events(a.trace):
        if e.get("cat") != "cpu_op":
            continue
        ts = e.get("ts")
        if ts is None or ts < lo or ts >= hi or not inw(ts):
            continue
        d = e.get("dur") or 0
        nm = e.get("name", "")
        while stack and stack[-1][0] + stack[-1][1] <= ts:
            stack.pop()
        if any(nm.startswith(s) for s in SYNC_RX):
            chain = " < ".join(x[2] for x in stack[-4:])
            chains[(nm, chain)] += 1
        stack.append((ts, d, nm))
    for (nm, chain), c in chains.most_common(12):
        print(f"{c/n:7.2f}/step  {nm[:30]:32s} parents: {chain[:110]}")
    if a.json:
        json.dump({"kernels": {f"{k[0]} <- {k[1]}": {"n_per_step": v[0] / n,
                                                     "ms_per_step": v[1] / 1000 / n}
                               for k, v in kern.items()},
                   "by_parent": {k: [v[0] / n, v[1] / 1000 / n] for k, v in by_parent.items()}},
                  open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
