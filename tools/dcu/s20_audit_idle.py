#!/usr/bin/env python3
"""S20: attribute GPU idle time to what the CPU was doing inside each gap.

For every engine-step window:
  * GPU busy = union of GPU kernel/memcpy intervals; idle = window - busy
  * each idle interval's length is split fractionally across the CPU-side events
    (cpu_op / cuda_runtime) that overlap it -> "during GPU idle, the CPU was
    inside <op>".
  * also reports the raw per-step totals of the host-sync suspects
    (aten::item, _local_scalar_dense, blocking HIP APIs, DtoH copies).

Usage: python3 s20_audit_idle.py TRACE [--kind decode|prefill|all] [--min-ms 150]
                                     [--top 20] [--json OUT]
"""
import argparse
import bisect
import collections
import json
import sys

sys.path.insert(0, "/home/lyh/z100-vllm/tools/dcu")
from s20_audit_trace import (collect_steps, iter_events, union_ms,  # noqa: E402
                             SYNC_APIS, ITEM_RX, GPU_CATS)

CPU_CATS = ("cpu_op", "cuda_runtime", "user_annotation", "gpu_user_annotation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--kind", default="decode", choices=["decode", "prefill", "all"])
    ap.add_argument("--min-ms", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    steps = [s for s in collect_steps(a.trace)
             if (a.kind == "all" or s.kind == a.kind) and s.dur / 1000.0 >= a.min_ms]
    wins = sorted((s.ts, s.ts + s.dur) for s in steps)
    n = len(wins)
    starts = [w[0] for w in wins]
    lo, hi = wins[0][0], wins[-1][1]
    print(f"kind={a.kind} steps={n} period_mean={sum(w[1]-w[0] for w in wins)/n/1000:.1f} ms")

    gpu = collections.defaultdict(list)     # step -> intervals
    cpu = collections.defaultdict(list)     # step -> (ts, dur, name)
    totals = collections.defaultdict(lambda: [0, 0.0])
    for e in iter_events(a.trace):
        ts = e.get("ts")
        if ts is None or ts < lo or ts >= hi:
            continue
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0 or ts >= wins[i][1]:
            continue
        cat = e.get("cat", "")
        dur = e.get("dur") or 0
        nm = e.get("name", "")
        if cat in GPU_CATS:
            gpu[i].append((ts, ts + dur))
        elif cat in CPU_CATS:
            cpu[i].append((ts, dur, nm))
            if cat == "cpu_op" and ITEM_RX.search(nm):
                t = totals["item:" + nm]
                t[0] += 1
                t[1] += dur
            if cat == "cuda_runtime" and nm in SYNC_APIS:
                t = totals["sync:" + nm]
                t[0] += 1
                t[1] += dur
            if cat == "cpu_op" and nm in ("aten::copy_", "aten::_to_copy", "aten::to"):
                t = totals["dtoh-path:" + nm]
                t[0] += 1
                t[1] += dur

    idle_total = 0.0
    attrib = collections.Counter()          # name -> us attributed
    attrib_cat = collections.Counter()
    gap_hist = collections.Counter()
    for i in range(n):
        ts0, ts1 = wins[i]
        busy = union_ms(gpu[i])
        idle_total += (ts1 - ts0) / 1000.0 - busy
        iv = sorted(gpu[i])
        gaps, cur = [], ts0
        for s, e in iv:
            if s > cur:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < ts1:
            gaps.append((cur, ts1))
        ce = sorted(cpu[i])
        for g0, g1 in gaps:
            glen = g1 - g0
            if glen <= 0:
                continue
            gap_hist[min(9, int(glen // 1000))] += 1     # ms buckets of 1ms
            j = bisect.bisect_left(ce, (g0,))
            if j:
                j -= 1
            seen = 0.0
            while j < len(ce) and ce[j][0] < g1:
                ct, cd, cnm = ce[j]
                o0, o1 = max(ct, g0), min(ct + cd, g1)
                if o1 > o0:
                    attrib[cnm] += o1 - o0
                    attrib_cat[cnm.split("::")[0].split("(")[0][:24]] += o1 - o0
                    seen += o1 - o0
                j += 1
            if seen < glen * 0.5:
                attrib_cat["<no CPU event recorded>"] += glen - seen

    print(f"\nGPU idle/step = {idle_total/n:.2f} ms  ({(idle_total/n)/(sum(w[1]-w[0] for w in wins)/n/1000)*100:.1f}% of period)")
    print(f"\n-- what the CPU was inside during GPU idle (fractional attribution, ms/step) --")
    tot_attr = sum(attrib.values())
    for nm, us in attrib.most_common(a.top):
        print(f"{us/1000/n:8.3f} ms/step  {100*us/tot_attr:5.1f}%  {nm[:80]}")
    print(f"\n-- same, grouped by op family --")
    for nm, us in attrib_cat.most_common(a.top):
        print(f"{us/1000/n:8.3f} ms/step  {100*us/tot_attr:5.1f}%  {nm}")
    print(f"\n-- raw per-step totals of sync suspects --")
    for nm, (c, us) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
        print(f"{c/n:8.2f} calls/step {us/1000/n:9.3f} ms/step  {nm[:80]}")
    print(f"\n-- GPU-idle gap histogram (count of gaps per step, by ms bucket) --")
    for b in sorted(gap_hist):
        print(f"  {b:2d}-{b+1:2d} ms : {gap_hist[b]/n:8.2f} gaps/step")
    if a.json:
        json.dump({"idle_ms_per_step": idle_total / n,
                   "attrib_ms_per_step": {k: v / 1000 / n for k, v in attrib.items()},
                   "totals": {k: [v[0] / n, v[1] / 1000 / n] for k, v in totals.items()}},
                  open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
