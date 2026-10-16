#!/usr/bin/env python3
"""Is the ~330-370 us/call NCCL AllReduce on the fast ranks real transfer cost,
or spin-wait for the slower Z100 ranks?

Reads the four per-rank atom pickles produced by the S-perf1 trace harvest
(/home/lyh/vllm-test/results/atoms/atoms_rank{0..3}.pkl) and decomposes every
AllReduce kernel into

    d_r(i) = A_last(i) - A_r(i) + transfer(i)          (ring lockstep)
           = wait_r(i)      + floor(i)

where A_r(i) is the arrival time of rank r at collective i and
floor(i) = min_r d_r(i) is the transfer floor (all ranks move the same bytes).
Summing gives, per rank: real busy time = sum(floor) + sum(wait), and the
critical-path communication cost of the whole run = sum(floor) only.

Usage: python3 tools/dcu/perf1_comm_skew.py [atoms_dir]
"""

import pickle
import statistics
import sys
from collections import Counter, defaultdict

ATOMS = sys.argv[1] if len(sys.argv) > 1 else "/home/lyh/vllm-test/results/atoms"
RANKS = [0, 1, 2, 3]


def load(rank):
    with open(f"{ATOMS}/atoms_rank{rank}.pkl", "rb") as fh:
        return pickle.load(fh)


def ar_events(d):
    """g_cat/g_nm are integer indices into d['cats'] / d['names']."""
    names = d["names"]
    cats = d["cats"]
    kernel = cats.index("kernel")
    out = []
    for ts, dur, nm, cat in zip(d["g_ts"], d["g_dur"], d["g_nm"], d["g_cat"]):
        if cat == kernel and "AllReduce" in names[nm]:
            out.append((float(ts), float(dur), names[nm]))
    out.sort()
    return out


def proto(nm):
    if "Ring_Simple" in nm:
        return "Ring_Simple"
    if "Ring_LL" in nm:
        return "Ring_LL"
    if "Tree" in nm:
        return "Tree"
    return nm


def main():
    ev = {r: ar_events(load(r)) for r in RANKS}
    n = min(len(v) for v in ev.values())
    print("=== census ===")
    for r in RANKS:
        tot = sum(e[1] for e in ev[r])
        c = Counter(proto(e[2]) for e in ev[r])
        span = ev[r][-1][0] - ev[r][0][0]
        print(
            f"rank{r}: n={len(ev[r]):5d} sum_dur={tot / 1000:8.1f} ms "
            f"mean={tot / len(ev[r]):7.1f} us span={span / 1000:9.1f} ms "
            f"proto={dict(c)}"
        )

    # index alignment (identical collective sequence on all ranks)
    idx = list(range(n))
    dur = {r: [ev[r][i][1] for i in idx] for r in RANKS}
    ts = {r: [ev[r][i][0] for i in idx] for r in RANKS}
    nm = {r: [ev[r][i][2] for i in idx] for r in RANKS}

    floor = [min(dur[r][i] for r in RANKS) for i in idx]
    excess = {r: [dur[r][i] - floor[i] for i in idx] for r in RANKS}

    print("\n=== decomposition (us per call, whole run) ===")
    print(
        f"{'rank':>5}{'sum_dur_ms':>12}{'sum_floor_ms':>14}{'sum_wait_ms':>13}"
        f"{'wait%':>8}{'mean_wait_us':>14}"
    )
    for r in RANKS:
        sd = sum(dur[r])
        sw = sum(excess[r])
        print(
            f"{r:>5}{sd / 1000:>12.1f}{sum(floor) / 1000:>14.1f}{sw / 1000:>13.1f}"
            f"{100 * sw / sd:>7.1f}%{sw / n:>14.1f}"
        )

    # per-step view (140 collectives per decode step, 50 steps)
    print("\n=== per-step view (assuming 140 collectives/step) ===")
    for per_step in (140,):
        print(
            f"collectives/step={per_step}: floor/step = "
            f"{sum(floor) / per_step / 50 / 1000:.1f} ms/step; "
            + "; ".join(
                f"r{r} wait/step = {sum(excess[r]) / per_step / 50 / 1000:.1f} ms"
                for r in RANKS
            )
        )

    # who is the straggler (rank with the minimum duration) per collective
    print("\n=== straggler census (argmin of per-collective duration) ===")
    cnt = Counter()
    for i in idx:
        m = min(RANKS, key=lambda r: dur[r][i])
        cnt[m] += 1
    for r in RANKS:
        print(f"rank{r}: argmin in {cnt[r]:5d}/{n} calls ({100 * cnt[r] / n:.1f}%)")

    # arrival skew must equal the wait of the earliest arriver
    print("\n=== internal check: start-time spread vs max excess ===")
    spread = [max(ts[r][i] for r in RANKS) - min(ts[r][i] for r in RANKS) for i in idx]
    maxexc = [max(excess[r][i] for r in RANKS) for i in idx]
    ok = sum(1 for i in idx if spread[i] > 0)
    print(
        f"median start spread={statistics.median(spread):.1f} us, "
        f"median max-excess={statistics.median(maxexc):.1f} us, "
        f"spread>0 in {ok}/{n}"
    )
    print(
        f"sum(spread)={sum(spread) / 1000:.1f} ms "
        f"sum(max excess)={sum(maxexc) / 1000:.1f} ms"
    )

    # protocol consistency across ranks per collective
    print("\n=== protocol mix per collective ===")
    mism = 0
    pair = Counter()
    for i in idx:
        ps = tuple(proto(nm[r][i]) for r in RANKS)
        pair[ps] += 1
        if len(set(ps)) > 1:
            mism += 1
    print(f"collectives with a uniform protocol: {n - mism}/{n}; mismatched: {mism}")
    for k, v in pair.most_common(6):
        print(f"  {k}: {v}")

    print("\n=== per-protocol per-rank duration (us/call) ===")
    byp = defaultdict(lambda: defaultdict(list))
    for i in idx:
        p = proto(nm[0][i])
        for r in RANKS:
            byp[p][r].append(dur[r][i])
    for p, per in byp.items():
        print(
            f"rank0-proto={p:12s} n={len(per[0]):5d} "
            + " ".join(
                f"r{r}={statistics.mean(per[r]):7.1f}(min {min(per[r]):6.1f})"
                for r in RANKS
            )
        )

    # ---- start vs end spread: the decisive "wait, not transfer" evidence ----
    ss, es = [], []
    for i in idx:
        s = [ts[r][i] for r in RANKS]
        e = [ts[r][i] + dur[r][i] for r in RANKS]
        ss.append(max(s) - min(s))
        es.append(max(e) - min(e))

    def pct(x, q):
        return sorted(x)[min(len(x) - 1, int(q * len(x)))]

    print("\n=== start vs end spread per collective (us) ===")
    print(
        f"START spread: median={statistics.median(ss):7.1f} p90={pct(ss, .9):7.1f} "
        f"mean={statistics.mean(ss):7.1f}"
    )
    print(
        f"END   spread: median={statistics.median(es):7.1f} p90={pct(es, .9):7.1f} "
        f"mean={statistics.mean(es):7.1f}"
    )
    print(
        "interpretation: the four ranks finish every collective together while "
        "starting up to hundreds of us apart => the duration difference is\n"
        "arrival wait, not data movement."
    )

    # ---- inter-collective compute gaps and the wait identity ----
    print("\n=== wait identity: wait_r == max_s gap_s - gap_r (gap = own compute) ===")
    gap = {r: [ts[r][i] - (ts[r][i - 1] + dur[r][i - 1]) for i in idx[1:]] for r in RANKS}
    m = len(idx) - 1
    observed = {r: sum(dur[r][i] - floor[i] for i in idx[1:]) for r in RANKS}
    predicted = {
        r: sum(max(gap[s][i] for s in RANKS) - gap[r][i] for i in range(m))
        for r in RANKS
    }
    for r in RANKS:
        print(
            f"  r{r}: observed wait={observed[r] / 1000:8.1f} ms "
            f"predicted={predicted[r] / 1000:8.1f} ms  "
            f"mean gap={statistics.mean(gap[r]):7.1f} us"
        )

    # ---- what card uniformity could recover ----
    mx = [max(gap[s][i] for s in RANKS) for i in range(m)]
    z100 = [
        statistics.mean([gap[2][i], gap[3][i]]) for i in range(m)
    ]
    z100l = [
        statistics.mean([gap[0][i], gap[1][i]]) for i in range(m)
    ]
    mn = [min(gap[s][i] for s in RANKS) for i in range(m)]
    per = n / 50.14
    print("\n=== recoverable by card uniformity (per step = %.0f collectives) ===" % per)
    for label, arr in (
        ("all-Z100   (max - mean Z100 )", z100),
        ("all-Z100L  (max - mean Z100L)", z100l),
        ("all-equal  (max - min       )", mn),
    ):
        d = [mx[i] - arr[i] for i in range(m)]
        print(
            f"  {label}: mean delta={statistics.mean(d):7.1f} us "
            f"median={statistics.median(d):7.1f} us -> "
            f"{per * statistics.mean(d) / 1000:5.1f} ms/step (mean) / "
            f"{per * statistics.median(d) / 1000:5.1f} ms/step (median)"
        )
    print(
        "  caveat: means include the 6 MTP draft intervals (1.3-2.0 s); the "
        "median column is the steady-state figure."
    )
    # who arrives last, per rank
    print("\n=== last arriver per collective (max gap) ===")
    last = Counter()
    for i in range(m):
        last[max(RANKS, key=lambda r: gap[r][i])] += 1
    for r in RANKS:
        print(f"  r{r}: {last[r]:5d}/{m} ({100 * last[r] / m:.1f}%)")
    print(
        "  pairwise interval difference Z100 - Z100L: "
        + ", ".join(
            f"r{b}-r{a} mean={statistics.mean([gap[b][i] - gap[a][i] for i in range(m)]):6.1f} "
            f"median={statistics.median([gap[b][i] - gap[a][i] for i in range(m)]):6.1f}"
            for a, b in ((0, 2), (1, 3))
        )
    )


if __name__ == "__main__":
    main()
