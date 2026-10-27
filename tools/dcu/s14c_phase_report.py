#!/usr/bin/env python3
"""S14c G1: phase-resolved host-memory attribution for one arm's memtrace.

Reads /mnt/game/s14c/hostlogs/<LABEL>-memtrace.log (S/P/M lines from
tools/dcu/s14c_memtrace.sh) and prints:

  1. a per-phase table: peak anonymous demand (all engine procs), min
     MemAvailable, swap use, and the [heap]/unnamed/file split at that peak,
  2. the global-peak sample and the lowest-MemAvailable sample with the full
     per-process breakdown,
  3. the steady-state medians,
  4. a downsampled curve.

Phase model (built to be monotonic -- the engine logs `Available KV cache
memory` *before* the second PIECEWISE capture pass finishes, so a naive
"latest marker wins" model produces overlapping phases):

  P0_bootstrap   before the first engine process exists
  P1_weight_load up to the first  `Model loading took`
  P2_compile     up to the first  `torch.compile took` / `Capturing CUDA graphs`
                 (for eager: up to the first `Available KV cache memory`)
  P3_capture_kv  up to the *last*  `Graph capturing finished` (all ranks done)
  P4_warmup      up to `Application startup complete`
  P5_steady      after that

Every boundary is `first marker at or after the previous boundary`; the capture
boundary uses the last occurrence so all ranks have finished.
"""

import argparse
import re
import statistics
import sys

RE_LOAD = re.compile(r"Model loading took")
RE_COMPILE = re.compile(r"torch\.compile took|Capturing CUDA graphs")
RE_KV = re.compile(r"Available KV cache memory|init engine \(profile")
RE_CAPDONE = re.compile(r"Graph capturing finished")
RE_READY = re.compile(r"Application startup complete")

PHASES = ["P0_bootstrap", "P1_weight_load", "P2_compile", "P3_capture_kv",
          "P4_warmup", "P5_steady"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--curve-every", type=int, default=12)
    args = ap.parse_args()

    S, P, marks = {}, {}, []          # marks: list of (epoch, text)
    for ln in open(args.path, errors="replace"):
        f = ln.split()
        if not f:
            continue
        if f[0] == "S" and len(f) >= 16:
            S[int(f[2])] = {
                "iso": f[1], "mtot": int(f[3]), "mfree": int(f[4]),
                "mavail": int(f[5]), "cached": int(f[6]), "shmem": int(f[7]),
                "stot": int(f[8]), "sfree": int(f[9]),
                "cgcur": f[10], "cgmax": f[11], "oomk": f[12],
                "nw": int(f[13]), "ne": int(f[14]), "vram": int(f[15]),
            }
        elif f[0] == "P" and len(f) >= 14:
            P.setdefault(int(f[1]), []).append(
                {"role": f[2], "pid": int(f[3]), "anon": int(f[4]),
                 "file": int(f[5]), "swap": int(f[6]), "heap": int(f[7]),
                 "unnamed": int(f[8]), "named": int(f[9]), "stack": int(f[10]),
                 "other": int(f[11]), "file_agg": int(f[12]), "atot": int(f[13]),
                 "vmsize": int(f[14]) if len(f) > 14 else 0})
        elif f[0] == "M" and len(f) >= 3:
            marks.append((int(f[1]), " ".join(f[3:]))) 

    if not S:
        print("no samples", file=sys.stderr)
        return 1
    epochs = sorted(S)
    t0 = epochs[0]
    marks.sort()

    # The tracer may outlive its own arm and then pick up a *later* arm's engine
    # (S14c restarted tracers / ran arms back-to-back).  Truncate at the first
    # sample after the peak where the device is empty -- that is this arm's stop.
    _peak = max(epochs, key=lambda e: sum(p["anon"] for p in P.get(e, [])))
    _cut = epochs[-1]
    for e in epochs:
        if e > _peak and S[e]["vram"] == 0:
            _cut = e
            break
    _n_all = len(epochs)
    epochs = [e for e in epochs if e <= _cut]

    def first_after(rx, lo):
        for ep, txt in marks:
            if ep >= lo and rx.search(txt):
                return ep
        return None

    def last_after(rx, lo):
        hit = None
        for ep, txt in marks:
            if ep >= lo and rx.search(txt):
                hit = ep
        return hit

    bA = first_after(RE_LOAD, t0)
    base = bA if bA else t0
    bB = first_after(RE_COMPILE, base) or first_after(RE_KV, base)
    base = bB if bB else base
    bC = last_after(RE_CAPDONE, base) or first_after(RE_KV, base)
    base = bC if bC else base
    bD = first_after(RE_READY, base)
    first_any = min(P) if P else None
    bounds = sorted((lo, nm) for lo, nm in (
        (first_any, "P1_weight_load"), (bA, "P2_compile"), (bB, "P3_capture_kv"),
        (bC, "P4_warmup"), (bD, "P5_steady")) if lo is not None)

    def phase_of(ep):
        cur = "P0_bootstrap"
        for bep, nm in bounds:
            if ep >= bep:
                cur = nm
        return cur

    def sums(ep):
        ps = P.get(ep, [])
        return {k: sum(p[k] for p in ps) for k in
                ("anon", "heap", "unnamed", "named", "stack", "other", "atot",
                 "file", "file_agg", "swap")}

    by_phase = {}
    for ep in epochs:
        by_phase.setdefault(phase_of(ep), []).append(ep)

    print(f"# {args.path}")
    print(f"# samples={len(epochs)}/{_n_all} (truncated at arm stop +{_cut-t0}s) span={epochs[-1]-epochs[0]}s "
          f"first={S[epochs[0]]['iso']} last={S[epochs[-1]]['iso']}")
    print("# boundaries (+N s from run start):")
    for bep, nm in bounds:
        print(f"#   +{bep-t0:<5}s {nm}")
    print()
    hdr = (f"{'phase':15} {'t0':>5} {'t1':>5} {'dur':>5} {'n':>3} {'minAvail':>8} "
           f"{'maxSwap':>7} {'peakAnon':>8} {'=heap':>6} {'unnamed':>8} "
           f"{'file':>5} {'maxCgGiB':>8} {'oomk':>4}")
    print(hdr)
    print("-" * len(hdr))
    for ph in PHASES:
        eps = by_phase.get(ph)
        if not eps:
            continue
        # samples with no engine holding VRAM are post-shutdown tail: they must
        # not dilute the medians / "peak - steady" of a live phase
        if ph != "P0_bootstrap":
            live = [e for e in eps if S[e]["vram"] > 1000]
            if live:
                eps = live
        peak_an, peak_ep = max((sums(ep)["anon"], ep) for ep in eps)
        peak_s = sums(peak_ep)
        print(f"{ph:15} {eps[0]-t0:>5} {eps[-1]-t0:>5} {eps[-1]-eps[0]:>5} {len(eps):>3} "
              f"{min(S[e]['mavail'] for e in eps)//1024:>8} "
              f"{max((S[e]['stot']-S[e]['sfree']) for e in eps)//1024:>7} "
              f"{peak_an:>8} {peak_s['heap']:>6} {peak_s['unnamed']:>8} "
              f"{peak_s['file']:>5} "
              f"{max(int(S[e]['cgcur']) for e in eps)/2**30:>8.2f} "
              f"{S[eps[-1]]['oomk']:>4}")

    if P:
        peak_ep = max(epochs, key=lambda e: sums(e)["anon"])
        low_ep = min(epochs, key=lambda e: S[e]["mavail"])
        for tag, ep in (("GLOBAL PEAK (max Sum RssAnon)  t=+%ds" % (peak_ep - t0), peak_ep),
                        ("LOWEST MemAvailable           t=+%ds" % (low_ep - t0), low_ep)):
            s = S[ep]
            print(f"\n## {tag}  {s['iso']}")
            print(f"   MemAvailable={s['mavail']//1024} MiB  MemFree={s['mfree']//1024} MiB "
                  f"Cached={s['cached']//1024} MiB  swap_used={(s['stot']-s['sfree'])//1024} MiB "
                  f"cg_current={int(s['cgcur'])/2**30:.2f} GiB oom_kill={s['oomk']} "
                  f"vram={s['vram']} MiB")
            print(f"   {'role':4} {'pid':>8} {'RssAnon':>8} {'[heap]':>7} {'unnamed':>8} "
                  f"{'RssFile':>8} {'VmSwap':>7} {'VmSize':>7}")
            for p in sorted(P.get(ep, []), key=lambda x: -x["anon"]):
                print(f"   {p['role']:4} {p['pid']:>8} {p['anon']:>8} {p['heap']:>7} "
                      f"{p['unnamed']:>8} {p['file']:>8} {p['swap']:>7} {p['vmsize']:>7}")
            t = sums(ep)
            print(f"   {'TOT':4} {'':>8} {t['anon']:>8} {t['heap']:>7} {t['unnamed']:>8} "
                  f"{t['file']:>8} {t['swap']:>7}")

    eps = [e for e in by_phase.get("P5_steady", []) if S[e]["vram"] > 1000]
    if len(eps) >= 2:

        def med(fn):
            return int(statistics.median(fn(e) for e in eps))

        print(f"\n## STEADY (P5) medians over n={len(eps)} samples "
              f"[+{eps[0]-t0}s..+{eps[-1]-t0}s]")
        print(f"   MemAvailable={med(lambda e: S[e]['mavail']//1024)} MiB  "
              f"swap_used={med(lambda e: (S[e]['stot']-S[e]['sfree'])//1024)} MiB  "
              f"SumRssAnon={med(lambda e: sums(e)['anon'])} MiB "
              f"(heap={med(lambda e: sums(e)['heap'])} unnamed={med(lambda e: sums(e)['unnamed'])})  "
              f"SumRssFile={med(lambda e: sums(e)['file'])} MiB  "
              f"SumVmSwap={med(lambda e: sums(e)['swap'])} MiB")
        gp = max(sums(e)["anon"] for e in epochs)
        st = med(lambda e: sums(e)["anon"])
        print(f"   peak SumRssAnon (whole run) = {gp} MiB ; steady median = {st} MiB ; "
              f"peak - steady = {gp - st} MiB")

    print(f"\n## CURVE (every {args.curve_every} samples)")
    print(f"{'t+s':>5} {'avail':>6} {'swapU':>6} {'cgGiB':>6} {'vram':>6} "
          f"{'SumAnon':>8} {'Sumheap':>7} {'Sumunnamed':>10} {'Sumfile':>7} per-proc RssAnon")
    for i, ep in enumerate(epochs):
        if i % args.curve_every:
            continue
        s = S[ep]
        ps = P.get(ep, [])
        t = sums(ep)
        per = "+".join(str(p["anon"]) for p in sorted(ps, key=lambda x: (x["role"], x["pid"])))
        print(f"{ep-t0:>5} {s['mavail']//1024:>6} {(s['stot']-s['sfree'])//1024:>6} "
              f"{int(s['cgcur'])/2**30:>6.2f} {s['vram']:>6} {t['anon']:>8} {t['heap']:>7} "
              f"{t['unnamed']:>10} {t['file']:>7} {per}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
