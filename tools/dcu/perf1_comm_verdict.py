#!/usr/bin/env python3
"""perf1: apply the Lead's decision rule to round-2 measurements.

Rule (Lead, 2026-10-13):
  recommend NCCL_PROTO=LL only if
    (a) the decode-size gain (10 KiB / 30 KiB / 120 KiB) reproduces at >= -25%
        versus the default configuration, AND
    (b) the large-message penalty is bounded: extra time on the ~140 x 20 MiB
        prefill all-reduces stays under ~5% of TTFT, AND
    (c) the staggered chain does not show the advantage vanishing under a
        300 us straggler (if it does, the verdict is "keep protocol auto"
        regardless of the microbenchmark).

Usage: python3 tools/dcu/perf1_comm_verdict.py [matrix_log] [chain_log]
"""

import math
import re
import sys

MATRIX = sys.argv[1] if len(sys.argv) > 1 else "/mnt/game/perf1/comm_matrix2.log"
CHAIN = sys.argv[2] if len(sys.argv) > 2 else "/mnt/game/perf1/comm_chain.log"

TTFT_MS_8K = 7160.0  # report section 2, 7443-token prompt at MBT=2048
CHUNKS_8K = 4  # ceil(8100 / 2048)
COLLECTIVES_PER_CHUNK = 140.0
TTFT_BUDGET_FRAC = 0.05
DECODE_GAIN_REQUIRED = -0.25
CHAIN_NULL_FRAC = 0.05  # an advantage below 5% of the collective is a null


def parse_matrix(path):
    rows = {}
    for line in open(path):
        m = re.match(r"###\s+variant=(\S+)", line)
        if m:
            cur = m.group(1)
            continue
        for m in re.finditer(r"\[(\w+)\] bytes=\s*(\d+) min=\s*([\d.]+)", line):
            rows.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(3))
    return rows


def parse_chain(path):
    out = {}
    for line in open(path):
        m = re.search(r"\[(\w+)\] chain per-collective us=\[+([^\]]+)\]", line)
        if m:
            us = [float(x) for x in m.group(2).split(",")]
            out[m.group(1)] = (us, max(us))
    return out


def main() -> None:
    rows = parse_matrix(MATRIX)
    if "r2_default" not in rows or "r2_LL" not in rows:
        print("matrix2 incomplete: have", sorted(rows))
        return
    base, ll = rows["r2_default"], rows["r2_LL"]
    print("=== round-2 ladder (us/call) ===")
    sizes = sorted(set(base) | set(ll))
    hdr = "payload".ljust(10) + "".join(f"{v:>12}" for v in sorted(rows))
    print(hdr)
    for s in sizes:  # s is a BYTE count (the log prints bytes=)
        lbl = f"{s / 1024:.0f} KiB" if s < 1 << 20 else f"{s >> 20} MiB"
        print(lbl.ljust(10) + "".join(f"{rows[v].get(s, float('nan')):>12.1f}" for v in sorted(rows)))

    dec = [s for s in sizes if 10240 <= s <= 122880]  # 10 KiB / 30 KiB / 120 KiB
    gains = [(ll[s] - base[s]) / base[s] for s in dec if s in ll and s in base]
    gain = sum(gains) / len(gains) if gains else float("nan")
    print(f"\n(a) decode-size mean gain vs default: {100 * gain:+.1f}% "
          f"(pooled over {len(gains)} sizes) -> "
          f"{'PASS' if gain <= DECODE_GAIN_REQUIRED else 'FAIL'}")

    big = max(sizes)
    pen_us = ll.get(big, float("nan")) - base.get(big, float("nan"))
    pen_ms_per_chunk = pen_us * COLLECTIVES_PER_CHUNK / 1000.0
    pen_ms_total = pen_ms_per_chunk * CHUNKS_8K
    frac = pen_ms_total / TTFT_MS_8K
    print(f"(b) {big >> 20} MiB penalty: {pen_us:+.1f} us/AR -> "
          f"{pen_ms_per_chunk:+.1f} ms/chunk, {pen_ms_total:+.1f} ms over {CHUNKS_8K} chunks "
          f"= {100 * frac:+.2f}% of a {TTFT_MS_8K:.0f} ms TTFT -> "
          f"{'PASS' if frac < TTFT_BUDGET_FRAC else 'FAIL'}")

    chain = parse_chain(CHAIN)
    print("\n=== staggered chain (us per collective, max over ranks = critical) ===")
    for k in ("c_def_s0", "c_LL_s0", "c_def_s300", "c_LL_s300"):
        if k in chain:
            print(f"  {k:12s} per-rank={[round(x, 1) for x in chain[k][0]]} "
                  f"critical={chain[k][1]:.1f}")
    ok_chain = True
    for s in ("s0", "s300"):
        a, b = chain.get(f"c_def_{s}"), chain.get(f"c_LL_{s}")
        if a and b:
            gainc = a[1] - b[1]
            rel = gainc / a[1]
            verdict = "LL wins" if rel > CHAIN_NULL_FRAC else "NULL (no gain)"
            if s == "s300" and rel <= CHAIN_NULL_FRAC:
                ok_chain = False
            print(f"  stagger={s}: LL saves {gainc:+.1f} us/collective "
                  f"({100 * rel:+.1f}%) -> {verdict}")
    print(f"(c) chain under a 300 us straggler: "
          f"{'PASS' if ok_chain else 'FAIL (keep protocol auto)'}")

    print("\n=== VERDICT ===")
    if gain <= DECODE_GAIN_REQUIRED and frac < TTFT_BUDGET_FRAC and ok_chain:
        print("RECOMMEND NCCL_PROTO=LL (pending a same-vintage parity check)")
    else:
        print("KEEP PROTOCOL AUTO; do not set NCCL_PROTO in the recipe")


if __name__ == "__main__":
    main()
