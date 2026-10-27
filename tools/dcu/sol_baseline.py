#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SOL (speed-of-light) baseline for Qwen3.8-27B-FP8 on 4x Z100L, TP=4.

Pure arithmetic, no GPU: every constant is either a vendor spec or a number
already measured on this host (see ``SOURCES``).  The point is to answer
"what is the hardware ceiling" before any optimisation, and to keep the
answer reproducible when someone questions an input.

Usage::

    .venv/bin/python tools/dcu/sol_baseline.py            # markdown tables
    .venv/bin/python tools/dcu/sol_baseline.py --svg out.svg

Model shape: Qwen3.8-27B, hybrid attention (64 layers, 16 full attention,
48 GDN), dense 27B, FP8 weights; served TP=4 with DFlash2 k=7.
"""

from __future__ import annotations

import argparse
import math

# SVG markup lines are long by construction.
# ruff: noqa: E501

# ---------------------------------------------------------------- constants
H = 5120  # hidden size == K of every linear layer
N_PARAMS = 27e9  # total parameters (dense)
W_BYTES = 27e9  # FP8: 1 byte/param, whole model
FLOP_PER_TOKEN = 2 * N_PARAMS  # 54 GFLOP
TP = 4
N_LAYERS = 64
N_FULL = 16  # full-attention layers (full_attention_interval=4)
N_COLL = 2 * N_LAYERS + 12  # all-reduces per decode step (measured 128-143)

# Per-device ceilings.
F_SPEC = 20.2e12  # vendor: FP16 20.2-24.5 TFLOPS
F_ROOF = 19.61e12  # measured rocBLAS fp16 4096^3 (18.2 TFLOP/s elsewhere)
BW_SPEC = 670e9  # spec for this SKU
BW_COPY = 776e9  # measured D2D copy peak (best case)
BW_HONEST = 621.7e9  # measured streaming reference
P2P = 28.4e9  # PCIe 4.0 x16 peer copy, ACS off (6.95 with ACS on)

# KV / state traffic per rank.
KV_FP8 = 4 * 1024  # 16 layers x 2 x 1 kv-head x 128 dim x 1B
KV_BF16 = 16 * 1024  # bf16 + 256 head-dim accounting (S22)
STATE = 36.9e6  # 48 GDN layers x 12 v-heads/rank x 128 x 128 x fp32

# Reference measurements on this host (docs/perf/PERF1-results.md).
CUR_PREFILL_TPS = 840.0  # @7469 tokens, DTK arm
CUR_TTFT_7469 = 8.894  # s
CUR_STEP_7469 = 129.0  # ms, k=7, acc 2.84
CUR_ACC_7469 = 2.84
CUR_ACC_512 = 5.82
CUR_STEP_512 = 167.4  # ms
CUR_MFU = 0.578  # 840 t/s / 1452 t/s
FIXED_TTFT = 0.56  # s: scheduler / first step / sampling / HTTP

SOURCES = [
    ("每 token FLOPs（TP4 每 rank 13.5 GFLOP）", "2 x 27e9 = 54 GFLOP"),
    ("FP16 峰值 20.2-24.5 TFLOPS", "Z100L 规格（报告 §0）"),
    ("fp16 GEMM 屋顶 19.61 TF/s", "z100-rocm101-perf-p3-construction.md:107"),
    ("HBM 670 GB/s / copy 776 GB/s", "PERF1-results.md:36-38"),
    ("PCIe P2P 28.4 GB/s（ACS off）", "z100-12-p2p-acs-impact-result.md:191"),
    ("每步 128-143 次 all-reduce", "PERF1-results.md:62-70"),
    ("KV 16 KB/token/卡（bf16）", ".s22/S22_LONGCTX_DECODE.md:44-48"),
    ("KV 4 KB/token/卡（fp8）", "z100-dual-prefill-reconciliation-projection.md:353"),
    ("现状 prefill 840 t/s / step 129 ms", "PERF1-results.md:324-330"),
]


def prefill_sol(tokens: float, flops: float, attn_eff: float = 0.25) -> float:
    """Seconds for one prefill chunk at `flops` (aggregate) and `attn_eff`."""
    linear = 2 * N_PARAMS * tokens
    attn = 2 * tokens * tokens * H * N_FULL
    return linear / flops + attn / (attn_eff * flops)


def decode_step(
    sum_len: float,
    batch: int = 1,
    kv: float = KV_FP8,
    amp: float = 1.0,
    acc: float = 1.0,
    rows: int = 1,
    bw: float = BW_COPY,
    lop: float = 20e-6,
) -> tuple[float, float]:
    """SOL decode step: (ms, aggregate tokens/s).

    `amp` is the KV re-scan factor of the verify width, `acc` the accepted
    tokens per step, `rows` the verify width (1 + draft tokens) that sets
    the all-reduce message size.
    """
    moved = W_BYTES / TP + kv * sum_len * amp + STATE * batch
    t_mem = moved / bw
    msg = rows * H * 2  # rows x hidden, fp16
    t_comm = N_COLL * (1.5 * msg / P2P + lop)  # ring: 2(N-1)/N
    step = (t_mem + t_comm) * 1e3
    return step, batch * acc * 1e3 / step


def md_table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---:"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def section_constants() -> str:
    return md_table(["常量", "出处"], [[k, v] for k, v in SOURCES])


def section_prefill() -> str:
    rows = []
    for t in (512, 2048, 4096, 8192, 32768, 131072):
        linear = 2 * N_PARAMS * t
        attn = 2 * t * t * H * N_FULL
        share = attn / (linear + attn) * 100
        s_best = prefill_sol(t, 4 * F_ROOF, attn_eff=1.0)
        s_real = prefill_sol(t, 4 * F_ROOF, attn_eff=0.25)
        rows.append(
            [
                f"{t}",
                f"{linear / 1e12:.1f}",
                f"{attn / 1e12:.1f}",
                f"{share:.1f}%",
                f"{t / s_best:.0f}",
                f"{t / s_real:.0f}",
                f"{FIXED_TTFT + s_real:.2f}",
            ]
        )
    head = [
        "prompt",
        "linear TFLOP",
        "attn TFLOP",
        "attn 占比",
        "SOL t/s (attn 满速)",
        "SOL t/s (attn 25%)",
        "SOL TTFT s",
    ]
    return md_table(head, rows)


def section_decode() -> str:
    rows = []
    for ctx in (512, 4096, 7469, 32768, 131072):
        for kv_name, kv in (("fp8", KV_FP8), ("bf16", KV_BF16)):
            step, tps = decode_step(ctx, 1, kv=kv, amp=2.5, acc=CUR_ACC_7469, rows=8)
            rows.append(
                [
                    f"{ctx}",
                    kv_name,
                    f"{kv * ctx * 2.5 / 1e9:.2f}",
                    f"{step:.1f}",
                    f"{tps:.0f}",
                ]
            )
    head = ["ctx", "KV dtype", "KV 读 GB/步/卡", "SOL step ms", "tok/s (acc 2.84)"]
    return md_table(head, rows)


def section_batch() -> str:
    rows = []
    for name, sum_len, batch in (
        ("1 x 7.5k", 7469, 1),
        ("4 x 7.5k", 4 * 7469, 4),
        ("4 x 32k", 4 * 32768, 4),
        ("4 x 121k", 4 * 121000, 4),
    ):
        step, tps = decode_step(sum_len, batch, kv=KV_BF16, amp=2.5, acc=2.84, rows=8)
        rows.append(
            [
                name,
                f"{sum_len / 1e3:.0f}k",
                f"{step:.1f}",
                f"{tps:.0f}",
                f"{tps / batch:.0f}",
                f"{1452 / tps:.2f}x",
            ]
        )
    head = ["batch", "ΣL", "SOL step ms", "聚合 tok/s", "单流 tok/s", "距算力封顶"]
    return md_table(head, rows)


def section_comm() -> str:
    rows = []
    for rows_per_step, name in ((8, "verify k=7 (M=8)"), (1, "裸 decode (M=1)")):
        msg = rows_per_step * H * 2
        xfer = 1.5 * msg / P2P * 1e6
        for lop_us in (0, 10, 20, 72):
            total_ms = N_COLL * (xfer + lop_us) / 1e3
            rows.append(
                [
                    name,
                    f"{msg / 1024:.0f} KB",
                    f"{xfer:.1f}",
                    f"{lop_us}",
                    f"{total_ms:.1f}",
                ]
            )
    head = ["消息", "每 op 字节", "1.5x 传输 µs", "per-op 延迟 µs", "ms/步"]
    return md_table(head, rows)


def section_gap() -> str:
    sol_prefill = 4 * F_ROOF / (2 * N_PARAMS)
    step, tps = decode_step(7469, 1, kv=KV_BF16, amp=2.5, acc=CUR_ACC_7469, rows=8)
    rows = [
        [
            "prefill @7469",
            f"{CUR_PREFILL_TPS:.0f} t/s",
            f"{int(sol_prefill)} t/s",
            f"{sol_prefill / CUR_PREFILL_TPS:.1f}x",
        ],
        [
            "decode @7.5k c=1 k=7",
            f"{CUR_STEP_7469:.0f} ms / {CUR_ACC_7469 * 1e3 / CUR_STEP_7469:.0f} t/s",
            f"{step:.1f} ms / {tps:.0f} t/s",
            f"{CUR_STEP_7469 / step:.1f}x",
        ],
        [
            "decode 聚合封顶",
            "—（未测过饱和点）",
            f"{int(sol_prefill)} t/s",
            "—",
        ],
    ]
    return md_table(["指标", "现状（本机实测）", "SOL", "差距"], rows)


# ------------------------------------------------- measured 2026-10-25
# tools/dcu/sol_measure.py + tools/dcu/ar_bench.py on 4 x Z100L, idle cards,
# DTK 26.04 / torch 2.11 / HIP 6.3 container.  See
# docs/perf/SOL-MEASURED-20261025.md for the raw tables.
MEAS_COPY = 815.2e9  # D2D copy, 1.5 GB fp16, 4/4 cards 815-828
MEAS_READ = 783.9e9  # x.sum() over 3 GB, 4/4 cards 783.7-784.1
MEAS_F = 19.65e12  # rocBLAS fp16 4096^3, 4/4 cards 19.6-19.7
MEAS_P2P = 28.4e9  # 12/12 pairs, ACS off
MEAS_GEMV_ROCBLAS = 280e9  # torch.mv / x@W.T, best of N=1280..8704
MEAS_GEMV_STREAM = 414e9  # PERF1 stream GEMV, N=8704 (fp8 W8A16)
MEAS_AR = [
    (10 * 1024, 98.6),
    (40 * 1024, 101.0),
    (80 * 1024, 90.0),
    (160 * 1024, 95.5),
    (1024 * 1024, 132.7),
    (4 * 1024 * 1024, 297.2),
    (20 * 1024 * 1024, 1205.1),
    (80 * 1024 * 1024, 4641.1),
]


def ar_us(msg_bytes: float) -> float:
    """Per-op all-reduce latency (us) interpolated from the measured curve."""
    if msg_bytes <= MEAS_AR[0][0]:
        return MEAS_AR[0][1]
    for (x0, y0), (x1, y1) in zip(MEAS_AR, MEAS_AR[1:]):
        if msg_bytes <= x1:
            return y0 + (msg_bytes - x0) / (x1 - x0) * (y1 - y0)
    return MEAS_AR[-1][1] * msg_bytes / MEAS_AR[-1][0]


def decode_meas(
    sum_len: float,
    batch: int = 1,
    kv: float = KV_BF16,
    amp: float = 2.5,
    acc: float = 2.84,
    rows: int = 8,
    w_bw: float = MEAS_GEMV_STREAM,
    bw: float = MEAS_READ,
) -> tuple[float, float]:
    """Decode step with the measured bands: (ms, aggregate tokens/s)."""
    t_w = W_BYTES / TP / w_bw
    t_kv = (kv * sum_len * amp + STATE * batch) / bw
    t_ar = N_COLL * ar_us(rows * H * 2) / 1e6
    step = (t_w + t_kv + t_ar) * 1e3
    return step, batch * acc * 1e3 / step


def section_meas() -> str:
    rows = [
        ["HBM copy（1.5 GB fp16，2x bytes）", "815–828 GB/s", "776（旧）"],
        ["HBM read（x.sum，3 GB）", "783.7–784.1 GB/s", "—"],
        ["fp16 GEMM 4096^3（rocBLAS）", "19.6–19.7 TF/s", "19.61（旧）"],
        ["P2P copy（128 MiB，12/12 pair）", "28.4 GB/s", "28.4（一致）"],
        ["GEMV 权重流（torch.mv / M=1）", "109–292 GB/s", "175（生产 fp8）"],
        ["GEMV 权重流（stream kernel，PERF1）", "414 GB/s", "175（生产 fp8）"],
        ["行归约 w.sum(dim=1)（无 GEMV）", "578–684 GB/s", "—"],
    ]
    return md_table(["项", "实测", "本仓历史"], rows)


def section_ar() -> str:
    rows = []
    for msg, us in MEAS_AR:
        rows.append(
            [
                f"{msg / 1024:.0f} KB" if msg < 2**20 else f"{msg / 2**20:.0f} MB",
                f"{us:.1f}",
                f"{msg / us / 1e3:.1f}",
                f"{N_COLL * us / 1e3:.1f}",
            ]
        )
    head = ["消息", "per-op µs (p50)", "有效 GB/s", f"{N_COLL} 次/步 ms"]
    return md_table(head, rows)


def prefill_meas(tokens: float, attn_eff: float = 0.25) -> float:
    """Prefill seconds with the measured fp16 roof and the measured AR tax."""
    linear = 2 * N_PARAMS * tokens
    attn = 2 * tokens * tokens * H * N_FULL
    ar = 2 * N_LAYERS * ar_us(tokens * H * 2) / 1e6
    return linear / (4 * MEAS_F) + attn / (attn_eff * 4 * MEAS_F) + ar


def section_prefill_meas() -> str:
    rows = []
    for t in (512, 2048, 8192, 32768, 131072):
        secs = prefill_meas(t)
        ar_ms = 2 * N_LAYERS * ar_us(t * H * 2) / 1e3
        rows.append(
            [f"{t}", f"{ar_ms:.0f}", f"{t / secs:.0f}", f"{FIXED_TTFT + secs:.2f}"]
        )
    head = ["prompt", "AR 税 ms（128 次）", "prefill t/s", "TTFT s"]
    return md_table(head, rows)


def section_bands() -> str:
    rows = []
    for wname, wb in (
        ("copy 815（做梦）", MEAS_COPY),
        ("stream GEMV 414", MEAS_GEMV_STREAM),
        ("rocBLAS GEMV 280", MEAS_GEMV_ROCBLAS),
    ):
        for ctx in (7469, 32768, 131072):
            step, tps = decode_meas(ctx, 1, w_bw=wb)
            rows.append([wname, f"{ctx}", f"{step:.1f}", f"{tps:.0f}"])
    head = ["权重流口径", "ctx", "step ms", "tok/s (acc 2.84)"]
    return md_table(head, rows)


def svg(path: str) -> None:
    """Two-panel chart: decode step vs ΣL, prefill TTFT vs prompt."""
    w, h = 980, 440
    pw, ph, pad = 390, 300, 80
    x0a, y0 = pad, 90
    x0b = pad + pw + 130
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace" font-size="11">'
    )
    out = [head, f'<rect width="{w}" height="{h}" fill="#fff"/>']

    def panel(x0: int, title: str, xlabel: str, ylabel: str) -> None:
        out.append(
            f'<text x="{x0}" y="52" font-size="13" font-weight="bold">{title}</text>'
        )
        out.append(
            f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>'
        )
        out.append(
            f'<text x="{x0 + pw / 2}" y="{y0 + ph + 44}" text-anchor="middle">{xlabel}</text>'
        )
        out.append(
            f'<text x="{x0 - 48}" y="{y0 + ph / 2}" text-anchor="middle" '
            f'transform="rotate(-90 {x0 - 48} {y0 + ph / 2})">{ylabel}</text>'
        )

    LO, HI = 512.0, 1e6

    def sx(x0: int, v: float) -> float:
        f = (math.log10(v) - math.log10(LO)) / (math.log10(HI) - math.log10(LO))
        return x0 + f * pw

    def sy(v: float, lo: float, hi: float) -> float:
        return y0 + ph - (v - lo) / (hi - lo) * ph

    def inside(x0: int, x: float, y: float) -> None:
        assert x0 - 1 <= x <= x0 + pw + 1, (x0, x)
        assert y0 - 1 <= y <= y0 + ph + 1, (y0, y)

    # Panel A: decode step vs sum context.
    panel(x0a, "decode step vs ΣL (c=1, k=7, TP4)", "ΣL (tokens, log)", "step (ms)")
    for gv in (1e3, 1e4, 1e5, 1e6):
        x = sx(x0a, gv)
        out.append(
            f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + ph}" stroke="#eee"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{y0 + ph + 16}" text-anchor="middle">{gv:.0e}</text>'
        )
    for gv in (10, 20, 50, 100, 200):
        y = sy(gv, 8, 220)
        out.append(
            f'<line x1="{x0a}" y1="{y:.1f}" x2="{x0a + pw}" y2="{y:.1f}" stroke="#eee"/>'
        )
        out.append(f'<text x="{x0a - 8}" y="{y + 4:.1f}" text-anchor="end">{gv}</text>')
    for kv, color, label, ly in (
        (KV_FP8, "#1f77b4", "fp8 KV", 40),
        (KV_BF16, "#d62728", "bf16 KV", 60),
    ):
        pts = []
        v = LO
        while v <= HI:
            step, _ = decode_step(v, 1, kv=kv, amp=2.5, acc=2.84, rows=8)
            x, y = sx(x0a, v), sy(min(step, 215), 8, 220)
            inside(x0a, x, y)
            pts.append(f"{x:.1f},{y:.1f}")
            v *= 1.12
        out.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        out.append(f'<text x="{x0a + 8}" y="{y0 + ly}" fill="{color}">{label}</text>')
    yw = sy(8.7, 8, 220)
    out.append(
        f'<line x1="{x0a}" y1="{yw:.1f}" x2="{x0a + pw}" y2="{yw:.1f}" '
        f'stroke="#999" stroke-dasharray="4 3"/>'
    )
    out.append(
        f'<text x="{x0a + pw - 4}" y="{yw - 6:.1f}" text-anchor="end" '
        f'fill="#666">权重下限 8.7ms（不含通信）</text>'
    )
    out.append(
        f'<circle cx="{sx(x0a, 7469):.1f}" cy="{sy(129, 8, 220):.1f}" r="4" fill="#000"/>'
    )
    out.append(
        f'<text x="{sx(x0a, 7469) + 8:.1f}" y="{sy(129, 8, 220) - 7:.1f}">'
        f"现状 129ms / 22 t/s</text>"
    )

    # Panel B: prefill TTFT vs prompt length.
    panel(x0b, "prefill TTFT vs prompt (TP4)", "prompt tokens (log)", "TTFT (s)")
    for gv in (1e3, 1e4, 1e5):
        x = sx(x0b, gv)
        out.append(
            f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + ph}" stroke="#eee"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{y0 + ph + 16}" text-anchor="middle">{gv:.0e}</text>'
        )
    for gv in (5, 10, 20, 40, 80):
        y = sy(gv, 0.5, 90)
        out.append(
            f'<line x1="{x0b}" y1="{y:.1f}" x2="{x0b + pw}" y2="{y:.1f}" stroke="#eee"/>'
        )
        out.append(f'<text x="{x0b - 8}" y="{y + 4:.1f}" text-anchor="end">{gv}</text>')
    for tps, color, label, ly in (
        (4 * F_ROOF / (2 * N_PARAMS), "#2ca02c", "SOL 1452 t/s", 40),
        (CUR_PREFILL_TPS, "#ff7f0e", "现状 840 t/s", 60),
    ):
        pts = []
        v = LO
        while v <= HI:
            secs = FIXED_TTFT + v / tps
            if secs <= 89:
                x, y = sx(x0b, v), sy(secs, 0.5, 90)
                inside(x0b, x, y)
                pts.append(f"{x:.1f},{y:.1f}")
            v *= 1.12
        out.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        out.append(f'<text x="{x0b + 8}" y="{y0 + ly}" fill="{color}">{label}</text>')
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", help="write the two-panel chart here")
    ap.add_argument(
        "--measured",
        action="store_true",
        help="also print the 2026-10-25 measured bands",
    )
    args = ap.parse_args()

    if args.svg:
        svg(args.svg)
        print(f"wrote {args.svg}")
        return

    print("## 常量\n")
    print(section_constants())
    print("\n## prefill SOL\n")
    print(section_prefill())
    print("\n## decode SOL (c=1, k=7)\n")
    print(section_decode())
    print("\n## decode 批量（bf16 KV, k=7）\n")
    print(section_batch())
    print("\n## all-reduce 预算（140 次/步）\n")
    print(section_comm())
    print("\n## 现状 vs SOL\n")
    print(section_gap())
    if args.measured:
        print("\n## 实测校准（2026-10-25）\n")
        print(section_meas())
        print("\n## all-reduce 实测曲线（4 rank，bf16）\n")
        print(section_ar())
        print("\n## 三种权重流口径下的 decode\n")
        print(section_bands())
        print("\n## 实测口径的 prefill（含 AR 税）\n")
        print(section_prefill_meas())


if __name__ == "__main__":
    main()
