#!/usr/bin/env python3
"""p3_score.py — P3 施工图的**唯一判读口径**（其它 agent 一律用这个，不要自己算）。

输入一个 probe run 目录（含 `probe-<case>/http_summary.json` 与 `probe-<case>/metrics_diff.json`），
输出标准指标表 + 机器可读 JSON。定义与历史 E 臂一致：

  TTFT  = http_summary.ttft_ms.p50                     （客户端，毫秒）
  TPOT  = http_summary.tpot_ms.p50                     （客户端，毫秒）
  step  = metrics_diff.histograms["vllm:inter_token_latency_seconds"].mean_ms
          （服务端 decode 步进均值；这是物理量，别用 itl_ms.p50 混比）
  accLen = (accepted + drafts) / drafts                （含 bonus token，与历史 E 臂同口径）
  acc%   = accepted / draft_tokens
  out/s  = accepted / (e2e 秒) 汇聚近似；并发档以 agg 行为准

之所以不直接用 `case_metrics.json`：它是另一套脚本派生的，字段名与算法在历史上漂过
（`accept_length` 是否含 bonus 就漂过一次）。本脚本只依赖 http_summary + metrics_diff
这两个**探针原始产物**，口径固定在这里。

用法：
  python3 tools/dcu/p3_score.py /mnt/game/port/perf-runs/20261024-053333-E2C
  python3 tools/dcu/p3_score.py <run> --json out.json --baseline <baseline_run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ITL = "vllm:inter_token_latency_seconds"
ACCEPT = "vllm:spec_decode_num_accepted_tokens_total"
DRAFT = "vllm:spec_decode_num_draft_tokens_total"
NDRAFTS = "vllm:spec_decode_num_drafts_total"


def _num(x, nd=2):
    return None if x is None else round(float(x), nd)


def score_case(case_dir: Path) -> dict:
    out: dict = {"case": case_dir.name.removeprefix("probe-")}
    hs = case_dir / "http_summary.json"
    md = case_dir / "metrics_diff.json"

    if hs.exists():
        d = json.loads(hs.read_text())
        cfg = d.get("config", {})
        out["ttft_ms"] = _num((d.get("ttft_ms") or {}).get("p50"))
        out["tpot_ms"] = _num((d.get("tpot_ms") or {}).get("p50"))
        out["e2e_ms"] = _num((d.get("e2e_ms") or {}).get("p50"))
        out["itl_p50_ms"] = _num((d.get("itl_ms") or {}).get("p50_ms"))
        out["n_ok"] = d.get("ok")
        out["n_fail"] = d.get("failed")
        out["nominal_prompt_tokens"] = cfg.get("prompt_tokens")
        out["concurrency"] = cfg.get("concurrency")
        out["actual_prompt_tokens"] = (d.get("prompt_tokens") or [None])[0]
        out["out_tokens_per_req"] = cfg.get("max_tokens")
    else:
        out["error"] = "no http_summary.json"

    if md.exists():
        m = json.loads(md.read_text())
        h = (m.get("histograms") or {}).get(ITL) or {}
        c = m.get("counters") or {}
        out["step_mean_ms"] = _num(h.get("mean_ms"))
        acc = c.get(ACCEPT)
        drf = c.get(DRAFT)
        ndr = c.get(NDRAFTS)
        out["accepted_tokens"] = acc
        out["draft_tokens"] = drf
        out["num_drafts"] = ndr
        if ndr:
            out["accept_length"] = _num((acc or 0) / ndr + 1.0, 3)
        if drf:
            out["accept_rate_pct"] = _num(100.0 * (acc or 0) / drf, 1)

    # 汇聚吞吐：accepted / decode 墙钟（无则退回 e2e）
    if out.get("accepted_tokens") and out.get("step_mean_ms") and out.get("num_drafts"):
        out["agg_tok_s"] = _num(
            1000.0 * out["accepted_tokens"] / (out["num_drafts"] * out["step_mean_ms"]), 2
        )
    return out


def route_evidence(run: Path) -> dict:
    """从 serve.log 抽路由证据。缺日志时返回空，不报错。"""
    log = run / "serve.log"
    if not log.exists():
        return {}
    ev: dict = {"serve_log": str(log)}
    w8a16 = []
    mats = 0
    gdn_chunked = 0
    gdn_triton = 0
    conv1d_triton = 0
    with log.open(errors="replace") as fh:
        for line in fh:
            if "W8A16 engaged" in line:
                w8a16.append(line.strip()[-120:])
            elif "first materialization route engaged" in line:
                mats += 1
            elif "chunked_gated_delta_rule" in line or "chunked prefill" in line:
                gdn_chunked += 1
            elif "gdn_mtp_triton" in line or "MTP_GDN_TRITON" in line:
                gdn_triton += 1
            elif "CAUSAL_CONV1D" in line:
                conv1d_triton += 1
    ev["w8a16"] = w8a16[:4]
    ev["materialization_hits"] = mats
    ev["gdn_chunked_lines"] = gdn_chunked
    ev["gdn_triton_lines"] = gdn_triton
    ev["conv1d_lines"] = conv1d_triton
    return ev


def stack_identity(run: Path) -> dict:
    p = run / "stack_identity.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {
        "ok": d.get("ok"),
        "torch": d.get("torch_version"),
        "triton": d.get("triton_version"),
        "so_md5": d.get("so_md5"),
        "src_dir": d.get("src_dir"),
    }


MUST = {
    "ttft_ms": ("<=", 2500.0),
    "step_mean_ms": ("<=", 210.0),
    "tpot_ms": ("<=", 45.0),
    "accept_length": (">=", 4.0),
    "accept_rate_pct": (">=", 45.0),
}


def gate(row: dict) -> dict:
    res = {}
    for k, (op, thr) in MUST.items():
        v = row.get(k)
        if v is None:
            res[k] = "n/a"
        else:
            res[k] = "PASS" if ((v <= thr) if op == "<=" else (v >= thr)) else "FAIL"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", default=None)
    ap.add_argument("--baseline", default=None, help="基线 run 目录，用于算 delta")
    a = ap.parse_args()

    run = Path(a.run)
    if not run.is_dir():
        print(f"❌ 不是目录：{run}", file=sys.stderr)
        return 2

    cases = sorted([p for p in run.iterdir() if p.is_dir() and p.name.startswith("probe-")])
    if not cases:
        print(f"❌ {run} 下没有 probe-* 目录", file=sys.stderr)
        return 2

    rows = [score_case(c) for c in cases]
    base = {}
    if a.baseline:
        base = {r["case"]: r for r in (score_case(c) for c in sorted(
            p for p in Path(a.baseline).iterdir()
            if p.is_dir() and p.name.startswith("probe-")))}

    hdr = (f"{'case':10s} {'tok':>5s} {'c':>2s} {'TTFT ms':>9s} {'step ms':>8s} "
           f"{'TPOT ms':>8s} {'accLen':>7s} {'acc%':>6s} {'agg t/s':>8s}  gate")
    print(f"== {run}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        d = ""
        if r["case"] in base:
            b = base[r["case"]]
            for k, short in (("ttft_ms", "TTFT"), ("step_mean_ms", "step")):
                if b.get(k) and r.get(k):
                    d += f" {short}{100.0 * (r[k] - b[k]) / b[k]:+.1f}%"
        g = gate(r)
        npass = sum(1 for v in g.values() if v == "PASS")
        print(f"{r['case']:10s} {str(r.get('actual_prompt_tokens') or '-'):>5s} "
              f"{str(r.get('concurrency') or '-'):>2s} "
              f"{str(r.get('ttft_ms')):>9s} {str(r.get('step_mean_ms')):>8s} "
              f"{str(r.get('tpot_ms')):>8s} {str(r.get('accept_length')):>7s} "
              f"{str(r.get('accept_rate_pct')):>6s} {str(r.get('agg_tok_s')):>8s}  "
              f"{npass}/5{d}")

    ev = route_evidence(run)
    si = stack_identity(run)
    print()
    if ev:
        print(f"路由证据：materialization={ev.get('materialization_hits')} "
              f"gdn_chunked={ev.get('gdn_chunked_lines')} gdn_triton={ev.get('gdn_triton_lines')} "
              f"conv1d={ev.get('conv1d_lines')}")
        for w in ev.get("w8a16", []):
            print(f"  {w}")
    else:
        print("路由证据：⚠️ 无 serve.log")
    if si:
        bad = [] if si.get("ok") else ["stack_identity.ok != true"]
        print(f"栈身份：ok={si.get('ok')} torch={si.get('torch')} triton={si.get('triton')}"
              + (f"  ⚠️ {bad}" if bad else ""))

    payload = {"run": str(run), "cases": rows,
               "route_evidence": ev, "stack_identity": si,
               "gates": {r["case"]: gate(r) for r in rows}}
    if a.json:
        Path(a.json).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        print(f"\nJSON -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
