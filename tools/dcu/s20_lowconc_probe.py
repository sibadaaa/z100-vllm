#!/usr/bin/env python3
"""S20: per-request timing at low concurrency (c <= 4) — the regime that matters.

The aggregate sweep (`s20_conc_sweep.py`) reports engine-level means, which mix
prefill steps and decode steps and therefore cannot say *where* a low-concurrency
request spends its time.  This probe issues N concurrent streaming requests and
records, per request:

  * TTFT (prefill + queueing)
  * inter-token latency distribution after the first token (pure decode)
  * total wall, and the ratio wall/(ttft + itl*(n-1)) so queueing shows up
    as a residual rather than being hidden in an average
  * the engine's own step period over the window, for cross-checking

  python3 s20_lowconc_probe.py --concurrency 1,4 --requests 4 --tag c4
"""

import argparse
import json
import os
import re
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RX = {k: re.compile(p, re.M) for k, p in {
    "itl": r"^vllm:inter_token_latency_seconds_sum\{[^}]*\}\s+([0-9.e+]+)",
    "itlc": r"^vllm:inter_token_latency_seconds_count\{[^}]*\}\s+([0-9.e+]+)",
    "itok_sum": r"^vllm:iteration_tokens_total_sum\{[^}]*\}\s+([0-9.e+]+)",
    "itok_cnt": r"^vllm:iteration_tokens_total_count\{[^}]*\}\s+([0-9.e+]+)",
    "pt": r"^vllm:prompt_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "gt": r"^vllm:generation_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "run": r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.e+]+)",
}.items()}


def met(base):
    txt = urllib.request.urlopen(f"{base}/metrics", timeout=15).read().decode()
    return {k: float(rx.search(txt).group(1)) for k, rx in RX.items() if rx.search(txt)}


def one(base, i, pt, mt):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "prompt": " ".join(f"q{i}z{j}" for j in range(pt)),
        "max_tokens": mt, "temperature": 0.0, "ignore_eos": True, "stream": True,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    stamps = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                stamps.append(time.time())
    if not stamps:
        return None
    ttft = stamps[0] - t0
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    return {
        "i": i, "chunks": len(stamps), "ttft_ms": ttft * 1000,
        "itl_mean_ms": statistics.mean(itls) * 1000 if itls else 0.0,
        "itl_p50_ms": statistics.median(itls) * 1000 if itls else 0.0,
        "itl_max_ms": max(itls) * 1000 if itls else 0.0,
        "wall_s": stamps[-1] - t0,
    }


def wait_idle(base, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if met(base).get("run", 1.0) == 0.0:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", default="1,4")
    ap.add_argument("--requests", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tag", default="lowconc")
    ap.add_argument("--out", default="/home/lyh/z100-vllm/.bench/ab20")
    args = ap.parse_args()

    out = []
    for c in (int(x) for x in args.concurrency.split(",")):
        wait_idle(args.base)
        one(args.base, 9999, 64, 8)  # warm
        time.sleep(1)
        m0 = met(args.base)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(one, args.base, i, args.prompt_tokens, args.max_tokens)
                    for i in range(args.requests)]
            res = [f.result() for f in futs if f.result()]
        wall = time.time() - t0
        m1 = met(args.base)
        steps = m1["itlc"] - m0["itlc"]
        gt = m1["gt"] - m0["gt"]
        itok = ((m1["itok_sum"] - m0["itok_sum"]) /
                (m1["itok_cnt"] - m0["itok_cnt"])
                if m1["itok_cnt"] != m0["itok_cnt"] else 0.0)
        rec = {
            "concurrency": c, "wall_s": wall,
            "engine_step_ms": (m1["itl"] - m0["itl"]) / steps * 1000 if steps else 0.0,
            "engine_fwd_per_step": itok,
            "engine_gen_tps": gt / wall if wall else 0.0,
            "per_req": {
                "ttft_ms": statistics.mean(r["ttft_ms"] for r in res),
                "ttft_min": min(r["ttft_ms"] for r in res),
                "ttft_max": max(r["ttft_ms"] for r in res),
                "itl_p50_ms": statistics.mean(r["itl_p50_ms"] for r in res),
                "itl_max_ms": max(r["itl_max_ms"] for r in res),
                "wall_s": statistics.mean(r["wall_s"] for r in res),
                "out_tps": statistics.mean(r["chunks"] / r["wall_s"] for r in res),
            },
        }
        out.append(rec)
        p = rec["per_req"]
        print(f"[c={c}] per-req: TTFT {p['ttft_ms']:.0f} ms ({p['ttft_min']:.0f}-{p['ttft_max']:.0f})  "
              f"ITL p50 {p['itl_p50_ms']:.0f} ms (worst gap {p['itl_max_ms']:.0f})  "
              f"wall {p['wall_s']:.2f} s  {p['out_tps']:.2f} t/s | "
              f"engine: step {rec['engine_step_ms']:.1f} ms, fwd/step {rec['engine_fwd_per_step']:.1f}, "
              f"gen {rec['engine_gen_tps']:.1f} t/s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"lowconc_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "results": out}, f, indent=1)
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
