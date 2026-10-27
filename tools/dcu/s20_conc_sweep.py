#!/usr/bin/env python3
"""S20: the same concurrency sweep the round-9 report used, so the numbers are
comparable to results/thr_L1_batch_20261014-031148.json.

Reports, per concurrency, from the engine's own counters over the window:
  * forward t/s = (prompt_tokens + generation_tokens) / wall   <- the robust arm-comparable metric
  * generation t/s, step period, tokens/step, accept length
  * client-side TTFT and per-request output rate

  python3 s20_conc_sweep.py --concurrency 1,4,16 --tag cfg16
"""

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RX = {k: re.compile(p, re.M) for k, p in {
    "pt": r"^vllm:prompt_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "gt": r"^vllm:generation_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "itl": r"^vllm:inter_token_latency_seconds_sum\{[^}]*\}\s+([0-9.e+]+)",
    "itlc": r"^vllm:inter_token_latency_seconds_count\{[^}]*\}\s+([0-9.e+]+)",
    "d": r"^vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([0-9.e+]+)",
    "a": r"^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "run": r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.e+]+)",
    "itok_sum": r"^vllm:iteration_tokens_total_sum\{[^}]*\}\s+([0-9.e+]+)",
    "itok_cnt": r"^vllm:iteration_tokens_total_count\{[^}]*\}\s+([0-9.e+]+)",
}.items()}


def met(base):
    txt = urllib.request.urlopen(f"{base}/metrics", timeout=15).read().decode()
    return {k: float(rx.search(txt).group(1)) for k, rx in RX.items() if rx.search(txt)}


def one(base, i, pt, mt):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "prompt": " ".join(f"p{i}x{j}" for j in range(pt)),
        "max_tokens": mt, "temperature": 0.0, "ignore_eos": True, "stream": True,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    return n, ttft or 0.0


def wait_idle(base, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            m = met(base)
            if m.get("run", 1.0) == 0.0:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--out", default="/home/lyh/z100-vllm/.bench/ab20")
    args = ap.parse_args()

    out = []
    for c in (int(x) for x in args.concurrency.split(",")):
        wait_idle(args.base)
        one(args.base, 9999, 64, 8)          # warm
        time.sleep(1)
        m0 = met(args.base)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(one, args.base, i, args.prompt_tokens, args.max_tokens)
                    for i in range(c)]
            res = [f.result() for f in futs]
        wall = time.time() - t0
        m1 = met(args.base)
        pt, gt = m1["pt"] - m0["pt"], m1["gt"] - m0["gt"]
        steps = m1["itlc"] - m0["itlc"]
        dr, ac = m1["d"] - m0["d"], m1["a"] - m0["a"]
        rec = {
            "concurrency": c, "wall_s": wall,
            "fwd_tps": (pt + gt) / wall if wall else 0.0,
            "gen_tps": gt / wall if wall else 0.0,
            "step_ms": (m1["itl"] - m0["itl"]) / steps * 1000 if steps else 0.0,
            "tok_per_step": gt / steps if steps else 0.0,
            # the real batch size: tokens the model forwarded per engine step
            "itok_per_step": ((m1["itok_sum"] - m0["itok_sum"]) /
                              (m1["itok_cnt"] - m0["itok_cnt"])
                              if m1["itok_cnt"] != m0["itok_cnt"] else 0.0),
            "accept": (ac + dr) / dr if dr else 0.0,
            "ttft_ms": sum(r[1] for r in res) / len(res) * 1000,
            "per_req_tps": (sum(r[0] for r in res) / len(res)) / wall if wall else 0.0,
        }
        out.append(rec)
        print(f"[c={c:2d}] fwd={rec['fwd_tps']:7.1f} t/s  gen={rec['gen_tps']:6.1f}  "
              f"step={rec['step_ms']:7.1f} ms  tok/step={rec['tok_per_step']:5.2f}  "
              f"accept={rec['accept']:4.2f}  fwd/step={rec['itok_per_step']:6.1f}  "
              f"TTFT={rec['ttft_ms']:7.0f} ms  "
              f"per-req={rec['per_req_tps']:5.2f} t/s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"sweep_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "results": out}, f, indent=1)
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
