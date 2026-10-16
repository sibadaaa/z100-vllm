#!/usr/bin/env python3
"""S20: high-power c=1 A/B probe.

The single-shot `s20_single_probe.py` cannot resolve a <5% step-time difference:
its per-arm spread across 4 runs is +/-5% and the first post-restart run is
systematically ~5% slow.  This probe measures a much longer window and reports
the engine's own counters over it, so the sampling error is dominated by real
drift rather than by one request.

Reported quantities (all over the same window):
  * p50 and trimmed-mean of per-request server ITL (the engine's step period);
  * server forward tokens/s = generation tokens / wall;
  * output tokens/s;
  * accept length (must not move: it is the correctness canary).

  python3 s20_ab_probe.py --requests 24 --tag cfg16
"""

import argparse
import json
import os
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MET = {
    "gen_tok": r"^vllm:generation_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "itl_sum": r"^vllm:inter_token_latency_seconds_sum\{[^}]*\}\s+([0-9.e+]+)",
    "itl_cnt": r"^vllm:inter_token_latency_seconds_count\{[^}]*\}\s+([0-9.e+]+)",
    "accept": r"^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "drafts": r"^vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([0-9.e+]+)",
    "gen_hist": None,
}
_RX = {k: re.compile(p, re.M) for k, p in MET.items() if p}


def metrics(base):
    txt = urllib.request.urlopen(f"{base}/metrics", timeout=15).read().decode()
    out = {k: float(rx.search(txt).group(1)) for k, rx in _RX.items() if rx.search(txt)}
    # histogram buckets of inter_token_latency give a real per-step p50
    buckets = []
    for m in re.finditer(
        r'^vllm:inter_token_latency_seconds_bucket\{[^}]*le="([0-9.e+]+)"[^}]*\}\s+([0-9.e+]+)',
        txt, re.M,
    ):
        buckets.append((float(m.group(1)), float(m.group(2))))
    out["_buckets"] = sorted(buckets)
    return out


def wait_idle(base, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        txt = urllib.request.urlopen(f"{base}/metrics", timeout=10).read().decode()
        r = re.search(r'^vllm:num_requests_running\{[^}]*\}\s+([0-9.e+]+)', txt, re.M)
        w = re.search(r'^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.e+]+)', txt, re.M)
        if r and w and float(r.group(1)) == 0 and float(w.group(1)) == 0:
            return True
        time.sleep(2)
    return False


def prompt_for(i, n_tokens):
    words = [f"p{i}x{j}" for j in range(1, n_tokens + 1)]
    return " ".join(words)


def one(base, model, prompt, max_tokens, timeout=900):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if raw.startswith(b"data: ") and b"[DONE]" not in raw:
                n += 1
    return n


def pctl(buckets, lo, hi, p):
    """Interpolate the p-quantile from cumulative histogram buckets."""
    sel = [(le, c) for le, c in buckets if lo <= le <= hi]
    if len(sel) < 3:
        return None
    base = sel[0][1]
    total = sel[-1][1] - base
    if total <= 0:
        return None
    target = base + total * p
    prev_le, prev_c = sel[0]
    for le, c in sel[1:]:
        if c >= target:
            return (prev_le + (le - prev_le) * ((target - prev_c) / max(c - prev_c, 1e-9))) * 1000.0
        prev_le, prev_c = le, c
    return sel[-1][0] * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--tag", default="ab")
    ap.add_argument("--out", default="/home/lyh/z100-vllm/.bench/ab20")
    args = ap.parse_args()

    if not wait_idle(args.base):
        print("[ab] WARN not idle; proceeding")

    for i in range(args.warmup):
        one(args.base, args.model, prompt_for(9000 + i, args.prompt_tokens), 16)
    if not wait_idle(args.base):
        print("[ab] WARN not idle after warmup")

    # Sequential by default: exactly one request in flight keeps the engine in
    # the c=1 decode regime for the whole window, and every request sees the
    # same prompt size, so the inter-token-latency histogram is a pure decode
    # distribution (a mixed arrival pattern inflates it with prefill steps).
    m0 = metrics(args.base)
    t0 = time.time()
    counts = []
    if args.parallel <= 1:
        for i in range(args.requests):
            counts.append(one(args.base, args.model,
                              prompt_for(1000 + i, args.prompt_tokens),
                              args.max_tokens))
            if (i + 1) % 6 == 0:
                print(f"[ab] {i + 1}/{args.requests} requests, "
                      f"{time.time() - t0:.0f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(one, args.base, args.model,
                              prompt_for(1000 + i, args.prompt_tokens), args.max_tokens)
                    for i in range(args.requests)]
            for f in futs:
                counts.append(f.result())
    wall = time.time() - t0
    m1 = metrics(args.base)

    gen = m1["gen_tok"] - m0["gen_tok"]
    steps = m1["itl_cnt"] - m0["itl_cnt"]
    itl = m1["itl_sum"] - m0["itl_sum"]
    acc = m1["accept"] - m0["accept"]
    dr = m1["drafts"] - m0["drafts"]
    step_mean = itl / steps * 1000 if steps else 0.0
    p50 = pctl(m1["_buckets"], 0.05, 0.60,
               (m0["_buckets"][0][1] if m0["_buckets"] else 0.0) and 0.5)
    # p50 of the *window*: quantile of the cumulative histogram shifted by m0
    shifted = []
    for (le, c) in m1["_buckets"]:
        prev = dict((l, cc) for l, cc in m0["_buckets"]).get(le)
        shifted.append((le, c - (prev if prev is not None else 0.0)))
    p50w = pctl(sorted(shifted), 0.05, 0.60, 0.5)

    res = {
        "tag": args.tag, "requests": args.requests, "warmup": args.warmup,
        "prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
        "parallel": args.parallel, "wall_s": wall,
        "out_tokens": sum(counts), "srv_gen_tokens": gen, "srv_steps": steps,
        "fwd_tok_per_s": gen / wall if wall else 0.0,
        "out_tok_per_s": sum(counts) / wall if wall else 0.0,
        "step_mean_ms": step_mean, "step_p50_ms": p50w,
        "tok_per_step": gen / steps if steps else 0.0,
        "accept_len": acc / dr if dr else 0.0,
    }
    print("[ABRESULT] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in res.items()}))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"s20abwin_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
