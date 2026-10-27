#!/usr/bin/env python3
"""S20: measure speculative-decoding acceptance **by draft position**.

Why: at c=1 the per-step cost is dominated by fixed terms (FP8 GEMM + NCCL +
launch), and the verify batch is M = k+1 rows.  Raising k costs almost nothing
per extra row -- the cost is M-flat in the tiled GEMM (which takes over at
M > 8) -- while each accepted extra position is a whole token.  So the right
question is not "how many tokens/s" but "how many accepted tokens per position
as a function of position": the step should stop at the k where marginal
acceptance falls below the marginal cost.

This probe reads the engine's own per-position acceptance histogram from
`/metrics` (vllm:spec_decode_num_accepted_tokens_per_pos) or, if absent,
reconstructs it from a batch of greedy requests by counting how many tokens
each request produced per engine step.

  python3 s20_accept_profile.py --base http://127.0.0.1:8000 --max-tokens 256
"""

import argparse
import json
import re
import statistics
import time
import urllib.request


def metrics(base, timeout=8):
    return urllib.request.urlopen(f"{base}/metrics", timeout=timeout).read().decode()


def per_pos(base):
    """Return the engine's per-position acceptance counters, if exposed."""
    txt = metrics(base)
    out = {}
    rx = re.compile(
        r"^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position=\"(\d+)\"[^}]*\}\s+([0-9.e+]+)",
        re.M,
    )
    for m in rx.finditer(txt):
        out[int(m.group(1))] = float(m.group(2))
    drafts = None
    m = re.search(r"^vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([0-9.e+]+)", txt, re.M)
    if m:
        drafts = float(m.group(1))
    acc = None
    m = re.search(
        r"^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([0-9.e+]+)", txt, re.M
    )
    if m:
        acc = float(m.group(1))
    return out, drafts, acc


def build_prompt(n, kind):
    if kind == "repeat":
        return " ".join(f"item{i}" for i in range(1, n + 1))
    # A prompt with real textual structure: instructions + a list to continue.
    head = (
        "You are a careful assistant. Write the answer as a numbered list, one "
        "item per line, and do not stop until you have written 200 items. "
        "Begin now.\n1. The first point about distributed systems is that "
    )
    words = head.split() + [f"detail{i}" for i in range(1, max(1, n - 40))]
    return " ".join(words)


def stream_once(base, model, prompt, max_tokens, timeout=900):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    n = 0
    ttft = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "ignore").strip()
            if not s.startswith("data: "):
                continue
            if s[6:] == "[DONE]":
                break
            obj = json.loads(s[6:])
            for ch in obj.get("choices") or []:
                if ch.get("text"):
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
    return n, (ttft or 0) * 1000, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--prompt-kind", default="structured", choices=["structured", "repeat"])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    per0, d0, a0 = per_pos(args.base)
    if per0:
        print(f"[metrics] per-position acceptance counters present: "
              f"{len(per0)} positions, drafts={d0}")
    else:
        print("[metrics] no per-position counters; falling back to aggregate only")

    runs = []
    for i in range(args.repeat):
        prompt = build_prompt(args.prompt_tokens, args.prompt_kind)
        m0 = metrics(args.base)
        n, ttft, wall = stream_once(args.base, args.model, prompt, args.max_tokens)
        m1 = metrics(args.base)
        c0 = float(re.search(r"^vllm:gen", "").group(0)) if False else None
        itl = re.search(r"^vllm:inter_token_latency_seconds_sum\{[^}]*\}\s+([0-9.e+]+)", m1, re.M)
        cnt = re.search(r"^vllm:inter_token_latency_seconds_count\{[^}]*\}\s+([0-9.e+]+)", m1, re.M)
        itl0 = re.search(r"^vllm:inter_token_latency_seconds_sum\{[^}]*\}\s+([0-9.e+]+)", m0, re.M)
        cnt0 = re.search(r"^vllm:inter_token_latency_seconds_count\{[^}]*\}\s+([0-9.e+]+)", m0, re.M)
        steps = float(cnt.group(1)) - float(cnt0.group(1)) if cnt and cnt0 else 0.0
        itl_s = float(itl.group(1)) - float(itl0.group(1)) if itl and itl0 else 0.0
        step_ms = itl_s / steps * 1000 if steps else 0.0
        tok_step = n / steps if steps else 0.0
        accept = tok_step - 1.0
        runs.append({"chunks": n, "ttft_ms": ttft, "wall_s": wall,
                     "srv_steps": steps, "step_ms": step_ms,
                     "tok_per_step": tok_step, "accept_est": accept,
                     "out_tps": n / wall if wall else 0.0})
        print(f"[run {i+1}/{args.repeat}] chunks={n} step={step_ms:.1f}ms "
              f"tok/step={tok_step:.2f} (accept~{accept:.2f}) "
              f"out={n/wall:.2f}t/s ttft={ttft:.0f}ms", flush=True)

    if per0:
        per1, d1, a1 = per_pos(args.base)
        print("\n=== per-position acceptance over this window ===")
        print(f"{'pos':>4}{'accepted':>12}{'rate':>9}")
        for p in sorted(per1):
            d = per1[p] - per0.get(p, 0.0)
            dd = (d1 - d0) if (d1 is not None and d0 is not None) else 0.0
            rate = d / dd if dd else 0.0
            print(f"{p:>4}{d:>12.0f}{rate:>9.3f}")
        if a1 is not None and a0 is not None:
            tot = a1 - a0
            print(f"total accepted tokens in window: {tot:.0f}")

    def agg(k):
        v = [r[k] for r in runs]
        return statistics.mean(v) if v else 0.0
    print("\n[SUMMARY] " + json.dumps({k: round(agg(k), 3) for k in
          ("step_ms", "tok_per_step", "accept_est", "out_tps", "ttft_ms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
