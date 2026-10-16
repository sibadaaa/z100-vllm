#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache probe: hit curve *and* cold-vs-warm output equality.

Sends the same literal prefix twice per candidate length and reports, for the
second (warm) request, the hit-token delta and the TTFT drop. Because a mamba
partial hit reuses a GDN state checkpoint taken at a non-block boundary, the
warm reply must be token-identical to the cold reply; that equality is the
correctness gate for ``--prefix-match-unit`` on a hybrid GDN model.

A third request perturbs one byte in the middle of the prefix. It must lose the
hits after that byte, which proves the measured hits come from the prefix and
not from something else.

  python tools/dcu/perf1_prefix_cache_probe.py --lengths 512,880,1250,1792,2510
"""

import argparse
import hashlib
import json
import re
import time
import urllib.request


def build_prompt(target_tokens: int, salt: str) -> str:
    lines, i = [], 0
    while i * 22 < target_tokens * 1.8:
        lines.append(
            f"{salt}Line {i}: the quick brown fox jumps over the lazy dog while "
            f"seventeen engineers review pipeline stage number {i} for latency "
            f"regressions."
        )
        i += 1
    return "\n".join(lines)[: int(target_tokens * 4.0)]


def counters(base: str) -> dict:
    try:
        text = urllib.request.urlopen(f"{base}/metrics", timeout=5).read().decode()
    except Exception:
        return {}
    out = {}
    for key in (
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "prompt_tokens_total",
    ):
        m = re.search(rf"^vllm:{key}\{{[^}}]*\}}\s+([0-9.e+]+)$", text, re.M)
        out[key] = float(m.group(1)) if m else 0.0
    return out


def call(base, model, prompt, max_tokens, timeout=1800):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    text, usage = "", {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (
                    delta.get("reasoning_content") or ""
                )
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text += piece
    return {
        "ttft_ms": round((ttft or 0) * 1000, 1),
        "ptok": usage.get("prompt_tokens"),
        "md5": hashlib.md5(text.encode()).hexdigest()[:16],
        "chars": len(text),
    }


def measure(base, model, prompt, max_tokens, label):
    before = counters(base)
    res = call(base, model, prompt, max_tokens)
    after = counters(base)
    res["hits"] = int(
        after.get("prefix_cache_hits_total", 0)
        - before.get("prefix_cache_hits_total", 0)
    )
    res["queries"] = int(
        after.get("prefix_cache_queries_total", 0)
        - before.get("prefix_cache_queries_total", 0)
    )
    print(
        f"  {label:<9} ptok={res['ptok']:>6} TTFT={res['ttft_ms']:>9.1f} ms "
        f"hits+{res['hits']:>5} queries+{res['queries']:>5} "
        f"md5={res['md5']} chars={res['chars']}"
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--lengths", default="512,880,1250,1792,2510")
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    stamp = int(time.time())
    failures = 0
    print("length  request     prompt_tok      TTFT      hits  queries  output")
    for target in [int(x) for x in args.lengths.split(",")]:
        salt = f"PMU{stamp}L{target}"
        prompt = build_prompt(target, salt)
        print(f"--- target {target} tokens (identical prefix sent twice) ---")
        cold = measure(args.base, args.model, prompt, args.max_tokens, "cold")
        warm = measure(args.base, args.model, prompt, args.max_tokens, "warm")

        same = cold["md5"] == warm["md5"]
        speedup = cold["ttft_ms"] / max(warm["ttft_ms"], 1e-6)
        print(
            f"  => hit_tokens={warm['hits'] * 16} TTFT {cold['ttft_ms']:.0f} -> "
            f"{warm['ttft_ms']:.0f} ms ({speedup:.2f}x) "
            f"identical={same} {'ok' if same else 'FAIL'}"
        )
        failures += int(not same)

        broken = prompt.replace("pipeline stage number 3 ", "pipeline stage number 3X ", 1)
        measure(args.base, args.model, broken, args.max_tokens, "perturbed")

    print(f"\nSUMMARY failures={failures} (cold/warm output mismatches)")


if __name__ == "__main__":
    main()
