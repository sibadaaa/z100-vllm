#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11k: greedy-token divergence probe.

`=1` and `=0` do not emit identical greedy tokens on the S11k prompts.  This
probe records the top-k logprobs around the divergence point so the difference
can be attributed: a *near tie* (tiny top-1/top-2 gap at the flip) is numeric
noise from a different int8 kernel / compiled graph, whereas a confident flip
would mean something structural.

Usage: s11k_logprob_probe.py --port 8021 --label sdot4 --max-tokens 24
"""

import argparse
import json
import urllib.request

PROMPT = "用一句话介绍杭州。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "return_token_ids": True,
        "logprobs": True,
        "top_logprobs": args.top_k,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        body = json.load(resp)
    choice = body["choices"][0]
    lp = choice.get("logprobs") or {}
    rows = []
    for entry in lp.get("content") or []:
        entries = [(t["token"], t["logprob"]) for t in entry.get("top_logprobs") or []]
        entries.sort(key=lambda e: -e[1])
        gap = entries[0][1] - entries[1][1] if len(entries) > 1 else float("inf")
        rows.append(
            {
                "token": entry["token"],
                "logprob": entry["logprob"],
                "gap_top1_top2": gap,
                "top": entries,
            }
        )
    out = {
        "label": args.label,
        "token_ids": choice.get("token_ids"),
        "finish_reason": choice.get("finish_reason"),
        "rows": rows,
    }
    print("S11K_LOGPROB " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
