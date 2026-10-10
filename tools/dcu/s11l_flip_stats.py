#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11l: per-token logprob recorder for the greedy-flip boundary study.

Sends a fixed battery of Chinese and English prompts with ``temperature=0``,
``logprobs=true`` and ``top_logprobs=5``, and writes one JSON object per prompt
to a JSONL file.  Every row carries the chosen token id, its logprob and its
top-1/top-2 margin, so two arms (e.g. ``=0`` vs ``=1``) can be compared offline
by ``s11l_flip_compare.py`` to locate every greedy flip and the margin at which
it happened.

Usage:
    s11l_flip_stats.py --port 8022 --label ctl0 --out /mnt/game/s11l/ctl0.jsonl
"""

import argparse
import json
import urllib.request

PROMPTS = (
    ("zh_hz", "用一句话介绍杭州。"),
    ("zh_ml", "解释一下什么是机器学习。"),
    ("zh_sort", "用 Python 写一个快速排序函数。"),
    ("zh_sky", "为什么天空是蓝色的？简单说一下。"),
    ("en_fr", "What is the capital of France?"),
    ("en_sea", "Write a short poem about the sea."),
    ("en_index", "Explain what a database index is."),
    ("en_math", "What is 17 times 23? Show the steps."),
)


def query(base_url: str, model: str, prompt: str, max_tokens: int, top_k: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "return_token_ids": True,
        "logprobs": True,
        "top_logprobs": top_k,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    with open(args.out, "a", encoding="utf-8") as fh:
        for pid, prompt in PROMPTS:
            body = query(base, args.model, prompt, args.max_tokens, args.top_k)
            choice = body["choices"][0]
            lp = choice.get("logprobs") or {}
            rows = []
            for i, entry in enumerate(lp.get("content") or []):
                top = [(t["token"], t["logprob"]) for t in entry.get("top_logprobs") or []]
                top.sort(key=lambda e: -e[1])
                gap = top[0][1] - top[1][1] if len(top) > 1 else float("inf")
                rows.append(
                    {
                        "i": i,
                        "token": entry["token"],
                        "logprob": entry["logprob"],
                        "gap": gap,
                        "top": top,
                    }
                )
            rec = {
                "label": args.label,
                "prompt_id": pid,
                "prompt": prompt,
                "finish_reason": choice.get("finish_reason"),
                "token_ids": choice.get("token_ids"),
                "rows": rows,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"[s11l-flip] label={args.label} prompt={pid} "
                f"tokens={len(rows)} finish={rec['finish_reason']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
