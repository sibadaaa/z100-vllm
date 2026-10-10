#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14: one fixed long greedy request, recorded for the TP=2 vs TP=4 study.

Unlike ``s11l_flip_stats.py`` (a battery of 8 short prompts), this sends a
single fixed prompt with a large ``max_tokens`` so the G1 acceptance item
"one complete greedy request" has an unambiguous artifact: the full text, the
full token-id list and per-token logprobs / top-1-top-2 margins, one JSON
object per line.

Usage:
    s14_greedy.py --port 8022 --label p4 --out /mnt/game/s14/greedy-p4.jsonl
"""

import argparse
import hashlib
import json
import urllib.request

PROMPT = (
    "请用中文逐步说明：在 31 GiB 内存的单机上，以张量并行度 TP=4 部署一个 27B 参数的"
    "INT8 量化大模型时，权重加载、KV 缓存与 CUDA graph 捕获各自会占用哪些资源，"
    "以及为什么 mmap 加载的权重计入 cgroup 的 memory.current。"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    payload = {
        "model": args.model,
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
        f"http://{args.host}:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = json.load(resp)

    choice = body["choices"][0]
    lp = choice.get("logprobs") or {}
    rows = []
    for i, entry in enumerate(lp.get("content") or []):
        top = [(t["token"], t["logprob"]) for t in entry.get("top_logprobs") or []]
        top.sort(key=lambda e: -e[1])
        gap = top[0][1] - top[1][1] if len(top) > 1 else float("inf")
        rows.append({"i": i, "token": entry["token"], "logprob": entry["logprob"],
                     "gap": gap, "top": top})
    token_ids = choice.get("token_ids") or []
    text = choice["message"].get("content") or ""
    rec = {
        "label": args.label,
        "prompt_id": "s14_greedy_long",
        "prompt": PROMPT,
        "finish_reason": choice.get("finish_reason"),
        "token_ids": token_ids,
        "text": text,
        "usage": body.get("usage"),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(
            ",".join(str(t) for t in token_ids).encode()
        ).hexdigest(),
        "rows": rows,
    }
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[s14-greedy] label={args.label} tokens={len(rows)} finish={rec['finish_reason']} "
          f"usage={rec['usage']}")
    print(f"[s14-greedy] text_sha256={rec['text_sha256']} token_ids_sha256={rec['token_ids_sha256']}")
    print(f"[s14-greedy] text_head={text[:200]!r}")


if __name__ == "__main__":
    main()
