#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14b: greedy requests over a prompt list, one JSONL record per prompt.

Extends ``s14_greedy.py`` (S14 baseline, kept frozen) in two ways that S14b
needs:

  * the prompt battery is read from ``--prompts-file`` (``id<TAB>text`` per
    line) instead of a single hard-coded prompt, so G1 can use n >= 3 prompts
    while keeping ``s14_greedy_long`` byte-identical to the S14 artifact,
  * every request can be replayed in a controlled ORDER (the file order), which
    is what the "first request on a fresh engine" control needs.

Usage:
    s14b_greedy.py --port 8022 --label A1 --out /mnt/game/s14b/greedy-A1.jsonl \
        --prompts-file /mnt/game/s14b/prompts.txt --max-tokens 256
"""

import argparse
import hashlib
import json
import urllib.request


def load_prompts(path: str):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            pid, _, text = line.partition("\t")
            out.append((pid.strip(), text))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    prompts = load_prompts(args.prompts_file)
    print(f"[s14b-greedy] label={args.label} prompts={len(prompts)} "
          f"max_tokens={args.max_tokens} order={[p[0] for p in prompts]}", flush=True)

    with open(args.out, "a", encoding="utf-8") as fh:
        for order, (pid, prompt) in enumerate(prompts):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
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
                "order": order,
                "prompt_id": pid,
                "prompt": prompt,
                "prompt_token_ids": body.get("prompt_token_ids"),
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
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[s14b-greedy] order={order} prompt={pid} tokens={len(rows)} "
                  f"finish={rec['finish_reason']} ids_sha={rec['token_ids_sha256'][:16]} "
                  f"head={text[:80]!r}", flush=True)


if __name__ == "__main__":
    main()
