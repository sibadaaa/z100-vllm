#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14b: same-context logit probe via ``prompt_logprobs``.

The G1 divergence between two arms can only be *mechanically* interpreted if the
two arms' logits are compared on an IDENTICAL context.  Greedy token streams
diverge and from then on the contexts differ, so comparing generated-token
logprobs only measures "some difference exists".  ``prompt_logprobs`` returns
the top-k logprobs at every PROMPT position, i.e. on the byte-identical input,
for both arms.

That gives the A/B discriminator this step needs:

  * max |delta logprob| over the top-k on a fixed context of order 1e-3..1e-1
    nat  ->  the two compilation modes differ only by rounding (mechanism B),
  * max |delta logprob| of order 1 nat or more  ->  one of the two modes is
    computing something materially different (mechanism A).

Requests are issued strictly in file order and the first request of the first
arm on a fresh engine is recorded with ``order == 0`` so "cold vs warmed
engine" can be told apart afterwards.

Usage:
    s14b_prompt_logprobs.py --port 8022 --label A1 --out /mnt/game/s14b/plp-A1.jsonl \
        --prompts-file /mnt/game/s14b/prompts.txt --top-k 20
"""

import argparse
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
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=1)
    args = ap.parse_args()

    prompts = load_prompts(args.prompts_file)
    print(f"[s14b-plp] label={args.label} prompts={len(prompts)} top_k={args.top_k} "
          f"order={[p[0] for p in prompts]}", flush=True)

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
                "top_logprobs": 5,
                "prompt_logprobs": args.top_k,
            }
            req = urllib.request.Request(
                f"http://{args.host}:{args.port}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3600) as resp:
                body = json.load(resp)
            choice = body["choices"][0]
            # vLLM-specific: prompt_logprobs / prompt_token_ids are TOP-LEVEL
            # fields of ChatCompletionResponse (chat_completion/protocol.py:138),
            # not per-choice.
            plp = body.get("prompt_logprobs") or []
            positions = []
            for i, entry in enumerate(plp):
                if not entry:
                    positions.append({"i": i, "top": []})
                    continue
                # vLLM may leave Logprob.token unset; the dict KEY is the token id,
                # which is the only stable alignment key (also across vLLM/HF).
                top = sorted(
                    (((v.get("token") or f"tok{k}"), int(k), v["logprob"])
                     for k, v in entry.items()),
                    key=lambda e: -e[2],
                )
                gap = top[0][2] - top[1][2] if len(top) > 1 else float("inf")
                positions.append({
                    "i": i,
                    "argmax": top[0][0],
                    "argmax_id": top[0][1],
                    "gap": gap,
                    "top": [list(t) for t in top],
                })
            ptok = body.get("prompt_token_ids") or []
            rec = {
                "label": args.label,
                "order": order,
                "prompt_id": pid,
                "prompt": prompt,
                "prompt_token_ids": ptok,
                "n_prompt_tokens": len(ptok),
                "gen_token": (choice.get("token_ids") or [None])[0],
                "positions": positions,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            argmax_last = positions[-1].get("argmax") if positions else None
            print(f"[s14b-plp] order={order} prompt={pid} n_prompt={len(ptok)} "
                  f"last_argmax={argmax_last!r}", flush=True)


if __name__ == "__main__":
    main()
