#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11l: one ~7k-token chat request against the 27B arm (8K regression).

Builds a long prompt with the model's own tokenizer, posts a single greedy chat
completion and reports HTTP status, usage and the first characters of the text.
Combined with the startup log (which must say ``TRITON_ATTN``) this is the S9 A4
8K evidence, re-run under ``VLLM_DCU_USE_INT8_SDOT4=1``.

Usage: s11l_8k_probe.py --port 8021 [--target-tokens 7500]
"""

import argparse
import json
import time
import urllib.request

UNIT = "The quick brown fox jumps over the lazy dog near the river bank. "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--target-tokens", type=int, default=7500)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=1200.0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(
            "/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8", trust_remote_code=True
        )
        which = "27B"
    except Exception as exc:  # noqa: BLE001
        print(f"[8k] 27B tokenizer unavailable ({exc}); falling back to Qwen3-0.6B")
        tok = AutoTokenizer.from_pretrained("/mnt/game/models/Qwen3-0.6B")
        which = "0.6B"
    ids = tok.encode(UNIT)
    text = tok.decode(
        (ids * (args.target_tokens // len(ids) + 1))[: args.target_tokens]
    )
    n_prompt = len(tok.encode(text))
    print(f"[8k] tokenizer={which} client_prompt_tokens={n_prompt}")

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": text + "\n\nReply with one word."}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        status = resp.status
        body = json.load(resp)
    dt = time.time() - t0
    usage = body.get("usage", {})
    text_out = (body["choices"][0].get("message") or {}).get("content") or ""
    print(f"[8k] HTTP {status} elapsed={dt:.2f}s usage={json.dumps(usage)}")
    print(f"[8k] text={json.dumps(text_out[:200], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
