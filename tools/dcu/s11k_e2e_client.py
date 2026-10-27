#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11k: minimal greedy decode client for the 27B W8A8 A/B runs.

Sends the same two prompts as S11e (`chat_zh` / `chat_en`) with
``max_tokens=129``, ``temperature=0`` and ``stream=True``, so that the
per-token latency can be measured from the arrival times of the chunks and the
generated text/token ids can be compared bit-for-bit between the
``VLLM_DCU_USE_INT8_SDOT4=0`` and ``=1`` arms.

Prints one JSON object per prompt:

    {"label": ..., "tokens": N, "ttft_ms": ..., "decode_ms_per_token": ...,
     "text": ..., "text_sha256": ..., "token_ids": [...]}

Only stdlib is used (urllib) so the container needs no extra packages.
"""

import argparse
import hashlib
import json
import time
import urllib.request

PROMPTS = (
    ("chat_zh", "用一句话介绍杭州。"),
    ("chat_en", "What is the capital of France?"),
)
MAX_TOKENS = 129


def _post_stream(base_url: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    marks: list[float] = []
    text_parts: list[str] = []
    token_ids: list[int] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    text_parts.append(piece)
                ids = choice.get("token_ids")
                if ids:
                    token_ids.extend(ids)
                if piece or ids:
                    marks.append(time.perf_counter())
    return t0, marks, "".join(text_parts), token_ids


def run_prompt(base_url: str, model: str, label: str, prompt: str, timeout: float):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "return_token_ids": True,
    }
    t0, marks, text, token_ids = _post_stream(base_url, payload, timeout)
    t_end = time.perf_counter()
    if not marks:
        raise RuntimeError(f"{label}: no chunks received")
    ttft_ms = (marks[0] - t0) * 1e3
    # Decode window = first chunk -> last chunk; tokens emitted in that window.
    decode_ms = (t_end - marks[0]) * 1e3
    return {
        "label": label,
        "prompt": prompt,
        "tokens": len(token_ids) if token_ids else None,
        "chunks": len(marks),
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "decode_ms_per_token": decode_ms / max(len(marks) - 1, 1),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids": token_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"
    for label, prompt in PROMPTS:
        out = run_prompt(base, args.model, label, prompt, args.timeout)
        line = json.dumps(out, ensure_ascii=False)
        print(line[:400] + ("..." if len(line) > 400 else ""), flush=True)
        print("S11K_RESULT " + line, flush=True)


if __name__ == "__main__":
    main()
