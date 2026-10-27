#!/usr/bin/env python3
"""One-shot: cache the exact 7469-token probe prompt ids (same as .s33/capture.py).

Run once; later bench arms load prompt_ids.json so they need no tokenizer.
"""
import importlib.util
import json
import sys

sys.path.insert(0, "/home/lyh/z100-vllm/tools/dcu")
import s20_single_probe as s20  # noqa: E402

DOC = "/home/lyh/z100-vllm/docs/design/z100-dual-prefill-reconciliation-projection.md"

tokenizer, errors = s20.load_tokenizer("/mnt/game/models/Qwen3.8-27B-FP8", "auto")
assert tokenizer is not None, f"tokenizer load failed: {errors}"
prompt_text, tokens_actual, truncated, chars = s20.read_prompt_file(DOC, tokenizer, 7469)
ids = tokenizer.encode(prompt_text)
print(f"chars={chars} tokens_actual={tokens_actual} ids={len(ids)}")
assert len(ids) == 7469, f"token count mismatch: {len(ids)}"
with open("/home/lyh/z100-vllm/runs/specbench/prompt_ids.json", "w") as h:
    json.dump(ids, h)
print("saved runs/specbench/prompt_ids.json")
