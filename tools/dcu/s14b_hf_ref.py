#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14b: CPU HF-transformers truth reference for the graph-vs-eager study.

Runs the SAME prompt battery as ``s14b_prompt_logprobs.py`` on CPU with plain
HF transformers in fp32 and writes the same JSONL shape, so
``s14b_compare_promptlp.py`` can compare a vLLM arm against the reference
directly.

Why this is the truth anchor: the two vLLM arms share the model, the tokenizer
and the int8 weights; only the *execution* changes.  A CPU fp32 forward through
the reference implementation has none of the fork's gfx906 kernels, no
inductor, no cudagraph, no tensor parallelism and no fp16 accumulate, so it
answers "which side is closer to the mathematical model".

Prompt token ids are cross-checked against the vLLM artifact by the caller: the
comparison is only meaningful if the two sides tokenised identically.

Usage:
    s14b_hf_ref.py --model /mnt/game/models/Qwen3.5-4B \
        --prompts-file /mnt/game/s14b/prompts.txt --out /mnt/game/s14b/plp-HF4B.jsonl \
        --top-k 20 --dtype float32
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="HF")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--ids-from", default=None,
                    help="JSONL from s14b_prompt_logprobs.py; use ITS prompt_token_ids "
                         "verbatim instead of re-templating, so a template mismatch "
                         "between HF and vLLM cannot masquerade as a numerical diff")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.eval()
    model.to(args.device)
    print(f"[hf-ref] loaded {args.model} dtype={args.dtype} in {time.time() - t0:.1f}s "
          f"params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B", flush=True)

    prompts = load_prompts(args.prompts_file)
    forced = {}
    if args.ids_from:
        with open(args.ids_from, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    forced[rec["prompt_id"]] = rec["prompt_token_ids"]
        print(f"[hf-ref] forced prompt_token_ids for {sorted(forced)}", flush=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for order, (pid, prompt) in enumerate(prompts):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if pid in forced:
                ids = torch.tensor([forced[pid]], dtype=torch.long, device=args.device)
            else:
                ids = tok(text, return_tensors="pt")["input_ids"].to(args.device)
            t1 = time.time()
            with torch.no_grad():
                out = model(ids)
            logits = out.logits[0].float()          # [seq, vocab]; logits[i] predicts token i+1
            logprobs = torch.log_softmax(logits, dim=-1)
            seq = ids.shape[1]
            topk = torch.topk(logprobs, args.top_k, dim=-1)

            def top_list(row_vals, row_idx):
                return [[tok.decode([int(j)]), int(j), float(v)]
                        for v, j in zip(row_vals, row_idx)]

            # vLLM's prompt_logprobs[i] is the logprob OF prompt token i, i.e. it comes
            # from logits[i-1]; position 0 has no entry.  Reproduce that convention so
            # an HF-vs-vLLM comparison is aligned without any shift.
            positions = [{"i": 0, "top": []}]
            for i in range(1, seq):
                top = top_list(topk.values[i - 1], topk.indices[i - 1])
                positions.append({
                    "i": i,
                    "argmax": top[0][0],
                    "argmax_id": top[0][1],
                    "gap": (top[0][2] - top[1][2]) if len(top) > 1 else float("inf"),
                    "top": top,
                })
            # The distribution that decides the FIRST generated token is logits[seq-1];
            # vLLM's prompt_logprobs does not contain it, so record it separately.
            next_top = top_list(topk.values[seq - 1], topk.indices[seq - 1])
            rec = {
                "label": args.label,
                "order": order,
                "prompt_id": pid,
                "prompt": prompt,
                "prompt_token_ids": ids[0].tolist(),
                "n_prompt_tokens": int(ids.shape[1]),
                "gen_token": None,
                "positions": positions,
                "next_token_top": next_top,
                "next_argmax": next_top[0][0],
                "hf_seconds": round(time.time() - t1, 2),
                "hf_dtype": args.dtype,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[hf-ref] order={order} prompt={pid} n_prompt={ids.shape[1]} "
                  f"last_prompt_argmax={positions[-1]['argmax']!r} "
                  f"next_argmax={next_top[0][0]!r} next_gap={next_top[0][2] - next_top[1][2]:.3f} "
                  f"({rec['hf_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
