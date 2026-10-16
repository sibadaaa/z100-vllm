#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18: measure MTP speculative-decoding acceptance from a live arm.
#
#   python3 /src/tools/dcu/s18_mtp_probe.py --port 8001 --label fp8-tp4-k2 \
#       [--mode greedy|prod] [--max-tokens 192] [--outdir /mnt/game/s18]
#
# Reads the vllm:spec_decode_* counters before/after a fixed prompt battery and
# reports: draft tokens, accepted tokens, accept rate (= accepted/draft_tokens),
# mean acceptance length (= 1 + accepted/drafts) and the per-position rates.
# `--mode prod` uses the production sampling config
# (temperature=0.7, repetition_penalty=1.05) because acceptance is
# penalty-sensitive (fork commit 5749599db: 84% -> 5% at rep=1.2).
#
# Stdlib only. Run it inside the container (DTK images need `import torch`
# before anything vllm-adjacent -- hard constraint 8).
try:  # noqa: SIM105 - probe should still work on a plain host python3
    import torch  # noqa: F401
except Exception:
    pass

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

PROMPTS = [
    "用三句话解释什么是张量并行（tensor parallelism），以及它为什么能降低单 token 延迟。",
    "Write a Python function that merges two sorted lists in O(n) time, then explain the invariant.",
    "简述海光 Z100 (gfx906) 与 MI50 在指令集上的异同，重点说明 MFMA 与 dot 指令的差别。",
    "Give a step-by-step plan to debug a CUDA (HIP) illegal memory access in a custom kernel.",
    "解释一下 vLLM 的前缀缓存（prefix caching）在混合线性注意力模型上为什么会变复杂。",
]


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def read_metrics(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        text = r.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("vllm:spec_decode"):
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return out


def total(metrics: dict[str, float], family: str) -> float:
    """Sum a metric family, tolerating both `name` and `name_total` suffixes."""
    s = 0.0
    for k, v in metrics.items():
        name = k.split("{")[0]
        if name in (family, family + "_total"):
            s += v
    return s


def per_pos(metrics: dict[str, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for k, v in metrics.items():
        if "accepted_tokens_per_pos" not in k:
            continue
        m = re.search(r'position="(\d+)"', k)
        if m:
            out[int(m.group(1))] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--mode", choices=["greedy", "prod"], default="greedy")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--outdir", default="/mnt/game/s18")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    os.makedirs(args.outdir, exist_ok=True)

    gen = {"temperature": 0.0} if args.mode == "greedy" else {
        "temperature": 0.7,
        "repetition_penalty": 1.05,
    }

    before = read_metrics(args.port)
    t0 = time.time()
    runs = []
    for rep in range(args.reps):
        for i, prompt in enumerate(PROMPTS):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                **gen,
            }
            t1 = time.time()
            resp = http_json(f"{base}/v1/chat/completions", payload)
            dt = time.time() - t1
            usage = resp.get("usage", {})
            runs.append(
                {
                    "rep": rep,
                    "prompt_idx": i,
                    "seconds": round(dt, 3),
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                }
            )
            print(
                f"[probe] rep={rep} p{i} {dt:6.2f}s "
                f"out={usage.get('completion_tokens')} "
                f"tok/s={((usage.get('completion_tokens') or 0) / dt):5.2f}"
            )
    wall = time.time() - t0
    after = read_metrics(args.port)

    drafts = total(after, "vllm:spec_decode_num_drafts") - total(
        before, "vllm:spec_decode_num_drafts"
    )
    draft_tokens = total(after, "vllm:spec_decode_num_draft_tokens") - total(
        before, "vllm:spec_decode_num_draft_tokens"
    )
    accepted = total(after, "vllm:spec_decode_num_accepted_tokens") - total(
        before, "vllm:spec_decode_num_accepted_tokens"
    )
    pp_before, pp_after = per_pos(before), per_pos(after)
    positions = sorted(set(pp_before) | set(pp_after))
    pp = {p: pp_after.get(p, 0.0) - pp_before.get(p, 0.0) for p in positions}

    out_tokens = sum(r["completion_tokens"] or 0 for r in runs)
    result = {
        "label": args.label,
        "mode": args.mode,
        "max_tokens": args.max_tokens,
        "reps": args.reps,
        "prompts": len(PROMPTS),
        "wall_seconds": round(wall, 3),
        "completion_tokens": out_tokens,
        "tok_per_s": round(out_tokens / wall, 4) if wall > 0 else None,
        "spec_metrics_present": bool(after),
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "accept_rate": (accepted / draft_tokens) if draft_tokens else None,
        "mean_acceptance_length": (1 + accepted / drafts) if drafts else None,
        "accepted_per_pos_total": {str(k): v for k, v in pp.items()},
        "accepted_per_pos_rate": (
            {str(k): (v / drafts) for k, v in pp.items()} if drafts else None
        ),
        "runs": runs,
    }
    path = os.path.join(args.outdir, f"probe-{args.label}-{args.mode}.json")
    with open(path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print("\n=== S18 MTP acceptance ===")
    if not after:
        print("no vllm:spec_decode_* metrics on this arm -> MTP is OFF (control arm)")
    else:
        print(f"drafts={drafts:.0f} draft_tokens={draft_tokens:.0f} "
              f"accepted={accepted:.0f}")
        ar, mal = result["accept_rate"], result["mean_acceptance_length"]
        print(
            "accept_rate=" + (f"{ar:.4f}" if ar is not None else "n/a")
            + " mean_acceptance_length=" + (f"{mal:.3f}" if mal is not None else "n/a")
        )
        if result["accepted_per_pos_rate"]:
            print("per-position rate: " + ", ".join(
                f"p{k}={v:.3f}" for k, v in result["accepted_per_pos_rate"].items()))
    print(f"tok/s={result['tok_per_s']} wall={wall:.1f}s out_tokens={out_tokens}")
    print(f"[probe] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
