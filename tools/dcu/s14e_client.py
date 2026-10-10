#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14d: deterministic token-stream + same-context logit recorder.

One engine, one label, one JSONL.  Requests are issued strictly sequentially
(single in-flight request) in a fixed order so that request history -- which
S14b showed moves the logits -- is identical across arms.

For every prompt we record:

  * the greedy token-id stream (``temperature=0``) and its sha256,
  * per generated token: logprob, top-1/top-2 margin and top-5,
  * ``prompt_logprobs`` top-k at every PROMPT position, i.e. logits on the
    byte-identical input context.  This is the arm-to-arm instrument that does
    not suffer from autoregressive divergence.

Battery (13 prompts, fixed order): the 8-prompt S11l/L1 flip battery, then the
4-prompt S14b ``p2_*`` battery, then the S14 long greedy prompt.

Usage:
    s14d_client.py --port 8601 --label W4 --out /mnt/game/s14d/probe-W4.jsonl
"""

import argparse
import hashlib
import json
import time
import urllib.request

FLIP = (
    ("zh_hz", "用一句话介绍杭州。"),
    ("zh_ml", "解释一下什么是机器学习。"),
    ("zh_sort", "用 Python 写一个快速排序函数。"),
    ("zh_sky", "为什么天空是蓝色的？简单说一下。"),
    ("en_fr", "What is the capital of France?"),
    ("en_sea", "Write a short poem about the sea."),
    ("en_index", "Explain what a database index is."),
    ("en_math", "What is 17 times 23? Show the steps."),
)

P2 = (
    ("p2_zh_slip",
     "请用中文写一段三百字左右的说明，解释量化交易中滑点的成因、度量方法，以及降低滑点的工程手段。"),
    ("p2_en_batch",
     "Explain step by step how a modern transformer inference server schedules continuous "
     "batching, and why prefix caching helps throughput."),
    ("p2_zh_code",
     "用 Python 实现一个带超时和重试的异步 HTTP 客户端，并解释其中的错误处理与资源释放。"),
)

LONG = (
    ("s14_greedy_long",
     "请用中文逐步说明：在 31 GiB 内存的单机上，以张量并行度 TP=4 部署一个 27B 参数的"
     "INT8 量化大模型时，权重加载、KV 缓存与 CUDA graph 捕获各自会占用哪些资源，"
     "以及为什么 mmap 加载的权重计入 cgroup 的 memory.current。"),
)

BATTERY = FLIP + P2 + LONG


def query(base_url: str, model: str, prompt: str, max_tokens: int, top_k: int, plp_k: int):
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
        "prompt_logprobs": plp_k,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--plp-k", type=int, default=20)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    with open(args.out, "a", encoding="utf-8") as fh:
        for order, (pid, prompt) in enumerate(BATTERY):
            t0 = time.time()
            body = query(base, args.model, prompt, args.max_tokens, args.top_k, args.plp_k)
            choice = body["choices"][0]
            lp = choice.get("logprobs") or {}
            rows = []
            for i, entry in enumerate(lp.get("content") or []):
                top = [(t["token"], t["logprob"]) for t in entry.get("top_logprobs") or []]
                top.sort(key=lambda e: -e[1])
                gap = top[0][1] - top[1][1] if len(top) > 1 else None
                rows.append({"i": i, "token_id": entry.get("token_id"),
                             "token": entry["token"], "logprob": entry["logprob"],
                             "gap": gap, "top": top})
            token_ids = choice.get("token_ids") or []
            plp_raw = body.get("prompt_logprobs") or choice.get("prompt_logprobs") or []
            plp = []
            for entry in plp_raw:
                if not entry:
                    plp.append(None)
                    continue
                items = sorted(
                    ((int(tid), tinfo["logprob"]) for tid, tinfo in entry.items()),
                    key=lambda e: -e[1],
                )[: args.plp_k]
                plp.append(items)
            rec = {
                "label": args.label,
                "order": order,
                "prompt_id": pid,
                "prompt": prompt,
                "finish_reason": choice.get("finish_reason"),
                "token_ids": token_ids,
                "n_tokens": len(token_ids),
                "text": choice["message"].get("content") or "",
                "text_sha256": hashlib.sha256(
                    (choice["message"].get("content") or "").encode()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(
                    ",".join(str(t) for t in token_ids).encode()).hexdigest(),
                "n_prompt_pos": len(plp),
                "prompt_logprobs": plp,
                "rows": rows,
                "elapsed_s": round(time.time() - t0, 3),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[s14d-client] label={args.label} order={order} prompt={pid} "
                  f"tokens={len(token_ids)} finish={rec['finish_reason']} "
                  f"plp_pos={len(plp)} {rec['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
