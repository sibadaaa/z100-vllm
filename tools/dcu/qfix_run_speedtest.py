#!/usr/bin/env python3
"""Drive the llm_speedtest backend headlessly over /ws/test.

Saves a run JSON in the same schema as runs/*.json:
{label, config, wall_seconds, results, events}
"""
import argparse
import asyncio
import json
import time

import websockets

BACKEND_WS = "ws://127.0.0.1:18000/ws/test"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--api-url", default="http://172.17.0.2:8523/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--output-length", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=120000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = {
        "api_url": a.api_url,
        "model_name": a.model,
        "api_key": a.api_key,
        "api_type": "openai",
        "min_length": a.length,
        "max_length": a.length,
        "step": 1,
        "test_lengths": [a.length],
        "output_length": a.output_length,
        "concurrency": a.concurrency,
        "timeout": a.timeout,
        "temperature": a.temperature,
        "top_p": a.top_p,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    t0 = time.time()
    results = []
    events = []
    async with websockets.connect(BACKEND_WS, max_size=128 * 1024 * 1024) as ws:
        await ws.send(json.dumps(cfg))
        while True:
            msg = json.loads(await ws.recv())
            mtype = msg.get("type")
            summary = {
                k: v
                for k, v in msg.items()
                if k not in ("concurrent_details", "all_results")
            }
            events.append(
                {"t": round(time.time() - t0, 2), "type": mtype, "summary": summary}
            )
            if mtype == "result":
                r = msg
                dets = r.get("concurrent_details") or []
                outs = [d.get("output_tokens") or 0 for d in dets]
                print(
                    f"[result] len={r.get('prompt_length')} ok={r.get('successful')}/"
                    f"{r.get('concurrency')} prefill={r.get('avg_prefill_speed')} "
                    f"decode={r.get('avg_output_speed')} ttft={r.get('avg_ttft_ms')} "
                    f"out_tokens={outs}",
                    flush=True,
                )
            elif mtype == "complete":
                if msg.get("all_results"):
                    results = msg["all_results"]
                break
            elif mtype == "error":
                events.append({"error": msg})
                break
            else:
                print(f"[{mtype}] {msg.get('message', '')}", flush=True)

    out = {
        "label": a.label,
        "config": cfg,
        "wall_seconds": round(time.time() - t0, 3),
        "results": results,
        "events": events,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("saved", a.out)


if __name__ == "__main__":
    asyncio.run(main())
