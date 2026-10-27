#!/usr/bin/env python3
"""S20: keep exactly one streaming request in flight, continuously.

Used to drive the engine while a torch profiler capture window is open: the
bounded profiler schedule only advances on engine iterations, so a load has to
be present from the moment the service is ready.  One request at a time keeps
the engine in the c=1 regime (the state the user cares about).

  python3 s20_c1_load.py --base http://127.0.0.1:8000 --duration 300
"""

import argparse
import json
import time
import urllib.request


def build_prompt(n):
    return " ".join(f"item{i}" for i in range(1, n + 1))


def one(base, model, prompt, max_tokens, timeout=900):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if raw.startswith(b"data: ") and b"[DONE]" not in raw:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()

    t_end = time.time() + args.duration
    prompt = build_prompt(args.prompt_tokens)
    rounds = 0
    t0 = time.time()
    while time.time() < t_end:
        try:
            n = one(args.base, args.model, prompt, args.max_tokens)
            rounds += 1
            if rounds % 5 == 0:
                print(f"[load] rounds={rounds} chunks={n} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[load] error {type(e).__name__}: {e}", flush=True)
            time.sleep(2)
    print(f"[load] done rounds={rounds} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
