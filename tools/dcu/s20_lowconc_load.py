#!/usr/bin/env python3
"""S20: sustained c<=4 load for a profiler window.

Keeps exactly `--concurrency` requests in flight (each worker issues one request,
waits for it, then issues the next) so the engine stays in the target regime for
the whole capture.  Used to profile the c=4 prefill/decode interference.

  python3 s20_lowconc_load.py --concurrency 4 --duration 300
"""

import argparse
import json
import threading
import time
import urllib.request


def one(base, wid, i, pt, mt):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "prompt": " ".join(f"L{wid}r{i}w{j}" for j in range(pt)),
        "max_tokens": mt, "temperature": 0.0, "ignore_eos": True, "stream": True,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    stop = time.time() + args.duration
    counts = [0] * args.concurrency

    def worker(wid):
        i = 0
        while time.time() < stop:
            try:
                counts[wid] += one(args.base, wid, i, args.prompt_tokens, args.max_tokens)
                i += 1
            except Exception as e:  # noqa: BLE001
                print(f"[load w{wid}] {type(e).__name__}: {e}", flush=True)
                time.sleep(2)

    ths = [threading.Thread(target=worker, args=(w,), daemon=True)
           for w in range(args.concurrency)]
    t0 = time.time()
    for t in ths:
        t.start()
    while time.time() < stop:
        time.sleep(15)
        print(f"[load] t={time.time() - t0:.0f}s chunks={counts}", flush=True)
    for t in ths:
        t.join(timeout=30)
    print(f"[load] done {time.time() - t0:.0f}s chunks={counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
