#!/usr/bin/env python3
"""Minimal OpenAI-compatible smoke client (stdlib only, no curl in the image).

Polls /v1/models until the server is ready, sends one greedy completion and
prints the response.  Exit codes: 0 ok, 1 not ready, 2 request failed.
"""
import argparse
import json
import sys
import time
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8020")
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--ready-timeout", type=float, default=240.0)
    args = ap.parse_args()

    t0 = time.time()
    ready = False
    while time.time() - t0 < args.ready_timeout:
        try:
            with urllib.request.urlopen(args.base + "/v1/models", timeout=5) as r:
                r.read()
            ready = True
            break
        except Exception:
            time.sleep(2)
    if not ready:
        print(f"NOT_READY after {time.time() - t0:.0f}s", flush=True)
        return 1
    print(f"READY after {time.time() - t0:.1f}s", flush=True)

    body = json.dumps(
        {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        args.base + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.load(r)
    except Exception as exc:
        print(f"REQUEST_FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 2

    choices = resp.get("choices") or []
    text = choices[0].get("text") if choices else None
    print("COMPLETION_TEXT=" + repr(text), flush=True)
    print("COMPLETION_JSON=" + json.dumps(resp)[:2000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
