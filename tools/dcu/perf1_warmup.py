#!/usr/bin/env python3
"""perf1: cold-start warmup (and first-request penalty measurement) for the
Qwen3.8-27B-FP8 TP=4 service.

Why this exists: the recipe runs `--enforce-eager`, and
`vllm/config/vllm.py:1793-1803` turns that into
`-cc.mode=none -cc.cudagraph_mode=none --kernel_config.enable_jit_warmup=False`
i.e. **no built-in JIT warmup whatsoever**.  Every kernel/Triton shape that the
first real request touches is compiled/loaded on the critical path of that
request; the S-perf1 baseline measured the first 8k prefill at 279 t/s vs 1039
t/s steady (3.7x, PERF_REPORT.md section 2).

Warm each shape class once with a synthetic request whose output is discarded:

    python3 tools/dcu/perf1_warmup.py --url http://127.0.0.1:8000 \
        --model qwen3.8-27b [--lens 2048,512,128] [--measure]

--measure runs the 2048-token request twice and reports cold/warm TTFT so the
penalty can be quantified (only meaningful when run right after a restart).
"""

import argparse
import json
import time
import urllib.request

DEFAULT_LENS = [2048, 512, 128]


def post(url, payload, timeout=900):
    req = urllib.request.Request(
        url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return time.time() - t0, body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--lens", default=",".join(map(str, DEFAULT_LENS)))
    ap.add_argument("--measure", action="store_true")
    args = ap.parse_args()

    lens = [int(x) for x in args.lens.split(",") if x]

    def run(n_tok, out_tok):
        payload = {
            "model": args.model,
            "prompt": list(range(100, 100 + n_tok)),  # token-id prompt, exact length
            "max_tokens": out_tok,
            "temperature": 0.0,
            "ignore_eos": True,
            "seed": 0,
        }
        dt, body = post(args.url, payload)
        usage = body.get("usage", {})
        return dt, usage

    if args.measure:
        for label in ("cold", "warm"):
            dt, usage = run(2048, 1)
            pt = usage.get("prompt_tokens", 2048)
            print(
                f"[warmup] {label}: prefill {pt} tok in {dt:.3f} s "
                f"= {pt / dt:.1f} t/s",
                flush=True,
            )

    for n in lens:
        dt, usage = run(n, 8)
        pt = usage.get("prompt_tokens", n)
        print(
            f"[warmup] len={n:5d} out=8: {dt:.3f} s ({pt / dt:.1f} prefill t/s)",
            flush=True,
        )
    print("[warmup] DONE - discard these outputs, the service is now warm", flush=True)


if __name__ == "__main__":
    main()
