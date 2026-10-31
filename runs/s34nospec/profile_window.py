#!/usr/bin/env python3
"""One profiled decode window on the no-MTP arm.

start_profile (prefix s34nospec, max_iterations=40) -> one 96-token greedy
stream on the hot 7469-token probe prompt -> wait -> stop_profile safety net.
Writes window metadata to runs/s34nospec/window.json.
"""
import json
import os
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
RB = os.environ.get("VLLM_PROBE_OUT", "/home/lyh/z100-vllm/runs/s34nospec")
APIKEY = open("/mnt/game/host-serve/api-key").read().strip()
HDR = {"Authorization": f"Bearer {APIKEY}", "Content-Type": "application/json"}
PROMPT_IDS = json.load(open("/home/lyh/z100-vllm/runs/specbench/prompt_ids.json"))
assert len(PROMPT_IDS) == 7469


def post(path, body=None, timeout=30):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else b"",
        headers=HDR, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def stream(body, on_first=None):
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers=HDR, method="POST")
    t0 = time.time()
    t_first = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if on_first is not None and t_first is None and obj.get("choices"):
                t_first = time.time()
                on_first(t0, t_first)
            if obj.get("usage"):
                n = obj["usage"].get("completion_tokens", n)
    return n, time.time() - t0


def main():
    meta = {}
    # 8-token warmup so the profiled request hits a fully warm graph/cache path.
    n, e2e = stream({"model": "qwen3.8-27b", "prompt": PROMPT_IDS, "max_tokens": 8,
                     "temperature": 0.0, "ignore_eos": True, "stream": True,
                     "stream_options": {"include_usage": True}})
    print(f"warmup n={n} e2e={e2e:.2f}s")

    st = post("/start_profile", {"profile_prefix": "s34nospec", "max_iterations": 40})
    print(f"start_profile -> {st}")
    t_start = time.time()

    first_box = {}

    def on_first(t0, t_first):
        first_box["ttft"] = t_first - t0
        print(f"first token at +{t_first - t0:.3f}s (profiled prefill)")

    n, e2e = stream({"model": "qwen3.8-27b", "prompt": PROMPT_IDS, "max_tokens": 96,
                     "temperature": 0.0, "ignore_eos": True, "stream": True,
                     "stream_options": {"include_usage": True}}, on_first)
    meta["req"] = dict(n=n, e2e_s=round(e2e, 3), ttft_s=round(first_box.get("ttft", -1), 3))
    print(f"profiled request n={n} e2e={e2e:.2f}s ttft={first_box.get('ttft')}")

    try:
        st = post("/stop_profile")
        print(f"stop_profile -> {st}")
    except Exception as exc:  # may already have auto-stopped via max_iterations
        print(f"stop_profile -> {type(exc).__name__}: {exc}")
    meta["wall_to_stop_s"] = round(time.time() - t_start, 3)

    # traces flush async; wait for all four ranks
    deadline = time.time() + 120
    while time.time() < deadline:
        ranks = [f for f in os.listdir("/mnt/game/s30c-prof")
                 if f.startswith("s34nospec_rank") and f.endswith(".trace.json.gz")]
        if len(ranks) >= 4:
            break
        time.sleep(3)
    meta["traces"] = sorted(ranks)
    print(f"traces: {meta['traces'] or 'NONE YET'}")
    with open(os.path.join(RB, "window.json"), "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
