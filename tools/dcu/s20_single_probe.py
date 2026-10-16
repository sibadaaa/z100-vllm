#!/usr/bin/env python3
"""S20: single-request (c=1) latency probe with a server-metric cross-check.

The physical quantities for a single interactive stream are the engine **step
period** and the accepted tokens per step -- not the client's aggregate decode
window (see THROUGHPUT_REPORT.md section 5.4 for the 22% artifact that punished
that window).  This probe therefore reports both:

  * client side: TTFT, ITL p50/mean, output tokens/s for one stream;
  * server side: /metrics deltas over the request window -> step period,
    tokens/step, and forward tokens/s, all measured from the engine's own
    counters.

Usage:
  python3 s20_single_probe.py [--base http://127.0.0.1:8000] [--model qwen3.8-27b]
                              [--prompt-tokens 512] [--max-tokens 128]
                              [--repeat 3] [--tag probe] [--out DIR]
"""

import argparse
import json
import os
import re
import statistics
import time
import urllib.request

MET = {
    "gen_tok": r"^vllm:generation_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "prompt_tok": r"^vllm:prompt_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "itl_sum": r"^vllm:inter_token_latency_seconds_sum\{[^}]*\}\s+([0-9.e+]+)",
    "itl_cnt": r"^vllm:inter_token_latency_seconds_count\{[^}]*\}\s+([0-9.e+]+)",
    "iter_sum": r"^vllm:iteration_tokens_total_sum\{[^}]*\}\s+([0-9.e+]+)",
    "iter_cnt": r"^vllm:iteration_tokens_total_count\{[^}]*\}\s+([0-9.e+]+)",
    "accept": r"^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "drafts": r"^vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([0-9.e+]+)",
}
_RX = {k: re.compile(p, re.M) for k, p in MET.items()}


def metrics(base, timeout=8):
    txt = urllib.request.urlopen(f"{base}/metrics", timeout=timeout).read().decode()
    return {k: float(rx.search(txt).group(1)) for k, rx in _RX.items() if rx.search(txt)}


def build_prompt(n_tokens):
    """A deterministic prompt of roughly n_tokens tokens (vLLM counts ~1/4 word)."""
    words = []
    i = 1
    while len(words) < n_tokens:
        words.append(f"item{i}")
        i += 1
    return " ".join(words)


def stream_once(base, model, prompt, max_tokens, timeout=600):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    stamps = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                txt = ch.get("text") or ""
                if txt:
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    stamps.append(now)
    t1 = time.time()
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    return {
        "wall_s": t1 - t0,
        "ttft_ms": (ttft or 0.0) * 1000.0,
        "completion_tokens": (usage or {}).get("completion_tokens", len(stamps)),
        "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
        "chunks": len(stamps),
        "itl_mean_ms": statistics.mean(itls) * 1000 if itls else 0.0,
        "itl_p50_ms": statistics.median(itls) * 1000 if itls else 0.0,
    }


def wait_idle(base, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        txt = urllib.request.urlopen(f"{base}/metrics", timeout=8).read().decode()
        run = re.search(r'^vllm:num_requests_running\{[^}]*\}\s+([0-9.e+]+)', txt, re.M)
        wait = re.search(r'^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.e+]+)', txt, re.M)
        if float(run.group(1)) == 0 and float(wait.group(1)) == 0:
            return True
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--out", default="/home/lyh/z100-vllm/.bench")
    ap.add_argument("--no-idle-gate", action="store_true")
    args = ap.parse_args()

    if not args.no_idle_gate and not wait_idle(args.base):
        print("[probe] WARN engine not idle after 120s; proceeding")

    runs = []
    for i in range(args.repeat):
        prompt = build_prompt(args.prompt_tokens)
        m0 = metrics(args.base)
        r = stream_once(args.base, args.model, prompt, args.max_tokens)
        m1 = metrics(args.base)
        steps = m1["iter_cnt"] - m0["iter_cnt"]
        gen = m1["gen_tok"] - m0["gen_tok"]
        itl_sum = m1["itl_sum"] - m0["itl_sum"]
        acc = m1["accept"] - m0["accept"]
        dr = m1["drafts"] - m0["drafts"]
        r["srv_steps"] = steps
        r["srv_gen_tok"] = gen
        r["srv_step_ms"] = (itl_sum / steps * 1000.0) if steps else 0.0
        r["srv_tok_per_step"] = (gen / steps) if steps else 0.0
        r["srv_itl_ms"] = (itl_sum / (m1["itl_cnt"] - m0["itl_cnt"]) * 1000.0
                           if m1["itl_cnt"] != m0["itl_cnt"] else 0.0)
        r["accept_len"] = (acc / dr) if dr else 0.0
        r["fwd_tok_per_s"] = (gen / r["wall_s"]) if r["wall_s"] else 0.0
        r["out_tok_per_s"] = (r["completion_tokens"] / r["wall_s"]) if r["wall_s"] else 0.0
        runs.append(r)
        print(
            f"[run {i + 1}/{args.repeat}] ttft={r['ttft_ms']:.0f}ms "
            f"step={r['srv_step_ms']:.1f}ms tok/step={r['srv_tok_per_step']:.2f} "
            f"accept={r['accept_len']:.2f} out={r['out_tok_per_s']:.2f}t/s "
            f"fwd={r['fwd_tok_per_s']:.1f}t/s wall={r['wall_s']:.2f}s",
            flush=True,
        )

    def agg(key):
        vals = [r[key] for r in runs]
        return statistics.mean(vals) if vals else 0.0

    summary = {k: agg(k) for k in (
        "ttft_ms", "srv_step_ms", "srv_tok_per_step", "accept_len",
        "out_tok_per_s", "fwd_tok_per_s", "itl_mean_ms", "itl_p50_ms", "wall_s",
    )}
    # step p50 across runs (per-run step is already a mean over the window)
    summary["step_p50_ms"] = statistics.median([r["srv_step_ms"] for r in runs]) if runs else 0.0
    print("[SUMMARY] " + json.dumps({k: round(v, 3) for k, v in summary.items()}))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"s20_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": runs, "summary": summary}, f, indent=1)
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
