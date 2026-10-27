#!/usr/bin/env python3
"""S24 long-context e2e probe: TTFT / prefill throughput / ITL vs prompt length.

Compared with .s24/probes/longctx_probe.py this version

* varies the prompt content per repetition (seeded by rep index), because the
  service runs with --enable-prefix-caching and a repeated identical prompt is
  a prefix-cache hit (the S24 plan measured 16.58 s -> 2.74 s for a repeat),
  which would silently turn a prefill A/B into a cache A/B;
* takes repeats and a tag, and writes one JSON line per (lens, rep) plus a
  p50/mean summary per lens, so the window driver can grep a verdict;
* records kv-cache usage from /metrics before each request.

Discipline (S24 plan section 2): same-window A/B, >= 4 probes per arm, report
p50 + mean, never compare across windows.

  .venv/bin/python tools/dcu/s24_longctx.py --lens 512,8192,32768 --reps 4 \
      --tag w1_l2_control --out .s24/w1/L2/<window>/control
"""

import argparse
import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("S24_BASE", "http://127.0.0.1:8000")
KEY_FILE = "/mnt/game/host-serve/api-key"


def headers():
    with open(KEY_FILE) as f:
        key = f.read().strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def quota():
    req = urllib.request.Request(BASE + "/metrics", headers=headers())
    txt = urllib.request.urlopen(req, timeout=20).read().decode()
    out = {}
    for line in txt.splitlines():
        for k in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                  "vllm:kv_cache_usage_perc"):
            if line.startswith(k + "{"):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out


def run_one(n_tokens, seed, gen, timeout):
    prompt = [((seed * 7919 + i * 104729) % 120000) + 100 for i in range(n_tokens)]
    body = json.dumps({
        "model": "qwen3.8-27b",
        "prompt": prompt,
        "max_tokens": gen,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers=headers())
    t0 = time.perf_counter()
    ttft = None
    stamps = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            try:
                ev = json.loads(line[6:])
            except Exception:
                continue
            piece = (ev.get("choices") or [{}])[0].get("text") or ""
            if piece:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                stamps.append(now)
    total = time.perf_counter() - t0
    itls = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
    return {
        "prompt_tokens": n_tokens,
        "seed": seed,
        "gen_tokens": len(stamps),
        "ttft_ms": round(ttft * 1e3, 1) if ttft else None,
        "prefill_tok_s": round(n_tokens / ttft, 1) if ttft else None,
        "itl_mean_ms": round(sum(itls) / len(itls), 1) if itls else None,
        "itl_max_ms": round(max(itls), 1) if itls else None,
        "out_tok_s": round(len(stamps) / total, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="512,8192,32768")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=917421,
                    help="base seed; rep r uses seed0 + 7919*r + lens")
    ap.add_argument("--tag", default="s24")
    ap.add_argument("--out", default=".")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    lens = [int(x) for x in args.lens.split(",")]
    os.makedirs(args.out, exist_ok=True)
    jsonl = os.path.join(args.out, "longctx.jsonl")
    print(f"# s24_longctx tag={args.tag} lens={lens} reps={args.reps} "
          f"gen={args.gen} out={jsonl}", flush=True)
    summary = []
    for n in lens:
        recs = []
        for rep in range(args.reps):
            seed = args.seed0 + 7919 * rep + n
            qb = quota()
            try:
                rec = run_one(n, seed, args.gen, args.timeout)
            except Exception as exc:  # noqa: BLE001
                rec = {"prompt_tokens": n, "seed": seed, "error": repr(exc)}
            rec["rep"] = rep
            rec["quota_before"] = qb
            rec["tag"] = args.tag
            print(json.dumps(rec), flush=True)
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if "ttft_ms" in rec:
                recs.append(rec)
            time.sleep(1.0)
        if recs:
            s = {
                "tag": args.tag,
                "prompt_tokens": n,
                "reps": len(recs),
                "ttft_p50_ms": round(statistics.median(r["ttft_ms"] for r in recs), 1),
                "ttft_mean_ms": round(statistics.fmean(r["ttft_ms"] for r in recs), 1),
                "prefill_p50_tok_s": round(statistics.median(r["prefill_tok_s"] for r in recs), 1),
                "itl_p50_ms": round(statistics.median(r["itl_mean_ms"] for r in recs), 1),
                "itl_mean_ms": round(statistics.fmean(r["itl_mean_ms"] for r in recs), 1),
            }
            summary.append(s)
            print("[SUMMARY] " + json.dumps(s), flush=True)
    if summary:
        with open(os.path.join(args.out, "longctx-summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
