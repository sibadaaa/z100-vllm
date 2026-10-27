#!/usr/bin/env python3
"""Greedy token-for-token parity between two S23 arms.

Plan §1 condition (1): eight greedy prompts must reproduce the control arm's
token stream exactly.  The harness records prompt token ids, generated token ids,
per-token logprobs and the decoded text, so a divergence is reported at the
first differing index instead of as "the text looks different".

Capture the control stream once per window (never reuse a baseline captured under
a different window/build -- the plan forbids cross-window claims):

  .venv/bin/python tools/dcu/s23_parity.py --mode capture \
      --out .s23/w1/O6/w1a/parity/control.json

Then check an arm against it:

  .venv/bin/python tools/dcu/s23_parity.py --mode check \
      --baseline .s23/w1/O6/w1a/parity/control.json \
      --out .s23/w1/O6/w1a/parity/o1.json

Exit status: 0 = byte-identical token streams, 1 = divergence, 2 = harness error.
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s23_common  # noqa: E402

# Eight prompts that between them cover the shapes the fp8 emu path takes in
# production: short decode (M small, streaming path), 512/2K/4K prefill (chunked
# prefill + tile path), mixed language, and one long continuation that keeps the
# decode path busy for the whole 64-token window.  Lengths are approximated from
# characters (~4 chars/token) and the realised ``prompt_tokens`` is recorded in
# the capture file, so the shape of every case is auditable.
_BASE = (
    "The pipeline stage reads a block of weights, widens each fp8 value through "
    "the lookup table and accumulates the products in fp32 before folding the "
    "per-group scales back in. Latency regressions in this loop dominate the "
    "decode step, so every stage is timed separately and the ratio between the "
    "fixed and the proportional part is what the experiment has to move."
)


def _fill(target_tokens, seed=""):
    """Deterministic filler of roughly target_tokens tokens."""
    body = (seed + " " + _BASE) if seed else _BASE
    want = max(8, target_tokens * 4)
    out = []
    while sum(len(x) + 1 for x in out) < want:
        out.append(body.strip())
    return " ".join(out)[:want]


PROMPTS = [
    ("p8_facts", 8, "List three prime numbers and nothing else."),
    ("p64_code", 64, _fill(64, "Write a python function that sums a list and "
                                 "explain each line briefly.")),
    ("p512_analysis", 512, _fill(512, "Describe how a decoder processes one "
                                        "token end to end.")),
    ("p512_zh", 512, _fill(512, "请说明解码阶段为什么主要受 host 派发开销影响。")),
    ("p2k_log", 2048, _fill(2048, "Summarise the two anomalies in this log "
                                   "and quote the line numbers.")),
    ("p2k_math", 2048, _fill(2048, "Work through the arithmetic step by "
                                    "step and check each line.")),
    ("p4k_repeat", 4096, _fill(4096, "Number every sentence of the following "
                                     "report and repeat it back.")),
    ("p8_stream", 8, "Continue this list for as long as you can: alpha, "
                     "beta, gamma,"),
]

MAX_TOKENS = 64


def _digest(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def run_prompt(prompt, max_tokens, want_body=False):
    """One greedy request, retried once on transport error.

    The request body is hashed alongside the result: a parity failure is only a
    numerics claim if both arms sent byte-identical bodies (temperature, seed,
    max_tokens, flags).  vllm-py asked for that evidence explicitly, and without
    it a template/flag difference could masquerade as a numeric regression.
    """
    last = None
    for _ in range(2):
        try:
            body = s23_common.completion_body(prompt, max_tokens=max_tokens)
            resp = s23_common.post("/v1/completions", body)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2)
            continue
        choice = resp["choices"][0]
        return {
            "body_sha": s23_common.body_sha(body),
            "text": choice.get("text"),
            "token_ids": choice.get("token_ids"),
            "prompt_token_ids": choice.get("prompt_token_ids"),
            "logprobs": [
                (step or {}).get("logprob")
                for step in ((choice.get("logprobs") or {}).get("tokens") or [])
            ],
            "finish_reason": choice.get("finish_reason"),
            "usage": resp.get("usage"),
        }
    raise RuntimeError(f"request failed twice: {last}")


def selected(cases):
    """Filter PROMPTS by --case (name out of the selection is kept for order)."""
    return [p for p in PROMPTS if not cases or p[0] in cases]


def capture(max_tokens, repeat, cases=None):
    """Record the reference token streams for all eight prompts."""
    cases_out = []
    for name, _ntok, prompt in selected(cases):
        t0 = time.time()
        runs = [run_prompt(prompt, max_tokens) for _ in range(repeat)]
        first = runs[0]["token_ids"]
        stable = all(r["token_ids"] == first for r in runs[1:])
        cases_out.append({
            "case": name,
            "prompt_sha": _digest(prompt),
            "prompt_chars": len(prompt),
            "prompt_tokens": (runs[0].get("usage") or {}).get("prompt_tokens"),
            "stable_within_arm": stable,
            **{k: runs[0][k] for k in ("text", "token_ids", "prompt_token_ids",
                                       "logprobs", "finish_reason",
                                       "body_sha")},
        })
        print(f"[capture] {name:12s} ptok={cases_out[-1]['prompt_tokens']} "
              f"gtok={len(first or [])} stable={stable} "
              f"{(time.time() - t0):.1f}s", flush=True)
    return {"cases": cases_out}


def _first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def check(baseline, max_tokens, retry, cases=None):
    """Compare the live token streams against the recorded baseline."""
    base = {c["case"]: c for c in baseline["cases"]}
    results, ok = [], True
    for name, _ntok, prompt in selected(cases):
        ref = base.get(name)
        if ref is None:
            print(f"[check] {name}: MISSING in baseline -> harness error")
            return results, None
        if ref["prompt_sha"] != _digest(prompt):
            print(f"[check] {name}: prompt text drifted from baseline")
            return results, None
        got = run_prompt(prompt, max_tokens)
        div = _first_divergence(ref["token_ids"] or [], got["token_ids"] or [])
        pdiv = _first_divergence(ref["prompt_token_ids"] or [],
                                 got["prompt_token_ids"] or [])
        rec = {
            "case": name,
            "body_sha_equal": got.get("body_sha") == ref.get("body_sha"),
            "body_sha": got.get("body_sha"),
            "token_ids_equal": div is None,
            "prompt_ids_equal": pdiv is None,
            "first_divergence": div,
            "first_prompt_divergence": pdiv,
            "n_tokens": len(got["token_ids"] or []),
            "text_equal": (ref["text"] or "") == (got["text"] or ""),
            "baseline_text": ref["text"],
            "arm_text": got["text"],
            "baseline_token_ids": ref["token_ids"],
            "arm_token_ids": got["token_ids"],
            "reproduced_after_retry": None,
        }
        if div is not None and retry:
            # Two different questions, both recorded (they were conflated before):
            #  * does the arm reproduce its OWN first answer?  -> determinism.
            #  * does the repeat still differ from the baseline?  -> not a fluke.
            # A non-self-consistent arm is nondeterminism and must be re-run, not
            # reported as a numeric regression.
            again = run_prompt(prompt, max_tokens)
            rec["retry_matches_arm_first_run"] = (
                (again["token_ids"] or []) == (got["token_ids"] or [])
            )
            rec["self_consistent"] = rec["retry_matches_arm_first_run"]
            rec["reproduced_after_retry"] = (
                _first_divergence(ref["token_ids"] or [], again["token_ids"] or [])
                is not None
            )
        bad = div is not None or pdiv is not None
        if bad and rec.get("body_sha_equal") is False:
            rec["note"] = "request bodies differ -> not a numerics claim"
        ok = ok and not bad
        results.append(rec)
        print(f"[check] {name:12s} ptok={ref['prompt_tokens']} "
              f"gtok={rec['n_tokens']} token_ids_equal={rec['token_ids_equal']} "
              f"prompt_ids_equal={rec['prompt_ids_equal']} "
              f"text_equal={rec['text_equal']}"
              + (f" first_div={div}" if div is not None else ""), flush=True)
    return results, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("capture", "check"), required=True)
    ap.add_argument("--baseline", default="", help="capture: required by check")
    ap.add_argument("--out", default="", help="json path for this run")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--repeat", type=int, default=1,
                    help="capture only: repeats per prompt for stability")
    ap.add_argument("--case", action="append", default=[],
                    help="run only these cases (debug/spot checks)")
    ap.add_argument("--retry-on-diverge", action="store_true",
                    help="check only: re-run a diverging prompt once")
    args = ap.parse_args()

    s23_common.install_auth()
    if not s23_common.wait_ready(timeout=120):
        print("[parity] service not ready", file=sys.stderr)
        return 2

    if args.mode == "capture":
        payload = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "max_tokens": args.max_tokens,
            "repeat": args.repeat,
            **capture(args.max_tokens, args.repeat, args.case),
        }
        out = args.out or f"parity-capture-{time.strftime('%Y%m%d-%H%M%S')}.json"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[parity] captured -> {out}")
        return 0

    if not args.baseline:
        print("[parity] --baseline required for check", file=sys.stderr)
        return 2
    with open(args.baseline) as f:
        baseline = json.load(f)
    results, ok = check(baseline, args.max_tokens, args.retry_on_diverge,
                        args.case)
    if ok is None:
        return 2
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": os.path.abspath(args.baseline),
        "max_tokens": args.max_tokens,
        "verdict": "parity-ok" if ok else "parity-broken",
        "cases": results,
    }
    out = args.out or f"parity-check-{time.strftime('%Y%m%d-%H%M%S')}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[parity] verdict={payload['verdict']} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
