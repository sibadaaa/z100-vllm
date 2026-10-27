#!/usr/bin/env python3
"""Single-stream e2e TPS bench for the z100 specbench arms.

Streams /v1/completions with the 7469-token probe prompt (same as the s33
deployment protocol).  Two workloads per arm:
  W1: 96 tokens, greedy, 3 reps  (s33 deploy continuity; also parity ids)
  W2: 512 tokens, temp 0.7 + rep penalty 1.05, 2 reps (agent-like sampling)

Output: <out>/e2e-<tag>.json + human-readable summary on stdout.
"""
import argparse
import json
import os
import time
import urllib.request

BASE = os.environ.get("VLLM_PROBE_BASE", "http://127.0.0.1:8000")
MODEL = "qwen3.8-27b"
RB = "/home/lyh/z100-vllm/runs/specbench"
APIKEY = None
if os.path.exists("/mnt/game/host-serve/api-key"):
    APIKEY = open("/mnt/game/host-serve/api-key").read().strip()


def _hdr(extra=None):
    h = dict(extra or {})
    if APIKEY:
        h["Authorization"] = f"Bearer {APIKEY}"
    return h


def health():
    with urllib.request.urlopen(
        urllib.request.Request(BASE + "/health", headers=_hdr()), timeout=5
    ) as r:
        return r.status


def stream_once(body, timeout=1200):
    """One streaming completion.  Returns (n_out, ttft_s, gen_tps, e2e_s, ids, text)."""
    r = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(body).encode(),
        headers=_hdr({"Content-Type": "application/json"}),
        method="POST",
    )
    t0 = time.time()
    t_first = None
    finish = None
    usage = {}
    ids = []
    text = []
    with urllib.request.urlopen(r, timeout=timeout) as resp:
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
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if ch.get("token_ids"):
                    ids.extend(ch["token_ids"])
                if ch.get("text"):
                    text.append(ch["text"])
            if t_first is None and (ids or text):
                t_first = time.time()
    t_end = time.time()
    n = usage.get("completion_tokens") or len(ids) or sum(len(t) for t in text)
    ttft = (t_first - t0) if t_first else None
    tps = (n / (t_end - t_first)) if (t_first and t_end > t_first and n) else None
    e2e = t_end - t0
    return n, ttft, tps, e2e, ids, "".join(text), finish, usage


def run_workload(name, reps, max_tokens, temperature, rep_penalty, greedy_ids):
    rows = []
    for i in range(reps):
        body = {
            "model": MODEL,
            "prompt": PROMPT_IDS,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if rep_penalty != 1.0:
            body["repetition_penalty"] = rep_penalty
        if greedy_ids:
            body["return_token_ids"] = True
            body["return_tokens_as_token_ids"] = True
        try:
            n, ttft, tps, e2e, ids, text, finish, usage = stream_once(body)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] rep{i+1} FAILED: {exc!r}")
            rows.append({"rep": i + 1, "error": repr(exc)})
            time.sleep(2)
            continue
        row = {
            "rep": i + 1,
            "n_out": n,
            "ttft_s": round(ttft, 3) if ttft else None,
            "gen_tps": round(tps, 2) if tps else None,
            "e2e_s": round(e2e, 3),
            "finish": finish,
        }
        if greedy_ids:
            row["out_ids"] = [int(str(t).replace("token_id:", ""))
                              for t in ids if str(t).startswith("token_id:")] or ids
            row["text"] = text
        rows.append(row)
        print(f"  [{name}] rep{i+1}: n={n} ttft={row['ttft_s']}s "
              f"tps={row['gen_tps']} e2e={row['e2e_s']}s finish={finish}")
        time.sleep(1)
    ok = [r for r in rows if r.get("gen_tps")]
    summary = {}
    if ok:
        tpss = sorted(r["gen_tps"] for r in ok)
        summary = {
            "tps_med": tpss[len(tpss) // 2] if len(tpss) % 2 else
                       round((tpss[len(tpss)//2 - 1] + tpss[len(tpss)//2]) / 2, 2),
            "tps_all": tpss,
            "ttft_med": sorted(r["ttft_s"] for r in ok)[len(ok) // 2],
        }
    return rows, summary


PROMPT_IDS = json.load(open(f"{RB}/prompt_ids.json"))
assert len(PROMPT_IDS) == 7469


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=RB)
    ap.add_argument("--w1-reps", type=int, default=3)
    ap.add_argument("--w2-reps", type=int, default=2)
    ap.add_argument("--skip-w2", action="store_true")
    ap.add_argument("--w2-max-tokens", type=int, default=512)
    args = ap.parse_args()

    code = health()
    assert code == 200, f"health {code}"
    print(f"[{args.tag}] health ok, base={BASE}")

    # warmup: populate prefix cache for the probe prompt (8 tokens)
    n, ttft, tps, e2e, _, _, finish, _ = stream_once({
        "model": MODEL, "prompt": PROMPT_IDS, "max_tokens": 8,
        "temperature": 0.0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    })
    print(f"[{args.tag}] warmup: n={n} ttft={ttft and round(ttft,2)}s e2e={round(e2e,2)}s")

    print(f"[{args.tag}] W1 greedy 96 x{args.w1_reps}")
    w1_rows, w1_sum = run_workload("W1", args.w1_reps, 96, 0.0, 1.0, greedy_ids=True)
    if all(isinstance(r.get("out_ids"), list) and r["out_ids"] for r in w1_rows):
        same = all(r["out_ids"] == w1_rows[0]["out_ids"] for r in w1_rows[1:])
        print(f"[{args.tag}] W1 in-arm streams identical: {same}")
        w1_sum["in_arm_identical"] = same
        with open(f"{args.out}/parity-{args.tag}.json", "w") as h:
            json.dump({"tag": args.tag, "ids": w1_rows[0]["out_ids"],
                       "text": w1_rows[0].get("text", "")}, h)

    w2_rows, w2_sum = [], {}
    if not args.skip_w2:
        print(f"[{args.tag}] W2 temp=0.7 rp=1.05 {args.w2_max_tokens} x{args.w2_reps}")
        w2_rows, w2_sum = run_workload("W2", args.w2_reps, args.w2_max_tokens,
                                       0.7, 1.05, greedy_ids=False)

    doc = {
        "schema": "z100-specbench-e2e-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base": BASE,
        "tag": args.tag,
        "w1": {"rows": w1_rows, "summary": w1_sum},
        "w2": {"rows": w2_rows, "summary": w2_sum},
    }
    with open(f"{args.out}/e2e-{args.tag}.json", "w") as h:
        json.dump(doc, h, indent=1)
    print(f"[{args.tag}] W1 med tps={w1_sum.get('tps_med')} ttft={w1_sum.get('ttft_med')}")
    print(f"[{args.tag}] W2 med tps={w2_sum.get('tps_med')} ttft={w2_sum.get('ttft_med')}")


if __name__ == "__main__":
    main()
