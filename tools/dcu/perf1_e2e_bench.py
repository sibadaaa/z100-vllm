#!/usr/bin/env python3
"""perf1 end-to-end client: streaming perf sweep + greedy token-parity capture.

Superset of /home/lyh/vllm-test/scripts/bench_baseline.py.  The JSON schema of
that script is preserved (``config`` / ``runs`` / ``spec`` / ``summary``) so
results stay directly comparable with ``/home/lyh/vllm-test/results/*.json``.
Additions, all additive keys:

* ``--lengths`` takes a comma list of *configured* prompt sizes.  The builds in
  the reference report (119/451/1806/7443 real tokens) come from
  ``build_prompt(128/512/2048/8192)`` -- the tokenizer is identical, so the same
  numbers reproduce the same prompts.
* ``--mode parity`` / ``parity_runs``: greedy token-parity capture over
  ``/v1/completions`` with ``return_token_ids`` + ``logprobs=1``.  The
  completions endpoint is used because the running server has a reasoning
  parser installed, and vLLM suppresses streamed ``token_ids``/``logprobs``
  whenever such a parser is active (chat_completion/serving.py
  ``hide_stream_metadata``).  The raw text prompt is passed verbatim, so the
  tokenizer -- not this client -- decides the prompt ids; the server returns
  them per chunk and they are recorded and compared too.
* ``cold_start``: the very first request against a freshly started server,
  measured separately from steady state.

Comparison is done by ``tools/dcu/perf1_e2e_parity.py``.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request


def metrics(base):
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
            return r.read().decode()
    except Exception:
        return ""


_COUNTER_RE = re.compile(
    r"^(?P<name>vllm:(?:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens)_total"
    r"|generation_tokens_total))"
    r"\{(?P<labels>[^}]*)\}\s+(?P<value>[0-9.eE+]+)$",
    re.M,
)


def parse_prom(base):
    """Prometheus counters, same semantics as bench_baseline.spec_counters."""
    txt = metrics(base)
    out = {}
    for m in _COUNTER_RE.finditer(txt):
        name, labels, val = m.group("name"), m.group("labels"), float(m.group("value"))
        if "position=" in labels:
            pos = re.search(r'position="(\d+)"', labels)
            out[f"accept_pos{pos.group(1)}"] = val
        elif name.endswith("num_drafts_total"):
            out["drafts"] = val
        elif name.endswith("num_draft_tokens_total"):
            out["draft_tokens"] = val
        elif name.endswith("num_accepted_tokens_total"):
            out["accepted"] = val
        elif name.endswith("generation_tokens_total"):
            out["gen_total"] = val
    return out


def prom_delta(before, after):
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    if d.get("drafts", 0) > 0:
        d["accept_rate"] = round(d.get("accepted", 0) / d["draft_tokens"], 4) if d.get("draft_tokens") else None
        d["tokens_per_step"] = round(d.get("gen_total", 0) / d["drafts"], 4)
    return d


def build_prompt(target_tokens):
    """Byte-identical to bench_baseline.build_prompt: keeps prompts comparable."""
    lines, i = [], 0
    while i * 22 < target_tokens * 1.8:
        lines.append(
            f"Line {i}: the quick brown fox jumps over the lazy dog while "
            f"seventeen engineers review pipeline stage number {i} for latency regressions."
        )
        i += 1
    return "\n".join(lines)[: int(target_tokens * 4.0)]


def _stream(base, path, body, timeout):
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def one_chat_request(base, model, prompt, max_tokens, timeout, ignore_eos, temperature):
    """Streaming chat request; body identical to bench_baseline.one_request."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    t0 = time.perf_counter()
    ttft = None
    chunk_times = []
    usage, finish = {}, None
    with _stream(base, "/v1/chat/completions", body, timeout) as resp:
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
                d = ch.get("delta") or {}
                txt = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if txt:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    chunk_times.append(now)
    t_end = time.perf_counter()
    total = t_end - t0
    itls = [chunk_times[i] - chunk_times[i - 1] for i in range(1, len(chunk_times))]
    gen_tokens = usage.get("completion_tokens") or len(chunk_times)
    prompt_tokens = usage.get("prompt_tokens") or 0
    decode_time = (t_end - t0 - ttft) if ttft else total
    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "chunks": len(chunk_times),
        "ttft_ms": round(ttft * 1000, 2) if ttft else None,
        "total_ms": round(total * 1000, 2),
        "prefill_tps": round(prompt_tokens / ttft, 2) if ttft else 0,
        "decode_tps_net": round(gen_tokens / decode_time, 2) if decode_time > 0 else 0,
        "step_ms_p50": round(statistics.median(itls) * 1000, 2) if itls else None,
        "step_ms_mean": round(statistics.mean(itls) * 1000, 2) if itls else None,
        "step_ms_max": round(max(itls) * 1000, 2) if itls else None,
        "tokens_per_step": round(gen_tokens / len(itls), 3) if itls else None,
        "finish": finish,
        "usage": usage,
    }


def one_parity_request(base, model, prompt, max_tokens, timeout):
    """Greedy token-id + top-1 logprob capture over /v1/completions."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "logprobs": 1,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    prompt_ids, token_ids, token_texts, logprobs, offsets = [], [], [], [], []
    usage, finish = {}, None
    with _stream(base, "/v1/completions", body, timeout) as resp:
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
                if ch.get("prompt_token_ids") and not prompt_ids:
                    prompt_ids = list(ch["prompt_token_ids"])
                ids = ch.get("token_ids") or []
                if ids:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    token_ids += list(ids)
                lp = ch.get("logprobs") or {}
                if lp.get("tokens"):
                    token_texts += list(lp["tokens"])
                    logprobs += [None if v is None else float(v) for v in (lp.get("token_logprobs") or [])]
                    offsets += list(lp.get("text_offset") or [])
    t_end = time.perf_counter()
    return {
        "mode": "parity",
        "prompt_tokens": usage.get("prompt_tokens") or len(prompt_ids),
        "gen_tokens": usage.get("completion_tokens") or len(token_ids),
        "ttft_ms": round(ttft * 1000, 2) if ttft else None,
        "total_ms": round((t_end - t0) * 1000, 2),
        "finish": finish,
        "prompt_token_ids": prompt_ids,
        "token_ids": token_ids,
        "logprobs": logprobs,
        "token_texts": token_texts,
        "text_offsets": offsets,
        "usage": usage,
    }


def summarize(rows):
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "prefill_tps": mean("prefill_tps"),
        "decode_tps": mean("decode_tps_net"),
        "step_ms_p50": mean("step_ms_p50"),
        "tokens_per_step": mean("tokens_per_step"),
        "ttft_ms": mean("ttft_ms"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--lengths", default="512",
                    help="configured prompt token sizes, comma separated")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ignore-eos", action="store_true", default=True)
    ap.add_argument("--mode", choices=["perf", "parity", "both"], default="both")
    ap.add_argument("--parity-lengths", default="512")
    ap.add_argument("--parity-max-tokens", type=int, default=64)
    ap.add_argument("--parity-repeat", type=int, default=2)
    ap.add_argument("--cold-length", type=int, default=0,
                    help="run this configured prompt size first, as a fresh-server "
                         "cold-start probe (0 disables)")
    ap.add_argument("--cold-max-tokens", type=int, default=32)
    ap.add_argument("--warmup", action="store_true",
                    help="run one discarded chat request at the cold length before "
                         "the cold probe (warm-start arm: measures the post-warmup "
                         "first-request penalty instead of the true cold one)")
    ap.add_argument("--tag", default="e2e")
    ap.add_argument("--arm", default=os.environ.get("PERF1_ARM", "unknown"))
    ap.add_argument("--so-md5", default=None,
                    help="md5 of the _rocm_C.abi3.so the arm loaded; ties every "
                         "number to an exact kernel build")
    ap.add_argument("--out", default="/mnt/game/perf1")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    parity_lengths = [int(x) for x in args.parity_lengths.split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"[cfg] arm={args.arm} tag={args.tag} lengths={lengths} "
          f"max_tokens={args.max_tokens} repeat={args.repeat} mode={args.mode} "
          f"temp={args.temperature} ignore_eos={args.ignore_eos}")

    result = {
        "config": {
            "base": args.base, "model": args.model, "prompt_tokens": lengths[0],
            "prompt_tokens_all": lengths, "max_tokens": args.max_tokens,
            "repeat": args.repeat, "timeout": args.timeout,
            "ignore_eos": args.ignore_eos, "temperature": args.temperature,
            "mode": args.mode, "parity_lengths": parity_lengths,
            "parity_max_tokens": args.parity_max_tokens,
            "parity_repeat": args.parity_repeat,
            "cold_length": args.cold_length, "cold_max_tokens": args.cold_max_tokens,
            "tag": args.tag, "arm": args.arm, "out": args.out,
            "so_md5": args.so_md5,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "runs": [],
        "spec": {},
        "summary": {},
    }

    before = parse_prom(args.base)

    if args.cold_length:
        prompt = build_prompt(args.cold_length)
        if args.warmup:
            wu = one_chat_request(args.base, args.model, prompt, 2,
                                  args.timeout, True, args.temperature)
            wu["phase"] = "warmup_discard"
            result["warmup_discard"] = wu
            print(f"[warmup] discarded p{wu['prompt_tokens']} TTFT={wu['ttft_ms']}ms "
                  f"prefill={wu['prefill_tps']}t/s", flush=True)
        row = one_chat_request(args.base, args.model, prompt, args.cold_max_tokens,
                               args.timeout, args.ignore_eos, args.temperature)
        row["phase"] = "cold_start"
        row["prompt_cfg"] = args.cold_length
        result["cold_start"] = row
        print(f"[cold] p{row['prompt_tokens']} TTFT={row['ttft_ms']}ms "
              f"prefill={row['prefill_tps']}t/s decode={row['decode_tps_net']}t/s")

    if args.mode in ("perf", "both"):
        for cfg_len in lengths:
            prompt = build_prompt(cfg_len)
            for r in range(args.repeat):
                row = one_chat_request(args.base, args.model, prompt, args.max_tokens,
                                       args.timeout, args.ignore_eos, args.temperature)
                row["prompt_cfg"] = cfg_len
                row["rep"] = r
                row["phase"] = "perf"
                result["runs"].append(row)
                print(f"[perf p{cfg_len} {r + 1}/{args.repeat}] prompt={row['prompt_tokens']}tok "
                      f"gen={row['gen_tokens']}tok chunks={row['chunks']} TTFT={row['ttft_ms']}ms "
                      f"prefill={row['prefill_tps']}t/s decode={row['decode_tps_net']}t/s "
                      f"step(p50)={row['step_ms_p50']}ms tok/step={row['tokens_per_step']} "
                      f"total={row['total_ms']}ms")

    if args.mode in ("parity", "both"):
        result["parity_runs"] = []
        for cfg_len in parity_lengths:
            prompt = build_prompt(cfg_len)
            for r in range(args.parity_repeat):
                row = one_parity_request(args.base, args.model, prompt,
                                         args.parity_max_tokens, args.timeout)
                row["prompt_cfg"] = cfg_len
                row["rep"] = r
                result["parity_runs"].append(row)
                print(f"[parity p{cfg_len} {r + 1}/{args.parity_repeat}] "
                      f"prompt={row['prompt_tokens']}tok gen={row['gen_tokens']}tok "
                      f"TTFT={row['ttft_ms']}ms ids={row['token_ids'][:8]}...")

    after = parse_prom(args.base)
    result["spec"] = prom_delta(before, after)

    by_len = {}
    for cfg_len in lengths:
        rows = [r for r in result["runs"] if r.get("prompt_cfg") == cfg_len]
        if rows:
            by_len[str(cfg_len)] = summarize(rows)
    result["summary_by_prompt"] = by_len
    result["summary"] = summarize(result["runs"]) if result["runs"] else {}

    os.makedirs(args.out, exist_ok=True)
    if args.out_name:
        out = os.path.join(args.out, args.out_name)
    elif len(lengths) == 1 and args.mode == "perf":
        out = f"{args.out}/{args.tag}_p{lengths[0]}_g{args.max_tokens}_{stamp}.json"
    else:
        span = f"p{lengths[0]}" if len(lengths) == 1 else f"p{lengths[0]}-p{lengths[-1]}"
        out = f"{args.out}/{args.tag}_{span}_g{args.max_tokens}_{stamp}.json"
    result["config"]["out_file"] = out
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[summary] {json.dumps(result['summary'])}")
    for k, v in by_len.items():
        print(f"[summary p{k}] {json.dumps(v)}")
    if result["spec"]:
        print(f"[spec] {json.dumps(result['spec'])}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.exit(main())
