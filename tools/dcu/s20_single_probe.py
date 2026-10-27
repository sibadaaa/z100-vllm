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

``--prompt-file`` feeds one real UTF-8 document instead of the synthetic
``item1 item2 ...`` prompt (BUILD WP0.5).  Synthetic prompts have a different
token distribution, so acceptance length is only comparable at a fixed prompt
source (PROJ section 1); the acceptance-vs-context sweep must truncate one real
document instead.  The actual token count is reported from the server model's
own tokenizer copy (never assumed), so ``--truncate-tokens`` and the summary's
``prompt_tokens_actual`` can be trusted.

Usage:
  python3 s20_single_probe.py [--base http://127.0.0.1:8000] [--model qwen3.8-27b]
                              [--prompt-tokens 512] [--max-tokens 128]
                              [--repeat 3] [--tag probe] [--out DIR]

  # real document (BUILD WP0.5 / WP1.3); mutually exclusive with --prompt-tokens
  python3 s20_single_probe.py --prompt-file docs/design/z100-...-construction.md \
      --max-tokens 96 --tag real-20k
  python3 s20_single_probe.py --prompt-file <doc> --truncate-tokens 4096 \
      --max-tokens 96 --tag real-4k
  # count tokens only, zero server contact (self-check without the GPU)
  python3 s20_single_probe.py --prompt-file <doc> --dry-run

Tokenizer: the server model's offline copy (``--tokenizer``, default
/mnt/game/models/Qwen3.8-27B-FP8).  ``transformers.AutoTokenizer`` is preferred
because it is exactly what vLLM uses; when the DTK/torch runtime is not
available (a pure client shell), it falls back to the Rust ``tokenizers``
library on the same ``tokenizer.json`` -- the two give identical ids for this
model.  ``--tokenizer-backend`` forces one path.
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
    out = {k: float(rx.search(txt).group(1)) for k, rx in _RX.items() if rx.search(txt)}
    # pure-decode services expose no spec-decode counters; default them so the
    # probe works on both configurations (accept_len reads 0.0)
    out.setdefault("accept", 0.0)
    out.setdefault("drafts", 0.0)
    return out


def build_prompt(n_tokens):
    """A deterministic prompt of roughly n_tokens tokens (vLLM counts ~1/4 word)."""
    words = []
    i = 1
    while len(words) < n_tokens:
        words.append(f"item{i}")
        i += 1
    return " ".join(words)


DEFAULT_TOKENIZER = "/mnt/game/models/Qwen3.8-27B-FP8"


class _TokenizerShim:
    """Uniform ``encode``/``decode`` over transformers or the Rust tokenizers."""

    def __init__(self, backend, tokenizer, max_len=None):
        self.backend = backend
        self._tok = tokenizer
        self.max_len = max_len

    def encode(self, text):
        if self.backend == "transformers":
            return list(self._tok(text, add_special_tokens=True)["input_ids"])
        return list(self._tok.encode(text, add_special_tokens=True).ids)

    def decode(self, ids):
        if self.backend == "transformers":
            return self._tok.decode(ids, skip_special_tokens=False)
        return self._tok.decode(list(ids), skip_special_tokens=False)


def _tokenizer_max_len(path):
    try:
        with open(os.path.join(path, "tokenizer_config.json"), encoding="utf-8") as f:
            value = json.load(f).get("model_max_length")
        return int(value) if value else None
    except (OSError, ValueError, TypeError):
        return None


def load_tokenizer(path, backend="auto"):
    """Load the served model's tokenizer offline.

    ``transformers.AutoTokenizer`` is preferred (it is what vLLM itself uses).
    Importing transformers pulls torch, so a pure client shell without the DTK
    runtime falls back to ``tokenizers`` on the same ``tokenizer.json``; both
    produce identical ids for this model (checked in the WP0.5 self-test).

    Returns:
        ``(shim, errors)``; ``shim`` is ``None`` when every backend failed.

    """
    errors = {}
    if backend in ("auto", "transformers"):
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            return _TokenizerShim("transformers", tok, _tokenizer_max_len(path)), errors
        except Exception as exc:  # noqa: BLE001 - fall back to tokenizers
            errors["transformers"] = f"{type(exc).__name__}: {exc}"
            if backend == "transformers":
                return None, errors
    if backend in ("auto", "tokenizers"):
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
            return _TokenizerShim("tokenizers", tok, _tokenizer_max_len(path)), errors
        except Exception as exc:  # noqa: BLE001
            errors["tokenizers"] = f"{type(exc).__name__}: {exc}"
    return None, errors


def read_prompt_file(path, tokenizer, truncate_tokens=0):
    """Read a UTF-8 document and return ``(text, tokens, truncated, chars)``.

    ``truncate_tokens`` > 0 keeps the first N token ids (needed by the
    acceptance-vs-context sweep) and re-decodes them; the reported count is the
    *actual* re-encoded count, never the requested one.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError(f"--prompt-file {path!r} is empty or whitespace-only")
    ids = tokenizer.encode(text)
    total = len(ids)
    truncated = False
    if truncate_tokens and total > truncate_tokens:
        ids = ids[:truncate_tokens]
        text = tokenizer.decode(ids)
        truncated = True
    return text, len(tokenizer.encode(text)), truncated, len(text)


def fetch_max_model_len(base, model, timeout=8):
    """The served max_model_len from /v1/models, or None when unreachable.

    Builds a ``Request`` (not a bare URL) so an ``install_auth`` wrapper
    (tools/dcu/s23_common.py) can attach the ``Authorization`` header; a keyed
    service returns 401 for bare-URL requests and this used to surface as a
    spurious "could not read max_model_len" warning.
    """
    try:
        request = urllib.request.Request(f"{base}/v1/models")
        data = json.loads(urllib.request.urlopen(request, timeout=timeout).read())
    except (OSError, ValueError):
        return None
    entries = data.get("data") or []
    for item in entries:
        if item.get("id") == model and item.get("max_model_len"):
            return int(item["max_model_len"])
    for item in entries:
        if item.get("max_model_len"):
            return int(item["max_model_len"])
    return None


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
        run = re.search(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.e+]+)", txt, re.M)
        wait = re.search(
            r"^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.e+]+)", txt, re.M
        )
        if float(run.group(1)) == 0 and float(wait.group(1)) == 0:
            return True
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    prompt_group = ap.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt-tokens",
        type=int,
        default=512,
        help="synthetic prompt length in ~tokens (default 512; old behaviour)",
    )
    prompt_group.add_argument(
        "--prompt-file",
        default=None,
        metavar="PATH",
        help="use one real UTF-8 document instead of the synthetic prompt",
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--out", default="/home/lyh/z100-vllm/.bench")
    ap.add_argument("--no-idle-gate", action="store_true")
    ap.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="offline tokenizer copy of the served model (default: %(default)s)",
    )
    ap.add_argument(
        "--tokenizer-backend",
        choices=("auto", "transformers", "tokenizers"),
        default="auto",
        help="tokenizer implementation; auto prefers transformers",
    )
    ap.add_argument(
        "--truncate-tokens",
        type=int,
        default=0,
        help="with --prompt-file: keep the first N token ids (0 = full document)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="with --prompt-file: report the token count and exit, no server contact",
    )
    ap.add_argument(
        "--allow-over-max-len",
        action="store_true",
        help="send even when the prompt reaches the served max_model_len",
    )
    args = ap.parse_args()

    if args.dry_run and not args.prompt_file:
        ap.error("--dry-run requires --prompt-file")
    if args.truncate_tokens < 0:
        ap.error("--truncate-tokens must be >= 0")
    if args.truncate_tokens and not args.prompt_file:
        ap.error("--truncate-tokens requires --prompt-file")

    # Real-document mode (BUILD WP0.5): tokenise the file with the server's own
    # tokenizer and report the *actual* count before touching the server.
    prompt_text = None
    tokens_actual = None
    if args.prompt_file:
        tokenizer, errors = load_tokenizer(args.tokenizer, args.tokenizer_backend)
        if tokenizer is None:
            detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
            print(
                f"[probe] ERROR cannot load tokenizer from {args.tokenizer}: {detail}",
                flush=True,
            )
            return 2
        print(
            f"[probe] tokenizer backend={tokenizer.backend} path={args.tokenizer}",
            flush=True,
        )
        try:
            prompt_text, tokens_actual, was_truncated, chars = read_prompt_file(
                args.prompt_file, tokenizer, args.truncate_tokens
            )
        except (OSError, ValueError) as exc:
            print(f"[probe] ERROR {exc}", flush=True)
            return 2
        print(
            f"[probe] prompt-file={args.prompt_file} chars={chars} "
            f"tokens_actual={tokens_actual} truncated={int(was_truncated)} "
            f"tokenizer_max_len={tokenizer.max_len or '?'}",
            flush=True,
        )
        if tokenizer.max_len and tokens_actual > tokenizer.max_len:
            print(
                f"[probe] WARN prompt has {tokens_actual} tokens > tokenizer "
                f"model_max_length={tokenizer.max_len}",
                flush=True,
            )
        if args.dry_run:
            print("[probe] dry-run: token count only, no server contact", flush=True)
            return 0

    if not args.no_idle_gate and not wait_idle(args.base):
        print("[probe] WARN engine not idle after 120s; proceeding")

    if args.prompt_file:
        max_len = fetch_max_model_len(args.base, args.model)
        if max_len is None:
            print(
                "[probe] WARN could not read max_model_len from /v1/models; "
                "verify the prompt fits the served limit",
                flush=True,
            )
        else:
            print(f"[probe] server max_model_len={max_len}", flush=True)
            if tokens_actual >= max_len:
                message = (
                    f"prompt has {tokens_actual} tokens >= server "
                    f"max_model_len {max_len}"
                )
                if not args.allow_over_max_len:
                    print(
                        f"[probe] ERROR {message}; refusing to send "
                        "(use --allow-over-max-len to override)",
                        flush=True,
                    )
                    return 2
                print(
                    f"[probe] WARN {message}; continuing (--allow-over-max-len)",
                    flush=True,
                )
            elif tokens_actual + args.max_tokens > max_len:
                print(
                    f"[probe] WARN prompt {tokens_actual} + max_tokens "
                    f"{args.max_tokens} exceeds max_model_len {max_len}; the "
                    "server may reject the request",
                    flush=True,
                )

    runs = []
    for i in range(args.repeat):
        if prompt_text is not None:
            prompt = prompt_text
        else:
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
        r["srv_itl_ms"] = (
            itl_sum / (m1["itl_cnt"] - m0["itl_cnt"]) * 1000.0
            if m1["itl_cnt"] != m0["itl_cnt"]
            else 0.0
        )
        r["accept_len"] = (acc / dr) if dr else 0.0
        r["fwd_tok_per_s"] = (gen / r["wall_s"]) if r["wall_s"] else 0.0
        r["out_tok_per_s"] = (
            (r["completion_tokens"] / r["wall_s"]) if r["wall_s"] else 0.0
        )
        if tokens_actual is not None:
            # Sanity check: the server tokenised the same text with the same
            # tokenizer, so its usage.prompt_tokens must match the offline count.
            r["prompt_tokens_actual"] = tokens_actual
            server_prompt_tokens = r.get("prompt_tokens") or 0
            if server_prompt_tokens and server_prompt_tokens != tokens_actual:
                print(
                    f"[run {i + 1}] WARN client tokens={tokens_actual} != "
                    f"server usage.prompt_tokens={server_prompt_tokens}",
                    flush=True,
                )
            elif not server_prompt_tokens:
                print(
                    f"[run {i + 1}] WARN server returned no usage.prompt_tokens; "
                    "cannot cross-check the token count",
                    flush=True,
                )
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

    summary = {
        k: agg(k)
        for k in (
            "ttft_ms",
            "srv_step_ms",
            "srv_tok_per_step",
            "accept_len",
            "out_tok_per_s",
            "fwd_tok_per_s",
            "itl_mean_ms",
            "itl_p50_ms",
            "wall_s",
        )
    }
    # step p50 across runs (per-run step is already a mean over the window)
    summary["step_p50_ms"] = (
        statistics.median([r["srv_step_ms"] for r in runs]) if runs else 0.0
    )
    if tokens_actual is not None:
        # Only in --prompt-file mode: keeps the historical summary unchanged.
        summary["prompt_tokens_actual"] = tokens_actual
        summary["prompt_file"] = args.prompt_file
    printed = {
        k: (round(v, 3) if isinstance(v, (int, float)) else v)
        for k, v in summary.items()
    }
    print("[SUMMARY] " + json.dumps(printed))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(
        args.out, f"s20_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": runs, "summary": summary}, f, indent=1)
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
