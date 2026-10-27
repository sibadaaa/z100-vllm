#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G9 gate: capture / compare greedy token streams over the OpenAI API.

L-1 (task-7) freezes the DTK same-tree token stream for three fixed prompts;
L3 replays the same capture on the new stack and ``--compare`` applies the
plan's rules (``docs/z100-torch216-rocm101-adapt-plan.md`` section 7-L3):

* prompt token ids must match (same tokenizer / request path),
* generated token ids must match element by element,
* otherwise the first divergence is excused only when the control's
  top-1/top-2 logprob margin at that position is below ``--near-tie``
  (default 0.01 nats) -- the plan's "near tie" rule,
* a non-finite logprob on either side invalidates the comparison: for L-1 it
  means the DTK baseline itself is suspect (F1, packed-SiLU fp16 regression).

Only the Python standard library is used, so the script runs unchanged in the
DTK container and in the ROCm10.1 runtime container.  Invoke it with the
*served stack's* interpreter; it never imports torch/vllm/triton.

Capture (against a live ``/v1`` server, e.g. inside ``z100-regress``)::

    python tools/dcu/g9_token_capture.py --base http://127.0.0.1:8021 \\
        --model qwen3.5-4b --label dtk \\
        --out /mnt/game/z100-baseline/dtk-g9-tokens-4b.json \\
        --serve-cmd-file /mnt/game/z100-baseline/run/g9-4b.launch.sh \\
        --meta stack=dtk --meta model_dir=/mnt/game/models/Qwen3.5-4B

Capture exit codes: 0 = captured and finite, 3 = captured but a NaN/inf
logprob was seen (an invalid L-1 baseline, e.g. the F1 regression), 2 = usage
or server-readiness error.

Compare (exit 0 = confirmed or near-tie, 1 = refuted, 2 = bad input)::

    python tools/dcu/g9_token_capture.py --compare \\
        /mnt/game/z100-baseline/dtk-g9-tokens-4b.json \\
        /mnt/game/z100-baseline/rocm101-g9-tokens-4b.json

The JSON keeps the prompt, the prompt ids, the generated ids, the per-token
logprobs and the top-1/top-2 margin at every position, so a later analysis
never needs to re-run the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request

SCHEMA = "g9-token-capture/1"

# The three prompts frozen by the L-1 baseline (plan section 7-L3).
DEFAULT_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "中国的首都是",
]

EXIT_OK = 0
EXIT_REFUTED = 1
EXIT_INPUT_ERROR = 2
EXIT_NONFINITE = 3

TOKEN_ID_PREFIX = "token_id:"


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_ready(base, timeout_s, interval=2.0):
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            return get_json(f"{base}/v1/models", timeout=10)
        except Exception as exc:  # noqa: BLE001 - poll until the deadline
            last_error = exc
            time.sleep(interval)
    raise RuntimeError(
        f"server at {base} not ready after {timeout_s}s: {last_error}"
    )


def token_id(key):
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.startswith(TOKEN_ID_PREFIX):
        try:
            return int(key[len(TOKEN_ID_PREFIX):])
        except ValueError:
            return None
    return None


def top_pairs(entry):
    """Return the top-N logprobs of one position as (logprob, token_id)."""
    if not entry:
        return []
    pairs = []
    for key, value in entry.items():
        tid = token_id(key)
        if tid is None or not isinstance(value, (int, float)):
            continue
        pairs.append((float(value), tid))
    pairs.sort(key=lambda pair: -pair[0])
    return pairs


def positions(choice):
    """Flatten one completion choice into one record per generated token."""
    logprobs = choice.get("logprobs") or {}
    tokens = list(logprobs.get("tokens") or [])
    token_logprobs = list(logprobs.get("token_logprobs") or [])
    top_logprobs = list(logprobs.get("top_logprobs") or [])
    rows = []
    count = max(len(tokens), len(token_logprobs), len(top_logprobs))
    for index in range(count):
        pairs = top_pairs(
            top_logprobs[index] if index < len(top_logprobs) else None
        )
        sampled = (
            token_logprobs[index] if index < len(token_logprobs) else None
        )
        sampled = float(sampled) if isinstance(sampled, (int, float)) else None
        tid = token_id(tokens[index]) if index < len(tokens) else None
        margin = round(pairs[0][0] - pairs[1][0], 6) if len(pairs) >= 2 else None
        rank = next(
            (i for i, (_, pair_id) in enumerate(pairs) if pair_id == tid), None
        )
        rows.append(
            {
                "token_id": tid,
                "logprob": sampled,
                "top": [[pair_id, round(value, 6)] for value, pair_id in pairs],
                "margin_top1_top2": margin,
                "sampled_rank": rank,
            }
        )
    return rows


def nonfinite_positions(rows):
    """Positions where the sampled or any top logprob is NaN/inf."""
    bad = []
    for index, row in enumerate(rows):
        values = [row["logprob"]] + [value for _, value in row["top"]]
        if any(
            value is not None and not math.isfinite(value) for value in values
        ):
            bad.append(index)
    return bad


def capture_prompt(base, model, prompt, args):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "logprobs": args.logprobs,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "stream": False,
    }
    started = time.time()
    response = post_json(f"{base}/v1/completions", body, args.timeout)
    latency_ms = round((time.time() - started) * 1000.0, 1)
    choice = (response.get("choices") or [{}])[0]
    rows = positions(choice)
    token_ids = list(choice.get("token_ids") or [])
    if not token_ids:
        # Older servers only expose the ids through the logprob token keys.
        token_ids = [row["token_id"] for row in rows]
    prompt_token_ids = list(choice.get("prompt_token_ids") or [])
    if not prompt_token_ids:
        prompt_token_ids = list(response.get("prompt_token_ids") or [])
    result = {
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "generated_text": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "token_ids": token_ids,
        "n_generated": len(token_ids),
        "usage": response.get("usage"),
        "latency_ms": latency_ms,
        "positions": rows,
    }
    result["nonfinite_positions"] = nonfinite_positions(rows)
    return result


def capture(args):
    models = wait_ready(args.base, args.wait_ready)
    server_model_ids = [item.get("id") for item in (models.get("data") or [])]
    results = []
    for prompt in args.prompts:
        print(f"[g9] capture {prompt!r} -> {args.base}", flush=True)
        result = capture_prompt(args.base, args.model, prompt, args)
        results.append(result)
        nonfinite = result["nonfinite_positions"]
        print(
            f"[g9]   text={result['generated_text']!r} "
            f"ids={result['token_ids']} finish={result['finish_reason']} "
            f"latency={result['latency_ms']}ms "
            f"nonfinite={nonfinite or 'none'}",
            flush=True,
        )
    nonfinite_all = [
        {"prompt": result["prompt"], "positions": result["nonfinite_positions"]}
        for result in results
        if result["nonfinite_positions"]
    ]
    document = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": args.label,
        "base": args.base,
        "model": args.model,
        "server_model_ids": server_model_ids,
        "request": {
            "max_tokens": args.max_tokens,
            "logprobs": args.logprobs,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "ignore_eos": False,
        },
        "serve_cmd": args.serve_cmd,
        "meta": args.meta,
        "prompts": list(args.prompts),
        "results": results,
        "finite_check": {
            "verdict": "PASS" if not nonfinite_all else "FAIL",
            "n_nonfinite_positions": sum(
                len(item["positions"]) for item in nonfinite_all
            ),
            "nonfinite": nonfinite_all,
            "method": (
                "all sampled logprobs and top-N logprobs are finite "
                "(the completions API does not expose raw logits)"
            ),
        },
        "texts": {
            result["prompt"]: result["generated_text"] for result in results
        },
    }
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
        print(f"[g9] saved {args.out}", flush=True)
    print(
        f"[g9] finite_check={document['finite_check']['verdict']} "
        f"nonfinite_positions={document['finite_check']['n_nonfinite_positions']}"
    )
    return EXIT_NONFINITE if nonfinite_all else EXIT_OK


def load_capture(path):
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path}: cannot read capture: {exc}")
    if document.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: not a {SCHEMA} file")
    return document


def first_divergence(left, right):
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def margin_at(result, index):
    rows = result.get("positions") or []
    if index is None or index >= len(rows):
        return None
    return rows[index].get("margin_top1_top2")


def compare(paths, near_tie):
    control = load_capture(paths[0])
    candidate = load_capture(paths[1])
    control_by_prompt = {
        result["prompt"]: result for result in control.get("results") or []
    }
    candidate_by_prompt = {
        result["prompt"]: result for result in candidate.get("results") or []
    }
    prompts = list(control.get("prompts") or control_by_prompt)
    rows = []
    for prompt in prompts:
        left = control_by_prompt.get(prompt)
        right = candidate_by_prompt.get(prompt)
        row = {
            "prompt": prompt,
            "verdict": "refuted",
            "notes": [],
        }
        if left is None or right is None:
            row["notes"].append("missing on one side")
            rows.append(row)
            continue
        if left.get("nonfinite_positions"):
            row["verdict"] = "nonfinite-control"
            row["notes"].append("control has non-finite logprobs: invalid baseline")
            rows.append(row)
            continue
        if right.get("nonfinite_positions"):
            row["verdict"] = "nonfinite-candidate"
            row["notes"].append("candidate has non-finite logprobs")
            rows.append(row)
            continue
        left_prompt_ids = left.get("prompt_token_ids") or []
        right_prompt_ids = right.get("prompt_token_ids") or []
        if left_prompt_ids and right_prompt_ids:
            row["prompt_ids_equal"] = left_prompt_ids == right_prompt_ids
            if left_prompt_ids != right_prompt_ids:
                row["notes"].append("prompt token ids differ (tokenizer/path)")
                rows.append(row)
                continue
        else:
            row["prompt_ids_equal"] = None
            row["notes"].append("prompt token ids unavailable on one side")
        left_ids = list(left.get("token_ids") or [])
        right_ids = list(right.get("token_ids") or [])
        row["n_tokens_control"] = len(left_ids)
        row["n_tokens_candidate"] = len(right_ids)
        row["text_equal"] = left.get("generated_text") == right.get(
            "generated_text"
        )
        divergence = first_divergence(left_ids, right_ids)
        if divergence is None:
            row["verdict"] = "confirmed"
            row["n_diff_positions"] = 0
        else:
            margin = margin_at(left, divergence)
            row["first_divergence"] = divergence
            row["margin_control_at_divergence"] = margin
            row["control_ids"] = left_ids[divergence:divergence + 6]
            row["candidate_ids"] = right_ids[divergence:divergence + 6]
            n_diff = sum(
                1 for a, b in zip(left_ids, right_ids) if a != b
            ) + abs(len(left_ids) - len(right_ids))
            row["n_diff_positions"] = n_diff
            if margin is not None and margin < near_tie:
                row["verdict"] = "near-tie"
                row["notes"].append(
                    f"first divergence at {divergence} but control margin "
                    f"{margin} < {near_tie} nat"
                )
            else:
                row["notes"].append(
                    f"first divergence at {divergence}, control margin "
                    f"{margin} >= {near_tie} nat"
                )
        rows.append(row)
    refuted = sum(
        1 for row in rows if row["verdict"] not in ("confirmed", "near-tie")
    )
    print(f"[g9-compare] control  : {paths[0]}")
    print(f"[g9-compare] candidate: {paths[1]}")
    for row in rows:
        detail = row.get("notes") or []
        if row.get("verdict") == "confirmed":
            detail = [f"{row.get('n_tokens_control')} ids identical"]
        print(
            f"[g9-compare] {row['verdict']:16s} {row['prompt']!r} "
            f"{'; '.join(detail)}"
        )
        if row.get("first_divergence") is not None:
            print(
                f"[g9-compare]   div={row['first_divergence']} "
                f"margin={row.get('margin_control_at_divergence')} "
                f"control={row.get('control_ids')} "
                f"candidate={row.get('candidate_ids')}"
            )
    if refuted:
        print(f"[g9-compare] RESULT: REFUTED ({refuted}/{len(rows)} prompts)")
        return EXIT_REFUTED
    print(f"[g9-compare] RESULT: CONFIRMED ({len(rows)}/{len(rows)} prompts)")
    return EXIT_OK


def parse_meta(pairs):
    meta = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--meta expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        meta[key] = value
    return meta


def parse_prompts(args):
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as handle:
            prompts = json.load(handle)
        if not isinstance(prompts, list) or not all(
            isinstance(item, str) for item in prompts
        ):
            raise SystemExit(f"{args.prompts_file}: expected a JSON list of strings")
        return prompts
    if args.prompt:
        return list(args.prompt)
    return list(DEFAULT_PROMPTS)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="g9_token_capture.py",
        description=(
            "Capture or compare G9 greedy token streams over the vLLM OpenAI "
            "API. Stdlib only; no torch/vllm import."
        ),
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CONTROL", "CANDIDATE"),
        help="compare two capture JSON files instead of capturing",
    )
    parser.add_argument(
        "--near-tie",
        type=float,
        default=0.01,
        help="top-1/top-2 margin (nats) below which a divergence is excused",
    )
    parser.add_argument(
        "--base",
        default="http://127.0.0.1:8021",
        help="server base URL of the stack under capture",
    )
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument(
        "--label",
        default="dtk",
        help="free-form stack label recorded in the JSON (dtk|rocm101|...)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="capture output JSON (required unless --compare is used)",
    )
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=5,
        help="top-N logprobs per position (also gives the top-1/top-2 margin)",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--wait-ready",
        type=int,
        default=60,
        help="seconds to poll /v1/models before capture (0 = no wait)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="prompt to capture (repeatable; default = the L-1 three prompts)",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="JSON list of prompts; overrides --prompt",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=None,
        help="KEY=VALUE metadata recorded in the capture (repeatable)",
    )
    parser.add_argument(
        "--serve-cmd",
        default="",
        help="serve command line recorded in the capture",
    )
    parser.add_argument(
        "--serve-cmd-file",
        default=None,
        help="file whose contents are recorded as serve_cmd",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.compare:
        return compare(args.compare, args.near_tie)
    if not args.out:
        raise SystemExit("--out is required for capture (or use --compare)")
    args.meta = parse_meta(args.meta)
    if args.serve_cmd_file:
        with open(args.serve_cmd_file, encoding="utf-8") as handle:
            args.serve_cmd = handle.read().strip()
    args.prompts = parse_prompts(args)
    if not args.prompts:
        raise SystemExit("no prompts to capture")
    try:
        return capture(args)
    except RuntimeError as exc:
        print(f"[g9] ERROR: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    sys.exit(main())
