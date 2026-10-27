#!/usr/bin/env python3
"""Offline prompt-lookup analysis for LABD / ngram-assisted DFlash2.

Answers one question *before* any GPU work: on this prompt, how much of the
generated output is reproducible from the request's own token history?

1Cat's lookup-augmented block drafting fills q16 positions 8-15 by scanning
the request's authoritative token history for the longest suffix match, and
only enters q16 after two consecutive strong copy signals
(``sm70_dflash2_lookup_augmented.md``).  Measured there: positions 8-15 were
accepted in 14 of 15 q16 rounds on a repeated-context workload.  That is a
property of the *workload*, not of the kernel, so it can be measured on CPU
in seconds -- if the hit rate is low, no amount of kernel work revives the
route for that prompt and it should not be scheduled.

The model is deliberately conservative: it credits a lookup row only when the
proposed token equals the token actually generated, which is an upper bound
on what the target verifier would accept.

Usage:
    labd_lookup_probe.py --input run.json
    labd_lookup_probe.py --prompt p.txt --output o.txt
    labd_lookup_probe.py --selftest

Input files: JSON ``{"prompt_ids": [...], "output_ids": [...]}``, or two text
files holding comma/whitespace separated integers.  Output: a summary table
plus, with ``--json``, the same numbers as one JSON object on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field


@dataclass
class Round:
    """One generation position where a lookup draft could be proposed."""

    position: int
    proposed: list[int] = field(default_factory=list)
    accepted: int = 0
    match_len: int = 0


def _longest_suffix_match(
    history: list[int],
    index: list[dict[tuple[int, ...], list[int]]],
    n_min: int,
    n_max: int,
    k_max: int,
) -> tuple[list[int], int]:
    """Propose up to ``k_max`` tokens from the most recent earlier suffix match.

    Args:
        history: All tokens known so far (prompt plus already generated).
        index: ``index[n - n_min]`` maps an n-gram to its start positions.
        n_min: Shortest suffix that may anchor a match.
        n_max: Longest suffix tried first.
        k_max: Maximum number of tokens to propose.

    Returns:
        ``(proposal, n)`` where ``n`` is the matched suffix length (0 on miss).

    """
    end = len(history)
    for n in range(min(n_max, end), n_min - 1, -1):
        positions = index[n - n_min].get(tuple(history[end - n :]))
        if not positions:
            continue
        # Recency tie-break, and the match must have at least one token after
        # it inside the history or there is nothing to propose.
        for start in reversed(positions):
            if start + n >= end:
                continue
            proposal = history[start + n : min(start + n + k_max, end)]
            if proposal:
                return proposal, n
    return [], 0


def _extend_index(
    history: list[int],
    index: list[dict[tuple[int, ...], list[int]]],
    n_min: int,
    n_max: int,
) -> None:
    """Register every n-gram that ends at the last token of ``history``."""
    end = len(history)
    for n in range(min(n_min, end), min(n_max, end) + 1):
        start = end - n
        index[n - n_min].setdefault(tuple(history[start:end]), []).append(start)


def analyze(
    prompt_ids: list[int],
    output_ids: list[int],
    k_max: int = 8,
    n_min: int = 3,
    n_max: int = 12,
) -> list[Round]:
    """Simulate greedy lookup rounds and score each against the real output.

    A round starts at the first token not yet retired.  Whatever the lookup
    finds there is compared to what the target actually produced; the round
    retires ``max(1, accepted)`` tokens, so a miss still costs one token, the
    same way a speculative round always yields at least its anchor.

    Args:
        prompt_ids: The request's input token ids.
        output_ids: The tokens the target actually generated.
        k_max: Lookup rows available per round (q16 minus the 8-row block gives
            8; pass 7 to model a q8-plus-lookup shape).
        n_min: Shortest anchoring suffix.
        n_max: Longest anchoring suffix tried first.

    Returns:
        One :class:`Round` per round actually taken, in order.

    """
    history = list(prompt_ids)
    index: list[dict[tuple[int, ...], list[int]]] = [{} for _ in range(n_max - n_min + 1)]
    for pos in range(1, len(history) + 1):
        _extend_index(history[:pos], index, n_min, n_max)

    rounds: list[Round] = []
    offset = 0
    while offset < len(output_ids):
        position = len(history)
        proposal, n = _longest_suffix_match(history, index, n_min, n_max, k_max)
        accepted = 0
        while accepted < len(proposal) and offset + accepted < len(output_ids):
            if proposal[accepted] != output_ids[offset + accepted]:
                break
            accepted += 1
        rounds.append(
            Round(position=position, proposed=proposal, accepted=accepted, match_len=n)
        )
        step = max(1, accepted)
        for token in output_ids[offset : offset + step]:
            history.append(token)
            _extend_index(history, index, n_min, n_max)
        offset += step
    return rounds


def summarize(rounds: list[Round], output_len: int) -> dict:
    """Reduce the rounds to the decision metrics.

    ``lookup_coverage`` is the share of output tokens the lookup could supply
    on its own; ``mean_extra_tokens_per_round`` is what a verifier would see as
    extra accepted rows per round.

    Args:
        rounds: Output of :func:`analyze`.
        output_len: ``len(output_ids)``.

    Returns:
        A JSON-serialisable dict of counts and rates.

    """
    hits = [r for r in rounds if r.proposed]
    served = sum(min(r.accepted, len(r.proposed)) for r in rounds)
    buckets: dict[int, int] = {}
    for r in rounds:
        buckets[r.accepted] = buckets.get(r.accepted, 0) + 1
    rounds_n = len(rounds) or 1
    return {
        "rounds": len(rounds),
        "output_len": output_len,
        "rounds_with_proposal": len(hits),
        "proposal_rate": round(len(hits) / rounds_n, 4),
        "rounds_with_hit": sum(1 for r in rounds if r.accepted),
        "hit_rate": round(sum(1 for r in rounds if r.accepted) / rounds_n, 4),
        "tokens_served_by_lookup": served,
        "lookup_coverage": round(served / (output_len or 1), 4),
        "mean_extra_tokens_per_round": round(served / rounds_n, 3),
        "mean_accepted_per_proposal": round(served / len(hits), 3) if hits else 0.0,
        "mean_proposed_len": round(sum(len(r.proposed) for r in hits) / len(hits), 3)
        if hits
        else 0.0,
        "accepted_histogram": {str(k): buckets[k] for k in sorted(buckets)},
    }


def _verdict(summary: dict) -> str:
    """Turn the numbers into a go/no-go line for scheduling LABD."""
    cov = summary["lookup_coverage"]
    if cov >= 0.40:
        return "LABD 值得立项：lookup 可覆盖 >=40% 输出 token，q16 的 8 个 lookup 行有真实内容可填。"
    if cov >= 0.15:
        return "LABD 边缘可行：覆盖率 15-40%，只在检测到强复制信号时启用 q16 才划算。"
    return "LABD 对这条 prompt 无效：覆盖率 <15%，q16 只是把 step 变贵而去填空气。"


def _load(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    """Read prompt/output token ids from the two supported input shapes."""
    if args.input:
        with open(args.input, encoding="utf-8") as handle:
            blob = json.load(handle)
        return list(blob["prompt_ids"]), list(blob["output_ids"])
    if args.prompt and args.output:
        def _read(path: str) -> list[int]:
            with open(path, encoding="utf-8") as handle:
                return [int(tok) for tok in handle.read().replace(",", " ").split()]

        return _read(args.prompt), _read(args.output)
    raise SystemExit("need --input, or both --prompt and --output (or --selftest)")


def _selftest() -> int:
    """Exercise the hit and miss paths on a hand-built case."""
    base = list(range(1000, 1040))
    prompt = base + base  # the repeated copy gives the tail a real suffix match
    output = base[:8] + [9999, 9998, 9997]
    rounds = analyze(prompt, output, k_max=8, n_min=3, n_max=12)
    summary = summarize(rounds, len(output))
    print(json.dumps(summary, indent=2))
    if summary["lookup_coverage"] > 1.0:
        print("FAIL: coverage must not exceed 1", file=sys.stderr)
        return 1
    if summary["tokens_served_by_lookup"] < 8:
        print("FAIL: expected the copied run to be servable", file=sys.stderr)
        return 1
    if rounds[-1].accepted != 0:
        print("FAIL: the novel tail must not be credited", file=sys.stderr)
        return 1
    print("selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, score one request, print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", help="JSON file with prompt_ids and output_ids")
    parser.add_argument("--prompt", help="text file of prompt token ids")
    parser.add_argument("--output", help="text file of output token ids")
    parser.add_argument("--k", type=int, default=8, help="lookup rows per round")
    parser.add_argument("--n-min", type=int, default=3, help="shortest anchor suffix")
    parser.add_argument("--n-max", type=int, default=12, help="longest anchor suffix")
    parser.add_argument("--json", action="store_true", help="emit only JSON")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    prompt_ids, output_ids = _load(args)
    rounds = analyze(prompt_ids, output_ids, args.k, args.n_min, args.n_max)
    summary = summarize(rounds, len(output_ids))
    summary["k"] = args.k
    summary["n_min"] = args.n_min
    summary["n_max"] = args.n_max
    summary["prompt_len"] = len(prompt_ids)
    summary["output_len"] = len(output_ids)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    print(f"prompt {len(prompt_ids)} tok, output {len(output_ids)} tok, k={args.k}")
    for key, value in summary.items():
        if key in ("accepted_histogram", "k", "n_min", "n_max", "prompt_len", "output_len"):
            continue
        print(f"  {key:31s} {value}")
    print(f"  accepted_histogram             {summary['accepted_histogram']}")
    print(_verdict(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
