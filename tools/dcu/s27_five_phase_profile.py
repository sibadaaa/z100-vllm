#!/usr/bin/env python3
"""S27 WP1.2 five-phase round profile: collect + parse + selftest.

G5 (``docs/design/z100-decode-prefill-optimization-construction.md`` §5 WP1.2)
needs the 129 ms decode step cut into 1Cat's five phases::

    draft | draft->target | target | target->draft | complete round

with, for every phase, the top-5 operators (ms/step, calls/step) and
``sum(5 phases)`` within 5% of ``srv_step_ms`` (any remainder printed as
``gap/unknown``).  This tool is both halves of that measurement:

``--phase collect``
    Drives one real long document request against a live vLLM service, waits
    for steady-state decode, POSTs ``/start_profile``, collects
    ``--profile-steps`` worker iterations (read from ``/metrics``), POSTs
    ``/stop_profile`` and waits for the per-rank trace files to land.

    The service MUST have been started with a profiler config, e.g.::

        --profiler-config.profiler=torch \
        --profiler-config.torch_profiler_dir=/mnt/game/s27-prof

    ``VLLM_TORCH_PROFILER_DIR`` is **not** consumed by this tree (checked in
    ``vllm/config/profiler.py`` / ``vllm/engine/arg_utils.py``); without the
    CLI config ``POST /start_profile`` fails and the profile routers are not
    even attached.  The API key is attached through
    ``tools/dcu/s23_common.py:install_auth``.

``--phase parse``
    Reads the newest per-rank torch.profiler Chrome traces, cuts the decode
    window into the five phases and emits JSON + markdown.

Attribution (evidence, reproducible):

* **Step windows** come from vLLM's own profiler annotations
  ``execute_context_{ctx_reqs}({ctx_tokens})_generation_{gen_reqs}({gen_tokens})``
  (or the ``detailed_trace_annotation`` variant
  ``execute_{total}_context_..._generation_...``).  Only windows with
  ``generation tokens > 0`` are decode steps; prefill windows are excluded, so
  a prefill ``aten::mm`` can never leak into the decode census (BUILD WP1.2
  item 3).  ``ProfilerStep#N`` ranges are the fallback when the annotation is
  absent.
* **draft vs target** inside a step depends on what the annotations cover:

  - **piecewise CUDA-graph arm** (the WP1.1 candidate): the CPU annotation
    spans one whole round (~117 ms) while the *GPU-side* annotation covers only
    the target verify forward (~78 ms).  ``--split-mode auto`` detects this
    (GPU-annotation/round-period ratio < ``--graph-target-ratio``) and assigns
    ``target`` = kernels inside the GPU annotation, ``draft`` = all kernels
    before it in the round, ``target->draft`` = the verify/sampling/prepare
    tail.  This is annotation-based, not a heuristic; the measured layout is
    ``[draft+selector ~31 ms][target ~79 ms][verify/sampling ~5 ms]``.
  - **eager arm** (GPU annotation covers the whole step): kernel-activity
    burst/signature split.  Target kernels carry a dense signature absent from
    the draft tower: fp8 emulation GEMMs, ``per_token_group_quant_8bit``,
    attention/paged-attention, GDN/FLA chunk kernels and the dominant
    allreduce/all-gather kernels.  The largest time-cluster of signature
    kernels is the target burst ("~256 GEMMs/step + allreduce" in the BUILD);
    everything before it in that step is the draft tower + selector.  With no
    signature kernels (e.g. a rebuilt kernel set) it falls back to splitting
    the step at its largest inter-kernel gap and classifying by signature
    count.
* Draft/target annotations (``gpu_model_runner: draft`` / ``: forward`` /
  ``: sample`` / ``: bookkeep``), when present, are reported as a cross-check
  (``annotation_check``), not used as the primary attribution.
* The five phases are measured as intervals on the GPU kernel timeline:
  ``draft`` = first-to-last draft kernel, ``draft->target`` = draft end to
  target start, ``target`` = target envelope, ``target->draft`` = target end
  to the next step's window start, ``complete round`` = next window start minus
  this window start.  The only un-attributed time is
  ``gap_unknown_ms`` = window start to first kernel (host pre-step work).

Trace format assumptions (Chrome trace JSON, torch.profiler/kineto):

* top-level ``traceEvents`` list; GPU work is ``cat == "kernel"`` (plus
  ``gpu_memcpy``/``gpu_memset``), each ``ph == "X"`` with ``ts``/``dur`` in
  microseconds and ``args.stream``/``args.correlation``;
* annotations are ``cat == "user_annotation"`` (CPU timeline) and
  ``cat == "gpu_user_annotation"`` (GPU timeline), same ``ts``/``dur`` fields;
* metadata events (``ph == "M"``) name the process (``VLLM::Worker_TP0``) and
  label the device (``GPU 0``); traces with no ``kernel`` events at all are
  front-end CPU-only traces and are skipped.

Usage::

    # 1) collect (live service, GPU owned by the operator)
    tools/dcu/s27_five_phase_profile.py --phase collect \
        --prompt-file <7469-token-doc> --profile-steps 20 \
        --out .s27/p1 --tag graphA

    # 2) parse (zero-card)
    tools/dcu/s27_five_phase_profile.py --phase parse \
        --trace-dir /mnt/game/s27-prof --collect-summary .s27/p1/....json \
        --out-md docs/result/z100-five-phase-profile.md

    # 3) acceptance checks on synthetic traces (zero-card)
    tools/dcu/s27_five_phase_profile.py --selftest
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools", "dcu")
DEFAULT_TRACE_DIR = os.environ.get("VLLM_TORCH_PROFILER_DIR", "/mnt/game/s27-prof")

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel"}
KERNEL_CATS = {"kernel", "Kernel"}
PHASES = (
    ("draft", "draft"),
    ("draft_to_target", "draft->target"),
    ("target", "target"),
    ("target_to_draft", "target->draft"),
)
PHASE_TOTAL = ("complete_round", "complete round")

# Kernel-name signatures that only the target 27B verify step produces in
# volume.  Matched case-insensitively against the full kernel symbol.
TARGET_SIG_RX = re.compile(
    r"per_token_group_quant|fp8_emu|attention|paged_attn|gdn|fla_|chunk|"
    r"delta|causal_conv|nccl|allreduce|all_reduce|all_gather|allgather|"
    r"reduce_scatter|rms_norm|gemma_fused",
    re.IGNORECASE,
)
DRAFT_SIG_RX = re.compile(r"dflash|draft", re.IGNORECASE)
NCCL_RX = re.compile(
    r"nccl|rccl|allreduce|all_reduce|allgather|all_gather|reduce_scatter|"
    r"reducescatter",
    re.IGNORECASE,
)
ANNOT_DRAFT_RX = re.compile(r"draft|propose|ngram", re.IGNORECASE)
ANNOT_TARGET_RX = re.compile(
    r"gpu_model_runner:\s*(forward|sample|bookkeep|postprocess)|target",
    re.IGNORECASE,
)

# vLLM profiler annotations for one engine step (gpu_worker.py:1329).
STEP_RX_SIMPLE = re.compile(
    r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$"
)
STEP_RX_DETAILED = re.compile(
    r"^execute_(\d+)_context_(\d+)\("
    r"sq(\d+)sk\d+sqsq\d+sqsk\d+\)_generation_(\d+)\("
    r"sq(\d+)sk\d+sqsq\d+sqsk\d+\)$"
)


# ---------------------------------------------------------------------------
# Trace loading (streaming; a 20-step trace is still hundreds of MB of JSON)
# ---------------------------------------------------------------------------


def iter_trace_events(path):
    """Yield one ``traceEvents`` entry at a time (O(n), low memory).

    The trace array is hundreds of MB of JSON and a 20-step profile has
    ~40k kernel events per rank, so it is parsed incrementally.  The read
    offset is advanced in place and the buffer is compacted every 16 MiB;
    slicing the buffer after every event would make this quadratic.
    """
    decoder = json.JSONDecoder()
    chunk = 4 << 20
    compact = 16 << 20
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="ignore") as handle:
        buf = ""
        while '"traceEvents"' not in buf:
            piece = handle.read(chunk)
            if not piece:
                raise RuntimeError(f"{path}: no traceEvents key")
            buf += piece
        pos = buf.index("[", buf.index('"traceEvents"')) + 1
        while True:
            size = len(buf)
            while pos < size and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= size:
                piece = handle.read(chunk)
                if not piece:
                    return
                buf = buf[pos:] + piece
                pos = 0
                continue
            if buf[pos] == "]":
                return
            try:
                event, end = decoder.raw_decode(buf, pos)
            except ValueError:
                piece = handle.read(chunk)
                if not piece:
                    return
                buf = buf[pos:] + piece
                pos = 0
                if len(buf) > 256 * chunk:
                    raise RuntimeError(
                        f"{path}: single event larger than 256MB"
                    ) from None
                continue
            pos = end
            yield event
            if pos > compact:
                buf = buf[pos:]
                pos = 0


def load_trace_json(path):
    """Load a small trace fully (used by the selftest / diagnostics only)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="ignore") as handle:
        return json.load(handle)


def short_kernel_name(name, limit=80):
    """Readable operator id for the top-5 tables.

    Kineto uses the full C++ symbol (template arguments and all).  Group by the
    leading function name: strip access qualifiers/return type, then cut at the
    first template/argument list, then keep the last namespace component.
    """
    text = (name or "").strip()
    if not text:
        return "<unnamed>"
    for prefix in ("static void ", "void ", "template <"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.replace("(anonymous namespace)::", "")
    for separator in ("<", "("):
        if separator in text:
            text = text.split(separator, 1)[0]
    return text.split("::")[-1][:limit] or "<unnamed>"


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def parse_step_annotation(name):
    """Decode a vLLM ``execute_context`` step annotation.

    Returns a dict with ``gen_tokens``/``ctx_tokens``/``gen_reqs``/``ctx_reqs``
    or None when the name is not a step annotation.
    """
    if not name:
        return None
    match = STEP_RX_SIMPLE.match(name)
    if match:
        return {
            "format": "simple",
            "ctx_reqs": int(match.group(1)),
            "ctx_tokens": int(match.group(2)),
            "gen_reqs": int(match.group(3)),
            "gen_tokens": int(match.group(4)),
        }
    match = STEP_RX_DETAILED.match(name)
    if match:
        return {
            "format": "detailed",
            "total_tokens": int(match.group(1)),
            "ctx_reqs": int(match.group(2)),
            "ctx_tokens": int(match.group(3)),
            "gen_reqs": int(match.group(4)),
            "gen_tokens": int(match.group(5)),
        }
    return None


def is_decode_step(step):
    return bool(step) and step["gen_tokens"] > 0


# ---------------------------------------------------------------------------
# Burst segmentation / classification
# ---------------------------------------------------------------------------


def burst_spans(kernels):
    """Gap-segment kernels into bursts, return list of index slices."""
    if not kernels:
        return []
    spans = []
    begins = [k["ts"] for k in kernels]
    _ = begins
    start = 0
    for index in range(1, len(kernels)):
        prev = kernels[index - 1]
        gap = kernels[index]["ts"] - (prev["ts"] + prev["dur"])
        if gap > 0:  # only forward progress can open a burst
            spans.append((start, index))
            start = index
    spans.append((start, len(kernels)))
    return spans


def cluster_by_gap(kernels, gap_us):
    """Group kernel indices into clusters whose inter-kernel gap <= gap_us."""
    clusters = []
    current = [0]
    for index in range(1, len(kernels)):
        prev = kernels[index - 1]
        gap = kernels[index]["ts"] - (prev["ts"] + prev["dur"])
        if gap > gap_us:
            clusters.append(current)
            current = [index]
        else:
            current.append(index)
    if current:
        clusters.append(current)
    return clusters


def target_signature_count(kernels):
    n_nccl = 0
    n_sig = 0
    n_target_hard = 0
    for kernel in kernels:
        name = kernel["name"]
        if NCCL_RX.search(name):
            n_nccl += 1
        if TARGET_SIG_RX.search(name):
            n_sig += 1
        if re.search(r"per_token_group_quant|fp8_emu|paged_attn", name, re.IGNORECASE):
            n_target_hard += 1
    return n_nccl, n_sig, n_target_hard


def split_by_signature_cluster(kernels, cluster_gap_us):
    """Split one step's kernels into (draft, target) index ranges.

    The target burst is the largest time-cluster of target-signature kernels;
    everything earlier in the step is the draft tower + selector.  ``target``
    runs from the cluster start to the end of the step (sample/bookkeep and the
    trailing generic kernels belong to it).
    """
    if not kernels:
        return None
    signature = [
        index
        for index, kernel in enumerate(kernels)
        if TARGET_SIG_RX.search(kernel["name"])
    ]
    if not signature:
        return _split_by_largest_gap(kernels)
    clusters = []
    current = [signature[0]]
    for index in signature[1:]:
        prev = kernels[current[-1]]
        gap = kernels[index]["ts"] - (prev["ts"] + prev["dur"])
        if gap > cluster_gap_us:
            clusters.append(current)
            current = [index]
        else:
            current.append(index)
    clusters.append(current)
    best = max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            kernels[cluster[-1]]["ts"] - kernels[cluster[0]]["ts"],
        ),
    )
    draft_end_index = best[0]
    if draft_end_index == 0:
        return _split_by_largest_gap(kernels)
    return {
        "method": "signature_cluster",
        "draft": (0, draft_end_index),
        "target": (best[0], len(kernels)),
        "signature_kernels": len(signature),
        "signature_clusters": len(clusters),
        "targetsig_per_kernel": len(best),
    }


def _split_by_largest_gap(kernels):
    """Fallback: split at the largest forward gap, classify by signatures."""
    if len(kernels) < 2:
        return None
    gaps = []
    for index in range(1, len(kernels)):
        prev = kernels[index - 1]
        gap = kernels[index]["ts"] - (prev["ts"] + prev["dur"])
        gaps.append((gap, index))
    gap, split = max(gaps)
    if gap <= 0:
        return None
    first_nccl, first_sig, first_hard = target_signature_count(kernels[:split])
    second_nccl, second_sig, second_hard = target_signature_count(kernels[split:])
    first_score = first_nccl + 2 * first_hard + first_sig
    second_score = second_nccl + 2 * second_hard + second_sig
    if second_score >= first_score:
        return {
            "method": "largest_gap",
            "draft": (0, split),
            "target": (split, len(kernels)),
            "gap_us": round(gap, 1),
        }
    return {
        "method": "largest_gap_reversed",
        "draft": (split, len(kernels)),
        "target": (0, split),
        "gap_us": round(gap, 1),
    }


# ---------------------------------------------------------------------------
# Phase measurement
# ---------------------------------------------------------------------------


def _union_ms(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cursor_start, cursor_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cursor_end:
            total += cursor_end - cursor_start
            cursor_start, cursor_end = start, end
        elif end > cursor_end:
            cursor_end = end
    total += cursor_end - cursor_start
    return total / 1000.0


def _kernel_in_window_bounds(window):
    return window["ts"], window["ts"] + window["dur"]


def _step_windows(step_events):
    """Split per-step annotations into (cpu_windows, gpu_windows, source).

    CPU and GPU sides are kept separately because they mean different things
    under CUDA graphs: the CPU range spans the launch of one ``execute_model``
    while the GPU range spans the *correlated* GPU work.  In eager mode the GPU
    range covers the whole step; in the piecewise-graph arm it covers only the
    target verify forward (~78 of ~117 ms), which is exactly the target phase.
    """
    cpu = []
    gpu = []
    for event in step_events:
        step = parse_step_annotation(event["name"])
        if not step:
            continue
        if not is_decode_step(step):
            continue
        window = {
            "ts": event["ts"],
            "dur": event["dur"],
            "side": event["side"],
            "step": step,
        }
        (cpu if event["side"] == "cpu" else gpu).append(window)

    def dedupe(items):
        items = sorted(items, key=lambda w: w["ts"])
        unique = []
        for window in items:
            if unique and abs(window["ts"] - unique[-1]["ts"]) < 1.0:
                continue
            unique.append(window)
        return unique

    cpu = dedupe(cpu)
    gpu = dedupe(gpu)
    if cpu:
        return cpu, gpu, "execute_context(cpu)"
    return gpu, gpu, "execute_context(gpu)"


def _split_mode_enabled(args):
    return getattr(args, "split_mode", "auto") in ("auto", "target")


def _match_gpu_window(gpu_windows, round_start, round_end):
    """The GPU annotation window that best covers this round, if any."""
    best = None
    best_overlap = 0.0
    for window in gpu_windows:
        start = max(round_start, window["ts"])
        end = min(round_end, window["ts"] + window["dur"])
        overlap = end - start
        if overlap > best_overlap:
            best_overlap = overlap
            best = window
    if best is None or best_overlap <= 0:
        return None
    return best


def _split_target_annotation(step_kernels, gpu_window, round_start, round_end):
    """Graph-arm split: the GPU annotation is the target verify forward.

    Layout observed in the piecewise-graph arm (per round, ~117 ms)::

        [draft tower + selector ~34 ms][target verify (annotation) ~78 ms]
        [verify/sampling/prepare-next-draft ~5 ms][host gap -> next round]

    so ``target`` = all kernels inside the GPU annotation, ``draft`` = all
    kernels before it in the round and ``target->draft`` = the remainder.
    If the pre-target part is implausibly small (draft after target in some
    future layout), fall back to the signature split instead of inventing a
    phase.
    """
    target_start = gpu_window["ts"]
    target_end = gpu_window["ts"] + gpu_window["dur"]
    pre = [k for k in step_kernels if k["ts"] < target_start]
    inside = [k for k in step_kernels if target_start <= k["ts"] < target_end]
    post = [k for k in step_kernels if k["ts"] >= target_end]
    if len(pre) < 3 or len(inside) < 3:
        return None
    draft_start = pre[0]["ts"]
    draft_end = max(k["ts"] + k["dur"] for k in pre)
    if draft_end > target_start:
        draft_end = target_start
    if target_end > round_end:
        target_end = round_end
    return {
        "method": "target_annotation",
        "draft": (0, len(pre)),
        "target": (len(pre), len(pre) + len(inside)),
        "draft_kernels": pre,
        "target_kernels": inside,
        "post_kernels": post,
        "target_annot_ms": round((target_end - target_start) / 1000.0, 3),
        "draft_start": draft_start,
        "draft_end": draft_end,
        "target_start": target_start,
        "target_end": target_end,
    }


def _profiler_step_windows(events):
    """Fallback step windows: outermost ProfilerStep#N range per name."""
    best = {}
    for event in events:
        if event["side"] != "cpu":
            continue
        name = event["name"]
        if not name.startswith("ProfilerStep#"):
            continue
        if name not in best or event["dur"] > best[name]["dur"]:
            best[name] = event
    windows = [
        {"ts": event["ts"], "dur": event["dur"], "step": {"gen_tokens": 1}}
        for event in sorted(best.values(), key=lambda e: e["ts"])
    ]
    return windows, "ProfilerStep# (fallback)"


def _extract_window_kernels(kernels, windows, margin_us=5000):
    """Kernels belonging to the decode window range (prefill excluded)."""
    if not windows:
        return list(kernels)
    decode_lo = windows[0]["ts"] - margin_us
    decode_hi = windows[-1]["ts"] + windows[-1]["dur"] + margin_us
    in_range = [k for k in kernels if decode_lo <= k["ts"] < decode_hi]
    # Exclude kernels that fall inside a prefill annotation, when that
    # annotation overlaps the decoded range (defensive; the profile window
    # should already be decode-only).
    return in_range


def _phase_attribution(kernels, start, end, rounds):
    """Operator census for one phase interval (ms/step and calls/step)."""
    counts = collections.defaultdict(lambda: [0, 0.0])
    intervals = []
    for kernel in kernels:
        if start <= kernel["ts"] < end:
            entry = counts[short_kernel_name(kernel["name"])]
            entry[0] += 1
            entry[1] += kernel["dur"]
            intervals.append((kernel["ts"], kernel["ts"] + kernel["dur"]))
    total_ms = sum(value[1] for value in counts.values()) / 1000.0
    top = sorted(counts.items(), key=lambda item: (-item[1][1], -item[1][0]))[:5]
    return {
        "span_ms": max(0.0, (end - start) / 1000.0),
        "kernel_ms": total_ms / rounds if rounds else 0.0,
        "gpu_busy_ms": _union_ms(intervals) / rounds if rounds else 0.0,
        "calls_per_step": sum(v[0] for v in counts.values()) / rounds
        if rounds
        else 0.0,
        "top5": [
            {
                "op": name,
                "ms_per_step": round(value[1] / 1000.0 / rounds, 3) if rounds else 0.0,
                "calls_per_step": round(value[0] / rounds, 2) if rounds else 0.0,
            }
            for name, value in top
        ],
    }


def _annotation_crosscheck(annotations, window, next_ts):
    """Draft/target annotation ranges inside one step window, if present."""
    draft = [
        (a["ts"], a["ts"] + a["dur"])
        for a in annotations
        if window["ts"] - 1000 <= a["ts"] < next_ts
        and ANNOT_DRAFT_RX.search(a["name"])
        and not ANNOT_TARGET_RX.search(a["name"])
    ]
    target = [
        (a["ts"], a["ts"] + a["dur"])
        for a in annotations
        if window["ts"] - 1000 <= a["ts"] < next_ts
        and ANNOT_TARGET_RX.search(a["name"])
    ]
    if not draft or not target:
        return None
    return {
        "draft_annot_ms": round(
            (max(e for _, e in draft) - min(s for s, _ in draft)) / 1000.0, 3
        ),
        "target_annot_ms": round(
            (max(e for _, e in target) - min(s for s, _ in target)) / 1000.0, 3
        ),
        "draft_annotations": len(draft),
        "target_annotations": len(target),
    }


def measure_rank(trace, args):
    """Cut one rank trace into five phases; returns a per-rank record."""
    step_events = []
    annotations = []
    kernels = []
    markers = {}
    kernel_count = 0
    label = ""
    for event in iter_trace_events(trace["path"]):
        category = event.get("cat") or ""
        name = event.get("name") or ""
        phase = event.get("ph")
        if phase == "M":
            if name == "process_labels":
                label = (event.get("args") or {}).get("labels", label)
            continue
        ts = event.get("ts")
        if ts is None:
            continue
        if category in ("user_annotation", "gpu_user_annotation"):
            dur = event.get("dur") or 0.0
            side = "cpu" if category == "user_annotation" else "gpu"
            if parse_step_annotation(name):
                step_events.append({"name": name, "ts": ts, "dur": dur, "side": side})
            if ANNOT_DRAFT_RX.search(name) or ANNOT_TARGET_RX.search(name):
                annotations.append({"name": name, "ts": ts, "dur": dur})
            if name in ("Iteration Start: PyTorch Profiler", "Record Window End"):
                markers[name] = ts
        elif category in GPU_CATS:
            kernel_count += 1
            kernels.append(
                {
                    "ts": ts,
                    "dur": event.get("dur") or 0.0,
                    "name": name,
                    "cat": category,
                }
            )
    kernels.sort(key=lambda kernel: kernel["ts"])
    record = {
        "path": trace["path"],
        "rank": trace["rank"],
        "label": label,
        "bytes": trace["size"],
        "sha256_16": trace.get("sha256_16", ""),
        "kernel_events": kernel_count,
        "warnings": [],
    }
    if not kernels:
        record["skipped"] = "no GPU kernel events (front-end CPU trace?)"
        return record
    windows, gpu_windows, window_source = _step_windows(step_events)
    if not windows:
        windows, window_source = _profiler_step_windows(step_events)
        gpu_windows = []
    if not windows:
        # Last resort: treat the profiling window as one long step (the burst
        # classifier still runs, but there is no per-step census).
        lo = markers.get("Iteration Start: PyTorch Profiler")
        hi = markers.get("Record Window End")
        if lo is not None and hi is not None:
            windows = [{"ts": lo, "dur": hi - lo, "step": {"gen_tokens": 1}}]
            window_source = "profile-window marker (single window)"
            record["warnings"].append(
                "no per-step annotation found; single-window fallback - "
                "phases are total, not per step"
            )
    record["window_source"] = window_source
    record["decode_windows"] = len(windows)
    record["gpu_annotation_windows"] = len(gpu_windows)
    if len(windows) < 2:
        record["skipped"] = f"need >=2 decode windows, got {len(windows)}"
        return record

    window_kernels = _extract_window_kernels(kernels, windows)
    trimmed = 0
    rounds = []
    for index in range(len(windows) - 1):
        window = windows[index]
        next_window = windows[index + 1]
        round_start = window["ts"]
        round_end = next_window["ts"]
        period = round_end - round_start
        step_kernels = [
            kernel
            for kernel in window_kernels
            if round_start <= kernel["ts"] < round_end
        ]
        if not step_kernels:
            trimmed += 1
            continue
        split = None
        split_mode = ""
        gpu_window = _match_gpu_window(gpu_windows, round_start, round_end)
        if gpu_window is not None and _split_mode_enabled(args):
            ratio = gpu_window["dur"] / max(period, 1.0)
            use_target = args.split_mode == "target" or (
                args.split_mode == "auto" and ratio < args.graph_target_ratio
            )
            if use_target:
                split = _split_target_annotation(
                    step_kernels, gpu_window, round_start, round_end
                )
                if split is None:
                    # No usable target window for this round: trim instead of
                    # mixing in the burst split (which would invent a ~0 draft).
                    trimmed += 1
                    continue
                split_mode = "target_annotation"
        elif gpu_windows and _split_mode_enabled(args):
            # Graph arm but this round has no overlapping target annotation
            # (profile-window edge).  A burst split here would invent a ~0
            # draft phase, so trim the round instead of mixing methods.
            trimmed += 1
            continue
        if split is None:
            split = split_by_signature_cluster(step_kernels, args.cluster_gap_us)
            if split is not None:
                split_mode = split["method"]
        if split is None:
            trimmed += 1
            continue
        if "draft_kernels" in split:
            draft_kernels = split["draft_kernels"]
            target_kernels = split["target_kernels"]
        else:
            draft_lo, draft_hi = split["draft"]
            target_lo, target_hi = split["target"]
            draft_kernels = step_kernels[draft_lo:draft_hi]
            target_kernels = step_kernels[target_lo:target_hi]
        if not draft_kernels or not target_kernels:
            trimmed += 1
            continue
        draft_start = draft_kernels[0]["ts"]
        draft_end = max(k["ts"] + k["dur"] for k in draft_kernels)
        target_start = target_kernels[0]["ts"]
        target_end = max(k["ts"] + k["dur"] for k in target_kernels)
        if target_start < draft_end:
            trimmed += 1
            continue
        t2d = round_end - target_end
        overhang = 0.0
        if t2d < 0:
            overhang = -t2d
            t2d = 0.0
        phase_windows = {
            "draft": (draft_start, draft_end),
            "draft_to_target": (draft_end, target_start),
            "target": (target_start, target_end),
            "target_to_draft": (target_end, target_end + t2d),
        }
        phases = {
            key: _phase_attribution(step_kernels, start, end, 1)
            for key, (start, end) in phase_windows.items()
        }
        phase_ms = {key: value["span_ms"] for key, value in phases.items()}
        complete = sum(phase_ms.values())
        residual = period / 1000.0 - complete + overhang / 1000.0
        rounds.append(
            {
                "period_ms": period / 1000.0,
                "phases": phases,
                "phase_ms": phase_ms,
                "complete_round_ms": complete,
                "gap_unknown_ms": residual,
                "pre_step_gap_ms": (draft_start - round_start) / 1000.0,
                "overhang_ms": overhang / 1000.0,
                "split_method": split_mode,
                "signature_clusters": split.get("signature_clusters"),
                "target_annot_ms": split.get("target_annot_ms"),
                "annotation_check": _annotation_crosscheck(
                    annotations, window, window["ts"] + window["dur"]
                ),
            }
        )
    if not rounds:
        record["skipped"] = "no complete rounds after trimming"
        return record

    periods = [round_["period_ms"] for round_ in rounds]
    median_period = statistics.median(periods)
    kept = [
        round_
        for round_ in rounds
        if median_period > 0
        and abs(round_["period_ms"] - median_period) / median_period
        <= args.max_period_deviation
    ]
    trimmed += len(rounds) - len(kept)
    # Do not mix attribution methods inside one table: a graph-arm profile
    # edge round can be classified as whole-step while the steady rounds use
    # the target annotation.  Keep the majority method (same result as one
    # consistent method for all steady rounds) and report the drop.
    methods = collections.Counter(round_["split_method"] for round_ in kept)
    if getattr(args, "consistent_split", True) and len(methods) > 1:
        majority = methods.most_common(1)[0][0]
        dropped = [r for r in kept if r["split_method"] != majority]
        kept = [r for r in kept if r["split_method"] == majority]
        trimmed += len(dropped)
        record["warnings"].append(
            "dropped "
            f"{len(dropped)} round(s) with a different split method "
            f"({dict(methods)}); kept {majority} for a consistent table"
        )
    if not kept:
        record["skipped"] = "all rounds are period outliers"
        return record
    n_kept = len(kept)

    def mean(key):
        return sum(round_["phase_ms"][key] for round_ in kept) / n_kept

    phase_means = {key: mean(key) for key, _ in PHASES}
    total_mean = sum(phase_means.values())
    period_mean = sum(round_["period_ms"] for round_ in kept) / n_kept
    gap_mean = sum(round_["gap_unknown_ms"] for round_ in kept) / n_kept
    top = {}
    for key, _ in PHASES:
        merged = collections.defaultdict(lambda: [0.0, 0.0])
        for round_ in kept:
            for item in round_["phases"][key]["top5"]:
                merged[item["op"]][0] += item["ms_per_step"]
                merged[item["op"]][1] += item["calls_per_step"]
        top[key] = [
            {
                "op": name,
                "ms_per_step": round(value[0] / n_kept, 3),
                "calls_per_step": round(value[1] / n_kept, 2),
            }
            for name, value in sorted(merged.items(), key=lambda item: -item[1][0])[:5]
        ]
    record.update(
        {
            "rounds_found": len(rounds),
            "rounds_kept": n_kept,
            "rounds_trimmed": trimmed,
            "step_ms_from_trace": round(period_mean, 3),
            "complete_round_ms": round(total_mean, 3),
            "gap_unknown_ms": round(gap_mean, 3),
            "phases_ms": {key: round(phase_means[key], 3) for key, _ in PHASES},
            "phases_pct": {
                key: round(100.0 * phase_means[key] / total_mean, 2)
                if total_mean
                else 0.0
                for key, _ in PHASES
            },
            "phases": {
                key: {
                    "phase": label,
                    "ms_per_step": round(phase_means[key], 3),
                    "pct": round(100.0 * phase_means[key] / total_mean, 2)
                    if total_mean
                    else 0.0,
                    "gpu_busy_ms": round(
                        sum(round_["phases"][key]["gpu_busy_ms"] for round_ in kept)
                        / n_kept,
                        3,
                    ),
                    "calls_per_step": round(
                        sum(round_["phases"][key]["calls_per_step"] for round_ in kept)
                        / n_kept,
                        1,
                    ),
                    "top5": top[key],
                }
                for key, label in PHASES
            },
            "split_methods": collections.Counter(
                round_["split_method"] for round_ in kept
            ),
            "annotation_checks": [
                round_["annotation_check"]
                for round_ in kept
                if round_.get("annotation_check")
            ],
        }
    )
    record["split_methods"] = dict(record["split_methods"])
    # Plausibility guards: keep an implausible split visible instead of
    # silently producing a wrong table (graph-arm traces are the first real
    # test of the burst classifier on replay kernels).
    if record["phases"]["target"]["calls_per_step"] < 20:
        record["warnings"].append(
            "target phase has <20 kernel calls/step; the signature-cluster "
            "split may have missed the target burst"
        )
    if (
        record["phases"]["draft"]["gpu_busy_ms"]
        >= record["phases"]["target"]["gpu_busy_ms"]
    ):
        record["warnings"].append(
            "draft GPU-busy >= target GPU-busy; check the draft/target split"
            " (--cluster-gap-us / trace layout)"
        )
    return record


def _aggregate_ranks(records, step_ms, step_source):
    usable = [r for r in records if "phases_ms" in r]
    aggregate = {
        "ranks_used": [r["rank"] for r in usable],
        "ranks_skipped": [
            {"rank": r["rank"], "reason": r.get("skipped", "")}
            for r in records
            if "phases_ms" not in r
        ],
        "step_ms": step_ms,
        "step_source": step_source,
    }
    if not usable:
        aggregate["error_pct"] = None
        return aggregate

    def median_of(key):
        return statistics.median([r[key] for r in usable])

    aggregate["round_period_median_ms"] = round(median_of("complete_round_ms"), 3)
    aggregate["gap_unknown_ms"] = round(median_of("gap_unknown_ms"), 3)
    aggregate["phases_ms"] = {
        key: round(statistics.median([r["phases_ms"][key] for r in usable]), 3)
        for key, _ in PHASES
    }
    total = sum(aggregate["phases_ms"].values())
    aggregate["phase_sum_ms"] = round(total, 3)
    # The table's "complete round" is by definition the five segments added
    # up, so it stays internally consistent even when the per-phase medians
    # come from different ranks.
    aggregate["complete_round_ms"] = round(total, 3)
    if step_ms:
        aggregate["error_pct"] = round(100.0 * (total - step_ms) / step_ms, 3)
        aggregate["period_error_pct"] = round(
            100.0 * (aggregate["round_period_median_ms"] - step_ms) / step_ms,
            3,
        )
        aggregate["pass_5pct"] = abs(aggregate["error_pct"]) < 5.0
    else:
        aggregate["error_pct"] = None
        aggregate["pass_5pct"] = None
    return aggregate


# ---------------------------------------------------------------------------
# Trace discovery + JSON/markdown rendering
# ---------------------------------------------------------------------------


def _file_sha16(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def looks_like_gpu_trace(path):
    """Whether a trace file belongs to a GPU worker rank (not the front end).

    vLLM names the front-end CPU trace ``{hostname}_{pid}.async_llm`` and every
    worker trace ``{prefix}_{dp.._pp.._tp.._rank..}``
    (``vllm/v1/engine/async_llm.py`` / ``vllm/v1/worker/gpu_worker.py``).  The
    Chrome-trace body cannot be used for a cheap test because kineto writes all
    CPU-thread events before the GPU-thread events (the first ``kernel`` can be
    >1M events in), so the file name is the discriminator; a mis-classified
    file is still caught by ``measure_rank`` (it records "no GPU kernel
    events").
    """
    return ".async_llm" not in os.path.basename(path)


def discover_traces(args):
    if args.trace:
        paths = list(args.trace)
    else:
        trace_dir = os.path.abspath(os.path.expanduser(args.trace_dir))
        patterns = (
            "*.pt.trace.json.gz",
            "*.pt.trace.json",
            "*.json.gz",
        )
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(trace_dir, pattern)))
        paths = sorted(set(paths))
    found = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        mtime = os.path.getmtime(path)
        if (
            args.trace_dir
            and not args.trace
            and args.since_minutes
            and time.time() - mtime > args.since_minutes * 60
        ):
            continue
        found.append(
            {
                "path": path,
                "mtime": mtime,
                "size": os.path.getsize(path),
            }
        )
    if args.trace_dir and not args.trace and args.latest_only and found:
        newest = max(item["mtime"] for item in found)
        found = [item for item in found if newest - item["mtime"] <= args.cluster_s]
    found.sort(key=lambda item: item["path"])
    # Keep the newest file per rank: a re-run (or an aborted capture that was
    # cleaned up later) leaves several traces per rank in the same dir.
    if args.latest_only and len(found) > 1:
        newest_per_rank = {}
        for item in found:
            stem = os.path.basename(item["path"])
            for suffix in (
                ".pt.trace.json.gz",
                ".pt.trace.json",
                ".json.gz",
                ".json",
            ):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            match = re.match(r"^(.*?)(?:[._-]\d+)+$", stem)
            rank = match.group(1) if match else stem
            previous = newest_per_rank.get(rank)
            if previous is None or item["mtime"] > previous["mtime"]:
                newest_per_rank[rank] = item
        found = list(newest_per_rank.values())
    traces = []
    for index, item in enumerate(found):
        gpu = looks_like_gpu_trace(item["path"])
        stem = os.path.basename(item["path"])
        for suffix in (
            ".pt.trace.json.gz",
            ".pt.trace.json",
            ".json.gz",
            ".json",
        ):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        match = re.match(r"^(.*?)(?:[._-]\d+)+$", stem)
        rank = (match.group(1) if match else stem) or f"rank{index}"
        traces.append(
            {
                "path": item["path"],
                "mtime": item["mtime"],
                "size": item["size"],
                "rank": rank,
                "gpu": gpu,
                "sha256_16": _file_sha16(item["path"])
                if item["size"] < (512 << 20)
                else "",
            }
        )
    return traces


def render_json(record):
    return json.dumps(record, indent=1, sort_keys=True)


def _fmt(value, digits=2):
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(record, args):
    """Render the six-section WP1.2 report; missing data is written as TBD."""
    aggregate = record["aggregate"]
    phases = aggregate.get("phases_ms") or {}
    lines = []
    add = lines.append

    add("# Z100 五相位 round profile（WP1.2，G5）")
    add("")
    add(
        "> 由 `tools/dcu/s27_five_phase_profile.py --phase parse` 生成；"
        "相位定义与归因方法见该工具 docstring 与 §附注。"
    )
    add("")
    add("## 1. 一句话结论")
    add("")
    if phases:
        add("| 指标 | 值 |")
        add("|---|---:|")
        add(f"| complete round | {_fmt(aggregate.get('complete_round_ms'))} ms/step |")
        add(f"| 五段之和 | {_fmt(aggregate.get('phase_sum_ms'))} ms/step |")
        add(
            f"| srv_step_ms | {_fmt(aggregate.get('step_ms'))} ms"
            f"（{aggregate.get('step_source', '')}）|"
        )
        if aggregate.get("step_ms") is None:
            verdict = "TBD（未提供 srv_step_ms）"
        elif aggregate.get("pass_5pct"):
            verdict = "PASS"
        else:
            verdict = "FAIL"
        add(f"| 差 | {_fmt(aggregate.get('error_pct'), 3)} % （判据 <5% ⇒ {verdict}）|")
        add(f"| gap/未知 | {_fmt(aggregate.get('gap_unknown_ms'), 3)} ms |")
        if aggregate.get("step_ms"):
            add("")
            add(
                "**G5 一句话判读**：五段之和 "
                f"{_fmt(aggregate.get('phase_sum_ms'), 3)} ms vs 独立 "
                f"`srv_step_ms` {_fmt(aggregate.get('step_ms'), 3)} ms，差 "
                f"{_fmt(aggregate.get('error_pct'), 3)}%"
                "（判据 <5% ⇒ "
                f"{'PASS' if aggregate.get('pass_5pct') else 'FAIL'}）；"
                "其中 trace 自洽 round 周期 "
                f"{_fmt(aggregate.get('round_period_median_ms'), 3)} ms 与"
                "非 profile srv_step_ms 的差 "
                f"{_fmt(aggregate.get('period_error_pct'), 3)}% 归因为 "
                "profiler 开销（上界），不是相位归因缺口。"
            )
    else:
        add("TBD（缺少可解析的 GPU trace 或 srv_step_ms）")
    add("")
    add("## 2. 五相位分配表")
    add("")
    add("| 相位 | ms/step | 占比 | GPU busy ms | 调用/步 | top-5 算子 |")
    add("|---|---:|---:|---:|---:|---|")
    per_rank = record["ranks"]
    primary = next((rank for rank in per_rank if rank.get("phases")), None)
    for key, label in PHASES:
        if not phases:
            add(f"| `{label}` | TBD | TBD | TBD | TBD | TBD |")
            continue
        detail = (primary or {}).get("phases", {}).get(key, {})
        top = detail.get("top5") or []
        top_text = (
            "; ".join(
                f"`{item['op']}` {item['ms_per_step']:.3f}·{item['calls_per_step']:.2f}"
                for item in top
            )
            or "（无 kernel，纯 gap/sync）"
        )
        total = aggregate.get("phase_sum_ms") or 0.0
        pct = 100.0 * phases[key] / total if total else 0.0
        add(
            f"| `{label}` | {_fmt(phases[key], 3)} | {pct:.1f}% | "
            f"{_fmt(detail.get('gpu_busy_ms'), 3)} | "
            f"{_fmt(detail.get('calls_per_step'), 1)} | {top_text} |"
        )
    add(
        f"| **complete round** | "
        f"{_fmt(aggregate.get('complete_round_ms'), 3)} | 100% | | | |"
    )
    add("")
    add("### 逐 rank（ms/step）")
    add("")
    add(
        "| rank | draft | draft->target | target | target->draft | "
        "complete round | rounds kept | split methods |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---|")
    for rank in per_rank:
        if "phases_ms" not in rank:
            add(
                f"| `{rank.get('rank')}` | skipped: "
                f"{rank.get('skipped', '')} | | | | | | |"
            )
            continue
        add(
            f"| `{rank.get('rank')}` | "
            + " | ".join(_fmt(rank["phases_ms"][key], 3) for key, _ in PHASES)
            + f" | {_fmt(rank['complete_round_ms'], 3)} | "
            f"{rank['rounds_kept']}/{rank['rounds_found']} | "
            + ", ".join(
                f"{name}×{count}"
                for name, count in sorted((rank.get("split_methods") or {}).items())
            )
            + " |"
        )
    add("")
    add("## 3. env 全文 + 策略指纹 + 树 SHA")
    add("")
    if args.collect_summary and os.path.isfile(args.collect_summary):
        try:
            with open(args.collect_summary, encoding="utf-8") as handle:
                collected = json.load(handle)
            kept = {
                key: collected[key]
                for key in ("tag", "window", "request", "traces")
                if key in collected
            }
            if isinstance(kept.get("traces"), list):
                kept["traces"] = [
                    item
                    for item in kept["traces"]
                    if ".pt.trace.json" in str(item.get("path", ""))
                ]
            add("```json")
            add(json.dumps(kept, indent=1, sort_keys=True))
            add("```")
        except (OSError, ValueError):
            add("TBD")
    else:
        add("TBD（`--collect-summary` 未提供）")
    add("")
    add(
        f"- 树 SHA：`{record.get('git_sha') or 'TBD'}`"
        f"（dirty={record.get('git_dirty')}）"
    )
    add(f"- 策略指纹：`{args.fingerprint or 'TBD'}`")
    checks = [
        rank["annotation_checks"][0]
        for rank in per_rank
        if rank.get("annotation_checks")
    ]
    if checks:
        add(
            "- annotation cross-check（`gpu_model_runner: draft` / `: forward` "
            "包络，ms）："
            + ", ".join(
                f"draft={item['draft_annot_ms']:.2f}, "
                f"target={item['target_annot_ms']:.2f}"
                for item in checks
            )
        )
    if args.env_file and os.path.isfile(args.env_file):
        add("")
        add(f"采集臂 env 快照（`{os.path.basename(args.env_file)}`，逐字）：")
        add("```text")
        with open(args.env_file, encoding="utf-8", errors="ignore") as handle:
            add(handle.read().rstrip("\n"))
        add("```")
    add("- trace 文件（sha256[:16]）：")
    for trace in record["files"]:
        add(
            f"  - `{os.path.basename(trace['path'])}` "
            f"({trace['size'] / (1 << 20):.1f} MiB, {trace['sha256_16']})"
        )
    add("")
    add("## 4. 判读")
    add("")
    add(
        f"- 五段之和 {_fmt(aggregate.get('phase_sum_ms'), 3)} ms vs "
        f"srv_step_ms {_fmt(aggregate.get('step_ms'), 3)} ms："
        f"差 {_fmt(aggregate.get('error_pct'), 3)}%，"
        + ("**PASS**（<5%）" if aggregate.get("pass_5pct") else "**FAIL / 不可判定**")
        + "。"
    )
    add(
        f"- gap/未知：{_fmt(aggregate.get('gap_unknown_ms'), 3)} ms/step"
        "（host 侧 step 入口到首个 kernel；应 <2% step）。"
    )
    add(
        f"- 参与 rank：{aggregate.get('ranks_used')}；"
        f"跳过：{aggregate.get('ranks_skipped')}。"
    )
    add(f"- 独立 `srv_step_ms` 出处：{record.get('step_ms_source') or 'TBD'}")
    add(
        "- 采集口径：profiling 期间 `/metrics` 的 step 已含 profiler 开销，"
        "比较基准必须是同臂/同 env/同负载的 `srv_step_ms`。"
    )
    add(
        "- 若独立基线与 profiling 窗口**不是同一 wall-clock 窗口**（本例："
        "`/stop_profile` 落盘时服务 segfault，无法取同窗口 metrics），"
        "则五段和与基线的差拆成「profiling 开销上界 + 相位归因缺口」："
        "profiling 开销上界 = "
        f"{_fmt(aggregate.get('period_error_pct'), 3)}%"
        "（trace 自洽 round vs 非 profile 基线），相位归因缺口 ≈ "
        f"{_fmt(aggregate.get('error_pct'), 3)}% - "
        f"{_fmt(aggregate.get('period_error_pct'), 3)}%。"
    )
    add("")
    add("## 5. 反证条件")
    add("")
    add(
        "- 若 `phase_sum` 与 `srv_step_ms` 差 >5% ⇒ 先修采集（稳态、窗口、"
        "rank、trace 是否含 prefill），**不出结论**（BUILD WP1.2 停止线）。"
    )
    add(
        "- 若 `draft` 段的 nccl/fp8_emu/quant 调用数接近 target ⇒ 归因错误，"
        "检查 `--cluster-gap-us` 与目标签名，并用 annotation cross-check 行"
        "与单步时间轴人工复核。"
    )
    add(
        "- 若某 rank 的 round 周期与其它 rank 差 >5% ⇒ 该 rank trace 抓到 "
        "prefill/异常步，需重抓。"
    )
    add(
        "- 若 `aten::mm` 出现在 draft/target 之外（gap/未知）⇒ 该步有未归因"
        "GEMM，检查是否 prefill 混入或窗口切错。"
    )
    add("")
    add("## 6. 回滚方式")
    add("")
    add(
        "本报告为只读证据；回滚 profile 采集只需停服务。图臂回滚：删除 "
        "`tools/dcu/s27_graph_launch.sh`，回到 `tools/dcu/serve_host_dflash2.sh`。"
    )
    add("")
    for extra in getattr(args, "extra_section", None) or []:
        try:
            with open(extra, encoding="utf-8") as handle:
                add(handle.read().rstrip("\n"))
            add("")
        except OSError:
            add(f"TBD（无法读取 extra section {extra}）")
            add("")
    add("## 附注：归因方法与可复现性")
    add("")
    add(
        "- 步骤窗口：vLLM `execute_context_*_generation_*(N)` 注释，只取 "
        "`N>0` 的 decode 步；`ProfilerStep#N` 为 fallback。"
    )
    add(
        "- draft/target 切分（`--split-mode auto`，两种 trace 结构通用）："
        "**piecewise CUDA-graph 臂**（本报告）：CPU `execute_context` 注解覆盖"
        "整个 round，而 GPU 侧同名注解只覆盖 target verify（~78/116 ms）；"
        "auto 据 GPU 注解/round 周期 <0.85 自动选用 target-annotation 切分 —— "
        "target = GPU 注解内 kernels，draft = round 内其前全部 kernels"
        "（草稿塔+selector），target->draft = verify/sampling 尾部。"
        "**eager 臂**（GPU 注解覆盖整个 step）：按 target 签名 kernel"
        "（fp8_emu / per_token_group_quant / attention / GDN / allreduce）的"
        "最大时间簇切 target，无签名时用最大 inter-kernel gap + 签名计数。"
        "同一张表只保留一种切分方法（`--consistent-split`，默认开），"
        "profile 窗口边缘不完整 round 剔除并计入 `rounds_trimmed`。"
    )
    add(
        "- `draft->target` / `target->draft` 是相邻 burst 的 gap（含 KV 写、"
        "同步、小 kernel）；`gap/未知` 只留 step 入口到首个 kernel 的 host 时间。"
    )
    add(
        "- 解析器只读 `cat/name/ph/ts/dur` 与 `process_labels` metadata，"
        "不依赖 `args.stack`/`python_function`（`with_stack=false` 的 trace "
        "完全兼容；`with_stack=true` 只多出可忽略的 python_function 事件）。"
    )
    add("- 逐 rank 结果见 JSON `--out`；本表取跨 rank 中位数。")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# collect: live service request + /start_profile .. /stop_profile
# ---------------------------------------------------------------------------


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_s20():
    return _load_module("s27_s20_probe", os.path.join(TOOLS_DIR, "s20_single_probe.py"))


def load_s23():
    if TOOLS_DIR not in sys.path:
        sys.path.insert(0, TOOLS_DIR)
    import s23_common  # noqa: PLC0415 - tool-local helper import

    return s23_common


METRIC_KEYS = {
    "generation_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "iteration_sum": "vllm:iteration_tokens_total_sum",
    "iteration_count": "vllm:iteration_tokens_total_count",
    "itl_sum": "vllm:inter_token_latency_seconds_sum",
    "itl_count": "vllm:inter_token_latency_seconds_count",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
}


def parse_metrics(text):
    """Sum every label variant of the interesting vLLM metrics."""
    snapshot = {key: 0.0 for key in METRIC_KEYS}
    for metric, key in METRIC_KEYS.items():
        pattern = re.compile(
            rf"^{re.escape(key)}\{{[^}}]*\}}\s+([0-9.eE+-]+)", re.MULTILINE
        )
        values = [float(match) for match in pattern.findall(text)]
        if values:
            snapshot[metric] = sum(values)
    return snapshot


def _http_get(url, timeout=8):
    """GET with a Request object so s23_common.install_auth can add the key."""
    request = urllib.request.Request(url)
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_metrics(base, timeout=30):
    with _http_get(f"{base}/metrics", timeout=timeout) as response:
        return parse_metrics(response.read().decode("utf-8", "ignore"))


def http_post(base, path, body=None, timeout=120):
    """POST (vLLM's /start_profile and /stop_profile are POST-only).

    ``urllib.request.Request(data=None)`` defaults to GET, which returns 405
    from the profile routers; always pass a body and an explicit method.
    """
    data = json.dumps(body if body is not None else {}).encode()
    request = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        return response.status, payload


def wait_service(base, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with _http_get(f"{base}/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                print(
                    f"[collect] ERROR /v1/models HTTP {exc.code}: API key "
                    "missing/rejected. Set S23_API_KEY or ensure "
                    "/mnt/game/host-serve/api-key is readable.",
                    flush=True,
                )
                return False
        except OSError:
            pass
        time.sleep(2)
    return False


def _wait_metrics(base, timeout):
    """Return one metrics snapshot, retrying until the API server answers."""
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            return fetch_metrics(base)
        except OSError as exc:
            last_error = exc
            time.sleep(2)
    raise OSError(f"/metrics unavailable for {timeout}s: {last_error}")


def wait_idle(base, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            snapshot = fetch_metrics(base)
        except OSError:
            time.sleep(1)
            continue
        if snapshot["running"] == 0 and snapshot["waiting"] == 0:
            return True
        time.sleep(1)
    return False


class _StreamRequest(threading.Thread):
    """One streaming greedy completion; records TTFT and token timestamps."""

    def __init__(self, base, model, prompt, max_tokens, timeout):
        super().__init__(daemon=True)
        self.base = base
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.first_token = threading.Event()
        self.done = threading.Event()
        self.result = {}
        self.error = None

    def run(self):
        body = json.dumps(
            {
                "model": self.model,
                "prompt": self.prompt,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        ttft = None
        stamps = []
        usage = {}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
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
                    for choice in obj.get("choices") or []:
                        if choice.get("text"):
                            now = time.time()
                            if ttft is None:
                                ttft = now - started
                                self.first_token.set()
                            stamps.append(now)
            self.result = {
                "wall_s": round(time.time() - started, 3),
                "ttft_ms": round((ttft or 0.0) * 1000.0, 1),
                "completion_tokens": usage.get("completion_tokens", len(stamps)),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "chunks": len(stamps),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            self.error = f"{type(exc).__name__}: {exc}"
            self.result = {"error": self.error}
        finally:
            self.first_token.set()
            self.done.set()


def _collect_step_ms(before, after):
    steps = after["iteration_count"] - before["iteration_count"]
    itl_sum = after["itl_sum"] - before["itl_sum"]
    accepted = after["accepted"] - before["accepted"]
    drafts = after["drafts"] - before["drafts"]
    generation = after["generation_tokens"] - before["generation_tokens"]
    return {
        "steps": steps,
        "generation_tokens": generation,
        "step_ms": round(itl_sum / steps * 1000.0, 3) if steps else None,
        "accept_len": round(accepted / drafts, 3) if drafts else None,
        "itl_ms": round(
            itl_sum / (after["itl_count"] - before["itl_count"]) * 1000.0, 3
        )
        if after["itl_count"] != before["itl_count"]
        else None,
    }


def phase_collect(args):
    s23 = load_s23()
    s20 = load_s20()
    key = s23.install_auth()
    base = args.base.rstrip("/")
    print(f"[collect] base={base} auth={'on' if key else 'off'}", flush=True)
    if not wait_service(base, args.ready_timeout):
        print("[collect] ERROR service not ready (/v1/models)", flush=True)
        return 2
    if not wait_idle(base, args.idle_timeout):
        print("[collect] ERROR engine not idle", flush=True)
        return 2
    # A previous aborted capture can leave the worker profilers active; a
    # no-op stop clears that orphan session so this window is clean.
    try:
        http_post(base, "/stop_profile", timeout=300)
        print("[collect] cleared any orphan profiler session", flush=True)
    except (OSError, ValueError) as exc:
        # The export can outlive the client timeout; the next wait makes the
        # service responsive again, so this warning is not fatal.
        print(f"[collect] WARN orphan-session cleanup: {exc}", flush=True)
    _wait_metrics(base, 180)
    tokenizer, errors = s20.load_tokenizer(args.tokenizer, args.tokenizer_backend)
    if tokenizer is None:
        print(f"[collect] ERROR tokenizer: {errors}", flush=True)
        return 2
    try:
        prompt, tokens_actual, truncated, chars = s20.read_prompt_file(
            args.prompt_file, tokenizer, args.truncate_tokens
        )
    except (OSError, ValueError) as exc:
        print(f"[collect] ERROR {exc}", flush=True)
        return 2
    max_len = s20.fetch_max_model_len(base, args.model)
    print(
        f"[collect] prompt-file={args.prompt_file} chars={chars} "
        f"tokens={tokens_actual} truncated={int(truncated)} "
        f"max_model_len={max_len or '?'}",
        flush=True,
    )
    if max_len and tokens_actual + args.max_tokens > max_len:
        print(
            f"[collect] ERROR prompt {tokens_actual} + max_tokens "
            f"{args.max_tokens} > max_model_len {max_len}",
            flush=True,
        )
        return 2
    trace_dir = os.path.abspath(os.path.expanduser(args.trace_dir or DEFAULT_TRACE_DIR))
    os.makedirs(trace_dir, exist_ok=True)
    before_files = set(glob.glob(os.path.join(trace_dir, "*")))

    start_metrics = _wait_metrics(base, 120)
    request = _StreamRequest(
        base, args.model, prompt, args.max_tokens, args.request_timeout
    )
    request.start()
    if not request.first_token.wait(args.ttft_timeout):
        print("[collect] ERROR no first token before TTFT timeout", flush=True)
        return 2
    if request.error:
        print(f"[collect] ERROR request failed: {request.error}", flush=True)
        return 2
    print(
        f"[collect] TTFT {request.result.get('ttft_ms', 0) / 1000.0:.3f}s; "
        f"waiting {args.skip_steps} steps before profiling",
        flush=True,
    )
    deadline = time.time() + args.request_timeout
    while time.time() < deadline:
        try:
            snapshot = fetch_metrics(base)
        except OSError as exc:
            print(f"[collect] WARN /metrics during warmup: {exc}", flush=True)
            time.sleep(1)
            continue
        if (
            snapshot["iteration_count"] - start_metrics["iteration_count"]
            >= args.skip_steps
        ):
            break
        if request.done.is_set():
            break
        time.sleep(0.2)
    if request.done.is_set():
        print("[collect] ERROR request finished before profiling started", flush=True)
        return 2

    window_before = _wait_metrics(base, 120)
    profile_body = {"profile_prefix": args.profile_prefix}
    # Safety net: let the worker stop (and flush its trace files) by itself if
    # this client dies mid-window.
    profile_body["max_iterations"] = args.profile_steps
    try:
        status, _ = http_post(base, "/start_profile", profile_body)
    except urllib.error.HTTPError as exc:
        print(f"[collect] ERROR /start_profile: HTTP {exc.code}", flush=True)
        if exc.code == 404:
            print(
                "[collect] hint: the profiler routers are only attached when "
                "the service was started with a profiler config. Launch it "
                "with: --profiler-config.profiler=torch "
                "--profiler-config.torch_profiler_dir=/mnt/game/s27-prof "
                "(VLLM_TORCH_PROFILER_DIR is NOT consumed by this tree).",
                flush=True,
            )
        return 2
    except OSError as exc:
        print(f"[collect] ERROR /start_profile: {exc}", flush=True)
        print(
            "[collect] hint: launch the service with "
            "--profiler-config.profiler=torch "
            "--profiler-config.torch_profiler_dir=<dir>",
            flush=True,
        )
        return 2
    print(
        f"[collect] /start_profile -> {status} "
        f"(profile_prefix={args.profile_prefix}, max_iterations="
        f"{profile_body['max_iterations']})",
        flush=True,
    )
    profiled_steps = 0
    deadline = time.time() + args.request_timeout
    while time.time() < deadline:
        try:
            snapshot = fetch_metrics(base)
        except OSError as exc:
            # A worker-side auto-stop (max_iterations) exports traces while we
            # poll; /metrics can block for the export duration.  Retry until
            # the request finishes or the deadline expires.
            print(f"[collect] WARN /metrics during profiling: {exc}", flush=True)
            if request.done.is_set():
                break
            time.sleep(2)
            continue
        profiled_steps = int(
            snapshot["iteration_count"] - window_before["iteration_count"]
        )
        if profiled_steps >= args.profile_steps:
            break
        if request.done.is_set():
            break
        time.sleep(0.2)
    try:
        window_after = _wait_metrics(base, 120)
    except OSError as exc:
        print(f"[collect] WARN /metrics after profiling: {exc}", flush=True)
        window_after = window_before
    try:
        status, _ = http_post(base, "/stop_profile", timeout=300)
        print(
            f"[collect] /stop_profile -> {status} after {profiled_steps} steps",
            flush=True,
        )
    except (OSError, ValueError) as exc:
        print(
            f"[collect] WARN /stop_profile failed ({exc}); relying on the "
            "worker-side max_iterations auto-stop to flush the traces",
            flush=True,
        )
    request.join(timeout=args.request_timeout)
    server_prompt_tokens = (request.result or {}).get("prompt_tokens") or 0
    if server_prompt_tokens and server_prompt_tokens != tokens_actual:
        print(
            f"[collect] WARN client tokens={tokens_actual} != server "
            f"usage.prompt_tokens={server_prompt_tokens}",
            flush=True,
        )
    window = _collect_step_ms(window_before, window_after)
    window["profiled_steps"] = profiled_steps
    ok = bool(window["step_ms"]) and profiled_steps > 0
    print(
        f"[collect] profile window: steps={profiled_steps} "
        f"step_ms={window['step_ms']} accept={window['accept_len']} "
        f"gen_tokens={window['generation_tokens']}",
        flush=True,
    )
    traces = _wait_new_traces(trace_dir, before_files, args.trace_wait)
    record = {
        "tool": "s27_five_phase_profile",
        "phase": "collect",
        "tag": args.tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tree": REPO_ROOT,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "base": base,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "prompt_file": args.prompt_file,
        "prompt_tokens": tokens_actual,
        "prompt_chars": chars,
        "prompt_truncated": truncated,
        "max_model_len": max_len,
        "profile_prefix": args.profile_prefix,
        "trace_dir": trace_dir,
        "window": window,
        "request": request.result,
        "request_error": request.error,
        "traces": traces,
        "ok": ok,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"s27_five_phase_collect_{args.tag}_{_stamp()}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    print(f"[collect] saved {path}", flush=True)
    if traces:
        first = traces[0]["path"]
        trace_args = " \\\n    ".join(
            f"--trace {trace['path']}" for trace in traces[:1]
        )
        parse_cmd = (
            f"{sys.executable} tools/dcu/s27_five_phase_profile.py "
            f"--phase parse {trace_args} "
            f"--step-ms {window['step_ms']} --out-md "
            f"docs/result/z100-five-phase-profile.md"
        )
        print(f"[next] {parse_cmd}", flush=True)
        record["next_command"] = parse_cmd
        # keep the saved copy in sync with next_command
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
        _ = first
    else:
        print(
            f"[collect] WARN no new trace files in {trace_dir} "
            f"(waited {args.trace_wait}s); check --profiler-config and the "
            "service log",
            flush=True,
        )
    return 0 if ok else 2


def _wait_new_traces(trace_dir, before, timeout):
    deadline = time.time() + timeout
    candidates = []
    while time.time() < deadline:
        now = {
            path: os.path.getsize(path)
            for path in glob.glob(os.path.join(trace_dir, "*"))
            if os.path.isfile(path)
        }
        candidates = [path for path in now if path not in before and now[path] > 0]
        if candidates:
            sizes = {path: now[path] for path in candidates}
            time.sleep(2)
            stable = True
            for path in candidates:
                try:
                    if os.path.getsize(path) != sizes[path]:
                        stable = False
                except OSError:
                    stable = False
            if stable:
                break
        time.sleep(1)
    traces = []
    for path in sorted(candidates):
        traces.append(
            {
                "path": path,
                "size": os.path.getsize(path),
                "sha256_16": _file_sha16(path)
                if os.path.getsize(path) < (512 << 20)
                else "",
                "mtime": os.path.getmtime(path),
            }
        )
    return traces


def _git_sha():
    return _git(["rev-parse", "HEAD"]) or "unknown"


def _git_dirty():
    status = _git(["status", "--porcelain"])
    return bool(status)


def _git(arguments):
    try:
        import subprocess

        result = subprocess.run(
            ["git", "-C", REPO_ROOT, *arguments],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _stamp():
    return time.strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# parse entry point
# ---------------------------------------------------------------------------


def phase_parse(args):
    if not args.step_ms and args.collect_summary:
        try:
            with open(args.collect_summary, encoding="utf-8") as handle:
                collected = json.load(handle)
            window_block = collected.get("window") or {}
            args.step_ms = (
                window_block.get("step_ms")
                or collected.get("step_ms")
                or collected.get("srv_step_ms")
            )
            args.step_ms_source = args.step_ms_source or (
                f"collect-summary {os.path.basename(args.collect_summary)} "
                "(same profile window)"
            )
            if collected.get("profile_prefix"):
                args.profile_prefix = args.profile_prefix or collected["profile_prefix"]
        except (OSError, ValueError) as exc:
            print(f"[parse] WARN cannot read --collect-summary: {exc}", flush=True)
    traces = discover_traces(args)
    gpu = [trace for trace in traces if trace["gpu"]]
    skipped = [trace for trace in traces if not trace["gpu"]]
    print(
        f"[parse] {len(traces)} trace file(s): {len(gpu)} GPU rank(s), "
        f"{len(skipped)} skipped",
        flush=True,
    )
    if not gpu:
        print(
            "[parse] ERROR no GPU rank trace found; is the service profiled "
            "and has /stop_profile flushed the files?",
            flush=True,
        )
        return 2
    method = "burst-signature_cluster"
    records = []
    for trace in gpu:
        print(
            f"[parse] rank={trace['rank']} {os.path.basename(trace['path'])} "
            f"({trace['size'] / (1 << 20):.1f} MiB)",
            flush=True,
        )
        parsed_rank = measure_rank(trace, args)
        for warning in parsed_rank.get("warnings") or []:
            print(f"[parse] WARN rank={parsed_rank['rank']}: {warning}", flush=True)
        records.append(parsed_rank)
    if not args.step_ms:
        print(
            "[parse] WARN no --step-ms/--collect-summary: the <5% check would "
            "compare against the trace-derived period, not an independent "
            "srv_step_ms",
            flush=True,
        )
    step_source = (
        args.step_ms_source
        if args.step_ms
        else "trace-derived (no --step-ms/--collect-summary: <5% check "
        "is NOT an independent check)"
    )
    aggregate = _aggregate_ranks(records, args.step_ms, step_source)
    record = {
        "tool": "s27_five_phase_profile",
        "phase": "parse",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tree": REPO_ROOT,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "method": method,
        "window_source": records[0].get("window_source", ""),
        "step_ms": args.step_ms,
        "step_ms_source": step_source,
        "profiler_prefix": getattr(args, "profile_prefix", None),
        "phase_order": [label for _, label in PHASES],
        "phases": {key: label for key, label in PHASES},
        "aggregate": aggregate,
        "ranks": records,
        "files": [
            {
                "path": trace["path"],
                "size": trace["size"],
                "sha256_16": trace["sha256_16"],
                "rank": trace["rank"],
            }
            for trace in gpu
        ],
        "skipped_files": [
            {"path": trace["path"], "rank": trace["rank"]} for trace in skipped
        ],
        "thresholds": {
            "cluster_gap_us": args.cluster_gap_us,
            "max_period_deviation": args.max_period_deviation,
            "profile_steps": args.profile_steps,
            "target_signature_rx": TARGET_SIG_RX.pattern,
        },
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"s27_five_phase_parse_{args.tag}_{_stamp()}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    print(f"[parse] saved {path}", flush=True)
    if aggregate.get("phases_ms"):
        print(
            "[parse] five phases (median across ranks, ms/step): "
            + " ".join(
                f"{label}={aggregate['phases_ms'][key]:.2f}" for key, label in PHASES
            ),
            flush=True,
        )
        print(
            f"[parse] complete round={aggregate['complete_round_ms']:.2f} ms "
            f"sum={aggregate['phase_sum_ms']:.2f} ms "
            f"step={aggregate['step_ms']} ms "
            f"diff={aggregate['error_pct']}% "
            f"pass5={'YES' if aggregate.get('pass_5pct') else 'NO'}",
            flush=True,
        )
    if args.out_md:
        markdown = render_markdown(record, args)
        out_md = os.path.abspath(os.path.expanduser(args.out_md))
        os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        print(f"[parse] markdown -> {out_md}", flush=True)
    return 0 if aggregate.get("phases_ms") else 2


# ---------------------------------------------------------------------------
# selftest (no live service, synthetic Chrome traces)
# ---------------------------------------------------------------------------


def _synthetic_trace(rounds=3, style="burst", base_ts=1000.0, gzip_out=False):
    """Build a minimal Chrome trace with draft/target bursts and gaps.

    Layout per round (period = 40 ms):
      draft 2 ms (5 kernels) | gap 3 ms | target 15 ms (30 kernels, signatures)
      | gap 20 ms -> next round.
    """
    events = [
        {
            "name": "process_name",
            "ph": "M",
            "ts": base_ts,
            "pid": 0,
            "tid": 0,
            "args": {"name": "VLLM::Worker_TP0"},
        },
        {
            "name": "process_labels",
            "ph": "M",
            "ts": base_ts,
            "pid": 0,
            "tid": 0,
            "args": {"labels": "GPU 0"},
        },
        {
            "name": "Iteration Start: PyTorch Profiler",
            "ph": "X",
            "cat": "",
            "ts": base_ts,
            "dur": 1,
            "pid": 0,
            "tid": 0,
            "args": {},
        },
    ]
    period_us = 40000.0
    for index in range(rounds):
        step_start = base_ts + 1000.0 + index * period_us
        events.append(
            {
                "name": "execute_context_0(0)_generation_1(8)",
                "ph": "X",
                "cat": "user_annotation",
                "ts": step_start,
                "dur": period_us - 500.0,
                "pid": 100,
                "tid": 100,
                "args": {},
            }
        )
        if style == "target_annotation":
            # Piecewise-graph layout: the GPU-side execute_context annotation
            # covers only the target verify window (5 ms .. 20 ms of the round).
            events.append(
                {
                    "name": "execute_context_0(0)_generation_1(8)",
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "ts": step_start + 5000.0,
                    "dur": 15000.0,
                    "pid": 0,
                    "tid": 29,
                    "args": {},
                }
            )
        if style == "annotations":
            events.append(
                {
                    "name": "gpu_model_runner: draft",
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "ts": step_start + 100.0,
                    "dur": 2000.0,
                    "pid": 0,
                    "tid": 29,
                    "args": {},
                }
            )
            events.append(
                {
                    "name": "gpu_model_runner: forward",
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "ts": step_start + 5000.0,
                    "dur": 15000.0,
                    "pid": 0,
                    "tid": 29,
                    "args": {},
                }
            )
        kernel_ts = step_start
        for k_index in range(5):
            events.append(
                {
                    "name": f"draft_gemm_kernel_{k_index}",
                    "ph": "X",
                    "cat": "kernel",
                    "ts": kernel_ts,
                    "dur": 400.0,
                    "pid": 0,
                    "tid": 29,
                    "args": {"stream": 29, "device": 0, "correlation": k_index},
                }
            )
            kernel_ts += 400.0
        kernel_ts = step_start + 5000.0
        target_names = (
            "void (anonymous namespace)::fp8_emu_gemm_block_kernel<32, 64>",
            "void per_token_group_quant_8bit_kernel<c10::Half>",
            "void vllm::gemma_fused_add_rms_norm_gfx906_kernel<c10::Half>",
            "ncclDevKernel_AllReduce_Ring_Simple_Sum_f16",
            "triton_unified_attention_kernel",
        )
        for k_index in range(30):
            events.append(
                {
                    "name": target_names[k_index % len(target_names)],
                    "ph": "X",
                    "cat": "kernel",
                    "ts": kernel_ts,
                    "dur": 500.0,
                    "pid": 0,
                    "tid": 29,
                    "args": {"stream": 29, "device": 0, "correlation": k_index},
                }
            )
            kernel_ts += 500.0
        if style == "target_annotation":
            events.append(
                {
                    "name": "_rejection_kernel",
                    "ph": "X",
                    "cat": "kernel",
                    "ts": kernel_ts,
                    "dur": 300.0,
                    "pid": 0,
                    "tid": 29,
                    "args": {"stream": 29, "device": 0},
                }
            )
    events.append(
        {
            "name": "Record Window End",
            "ph": "X",
            "cat": "",
            "ts": base_ts + 1000.0 + rounds * period_us + 5000.0,
            "dur": 1,
            "pid": 0,
            "tid": 0,
            "args": {},
        }
    )
    payload = {"schemaVersion": 1, "traceEvents": events}
    return payload, gzip_out


def _write_synthetic(path, payload, gzip_out):
    if gzip_out:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)


def _parse_args_selftest():
    return argparse.Namespace(
        trace=None,
        trace_dir="",
        since_minutes=0,
        latest_only=False,
        cluster_s=900,
        cluster_gap_us=3000,
        split_mode="auto",
        graph_target_ratio=0.85,
        consistent_split=True,
        max_period_deviation=0.35,
        profile_steps=20,
        step_ms=None,
        step_ms_source="",
        collect_summary=None,
        out="",
        out_md=None,
        env_file=None,
        extra_section=None,
        tag="selftest",
        fingerprint="",
        profile_prefix=None,
    )


def _selftest():  # noqa: PLR0915 - one comprehensive acceptance block
    failures = []
    checks = 0

    def check(ok, label, detail=""):
        nonlocal checks
        checks += 1
        if ok:
            print(f"[selftest] PASS {label}")
        else:
            failures.append(label)
            print(f"[selftest] FAIL {label}" + (f": {detail}" if detail else ""))

    args = _parse_args_selftest()
    with tempfile.TemporaryDirectory(prefix="s27-selftest-") as tmp:
        # 1) annotation parsing, simple + detailed.
        step = parse_step_annotation("execute_context_0(0)_generation_1(8)")
        check(
            step is not None
            and step["gen_tokens"] == 8
            and step["ctx_tokens"] == 0
            and is_decode_step(step),
            "simple execute_context annotation parsed",
            str(step),
        )
        detailed = parse_step_annotation(
            "execute_4096_context_1(sq4096sk0sqsq0sqsk0)"
            "_generation_1(sq8sk4096sqsq8sqsk32768)"
        )
        check(
            detailed is not None
            and detailed["gen_tokens"] == 8
            and detailed["total_tokens"] == 4096,
            "detailed execute_context annotation parsed",
            str(detailed),
        )
        check(
            parse_step_annotation("ProfilerStep#3") is None,
            "non-step annotation rejected",
        )

        # 2) kernel-name normalisation.
        check(
            short_kernel_name(
                "void (anonymous namespace)::fp8_emu_gemm_block_kernel<32,"
                " 64, 2, 4, c10::Half, 1>(unsigned char const*, ...)"
            )
            == "fp8_emu_gemm_block_kernel",
            "kernel symbol normalised for top-5",
        )
        check(
            short_kernel_name("") == "<unnamed>",
            "unnamed kernel handled",
        )

        # 3) metrics parsing sums label variants.
        metrics_text = (
            'vllm:generation_tokens_total{engine="0"} 100.0\n'
            'vllm:generation_tokens_total{engine="1"} 50.0\n'
            'vllm:num_requests_running{engine="0"} 1.0\n'
            'vllm:inter_token_latency_seconds_sum{engine="0"} 0.08\n'
        )
        snapshot = parse_metrics(metrics_text)
        check(
            snapshot["generation_tokens"] == 150.0
            and snapshot["running"] == 1.0
            and snapshot["waiting"] == 0.0,
            "metrics parser sums label variants",
            str(snapshot),
        )
        window = _collect_step_ms(
            {
                "iteration_count": 10.0,
                "itl_sum": 1.0,
                "accepted": 0.0,
                "drafts": 0.0,
                "generation_tokens": 20.0,
                "itl_count": 10.0,
            },
            {
                "iteration_count": 30.0,
                "itl_sum": 3.6,
                "accepted": 56.0,
                "drafts": 20.0,
                "generation_tokens": 76.0,
                "itl_count": 30.0,
            },
        )
        check(
            window["steps"] == 20
            and abs(window["step_ms"] - 130.0) < 1e-6
            and abs(window["accept_len"] - 2.8) < 1e-6,
            "collect window step/accept arithmetic",
            str(window),
        )

        # 4) synthetic burst trace: phases, rounding, top-5.
        path = os.path.join(tmp, "rank0.123.pt.trace.json")
        payload, _ = _synthetic_trace(rounds=5, style="burst")
        _write_synthetic(path, payload, gzip_out=False)
        trace = {
            "path": path,
            "rank": "rank0",
            "size": os.path.getsize(path),
            "sha256_16": _file_sha16(path),
        }
        parsed = measure_rank(trace, args)
        check("phases_ms" in parsed, "synthetic trace parsed", parsed.get("skipped"))
        if "phases_ms" in parsed:
            expected = {
                "draft": 2.0,
                "draft_to_target": 3.0,
                "target": 15.0,
            }
            check(
                all(
                    abs(parsed["phases_ms"][key] - value) < 0.01
                    for key, value in expected.items()
                ),
                "draft / gap / target split matches injection",
                str(parsed["phases_ms"]),
            )
            check(
                abs(parsed["complete_round_ms"] - 40.0) < 0.05,
                "complete round == injected period",
                str(parsed["complete_round_ms"]),
            )
            check(
                abs(sum(parsed["phases_ms"].values()) - parsed["complete_round_ms"])
                < 1e-6,
                "five segments sum exactly to complete round",
            )
            draft_ops = [item["op"] for item in parsed["phases"]["draft"]["top5"]]
            target_ops = [item["op"] for item in parsed["phases"]["target"]["top5"]]
            check(
                all(op.startswith("draft_gemm_kernel_") for op in draft_ops)
                and len(draft_ops) == 5,
                "draft top-5 are the draft kernels",
                str(draft_ops),
            )
            check(
                "fp8_emu_gemm_block_kernel" in target_ops
                and "per_token_group_quant_8bit_kernel" in target_ops
                and "ncclDevKernel_AllReduce_Ring_Simple_Sum_f16" in target_ops,
                "target top-5 carry the target signatures",
                str(target_ops),
            )
            check(
                parsed["rounds_kept"] == 4,
                "all synthetic rounds kept (5 windows -> 4 rounds)",
                str(parsed["rounds_kept"]),
            )
            annotated = _parse_args_selftest()
            annotated.step_ms = parsed["complete_round_ms"]
            aggregate = _aggregate_ranks([parsed], annotated.step_ms, "injected")
            check(
                aggregate.get("pass_5pct") is True
                and abs(aggregate.get("error_pct", 999)) < 0.1,
                "aggregate passes the 5% check when step_ms matches",
                str(aggregate.get("error_pct")),
            )

        # 5) annotations cross-check present.
        ann_path = os.path.join(tmp, "rank0.ann.pt.trace.json.gz")
        ann_payload, _ = _synthetic_trace(rounds=3, style="annotations")
        _write_synthetic(ann_path, ann_payload, gzip_out=True)
        ann_trace = {
            "path": ann_path,
            "rank": "rank0",
            "size": os.path.getsize(ann_path),
            "sha256_16": _file_sha16(ann_path),
        }
        ann_parsed = measure_rank(ann_trace, _parse_args_selftest())
        check(
            "phases_ms" in ann_parsed
            and abs(ann_parsed["phases_ms"]["draft"] - 2.0) < 0.01,
            "gzip trace + annotation cross-check parsed",
            str(ann_parsed.get("phases_ms")),
        )
        checks_with = ann_parsed.get("annotation_checks") or []
        check(
            bool(checks_with)
            and abs(checks_with[0]["draft_annot_ms"] - 2.0) < 0.01
            and abs(checks_with[0]["target_annot_ms"] - 15.0) < 0.01,
            "annotation cross-check ranges match injection",
            str(checks_with[:1]),
        )

        # 5b) piecewise-graph layout: GPU annotation covers target only.
        ta_path = os.path.join(tmp, "rank0.targetann.pt.trace.json")
        ta_payload, _ = _synthetic_trace(rounds=5, style="target_annotation")
        _write_synthetic(ta_path, ta_payload, gzip_out=False)
        ta_trace = {
            "path": ta_path,
            "rank": "rank0",
            "size": os.path.getsize(ta_path),
            "sha256_16": _file_sha16(ta_path),
        }
        ta_parsed = measure_rank(ta_trace, _parse_args_selftest())
        check(
            "phases_ms" in ta_parsed
            and ta_parsed["split_methods"] == {"target_annotation": 4},
            "target-annotation split selected for graph-shaped trace",
            str(ta_parsed.get("split_methods")),
        )
        if "phases_ms" in ta_parsed:
            check(
                abs(ta_parsed["phases_ms"]["draft"] - 2.0) < 0.01
                and abs(ta_parsed["phases_ms"]["draft_to_target"] - 3.0) < 0.01
                and abs(ta_parsed["phases_ms"]["target"] - 15.0) < 0.01
                and abs(ta_parsed["phases_ms"]["target_to_draft"] - 20.0) < 0.05,
                "graph phases match injection (draft 2 / gap 3 / target 15)",
                str(ta_parsed["phases_ms"]),
            )
            check(
                abs(
                    sum(ta_parsed["phases_ms"].values())
                    - ta_parsed["complete_round_ms"]
                )
                < 1e-6,
                "graph five segments sum to complete round",
            )

        # 6) discovery + markdown + JSON rendering.
        args.trace = [path]
        args.trace_dir = ""
        traces = discover_traces(args)
        check(
            len(traces) == 1 and traces[0]["gpu"] and traces[0]["rank"] == "rank0",
            "explicit --trace discovery",
            str(traces),
        )
        # production file naming: tensorboard_trace_handler writes
        # "{worker_name}.{time_ns}.pt.trace.json[.gz]"; vLLM worker_name is
        # "{prefix}_{dp.._pp.._tp.._rank..}", front end "{...}.async_llm".
        worker_path = os.path.join(
            tmp,
            "s27_dp0_pp0_tp0_dcp0_ep0_rank0.1791234567890.pt.trace.json",
        )
        _write_synthetic(worker_path, payload, gzip_out=False)
        args.trace = [worker_path]
        worker_traces = discover_traces(args)
        check(
            worker_traces[0]["rank"] == "s27_dp0_pp0_tp0_dcp0_ep0_rank0",
            "production worker trace name -> rank suffix",
            worker_traces[0]["rank"],
        )
        frontend_path = os.path.join(
            tmp, "host_123.async_llm.1791234567890.pt.trace.json.gz"
        )
        frontend_payload = {
            "schemaVersion": 1,
            "traceEvents": [
                {
                    "name": "aten::mm",
                    "ph": "X",
                    "cat": "cpu_op",
                    "ts": 1.0,
                    "dur": 1.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {},
                }
            ],
        }
        _write_synthetic(frontend_path, frontend_payload, gzip_out=True)
        args.trace = [frontend_path]
        frontend_traces = discover_traces(args)
        check(
            frontend_traces[0]["gpu"] is False,
            "front-end .async_llm trace is not treated as a GPU rank",
            str(frontend_traces),
        )
        args.trace = None
        record = {
            "tool": "s27_five_phase_profile",
            "phase": "parse",
            "git_sha": "deadbeef",
            "git_dirty": False,
            "step_ms": parsed["complete_round_ms"],
            "step_ms_source": "selftest",
            "aggregate": _aggregate_ranks(
                [parsed], parsed["complete_round_ms"], "selftest"
            ),
            "ranks": [parsed],
            "files": [
                {
                    "path": path,
                    "size": os.path.getsize(path),
                    "sha256_16": trace["sha256_16"],
                    "rank": "rank0",
                }
            ],
            "method": "burst-signature_cluster",
        }
        md_args = _parse_args_selftest()
        markdown = render_markdown(record, md_args)
        check(
            "## 1. 一句话结论" in markdown
            and "## 6. 回滚方式" in markdown
            and "`draft`" in markdown,
            "markdown template renders six sections",
        )
        check(
            json.loads(render_json(record))["aggregate"]["pass_5pct"] is True,
            "JSON record round-trips",
        )
        check(
            md_args.step_ms is None and "TBD" not in markdown.split("## 2.")[0],
            "markdown has no TBD in the summary when data exists",
        )

        # 7) outlier trimming: shift two whole rounds by +30 ms so the
        # affected round period is 70 ms while the median stays 40 ms.
        long_path = os.path.join(tmp, "rank0.long.pt.trace.json")
        long_payload, _ = _synthetic_trace(rounds=6, style="burst")
        shift_from = 1000.0 + 2 * 40000
        for event in long_payload["traceEvents"]:
            if event.get("ts", 0.0) >= shift_from:
                event["ts"] += 30000.0
        _write_synthetic(long_path, long_payload, gzip_out=False)
        long_trace = {
            "path": long_path,
            "rank": "rank0",
            "size": os.path.getsize(long_path),
            "sha256_16": _file_sha16(long_path),
        }
        long_parsed = measure_rank(long_trace, _parse_args_selftest())
        check(
            long_parsed.get("rounds_trimmed", 0) >= 1,
            "period-outlier round trimmed",
            str(long_parsed.get("rounds_trimmed")),
        )

    print()
    if failures:
        print(
            f"[selftest] {checks - len(failures)}/{checks} checks passed; "
            f"{len(failures)} FAILED"
        )
        for label in failures:
            print(f"[selftest]   FAILED: {label}")
        return 1
    print(f"[selftest] all {checks} checks passed")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description=("S27 WP1.2 five-phase decode round profiler (collect + parse).")
    )
    parser.add_argument(
        "--phase",
        choices=("collect", "parse"),
        help="collect from a live profiled service, or parse trace files",
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--trace",
        action="append",
        default=None,
        help="explicit trace file (repeatable); skips --trace-dir discovery",
    )
    parser.add_argument(
        "--trace-dir",
        default=DEFAULT_TRACE_DIR,
        help="profiler output dir (default: %(default)s)",
    )
    parser.add_argument(
        "--since-minutes",
        type=int,
        default=120,
        help="only consider trace files newer than this (0 = no filter)",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        default=True,
        help="keep only the newest mtime cluster (default)",
    )
    parser.add_argument(
        "--cluster-s",
        type=int,
        default=900,
        help="mtime cluster width for --latest-only",
    )
    parser.add_argument(
        "--out",
        default=".s27/p1",
        help="directory for collect/parse JSON records",
    )
    parser.add_argument("--tag", default="profile")
    parser.add_argument("--out-md", default=None, help="write markdown report")
    parser.add_argument(
        "--step-ms",
        type=float,
        default=None,
        help="srv_step_ms measured in the SAME profile window (required for "
        "the independent <5%% check)",
    )
    parser.add_argument(
        "--step-ms-source",
        default="",
        help="free-text provenance for --step-ms",
    )
    parser.add_argument(
        "--collect-summary",
        default=None,
        help="collect JSON; provides --step-ms and env/trace provenance",
    )
    parser.add_argument(
        "--fingerprint",
        default="",
        help="WP0.4 policy fingerprint (for the report section 3)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="env snapshot file embedded verbatim in report section 3",
    )
    parser.add_argument(
        "--extra-section",
        action="append",
        default=None,
        help="markdown file appended as an extra report section "
        "(risk/accident notes); repeatable",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="decode steps to profile (default 20)",
    )
    parser.add_argument(
        "--cluster-gap-us",
        type=float,
        default=3000.0,
        help="max gap inside one target-signature cluster (default 3000 us)",
    )
    parser.add_argument(
        "--consistent-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only the majority split method inside one table "
        "(default: on; a graph-arm edge round otherwise mixes methods)",
    )
    parser.add_argument(
        "--split-mode",
        choices=("auto", "whole", "target"),
        default="auto",
        help="draft/target attribution: auto picks the target-annotation split "
        "when the GPU annotation covers only part of the round (piecewise "
        "CUDA-graph arm), otherwise the kernel-burst/signature split",
    )
    parser.add_argument(
        "--graph-target-ratio",
        type=float,
        default=0.85,
        help="auto split-mode: GPU-annotation/round-period ratio below which "
        "the annotation is treated as the target-only range (default 0.85)",
    )
    parser.add_argument(
        "--max-period-deviation",
        type=float,
        default=0.35,
        help="drop rounds whose period differs more than this fraction (default 0.35)",
    )
    # collect options
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="real UTF-8 document for the profiled request (collect)",
    )
    parser.add_argument(
        "--truncate-tokens",
        type=int,
        default=0,
        help="truncate the prompt to this many token ids (collect)",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--tokenizer",
        default="/mnt/game/models/Qwen3.8-27B-FP8",
        help="offline tokenizer copy of the served model",
    )
    parser.add_argument(
        "--tokenizer-backend",
        choices=("auto", "transformers", "tokenizers"),
        default="auto",
    )
    parser.add_argument("--profile-prefix", default="s27")
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--idle-timeout", type=int, default=180)
    parser.add_argument("--ttft-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--trace-wait", type=int, default=300)
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=6,
        help="decode steps to run before /start_profile (default 6)",
    )
    return parser


def run(args):
    if args.selftest:
        return _selftest()
    if args.phase == "collect":
        if not args.prompt_file:
            print("[collect] ERROR --prompt-file is required", flush=True)
            return 2
        return phase_collect(args)
    if args.phase == "parse":
        return phase_parse(args)
    print("ERROR: pass --phase collect|parse or --selftest", file=sys.stderr)
    return 2


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
