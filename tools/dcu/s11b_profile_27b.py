#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11b (P0): kernel-level profiling of 27B W8A8 INT8 decode on gfx906.

Runs the 27B compressed-tensors W8A8 model offline with TP=2 and profiles
32 greedy decode steps (`ignore_eos=True`).

Why the worker profiler and not `torch.profiler` in this process: with TP=2
the offline `LLM` driver never touches the GPU -- all model execution happens
in `WorkerProc` child processes -- so a profiler owned by the driver would
capture nothing. The engine therefore runs with `profiler_config` set and the
window is opened/closed with `LLM.start_profile()` / `LLM.stop_profile()`;
each worker dumps its own chrome trace into `--out`.

Why `VLLM_WORKER_MULTIPROC_METHOD=spawn` (see tools/dcu/s11b_run.sh): with the
default `fork`, the 27B worker processes die in `init_device()` with
`RuntimeError: CUDA driver initialization failed` (reproduced on cards 0,1 and
2,3; Qwen3-0.6B forks fine). Spawn sidesteps it.

Measurements per run mode (``default`` = FULL_AND_PIECEWISE, ``none`` = no
CUDA graphs, everything else identical):

1. warm-up request (16 tokens) - materializes torch.compile/graph artifacts;
2. sync run - `torch.cuda.synchronize()` around every `engine.step()`:
   host launch span vs GPU-bound step wall, plus the engine's own
   first/last-token timestamps and the NaN (`is_corrupted`) flag;
3. profiled run - prefill outside the window, then exactly 32 decode steps
   inside it.

Usage (inside the container, via tools/dcu/s11b_run.sh).
"""

import argparse
import json
import os
import subprocess
import threading
import time

import torch
from vllm import LLM, SamplingParams

PROMPT = (
    "Please write a short technical explanation of how tensor parallelism "
    "splits a transformer layer across accelerators."
)

# ---------------------------------------------------------------------------
# GEMM shape recorders.
#
# torch.compile inlines the int8 linear path into the compiled graph, so the
# profiler sees the kernels but not the M/N/K that produced them. These
# wrappers are installed in every process (including the spawned workers, which
# re-import this module) and dump one TSV line per GEMM call when
# S11B_SHAPE_DIR is set. They are passive observers: arguments are forwarded
# unchanged.
# ---------------------------------------------------------------------------
_RECORDERS_INSTALLED = False


class _ShapeRecorder:
    def __init__(self, kind: str) -> None:
        out_dir = os.environ.get("S11B_SHAPE_DIR")
        self.path = (
            os.path.join(out_dir, f"shapes_{kind}_pid{os.getpid()}.tsv")
            if out_dir
            else None
        )
        self.buf: list[str] = []

    def add(self, *fields: object) -> None:
        if self.path is None:
            return
        self.buf.append("\t".join(str(f) for f in fields))
        if len(self.buf) >= 64:
            with open(self.path, "a") as fh:
                fh.write("\n".join(self.buf) + "\n")
            self.buf.clear()


def install_shape_recorders() -> None:
    global _RECORDERS_INSTALLED
    if _RECORDERS_INSTALLED or not os.environ.get("S11B_SHAPE_DIR"):
        return
    _RECORDERS_INSTALLED = True

    try:
        from vllm.model_executor.layers import utils as layer_utils

        orig_tm = layer_utils.triton_matmul
        rec_tm = _ShapeRecorder("triton_matmul")

        def triton_matmul_spy(a, b, *args, **kwargs):
            try:
                rec_tm.add(a.shape[0], a.shape[1], b.shape[0], "")
            except Exception:  # noqa: BLE001 - never break the model
                pass
            return orig_tm(a, b, *args, **kwargs)

        layer_utils.triton_matmul = triton_matmul_spy
    except Exception as exc:  # noqa: BLE001
        print(f"[s11b] triton_matmul recorder not installed: {exc}", flush=True)


install_shape_recorders()


def log(msg: str) -> None:
    print(f"[s11b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class HySmiSampler:
    """Sample hy-smi in the background while the profiled loop runs."""

    def __init__(self, interval: float = 3.0) -> None:
        self.interval = interval
        self.samples: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["hy-smi"], capture_output=True, text=True, timeout=30
                )
                self.samples.append(
                    f"--- {time.strftime('%H:%M:%S')} rc={out.returncode}\n"
                    f"{out.stdout}{out.stderr}"
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                self.samples.append(f"--- hy-smi failed: {exc!r}")
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


def build_llm(args: argparse.Namespace) -> LLM:
    compilation_config: dict = {"cudagraph_capture_sizes": [1, 2]}
    if args.mode == "none":
        compilation_config["cudagraph_mode"] = "NONE"
    log(f"compilation_config={compilation_config}")
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": args.out,
        "torch_profiler_activities": ["CPU", "CUDA"],
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_use_gzip": False,
    }
    llm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=2,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.9,
        disable_custom_all_reduce=True,
        skip_mm_profiling=True,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        compilation_config=compilation_config,
        cudagraph_metrics=True,
        profiler_config=profiler_config,
    )
    log("LLM ready")
    return llm


def sampling_params(n_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=n_tokens, ignore_eos=True)


def enqueue(llm: LLM, prompt: str, n_tokens: int) -> str:
    ids = llm.enqueue(prompt, sampling_params(n_tokens), use_tqdm=False)
    return ids[0]


def drain(llm: LLM, cap: int = 400) -> list:
    """Finish everything left in the engine before the next request."""
    finished = []
    for _ in range(cap):
        if not llm.llm_engine.has_unfinished_requests():
            break
        for out in llm.llm_engine.step():
            if getattr(out, "finished", False):
                finished.append(out)
    return finished


def summarize(outputs: list) -> dict:
    """Non-NaN / non-degenerate check plus engine-side per-token timing."""
    if not outputs:
        return {"finished": False}
    out = outputs[-1]
    seq = out.outputs[0]
    stats = out.metrics
    info = {
        "finished": True,
        "finish_reason": str(seq.finish_reason),
        "n_tokens": len(seq.token_ids),
        "distinct_tokens": len(set(seq.token_ids)),
        "text_head": seq.text[:200],
    }
    if stats is not None:
        info.update(
            {
                "is_corrupted": bool(getattr(stats, "is_corrupted", False)),
                "first_token_latency_ms": float(stats.first_token_latency) * 1e3,
                "num_generation_tokens": stats.num_generation_tokens,
            }
        )
        span = stats.last_token_ts - stats.first_token_ts
        if span > 0 and stats.num_generation_tokens > 1:
            info["decode_ms_per_token_engine"] = (
                span / (stats.num_generation_tokens - 1) * 1e3
            )
    return info


def warmup(llm: LLM, steps: int = 16) -> dict:
    rid = enqueue(llm, PROMPT, steps)
    n = 0
    t0 = time.perf_counter()
    while llm.llm_engine.has_unfinished_requests() and n < steps + 8:
        llm.llm_engine.step()
        n += 1
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    tail = summarize(drain(llm)) if llm.llm_engine.has_unfinished_requests() else {}
    log(f"warmup: {n} steps in {wall_ms:.1f} ms ({wall_ms / max(n, 1):.1f} ms/step)")
    return {"request_id": rid, "steps": n, "wall_ms": wall_ms, "tail": tail}


def run_sync(llm: LLM, steps: int) -> dict:
    """Per-step driver CPU span vs GPU-bound wall, with a sync every step."""
    rid = enqueue(llm, PROMPT, steps + 1)
    t0 = time.perf_counter()
    llm.llm_engine.step()  # prefill
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3
    rows = []
    finished: list = []
    for i in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.llm_engine.step()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        finished += [o for o in outs if getattr(o, "finished", False)]
        rows.append({"step": i, "cpu_ms": (t1 - t0) * 1e3, "wall_ms": (t2 - t0) * 1e3})
    mean_wall = sum(r["wall_ms"] for r in rows) / len(rows)
    mean_cpu = sum(r["cpu_ms"] for r in rows) / len(rows)
    log(f"sync run: prefill={prefill_ms:.1f} ms, decode {mean_wall:.1f} ms/token")
    tail = summarize(finished or drain(llm))
    log(f"sync run tail: {json.dumps(tail)}")
    return {
        "request_id": rid,
        "prefill_ms": prefill_ms,
        "mean_wall_ms_per_step": mean_wall,
        "mean_cpu_ms_per_step": mean_cpu,
        "steps": rows,
        "tail": tail,
    }


def run_profiled(llm: LLM, steps: int, out_dir: str, prefix: str) -> dict:
    """Profile exactly `steps` decode steps (prefill runs outside the window)."""
    hysmi = HySmiSampler()
    hysmi.start()

    rid = enqueue(llm, PROMPT, steps + 1)
    # Step 1: prefill, deliberately outside the profiler window.
    t0 = time.perf_counter()
    llm.llm_engine.step()
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3

    torch.cuda.synchronize()
    llm.start_profile(prefix)
    log(f"profiler started (prefix={prefix}); running {steps} decode steps")
    spans = []
    finished: list = []
    t_loop0 = time.perf_counter()
    for i in range(steps):
        s0 = time.perf_counter()
        outs = llm.llm_engine.step()
        spans.append((time.perf_counter() - s0) * 1e3)
        finished += [o for o in outs if getattr(o, "finished", False)]
        if (i + 1) % 8 == 0:
            log(f"  profiled step {i + 1}/{steps}")
    t_loop1 = time.perf_counter()
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    llm.stop_profile()
    hysmi.stop()
    log("profiler stopped; workers exported their traces")

    loop_ms = (t_loop1 - t_loop0) * 1e3
    drained_ms = (t_end - t_loop0) * 1e3
    log(
        f"profiled loop: driver step spans {sum(spans) / len(spans):.1f} ms/step "
        f"(max {max(spans):.1f}), driver wall {loop_ms / steps:.1f} ms/step, "
        f"incl. final sync {drained_ms / steps:.1f} ms/step"
    )
    tail = summarize(finished)
    log(f"profiled run tail: {json.dumps(tail)}")
    with open(os.path.join(out_dir, "hysmi.txt"), "w") as fh:
        fh.write("\n".join(hysmi.samples))
    return {
        "request_id": rid,
        "prefill_ms": prefill_ms,
        "steps": steps,
        "driver_cpu_spans_ms": spans,
        "driver_cpu_span_mean_ms": sum(spans) / len(spans),
        "driver_wall_ms_per_step": loop_ms / steps,
        "driver_wall_ms_per_step_synced": drained_ms / steps,
        "tail": tail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8")
    ap.add_argument("--mode", choices=["default", "none"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--skip-sync-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log(
        f"mode={args.mode} out={args.out} steps={args.steps} "
        f"torch={torch.__version__} hip={torch.version.hip} "
        f"mp={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')}"
    )

    llm = build_llm(args)
    vcfg = llm.llm_engine.vllm_config
    resolved = {
        "cudagraph_mode": str(vcfg.compilation_config.cudagraph_mode),
        "cudagraph_capture_sizes": vcfg.compilation_config.cudagraph_capture_sizes,
        "max_cudagraph_capture_size": vcfg.compilation_config.max_cudagraph_capture_size,
        "max_model_len": vcfg.model_config.max_model_len,
        "dtype": str(vcfg.model_config.dtype),
        "tp": vcfg.parallel_config.tensor_parallel_size,
        "enforce_eager": vcfg.model_config.enforce_eager,
        "compile_mode": str(vcfg.compilation_config.mode),
        "kv_cache_dtype": str(vcfg.cache_config.cache_dtype),
        "worker_multiproc_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
    }
    log(f"resolved config: {json.dumps(resolved)}")

    result: dict = {
        "mode": args.mode,
        "resolved": resolved,
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }

    result_path = os.path.join(args.out, "result.json")

    def dump() -> None:
        with open(result_path, "w") as fh:
            json.dump(result, fh, indent=1)

    dump()
    result["warmup"] = warmup(llm, steps=16)
    dump()
    if not args.skip_sync_run:
        result["sync_run"] = run_sync(llm, args.steps)
    dump()
    prefix = f"s11b-{args.mode}"
    result["profiled_run"] = run_profiled(llm, args.steps, args.out, prefix)

    trace_files = sorted(
        f
        for f in os.listdir(args.out)
        if f.endswith(".pt.trace.json") or f.endswith(".pt.trace.json.gz")
    )
    result["trace_files"] = trace_files
    log(f"traces: {trace_files}")
    dump()
    log("done")
    # Hard exit: the engine teardown can block after `stop_profile` on this
    # stack (observed 2026-10-09: driver waited on a worker forever after the
    # profiled window). Everything is already flushed to disk.
    os._exit(0)


if __name__ == "__main__":
    main()
