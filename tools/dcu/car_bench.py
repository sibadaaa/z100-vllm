#!/usr/bin/env python3
"""WP2: vLLM custom all-reduce ladder against RCCL on 4 gfx906 (Z100) ranks.

Measures per-call latency of vLLM's ``CustomAllreduce`` kernel against
``torch.distributed`` NCCL at the sizes WP2 cares about, and validates every
reduction elementwise against NCCL (and against a float64 reference) on every
rank.

The custom path is driven through ``CustomAllreduce.custom_all_reduce`` -- the
exact call ``CudaCommunicator`` makes -- so the timing includes the copy of the
input into the pre-registered IPC buffer. NCCL is measured in place. Both are
timed with CUDA events while alternating NCCL/custom per sample, so slow drift
cancels instead of biasing one side.

Nothing here silently falls back: ``--custom-ar`` either measures the custom
kernel or fails with a distinct exit code. (S13 was void because the "A" arm
was RCCL vs RCCL after the gate refused the custom kernel.)

A watchdog turns a hang into a JSON record plus exit code 124 (60 s default per
configuration). JSON output has ``hang``/``exit_code`` markers, mean/p50/p99
per implementation and the correctness verdict.

Usage (through the systemd-run wrapper in .r3/WP2/RUNBOOK.md):

  torchrun --nproc_per_node=4 --standalone tools/dcu/car_bench.py \
      --custom-ar --alloc finegrained --algo 2stage \
      --out /home/lyh/z100-vllm/.r3/WP2/data/car_bench_custom_ar.json

  torchrun --nproc_per_node=4 --standalone tools/dcu/car_bench.py \
      --out /home/lyh/z100-vllm/.r3/WP2/data/car_bench_nccl.json

``--out`` belongs to rank 0 and holds the merged (per-rank) result; every other
rank writes ``<out>.rank<N>.json`` with its own view.

Exit codes: 0 ok, 2 custom all-reduce unavailable/disabled, 3 the gate refused
the custom kernel, 4 init/runtime error, 124 hang (per-config timeout).
"""

import argparse
import contextlib
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

EXIT_OK = 0
EXIT_DISABLED = 2
EXIT_GATE = 3
EXIT_ERROR = 4
EXIT_HANG = 124

# 10 KB, 40 KB, 160 KB, 1 MB, 4 MB (WP2 card), per rank byte size.
DEFAULT_SIZES = "10240,40960,163840,1048576,4194304"
DEFAULT_SAMPLES = 50
DEFAULT_WARMUP = 10
DEFAULT_TIMEOUT_S = 60.0
DTYPE_ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: str, payload: dict) -> None:
    """Atomically write ``payload`` as JSON to ``path``."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
    os.replace(tmp, path)


def _stats(samples_us: list[float]) -> dict:
    """mean/p50/p99/min/max plus the raw sample count."""
    ordered = sorted(samples_us)
    n = len(ordered)
    return {
        "samples": n,
        "mean_us": round(statistics.fmean(ordered), 3),
        "p50_us": round(ordered[n // 2], 3),
        "p99_us": round(ordered[min(n - 1, max(0, (99 * n + 99) // 100 - 1))], 3),
        "min_us": round(ordered[0], 3),
        "max_us": round(ordered[-1], 3),
        "stdev_us": round(statistics.pstdev(ordered), 3) if n > 1 else 0.0,
    }


def _git_head() -> str:
    repo = os.environ.get("REPO", os.getcwd())
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "-"
    except Exception:  # noqa: BLE001 - metadata must never fail the bench
        return "-"


def _so_md5() -> str:
    repo = os.environ.get("REPO", os.getcwd())
    path = os.path.join(repo, "vllm", "_rocm_C.abi3.so")
    try:
        with open(path, "rb") as handle:
            return hashlib.md5(handle.read()).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return "-"


def _env_echo() -> dict:
    keys = (
        "VLLM_DCU_CUSTOM_AR",
        "VLLM_DCU_CUSTOM_AR_ALLOC",
        "VLLM_CUSTOM_ALLREDUCE_ALGO",
        "HSA_FORCE_FINE_GRAIN_PCIE",
        "HIP_VISIBLE_DEVICES",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_MIN_NCHANNELS",
        "NCCL_BUFFSIZE",
        "NCCL_DEBUG",
    )
    return {key: os.environ.get(key, "<unset>") for key in keys}


class HangWatchdog:
    """Kill the process with EXIT_HANG if one configuration takes too long."""

    def __init__(self, timeout_s: float, path: str, rank: int, world: int):
        self.timeout_s = timeout_s
        self.path = path
        self.rank = rank
        self.world = world
        self._timer: threading.Timer | None = None
        self._label = ""

    def start(self, label: str) -> None:
        self._label = label
        self._timer = threading.Timer(self.timeout_s, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _fire(self) -> None:
        payload = {
            "wp": "WP2",
            "bench": "tools/dcu/car_bench.py",
            "rank": self.rank,
            "world_size": self.world,
            "hang": True,
            "hang_config": self._label,
            "timeout_s": self.timeout_s,
            "exit_code": EXIT_HANG,
            "finished_at": _now(),
            "note": "per-configuration watchdog fired; os._exit(124)",
        }
        try:
            write_json(self.path, payload)
        except Exception:  # noqa: BLE001 - the exit code is the primary signal
            pass
        print(
            f"[car_bench][rank{self.rank}] HANG config='{self._label}' "
            f"(>{self.timeout_s:.0f}s) -> exit {EXIT_HANG}",
            flush=True,
        )
        os._exit(EXIT_HANG)


def _guard(watchdog: HangWatchdog, label: str):
    @contextlib.contextmanager
    def _cm():
        watchdog.start(label)
        try:
            yield
        finally:
            watchdog.stop()

    return _cm()


def _csv_positive_ints(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"size must be positive: {token}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("no sizes given")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WP2 custom all-reduce vs RCCL ladder (4 ranks)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--custom-ar",
        action="store_true",
        help="also measure vLLM CustomAllreduce (fails instead of falling back)",
    )
    parser.add_argument(
        "--out",
        default="car_bench_result.json",
        help="JSON output path (rank 0; other ranks write <out>.rankN.json)",
    )
    parser.add_argument(
        "--sizes", type=_csv_positive_ints, default=_csv_positive_ints(DEFAULT_SIZES)
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="per-configuration hang timeout (seconds)",
    )
    parser.add_argument(
        "--alloc",
        default="finegrained",
        choices=["finegrained", "default", "uncached"],
        help="VLLM_DCU_CUSTOM_AR_ALLOC for the IPC buffers",
    )
    parser.add_argument(
        "--algo",
        default="2stage",
        choices=["1stage", "oneshot", "2stage", "twoshot"],
        help="VLLM_CUSTOM_ALLREDUCE_ALGO (the C++ dispatch needs it off XGMI)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=sorted(DTYPE_ITEMSIZE),
        help="payload dtype (serve uses float16)",
    )
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--tag", default="wp2", help="label recorded in the JSON")
    parser.add_argument(
        "--rank-delay-us",
        type=float,
        default=0.0,
        help="S33 WP2 skew 档: busy-wait this long on --delay-rank before "
        "every timed all-reduce enqueue (arrival skew; applies to BOTH the "
        "NCCL and custom sections of each sample)",
    )
    parser.add_argument(
        "--delay-rank",
        type=int,
        default=3,
        help="which rank hosts the artificial arrival delay",
    )
    args = parser.parse_args(argv)
    if args.samples < 50:
        parser.error("--samples must be >= 50 (WP2 evidence rule)")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = _now()

    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")
    if rank_env is None or world_env is None:
        print("[car_bench] run under torchrun (RANK/WORLD_SIZE unset)", flush=True)
        return EXIT_ERROR
    rank = int(rank_env)
    world = int(world_env)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    out_path = os.path.abspath(args.out)
    rank_path = out_path if rank == 0 else f"{out_path}.rank{rank}.json"

    watchdog = HangWatchdog(args.timeout_s, rank_path, rank, world)

    # All knobs are read by the custom all-reduce code (Python gate lazily, C++
    # once per process for the alloc flag); set them before importing vllm.
    if args.custom_ar:
        os.environ["VLLM_DCU_CUSTOM_AR"] = "1"
        os.environ["VLLM_DCU_CUSTOM_AR_ALLOC"] = args.alloc
        os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = args.algo

    base = {
        "wp": "WP2",
        "bench": "tools/dcu/car_bench.py",
        "tag": args.tag,
        "mode": "custom_ar_vs_nccl" if args.custom_ar else "nccl_only",
        "rank": rank,
        "world_size": world,
        "started_at": started,
        "hang": False,
        "exit_code": EXIT_OK,
        "cmd": [sys.executable, *sys.argv],
        "host": socket.gethostname(),
        "git_head": _git_head(),
        "so_md5": _so_md5() if rank == 0 else None,
        "env": _env_echo(),
        "args": vars(args),
        "skew": {
            "delay_rank": args.delay_rank if args.rank_delay_us > 0 else None,
            "delay_us": args.rank_delay_us,
        },
    }

    def fail(code: int, message: str, exc: BaseException | None = None) -> int:
        payload = dict(base)
        payload.update(
            {
                "exit_code": code,
                "error": message,
                "finished_at": _now(),
                "traceback": (
                    "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    if exc is not None
                    else None
                ),
            }
        )
        write_json(rank_path, payload)
        print(f"[car_bench][rank{rank}] {message} -> exit {code}", flush=True)
        return code

    try:
        if world != 4:
            return fail(EXIT_ERROR, f"this bench requires 4 ranks, got {world}")

        with _guard(watchdog, "import torch"):
            import torch
            import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        with _guard(watchdog, "dist.init_process_group"):
            dist.init_process_group(
                "nccl", timeout=timedelta(seconds=max(120.0, 4 * args.timeout_s))
            )
        if dist.get_world_size() != world:
            return fail(
                EXIT_ERROR,
                f"torchrun world_size {dist.get_world_size()} != {world}",
            )

        # CustomAllreduce exchanges IPC handles over a CPU (gloo) group; the
        # data path stays on NCCL/RCCL.
        if not dist.is_gloo_available():
            return fail(EXIT_ERROR, "torch.distributed gloo backend unavailable")
        with _guard(watchdog, "new_group(gloo)"):
            cpu_group = dist.new_group(ranks=list(range(world)), backend="gloo")
        dist.barrier(group=cpu_group)

        ca = None
        ca_info = None
        if args.custom_ar:
            with _guard(watchdog, "CustomAllreduce.init"):
                from vllm.distributed.device_communicators import (
                    custom_all_reduce as car,
                )

                if not car.custom_ar:
                    return fail(
                        EXIT_DISABLED,
                        "vllm custom all-reduce extension is not available "
                        "(ops.meta_size() failed)",
                    )
                ca = car.CustomAllreduce(group=cpu_group, device=device)
            ok = torch.tensor([0 if ca.disabled else 1], dtype=torch.int32)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=cpu_group)
            if not bool(ok.item()):
                return fail(
                    EXIT_DISABLED,
                    "CustomAllreduce is disabled on at least one rank "
                    "(check VLLM_DCU_CUSTOM_AR=1 and the all-gather logs)",
                )
            ca_info = {
                "disabled": bool(ca.disabled),
                "fully_connected": bool(ca.fully_connected),
                "world_size": int(ca.world_size),
                "max_size": int(ca.max_size),
                "alloc": os.environ.get("VLLM_DCU_CUSTOM_AR_ALLOC"),
                "algo": os.environ.get("VLLM_CUSTOM_ALLREDUCE_ALGO"),
            }

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
        itemsize = DTYPE_ITEMSIZE[args.dtype]

        per_size = []
        for nbytes in args.sizes:
            with _guard(watchdog, f"size {nbytes} bytes"):
                numel = nbytes // itemsize
                if numel * itemsize != nbytes:
                    return fail(
                        EXIT_ERROR, f"size {nbytes} not divisible by {itemsize}"
                    )
                torch.manual_seed(args.seed + 100003 * rank + nbytes)
                x = (
                    torch.randn(numel, dtype=torch.float32)
                    .to(device=device)
                    .to(torch_dtype)
                )
                x0 = x.clone()

                # --- correctness: NCCL vs fp64 truth, custom vs NCCL ---
                ref = x0.clone()
                dist.all_reduce(ref)
                truth = x0.double()
                dist.all_reduce(truth)
                torch.cuda.synchronize()
                nccl_err = (ref.double() - truth).abs().max().item()
                record = {
                    "bytes": nbytes,
                    "numel": numel,
                    "dtype": args.dtype,
                    "nccl": None,
                    "custom_ar": None,
                    "custom_ar_used": False,
                    "should_custom_ar": None,
                    "improvement_pct_p50": None,
                    "correctness": {
                        "nccl_vs_fp64_max_abs_diff": nccl_err,
                        "custom_vs_nccl_max_abs_diff": None,
                        "bitwise_equal": None,
                        "nan_or_inf": bool(
                            torch.isnan(ref).any() or torch.isinf(ref).any()
                        ),
                    },
                }

                if ca is not None:
                    record["should_custom_ar"] = bool(ca.should_custom_ar(x0))
                    out = ca.custom_all_reduce(x0)
                    torch.cuda.synchronize()
                    if out is None:
                        return fail(
                            EXIT_GATE,
                            f"gate refused the custom kernel at {nbytes} bytes "
                            f"(should_custom_ar={record['should_custom_ar']}); "
                            "VLLM_CUSTOM_ALLREDUCE_ALGO is required off XGMI",
                        )
                    record["custom_ar_used"] = True
                    record["correctness"].update(
                        {
                            "custom_vs_nccl_max_abs_diff": (
                                (out.float() - ref.float()).abs().max().item()
                            ),
                            "custom_vs_fp64_max_abs_diff": (
                                (out.double() - truth).abs().max().item()
                            ),
                            "bitwise_equal": bool(torch.equal(out, ref)),
                            "nan_or_inf": bool(
                                torch.isnan(out).any() or torch.isinf(out).any()
                            ),
                        }
                    )

                # --- latency: alternate NCCL / custom per sample ---
                for _ in range(args.warmup):
                    dist.all_reduce(x)
                    x.copy_(x0)
                    if ca is not None:
                        ca.custom_all_reduce(x)
                        x.copy_(x0)
                torch.cuda.synchronize()
                dist.barrier()  # all ranks start timing together

                nccl_us: list[float] = []
                ca_us: list[float] = []
                ev = [
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(args.samples)
                ]
                ev_ca = [
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(args.samples)
                ]
                for i in range(args.samples):
                    # S33 WP2 skew 档: host busy-wait on the delayed rank
                    # delays this rank's kernel ENQUEUE, so every other rank's
                    # all-reduce spins in the kernel waiting for the laggard
                    # -- arrival skew, charged to BOTH implementations.
                    if args.rank_delay_us > 0 and rank == args.delay_rank:
                        _t_end = time.perf_counter() + args.rank_delay_us * 1e-6
                        while time.perf_counter() < _t_end:
                            pass
                    start, end = ev[i]
                    start.record()
                    dist.all_reduce(x)
                    end.record()
                    x.copy_(x0)
                    if ca is not None:
                        if args.rank_delay_us > 0 and rank == args.delay_rank:
                            _t_end = time.perf_counter() + args.rank_delay_us * 1e-6
                            while time.perf_counter() < _t_end:
                                pass
                        start_ca, end_ca = ev_ca[i]
                        start_ca.record()
                        ca.custom_all_reduce(x)
                        end_ca.record()
                        x.copy_(x0)
                torch.cuda.synchronize()
                dist.barrier()

                nccl_us = [
                    ev[i][0].elapsed_time(ev[i][1]) * 1e3
                    for i in range(args.samples)
                ]
                record["nccl"] = _stats(nccl_us)
                if ca is not None:
                    ca_us = [
                        ev_ca[i][0].elapsed_time(ev_ca[i][1]) * 1e3
                        for i in range(args.samples)
                    ]
                    record["custom_ar"] = _stats(ca_us)
                    p50_nccl = record["nccl"]["p50_us"]
                    record["improvement_pct_p50"] = (
                        round(
                            100.0
                            * (p50_nccl - record["custom_ar"]["p50_us"])
                            / p50_nccl,
                            2,
                        )
                        if p50_nccl > 0
                        else None
                    )
                per_size.append(record)
                if rank == 0:
                    line = (
                        f"[car_bench] bytes={nbytes:8d}"
                        f" nccl_p50={record['nccl']['p50_us']:8.1f}us"
                    )
                    if record["custom_ar"] is not None:
                        line += (
                            f" ca_p50={record['custom_ar']['p50_us']:8.1f}us"
                            f" delta={record['improvement_pct_p50']:+6.1f}%"
                            f" bitwise={record['correctness']['bitwise_equal']}"
                            f" max|d|="
                            f"{record['correctness']['custom_vs_nccl_max_abs_diff']}"
                        )
                    print(line, flush=True)

        with _guard(watchdog, "all_gather_object(results)"):
            gathered: list[dict | None] = [None] * world
            dist.all_gather_object(
                gathered,
                {"rank": rank, "sizes": per_size, "ca_info": ca_info},
                group=cpu_group,
            )

        payload = dict(base)
        payload.update(
            {
                "finished_at": _now(),
                "torch": {
                    "version": torch.__version__,
                    "hip": getattr(torch.version, "hip", None),
                    "device_name": torch.cuda.get_device_name(local_rank),
                    "gcn_arch": torch.cuda.get_device_properties(
                        local_rank
                    ).gcnArchName,
                },
                "custom_ar": ca_info,
                "sizes": per_size,
                "per_rank": [entry for entry in gathered if entry is not None],
                "notes": [
                    "custom path = CustomAllreduce.custom_all_reduce() outside "
                    "graph capture (includes the input copy into the IPC buffer)",
                    "nccl path = torch.distributed all_reduce in place",
                    "latency = CUDA event elapsed per sample; samples>=50",
                ],
            }
        )
        if rank == 0:
            for index, record in enumerate(per_size):
                record["per_rank"] = [
                    {
                        "rank": entry["rank"],
                        "nccl": entry["sizes"][index]["nccl"],
                        "custom_ar": entry["sizes"][index]["custom_ar"],
                        "correctness": entry["sizes"][index]["correctness"],
                    }
                    for entry in gathered
                    if entry is not None
                ]
            payload["sizes"] = per_size
            write_json(out_path, payload)
            print(f"[car_bench] wrote {out_path}", flush=True)
        else:
            write_json(rank_path, payload)

        if ca is not None:
            try:
                ca.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the result
                pass
        dist.destroy_process_group()
        return EXIT_OK

    except BaseException as exc:  # noqa: BLE001 - every failure needs a JSON record
        return fail(EXIT_ERROR, f"unhandled {type(exc).__name__}: {exc}", exc)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Exit hard: a half-torn-down process group or a wedged HIP context must
    # not turn a finished measurement into a hang.
    os._exit(code)
