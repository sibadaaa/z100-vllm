#!/usr/bin/env python3
"""DCU microbenchmark and power sampler.

This script is intended to run on a Hygon DCU / ROCm-compatible PyTorch
environment. It records power/temperature/clock/utilization samples while
running a small set of compute and memory benchmarks.

Reconstructed from https://blog.muwinds.com/archives/373/ (the published text
is truncated mid-``main``; the tail follows the article's described 8 tests and
output files) with one addition: ``--warmup`` (default 2) is a CLI flag instead
of a per-benchmark constant.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

POWER_COLUMNS = [
    "ts_epoch",
    "phase",
    "bdf",
    "power_w",
    "temp_edge_c",
    "temp_junction_c",
    "temp_mem_c",
    "sclk_mhz",
    "mclk_mhz",
    "gpu_busy_pct",
    "mem_busy_pct",
]


@dataclass
class BenchResult:
    name: str
    dtype: str
    op: str
    device_count: int
    seconds: float
    iterations: int
    throughput: float | None
    throughput_unit: str
    matrix_m: int | None = None
    matrix_n: int | None = None
    matrix_k: int | None = None
    notes: str = ""


class Phase:
    def __init__(self, initial: str = "idle") -> None:
        self._value = initial
        self._lock = threading.Lock()

    def set(self, value: str) -> None:
        with self._lock:
            self._value = value

    def get(self) -> str:
        with self._lock:
            return self._value


def run_cmd(args: list[str], timeout: float = 5.0) -> str:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.stdout


def pick_smi() -> list[str] | None:
    if shutil.which("hy-smi"):
        return ["hy-smi"]
    if shutil.which("rocm-smi"):
        return ["rocm-smi"]
    return None


def extract_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if match:
        return float(match.group(0))
    return None


def parse_smi_text(text: str) -> list[dict[str, str | float | int | None]]:
    """Best-effort parser for hy-smi/rocm-smi human and CSV-like output.

    Accepts both this host's one-metric-per-line ``HCU[<n>] : <label>: <value>``
    layout and the published per-device/BDF line layout (plus the plain summary
    table), accumulating readings per device.
    """
    rows: dict[str, dict] = {}

    def row_for(bdf):
        return rows.setdefault(bdf, {
            "bdf": bdf, "power_w": None,
            "temp_edge_c": None, "temp_junction_c": None, "temp_mem_c": None,
            "sclk_mhz": None, "mclk_mhz": None,
            "gpu_busy_pct": None, "mem_busy_pct": None,
        })

    def first_number(fragment):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", fragment)
        return float(m.group(0)) if m else None

    hcu_re = re.compile(r"HCU\[(\d+)\]")
    bdf_re = re.compile(r"(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]")
    table_re = re.compile(r"^\s*(\d+)\s+([\d.]+)C\s+([\d.]+)W")
    for line in text.splitlines():
        if not line.strip():
            continue
        lower = line.lower()
        if "error" in lower or "fail" in lower:
            continue
        table = table_re.match(line)
        if table:
            row = row_for(f"HCU{table.group(1)}")
            row["temp_edge_c"] = float(table.group(2))
            row["power_w"] = float(table.group(3))
            pcts = re.findall(r"(\d+)%", line)
            if len(pcts) >= 1:
                row["mem_busy_pct"] = int(pcts[0])
            if len(pcts) >= 2:
                row["gpu_busy_pct"] = int(pcts[1])
            continue
        hcu = hcu_re.search(line)
        bdf_match = bdf_re.search(line)
        if hcu:
            row = row_for(f"HCU{hcu.group(1)}")
        elif bdf_match:
            row = row_for(bdf_match.group(0))
        elif "GPU[" in line or "card" in lower:
            row = row_for(line.split()[0].strip(":,"))
        else:
            continue
        body = line[line.find(":") + 1:] if ":" in line else line

        def labelled(labels):
            for label in labels:
                idx = lower.find(label)
                if idx >= 0:
                    return first_number(line[idx + len(label):idx + len(label) + 48])
            return None

        def clock(label):
            m = re.search(r"\((\d+)\s*[mM][hH]z\)", line)
            return int(m.group(1)) if m else None

        if "sclk" in lower:
            row["sclk_mhz"] = clock("sclk") or labelled(["sclk"])
        elif "mclk" in lower:
            row["mclk_mhz"] = clock("mclk") or labelled(["mclk"])
        elif "clock" in lower:
            pass
        elif "power" in lower or "pwr" in lower:
            row["power_w"] = labelled(["power", "pwr"])
        elif "temp" in lower:
            if "junction" in lower or "hotspot" in lower:
                row["temp_junction_c"] = labelled(["junction", "hotspot"])
            elif "mem" in lower or "hbm" in lower:
                row["temp_mem_c"] = labelled(["sensor mem", "mem", "hbm"])
            elif "edge" in lower:
                row["temp_edge_c"] = labelled(["edge"])
        elif "use" in lower or "busy" in lower or "util" in lower:
            gpu = labelled(["gpu use", "gpu busy", "gpu%", "hcu%", "hcu use"])
            mem = labelled(["mem use", "mem busy", "vram%"])
            row["gpu_busy_pct"] = int(gpu) if gpu is not None else None
            row["mem_busy_pct"] = int(mem) if mem is not None else None
        else:
            value = first_number(body)
            if value is not None and row["power_w"] is None:
                row["power_w"] = value
    return list(rows.values())


def sample_smi():
    smi = pick_smi()
    if smi is None:
        return []
    commands = [
        smi + ["--showpower", "--showtemp", "--showuse", "--showclocks"],
        smi + ["--showpower", "--showtemp", "--showuse"],
        smi,
    ]
    for cmd in commands:
        try:
            rows = parse_smi_text(run_cmd(cmd))
            if rows:
                return rows
        except Exception:
            continue
    return []


def power_sampler(path, phase, stop, interval):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=POWER_COLUMNS)
        writer.writeheader()
        while not stop.is_set():
            ts = int(time.time())
            rows = sample_smi()
            for row in rows:
                out = {col: "" for col in POWER_COLUMNS}
                out.update(row)
                out["ts_epoch"] = ts
                out["phase"] = phase.get()
                writer.writerow(out)
            f.flush()
            stop.wait(interval)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_loop(fn, seconds, warmup=5):
    for _ in range(warmup):
        fn()
    synchronize()
    iters = 0
    start = time.perf_counter()
    while True:
        fn()
        iters += 1
        if time.perf_counter() - start >= seconds:
            break
    synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, iters


def all_devices():
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def bench_matmul(dtype, size, seconds, warmup=5):
    devices = all_devices()
    tensors = []
    for dev in devices:
        a = torch.randn((size, size), device=dev, dtype=dtype)
        b = torch.randn((size, size), device=dev, dtype=dtype)
        tensors.append((a, b))

    def step():
        for a, b in tensors:
            torch.matmul(a, b)

    elapsed, iters = time_loop(step, seconds, warmup)
    flops = 2.0 * size * size * size * iters * len(devices)
    return BenchResult(
        name=f"{str(dtype).split('.')[-1]}_matmul",
        dtype=str(dtype).split(".")[-1], op="matmul",
        device_count=len(devices),
        matrix_m=size, matrix_n=size, matrix_k=size,
        seconds=elapsed, iterations=iters,
        throughput=flops / elapsed / 1e12, throughput_unit="TFLOP/s",
    )


def bench_int8_matmul(size, seconds, warmup=5):
    devices = all_devices()
    if not hasattr(torch, "_int_mm"):
        return BenchResult(name="int8_matmul", dtype="int8", op="torch._int_mm",
                           device_count=len(devices), matrix_m=size, matrix_n=size,
                           matrix_k=size, seconds=0.0, iterations=0, throughput=None,
                           throughput_unit="TOPS",
                           notes="torch._int_mm is unavailable in this PyTorch build")
    tensors = []
    try:
        for dev in devices:
            a = torch.randint(-128, 127, (size, size), device=dev, dtype=torch.int8)
            b = torch.randint(-128, 127, (size, size), device=dev, dtype=torch.int8)
            tensors.append((a, b))

        def step():
            for a, b in tensors:
                torch._int_mm(a, b)

        elapsed, iters = time_loop(step, seconds, warmup)
        ops = 2.0 * size * size * size * iters * len(devices)
        return BenchResult(name="int8_matmul", dtype="int8", op="torch._int_mm",
                           device_count=len(devices), matrix_m=size, matrix_n=size,
                           matrix_k=size, seconds=elapsed, iterations=iters,
                           throughput=ops / elapsed / 1e12, throughput_unit="TOPS")
    except Exception as exc:
        return BenchResult(name="int8_matmul", dtype="int8", op="torch._int_mm",
                           device_count=len(devices), matrix_m=size, matrix_n=size,
                           matrix_k=size, seconds=0.0, iterations=0, throughput=None,
                           throughput_unit="TOPS",
                           notes=f"unsupported: {type(exc).__name__}: {exc}")


def bench_integer_elementwise(dtype, elems, seconds, warmup=5):
    devices = all_devices()
    tensors = []
    for dev in devices:
        high = 127 if dtype is torch.int8 else 2 ** 30
        a = torch.randint(0, high, (elems,), device=dev, dtype=dtype)
        b = torch.randint(0, high, (elems,), device=dev, dtype=dtype)
        tensors.append((a, b))

    def step():
        for a, b in tensors:
            torch.bitwise_xor(a, b)
            torch.add(a, b)

    elapsed, iters = time_loop(step, seconds, warmup)
    ops = 2.0 * elems * iters * len(devices)
    return BenchResult(
        name=f"{str(dtype).split('.')[-1]}_elementwise",
        dtype=str(dtype).split(".")[-1], op="xor+add",
        device_count=len(devices),
        seconds=elapsed, iterations=iters,
        throughput=ops / elapsed / 1e9, throughput_unit="GOP/s")


def bench_int4_unpack(elems, seconds, warmup=5):
    devices = all_devices()
    packed_elems = max(1, elems // 2)
    tensors = []
    for dev in devices:
        x = torch.randint(0, 255, (packed_elems,), device=dev, dtype=torch.uint8)
        tensors.append(x)

    def step():
        for x in tensors:
            lo = x & 0x0F
            hi = (x >> 4) & 0x0F
            torch.add(lo, hi)

    elapsed, iters = time_loop(step, seconds, warmup)
    ops = 3.0 * packed_elems * iters * len(devices)
    return BenchResult(name="int4_unpack_elementwise", dtype="int4_packed_uint8",
                       op="unpack_nibbles+add", device_count=len(devices),
                       seconds=elapsed, iterations=iters,
                       throughput=ops / elapsed / 1e9, throughput_unit="GOP/s",
                       notes="packed INT4 path; measures unpack/bitwise throughput, "
                             "not tensor-core INT4 GEMM")


def bench_memory_bandwidth(elems, seconds, warmup=5):
    devices = all_devices()
    tensors = []
    for dev in devices:
        a = torch.empty((elems,), device=dev, dtype=torch.float32)
        b = torch.empty((elems,), device=dev, dtype=torch.float32)
        tensors.append((a, b))

    def step():
        for a, b in tensors:
            b.copy_(a)

    elapsed, iters = time_loop(step, seconds, warmup)
    bytes_moved = elems * 4 * iters * len(devices)
    return BenchResult(name="memory_copy", dtype="float32", op="copy",
                       device_count=len(devices),
                       seconds=elapsed, iterations=iters,
                       throughput=bytes_moved / elapsed / 1e9, throughput_unit="GB/s")


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="dcu_benchmark_out")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--idle-seconds", type=float, default=8.0)
    parser.add_argument("--cooldown-seconds", type=float, default=8.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--matmul-size", type=int, default=4096)
    parser.add_argument("--int8-size", type=int, default=4096)
    parser.add_argument("--elementwise-elems", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase = Phase("idle")
    stop = threading.Event()
    power_csv = out_dir / "dcu_power_curve.csv"
    sampler = threading.Thread(
        target=power_sampler,
        args=(power_csv, phase, stop, args.sample_interval),
        daemon=True,
    )
    sampler.start()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results: list[BenchResult] = []
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_names = (
        [torch.cuda.get_device_name(i) for i in range(device_count)]
        if device_count
        else []
    )
    try:
        time.sleep(args.idle_seconds)
        tests = [
            ("fp32_matmul",
             lambda: bench_matmul(torch.float32, args.matmul_size, args.seconds, args.warmup)),
            ("fp16_matmul",
             lambda: bench_matmul(torch.float16, args.matmul_size, args.seconds, args.warmup)),
            ("bf16_matmul",
             lambda: bench_matmul(torch.bfloat16, args.matmul_size, args.seconds, args.warmup)),
            ("int8_matmul",
             lambda: bench_int8_matmul(args.int8_size, args.seconds, args.warmup)),
            ("int32_elementwise",
             lambda: bench_integer_elementwise(torch.int32, args.elementwise_elems,
                                               args.seconds, args.warmup)),
            ("int8_elementwise",
             lambda: bench_integer_elementwise(torch.int8, args.elementwise_elems,
                                               args.seconds, args.warmup)),
            ("int4_unpack",
             lambda: bench_int4_unpack(args.elementwise_elems, args.seconds, args.warmup)),
            ("memory_copy",
             lambda: bench_memory_bandwidth(args.elementwise_elems, args.seconds, args.warmup)),
        ]
        for name, test in tests:
            phase.set(name)
            print(f"[bench] {name} ...", flush=True)
            try:
                res = test()
                results.append(res)
                print(
                    f"[bench] {name}: {res.throughput} {res.throughput_unit} "
                    f"({res.iterations} iters in {res.seconds:.2f} s)"
                    + (f"  [{res.notes}]" if res.notes else ""),
                    flush=True,
                )
            except Exception as exc:
                results.append(BenchResult(
                    name=name, dtype="", op=name, device_count=device_count,
                    matrix_m=None, matrix_n=None, matrix_k=None, seconds=0.0,
                    iterations=0, throughput=None, throughput_unit="",
                    notes=f"failed: {type(exc).__name__}: {exc}",
                ))
                print(f"[bench] {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
            phase.set("idle")
        phase.set("cooldown")
        time.sleep(args.cooldown_seconds)
    finally:
        phase.set("idle")
        stop.set()
        sampler.join(timeout=10)
        synchronize()

    payload = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device_count": device_count,
        "device_names": device_names,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "args": vars(args),
        "power_csv": str(power_csv),
        "results": [asdict(r) for r in results],
    }
    write_json(out_dir / "results.json", payload)

    print()
    header = f"{'name':<24}{'dtype':<10}{'throughput':>12}  unit"
    print(header)
    print("-" * len(header))
    for res in results:
        value = "n/a" if res.throughput is None else f"{res.throughput:.2f}"
        print(f"{res.name:<24}{res.dtype:<10}{value:>12}  {res.throughput_unit}")
    print()
    print(f"[out] {out_dir / 'results.json'}")
    print(f"[out] {power_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
