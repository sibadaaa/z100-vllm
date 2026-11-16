# Hygon DCU (Z100 / gfx906) helper scripts

Tooling for building, serving and validating vLLM on Hygon DCU boards running
DTK 26.04.

> **2026-11-16**: the one-off probe and analysis scripts that used to live here
> (per-round A/B drivers, PMC/ISA probes, trace interpreters) were removed.
> Their measurements are recorded under [`docs/result/`](../../docs/result) and
> [`docs/perf/`](../../docs/perf); the scripts themselves stay reachable in git
> history. Only the pieces needed to **build, serve, gate and fingerprint** the
> stack remain, plus the reports that explain them.

## What is here

| Path | Purpose |
|---|---|
| `env.sh` | DTK/HIP environment for the host (sourced by the launcher) |
| `env-rocm101.sh` | same for the ROCm 10.1 + torch 2.16 tree |
| `use_stack.sh` | switch/inspect the four extensions per stack (`status`/`dtk`) |
| `dcu_policy_fingerprint.py` | effective-policy hash for every A/B report |
| `dcu_benchmark.py` | latency/throughput entry point |
| `dcu_p2p_acs.py` | PCIe ACS state check for 4-rank P2P |
| `build_triton_gfx906.sh` + `triton-gfx906-3.6.0.patch` | self-built Triton (required on gfx906) |
| `probe_triton.py` | Triton sanity probe (`TRITON_OK` / `REDUCE_OK` / `INT8DOT_OK`) |
| `check_bf16.py` | whether bf16 numerics are usable on this device |
| `install_torchvision_shim.sh` | optional torchvision shim |
| `rccl-gfx906-build.sh` | RCCL build helper for the gfx906 tree |
| `attn_stage/` | staging-timed attention build recipe (used by `BUILD.md` §3.3) |
| `z100-vllm-*.service`, `dcu-p2p-acs.service` | auxiliary units |
| `*.md` | per-topic reports kept as documentation |

## `env.sh`

Sourced, not executed:

```bash
export DTK_ROOT=/mnt/game/host-runtime/dtk2604   # host /opt/dtk is 26.10
source tools/dcu/env.sh
```

It sources DTK's own `env.sh` and then applies what a plain DTK environment
does not provide:

1. prefers a `librocm_smi64` that exports `rsmi_init` (some DTK images ship a
   stub first, making `import torch` fail with `undefined symbol: rsmi_init`);
2. adds the host's gcc C++ headers, because DTK's clang looks for a matching gcc
   and otherwise fails with `fatal error: 'cmath' file not found`;
3. exports `VLLM_DCU_GCN_ARCH=gfx906` once every KFD node reports
   `gfx_target_version 90106` (a raced boot must not silently pick another
   compile-cache key — see the S89 incident);
4. sets `TRITON_HIP_LLD_PATH`, `TRITON_CACHE_DIR`, `HIP_VISIBLE_DEVICES`;
5. defaults `VLLM_ROCM_USE_SKINNY_GEMM=0` (gfx906 has no MFMA);
6. defaults `VLLM_CACHE_ROOT=/mnt/game/vllm-cache` when `/mnt/game` exists.

All of it is overridable by pre-setting the same variables.

**Do not set `ROCBLAS_TENSILE_LIBPATH`.** On Z100 / DTK 26.04 the gfx906
Tensile directory only ships the lazy library; exporting the variable makes
rocBLAS load the absent legacy one and Tensile host init aborts the process:

```text
rocBLAS error: Could not initialize Tensile host: std::bad_alloc   # SIGABRT
```

## `check_bf16.py`

```bash
source tools/dcu/env.sh
python3 tools/dcu/check_bf16.py
```

Exit `0` = all bf16 checks passed, `1` = bf16 numerics unreliable (keep
`--dtype float16`), `2` = no GPU visible. gfx906 has no bf16 instructions, so
bf16 is emulated; `torch.cuda.is_bf16_supported()` returns True on every ROCm
build and is reported but never used as a decision input.

Related knobs: `VLLM_DCU_BF16=auto|1|0` (offer/force/disable).

## `compressed-tensors` must be >= 0.15.0.1

The DTK image ships 0.14.0.1, which lacks
`compressed_tensors.compressors.pack_quantized`; every compressed-tensors model
then fails at import. `requirements/common.txt` pins 0.15.0.1, so
`pip install -r requirements/dcu.txt` fixes it; the pre-downloaded wheel is kept
at `/mnt/game/port/s8/pip-wheels/` for hosts without PyPI access.

## Reading decode traces

The serving unit runs with
`--profiler-config {"profiler":"torch","torch_profiler_dir":"/mnt/game/s30c-prof"}`.
The three rules that survived the script cleanup:

- **Never trust a bucket median.** `down_smul` and plain `down` share the kernel
  name `fp8_emu_gemv_wholea_m1`; only per-shape rows expose a single-shape
  regression (S45: 41.8 µs/534 GB/s → 69.8 µs/319 GB/s, bucket median moved
  +1.8 ms only).
- Attribute GPU idle separately from in-step gaps; step-boundary stalls are a
  different fix from kernel-level ones.
- Sample with one request, `/start_profile` then `/stop_profile` immediately;
  if the service wedges, restart rather than retrying.

## Window microbenchmarks

Measure on **idle card 3** (card 0 is held by the serving unit; card 3 is
11-13% slower at equal shapes, which is fine for ratios). Pin the environment
to the serving one (`VLLM_DCU_FP8_GEMV_WA=1`,
`VLLM_DCU_FP8_GEMV_STREAM_ROWS=4`) or the numbers are not comparable, and
validate the device state against an anchor (read-only bandwidth ~780-800 GB/s)
before trusting a run.

See [`BUILD.md`](../../BUILD.md) for the build/relink workflow and
[`DEPLOY.md`](../../DEPLOY.md) for the serving configuration.
