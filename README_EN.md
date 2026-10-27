[简体中文](./README.md)

# vLLM for Hygon DCU Z100

[![vLLM](https://img.shields.io/badge/vLLM-Hygon%20DCU-red)](https://github.com/vllm-project/vllm)
[![DTK](https://img.shields.io/badge/DTK-26.04-purple)](https://www.hygon.cn/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/vllm-project/vllm/blob/main/LICENSE)

**An LLM inference engine optimized for Hygon DCU Z100/Z100L (gfx906).**

> **Current production stack**: DTK 26.04 + torch 2.11.0+das + self-built Triton 3.6
> + Qwen3.8-27B-FP8, TP=4.
> Measured 2026-10-27: **51.6 t/s** at 7.5k single-stream (MTP k=2, engine generation
> phase), **31.4 t/s** at 128k, 30-minute decode run with 0 failures.
> Detailed performance notes (Chinese):
> [STATUS](docs/perf/STATUS-20261027.md) ·
> [ROADMAP](docs/perf/ROADMAP-20261027.md) ·
> [OPT-EFFECTIVE](docs/perf/OPT-EFFECTIVE-20261027.md).

## Status

This is a community-maintained fork, **not** a generic GPU build of upstream vLLM.
It contains Z100-specific paths for ISA limitations, DTK runtime differences, and the
absence of MFMA / native fp8 on this hardware. Changing DTK, torch, Triton, or mixing
non-DTK wheels can crash the engine or silently produce wrong results.

| Item | Value |
|---|---|
| GPU | 4× Hygon DCU Z100L, gfx906, 60 CU, 32 GiB |
| Host | Ubuntu 22.04, 12 vCPU, 31 GiB RAM + 51 GiB swap |
| DTK | **26.04** (HIP 6.3.26113) |
| Python | **3.10** (DTK wheels are cp310) |
| torch | 2.11.0+das.opt1.dtk2604 |
| Triton | **self-built** 3.6.0+git7c56a5e4 (gfx906 patch + SFU hazard fix) |
| vLLM extensions | `_C`, `_C_stable_libtorch`, `_moe_C_stable_libtorch`, `_rocm_C` |
| Production model | Qwen3.8-27B-FP8, TP=4, MTP k=2, piecewise graph |

> Only the **DTK 26.04** stack is covered here. DTK 26.10 and other vendor stacks are
> not validated.
>
> Some DTK build variables/artifacts still use legacy `rocm_*` names (for example
> `PYTORCH_ROCM_ARCH` and `_rocm_C`); these are DTK ABI identifiers, not a second
> vendor stack.

## Performance

Qwen3.8-27B-FP8, TP=4, single stream, 7469-token document, MTP k=2.
`t/s` is the engine generation-phase rate (`tok/step ÷ step_p50`, excluding TTFT).

| Context | step p50 | tok/step | t/s |
|---:|---:|---:|---:|
| 7.5k | 46.5 ms | 2.40 | **51.6** |
| 32k | 52.6 ms | 2.27 | 43.0 |
| 60k | 60.6 ms | 2.23 | 36.7 |
| 128k | 62.2 ms | 1.95 | 31.4 |

- No-spec baseline: 36.9 t/s → MTP k=2: 51.6 t/s (greedy, +40%); sampled 34.7 → 40.6 t/s (+17%).
- MTP raises warm TTFT from 0.72 s to 2.40 s; below ~220–230 output tokens it can be slower end-to-end.
- c=4 aggregate: 59–70 t/s (decode phase 66–70); only 1.2–1.4× single-stream.

## Get DTK

DTK is not bundled in this repository.

### DTK 26.04 container image (recommended)

```bash
# Requires a Hygon Harbor account; ask your cluster administrator.
docker login harbor.sourcefind.cn:5443

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413
```

The image is ~88.5 GB and contains `/opt/dtk` (DTK 26.04), rocBLAS, RCCL, MIOpen,
Python 3.10, and `hy-smi`.

### Host DTK tree (production uses this)

```bash
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
test -f "$DTK_ROOT/env.sh"
cat "$DTK_ROOT/.info/rocm_version"    # expect 26.04
source tools/dcu/env.sh
```

DTK installation packages are usually delivered offline by Hygon; there is no public
CDN. Extract the package to `/opt/dtk-26.04` or `/mnt/game/host-runtime/dtk2604`,
keeping the `env.sh` and `.info/rocm_version` layout.

> On this host `/opt/dtk` points to DTK **26.10**; do not build or run 26.04 artifacts
> against it. Inside the container `/opt/dtk` is 26.04.

### Matching torch/triton wheels

`requirements/dcu.txt` and `requirements/dcu-build.txt` pin the DTK 26.04 wheels
(cp310):

```text
torch @ https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.8.5/torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl
triton @ https://download.sourcefind.cn:65024/directlink/4/triton/DAS1.8.5/triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /path/to/z100-vllm
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements/dcu-build.txt
```

Do not mix non-DTK torch/triton wheels; the ABIs are incompatible.

## Container (Docker / rocker)

```bash
IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker run -d --name z100-dev \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --security-opt label=disable --ipc host \
  --shm-size=16g --memory=8g --memory-swap=8g \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$PWD":/src \
  -v /mnt/game:/mnt/game \
  -w /src \
  "$IMAGE" bash -c 'sleep infinity'
```

- Always pass all three devices: `/dev/kfd`, `/dev/mkfd`, `/dev/dri`.
- For a serving container use `--memory=0 --memory-swap=0`; a cgroup memory limit
  can OOM the 27B load.
- This host does not have `rocker` installed. If you use OSRF rocker, pass the same
  device/group/ipc/shm/volume flags explicitly; it will not discover DCU devices
  automatically.

## Build

```bash
cd /src
source /opt/dtk/env.sh          # inside the container
source tools/dcu/env.sh

export PYTORCH_ROCM_ARCH=gfx906
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4

python setup.py build_ext --inplace
```

- `PYTORCH_ROCM_ARCH=gfx906` is mandatory: DTK's enumerator also lists gfx926/gfx928,
  which triples the binary size and can emit ISA the Z100 cannot execute.
- First full build takes ~1.5–3 h. Start with 4 parallel jobs on a 31 GiB host.
- Prefer `build_ext --inplace`; use `PYTHONPATH=$PWD` or
  `uv pip install --no-build-isolation --no-deps -e .` to make the Python package
  importable.

### Triton for gfx906

The DTK-bundled Triton does not support gfx906. Build the patched version:

```bash
bash tools/dcu/build_triton_gfx906.sh

export PYTHONPATH=/mnt/game/triton-gfx906/install:$PYTHONPATH
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
```

The script clones Triton v3.6.0, applies `tools/dcu/triton-gfx906-3.6.0.patch`
(gfx906 ISA family, DTK `hipDeviceProp_t`, SFU hazard `s_nop 2`), and installs to
`/mnt/game/triton-gfx906/install`. The DTK-bundled Triton is left untouched.

### Incremental build

Use an out-of-tree build directory and back up the current extension before swapping:

```bash
export BUILD_DIR=/path/to/vllm-build
cp vllm/_rocm_C.abi3.so "vllm/_rocm_C.abi3.so.bak-$(date +%Y%m%d-%H%M)"
cmake --build "$BUILD_DIR" -j "${MAX_JOBS:-4}" --target _rocm_C
cp "$BUILD_DIR/_rocm_C.abi3.so" vllm/
```

Targets: `_C` (GDN/Mamba/GGUF), `_C_stable_libtorch` (stable `torch.ops._C`, push AR),
`_moe_C_stable_libtorch` (MoE), `_rocm_C` (fp8/W8A16/sdot4).

> Restart the service and verify the policy fingerprint/checksum after swapping.
> If the build log shows hipify produced nothing and the artifact did not change,
> rerun the build; do not hand-edit generated files.

## Run

```bash
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
source tools/dcu/env.sh

vllm serve /mnt/game/models/Qwen3.8-27B-FP8 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4]}' \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8000
# Add --api-key "$VLLM_API_KEY" if authentication is required.
```

The production recipe adds a set of `VLLM_DCU_*` / `VLLM_GFX906_*` switches
(push AR, GEMV stack, AR size gate, W8A16, GDN M1, BM16). See the Chinese
[README](README.md) §6.3.

## Known limitations

1. First inference is slow while Triton compiles kernels (1–2 minutes).
2. KV cache pre-allocation uses a large amount of device memory; use `util 0.85` for 27B.
3. MTP k=2 raises warm TTFT (0.72 → 2.4 s); short replies can be slower end-to-end.
4. Long-context cold prefill is expensive (60k/128k ≈ 178 s/520 s); prefix caching is critical.
5. `hipBLASLt` has no usable gfx906 kernel; `torch.mm` uses rocBLAS.
6. `dccobjdump` is broken in the DTK image; use `strings`/`clang` for ISA checks.

## Documentation

- Repository: [sibadaaa/z100-vllm](https://github.com/sibadaaa/z100-vllm)
- Issues: [sibadaaa/z100-vllm/issues](https://github.com/sibadaaa/z100-vllm/issues)
- Detailed Chinese README: [README.md](README.md)
- Performance: [STATUS](docs/perf/STATUS-20261027.md),
  [ROADMAP](docs/perf/ROADMAP-20261027.md),
  [OPT-EFFECTIVE](docs/perf/OPT-EFFECTIVE-20261027.md)
- Design/construction: [Z100 docs index](docs/design/z100/README.md)
- Results: `docs/result/z100-*.md`
- Tools: `tools/dcu/`

## Borrowed from 1Cat-vLLM

This repository references [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) — another
vLLM branch for a platform without native fp8 — for **control-plane and measurement
ideas only**. No data-plane kernels were ported: its attention/GEMM/quantization
kernels target a different instruction set and memory hierarchy and are not usable on
the Z100.

Implemented borrowings:

- **W8A16 contract**: fp8 weights stay resident in HBM, activations enter as fp16
  directly, activation quantization is removed (`apply_input_quant=False`). Landed as
  `VLLM_DCU_FP8_W8A16` plus fp16 prefill materialization; decode 1.2–2.1×, prefill
  M≥512 1.4–2×.
- **Five-phase round profile**: `draft / draft→target / target / target→draft / round`,
  implemented in `tools/dcu/s27_five_phase_profile.py`.
- **Gating discipline**: per-optimization admission predicates, rollback switches,
  bitwise/quality gates, 1K/2K/4K/8K non-regression, same-window A/B, N≥2 restarts.
  Z100 additionally uses its own policy fingerprint.

Researched but not landed:

- Balanced prefill chunks / mixed-prefill budget.
- DFlash2 `draft_block` / lookup augmentation.
- 1Cat's data-plane attention/GEMM/quantization kernels.

Negative results that were not copied: 1Cat's mixed-batch decode had silent-corruption
cases; its persistent tile runtime measured slower; its 676 switch names are not copied.

Research notes (Chinese):

- [1Cat transfer research v2](docs/design/z100/research/z100-1cat-transfer-research-v2.md)
- [W8A16 construction](docs/design/z100/w8a16/z100-w8a16-prefill-fp16-construction.md)
- [Z100 docs index](docs/design/z100/README.md)

## Acknowledgements / License

- Original vLLM: [UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu)
- Hygon DTK: DCU runtime, compiler, and driver
- 1Cat-vLLM: control-plane and measurement-method reference; no data-plane kernels used

Licensed under Apache License 2.0. See [LICENSE](LICENSE).
