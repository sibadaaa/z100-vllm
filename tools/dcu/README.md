# Hygon DCU (Z100 / gfx906) helper scripts

Tools for building and validating vLLM on Hygon DCU boards running DTK.
See [`docs/hygon_z100.md`](../../docs/hygon_z100.md) for the measured evidence
and the design rationale.

## `env.sh`

Sourced, not executed:

```bash
source tools/dcu/env.sh
```

It sources DTK's own `env.sh` (`DTK_ROOT`, default `/opt/dtk`) and then applies
the fixes that a plain DTK environment does not provide:

1. puts a `librocm_smi64` that actually exports `rsmi_init` first on
   `LD_LIBRARY_PATH` (some DTK images ship a stub before it and `import torch`
   fails with `undefined symbol: rsmi_init`);
2. adds the host's gcc C++ header directories, because DTK's clang looks for a
   gcc installation matching its own defaults and otherwise fails with
   `fatal error: 'cmath' file not found`;
3. exports `VLLM_DCU_GCN_ARCH=gfx906` when every KFD GPU node reports
   `gfx_target_version 90106` (this also avoids initialising a HIP context
   while importing vLLM);
4. sets `TRITON_HIP_LLD_PATH`, `TRITON_CACHE_DIR` and `HIP_VISIBLE_DEVICES`;
5. defaults `VLLM_ROCM_USE_SKINNY_GEMM=0` (gfx906 has no MFMA; the ROCm skinny
   GEMM kernels are not built for it);
6. defaults `VLLM_CACHE_ROOT=/mnt/game/vllm-cache` when `/mnt/game` exists.
   vLLM otherwise writes `torch.compile`/inductor artifacts to `~/.cache/vllm`,
   i.e. the container writable layer — one small-model smoke run was measured
   at 2.1 GB. Pre-set `VLLM_CACHE_ROOT` to choose another location.

All of it is overridable by pre-setting the same variables.

**Do not set `ROCBLAS_TENSILE_LIBPATH` here.** On Z100 / DTK 26.04 the gfx906
Tensile directory only ships the lazy library
(`TensileLibrary_lazy_gfx906.dat`); exporting the variable makes rocBLAS look
for the legacy non-lazy `TensileLibrary.dat`, which is absent, and Tensile host
init then aborts the process on the first GEMM:

```text
rocBLAS error: Could not initialize Tensile host:
std::bad_alloc            # then SIGABRT (134) or SIGSEGV (139)
```

The DTK default path is already compiled into rocBLAS, so leaving the variable
unset is both sufficient and required.

## `check_bf16.py`

Verifies whether bfloat16 is numerically usable on the device and reports a
performance reference against float16.

```bash
source tools/dcu/env.sh
python3 tools/dcu/check_bf16.py
```

Exit code `0` means every bf16 check passed; `1` means bf16 numerics are not
reliable (keep `--dtype float16`); `2` means no GPU was visible.

Why a probe at all: gfx906 has no bf16 instructions — DTK's assembler rejects
both `v_cvt_f32_bf16` and `v_dot2_f32_bf16` for gfx906 while accepting
`v_pk_fma_f16` and `v_dot2_f32_f16` — so bf16 is emulated in software. Note
that `torch.cuda.is_bf16_supported()` returns True on every ROCm build and is
therefore reported but never used as a decision input.

Related environment variables:

- `VLLM_DCU_BF16=auto` (default): offer bf16 only if the built-in probe passes.
- `VLLM_DCU_BF16=1`: force bf16 on (experiments).
- `VLLM_DCU_BF16=0`: force bf16 off.

## `compressed-tensors` must be >= 0.15.0.1

The DTK image ships `compressed-tensors 0.14.0.1`, which is **not enough**. The
fork needs 0.15.0.1 (`requirements/common.txt`, restated in
`requirements/dcu.txt`): 0.14.0.1 has no
`compressed_tensors.compressors.pack_quantized`, so importing any
compressed-tensors scheme fails with

```text
ModuleNotFoundError: No module named 'compressed_tensors.compressors.pack_quantized'
  compressed_tensors_wNa8o8.py:14  <- schemes/__init__.py:15
```

which takes down every compressed-tensors model, including the production W8A8
INT8 one. A plain `pip install -r requirements/dcu.txt` upgrades it, because
`requirements/common.txt` already pins `compressed-tensors == 0.15.0.1`; the
pre-downloaded wheel is kept at `/mnt/game/port/s8/pip-wheels/` for hosts
without PyPI access:

```bash
pip install --no-deps /mnt/game/port/s8/pip-wheels/compressed_tensors-0.15.0.1-py3-none-any.whl
```

The production container `qwen38-vllm` also carries 0.14.0.1; the vLLM build it
runs is not this fork.

## `s8b_gate.sh`

Run **inside** a DTK container (it sources `s8_env.sh`, so it needs the
gfx906-capable self-built triton and the gfx906-only vLLM tree):

```bash
docker exec z100-s8 bash /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_gate.sh
```

It turns the gfx906 kill-list and the S8b conventions into executable checks
and exits non-zero if any of them regresses:

| check | expectation |
|---|---|
| arch predicates | `on_gfx906()=True`, `on_gfx9()=False`, `get_cdna_version()==0` |
| skinny GEMM | `ROCM_SKINNY_GEMM_ENABLED is False`, `VLLM_ROCM_USE_SKINNY_GEMM=0` |
| `_rocm_C` registration | exactly `['_rocm_C::paged_attention']` (no `LLMM1`/`wvSplitK*`) |
| aiter | `is_aiter_found_and_supported() is False`, all `is_*_enabled()` `None` |
| hipBLASLt | no `*gfx906*` kernel under `/opt/dtk/lib/hipblaslt/library` |
| flash-attn | `is_flash_attn_varlen_func_available() is False` |
| FP8 per-tensor / per-channel | selection raises (no silent fallback) |
| FP8 block-scaled | selects `TritonFp8BlockScaledMMKernel` **and** passes the numeric sweep in `s8b_fp8_block_probe.py` |
| W4A16 selection | `tests/kernels/quantization/test_w4a16_kernel_selection.py` passes |
| cache root | `VLLM_CACHE_ROOT` set and under `/mnt/game` |
| `compressed-tensors` | installed version >= 0.15.0.1 |

Note on the FP8 block-scaled row: S8b measured the emulated triton path on
gfx906 and found it numerically sound (mean relative error <= 1.5e-7 against the
fp32-dequantized reference over 17 shapes, all output dtypes), so the kernel is
deliberately **not** arch-gated. It is however ~36x slower than fp16 on this
part (`4096^3`: 304 ms vs 8.4 ms), because gfx906 has no FP8 unit at all. See
`docs/result/z100-08b-gating-and-requirements-result.md`.

## decode trace 判读(S49 起)

服务以 `--profiler-config {"profiler":"torch","torch_profiler_dir":"/mnt/game/s30c-prof"}`
在役时,采样规程(acceptance §8):单请求(99 tok 提示 → 256 out),
`POST /start_profile` 后**立即** `POST /stop_profile`;楔死 → restart。得到的
per-rank `*.pt.trace.json(.gz)` 用**两件套**判读,禁止只看桶中位数:

```bash
.venv/bin/python runs/tritonq/decode_stages.py <trace> runs/fulldec/<tag>_stages
.venv/bin/python runs/fulldec/classify_wholea_pos.py <trace> runs/fulldec/<tag>_shapes.json
.venv/bin/python runs/fulldec/gap_attribution.py <trace> runs/fulldec/<tag>_gaps.json
```

- `decode_stages.py`:按内核名类聚出每步分段耗时(gemv/attention/AR/idle…),
  回答"decode 各环节速度"。
- `classify_wholea_pos.py`:wholea GEMV 按发射序位分类(attn: qkv/o/gate_up/
  down;gdn: in/out/gate_up/down),给出逐形状 µs 与 GB/s。
- `gap_attribution.py`:GPU idle 归因(步间大停顿 + 步内小间隙,含边界核名)。

**告示:GEMV 桶必须看逐形状行。** down_smul 与 plain down 是**同一内核名**
(`fp8_emu_gemv_wholea_m1`),桶粒度会掩盖单形状回退——S45/S49 两次教训
(S45 N2a 的整行 staging 把 down 从 41.8µs/534 GB/s 打到 69.8µs/319 GB/s,
桶中位只 +1.8ms,逐形状分类才定罪;见 `runs/fulldec/graph_recalibration.md`)。

## 窗口微测标准跑法(S49 起)

微测一律在**空闲卡 3** 上跑(卡 3 较卡 0 同形状慢 11-13%,比值口径有效;
卡 0 被在役服务占用):

```bash
source /mnt/game/host-runtime/dtk2604/env.sh
export HIP_VISIBLE_DEVICES=3
export VLLM_DCU_FP8_GEMV_WA=1 VLLM_DCU_FP8_GEMV_STREAM_ROWS=4
export PYTHONPATH=/home/lyh/z100-vllm
.venv/bin/python .s40/gemv_bw_bench.py --anchor   # 锚:read-only 792 GB/s
```

锚点带外(历史 780-800)说明设备状态漂移,当轮数字作废。示例脚本:
`runs/fulldec/recal_down_bench.py`(down plain/smul 对照)、
`runs/fulldec/recal_smul_sweep.py`(K/N 惩罚扫描)。服务级 GEMV 环境
(`.s34/serve_deploy.sh`)与微测 env 保持一致,否则数字不可比。
