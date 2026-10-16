# S5 csrc gfx906 内核 + `_rocm_C` + skinny GEMM 排除

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md)
> 前置：[S4](z100-04-build-system-dtk.md) · 状态：⬜ 待做

## 目的

1. 用 DTK 的 clang/hipcc 编译本 fork 的 gfx906 专用内核；
2. 补齐 das 轮子缺失的 `_rocm_C`（`LLMM1` / `wvSplitK` / `wvSplitKQ` / `paged_attention_rocm`）；
3. 确保 **gfx906 不会进入任何 MFMA 假设路径**。

## 关键事实

- `csrc/rocm/attention.cu`、`csrc/rocm/skinny_gemms.cu` 的 `__HIP__GFX9__` 只在 `gfx90a/gfx942/gfx950` 下定义 ⇒ gfx906 自动走非 MFMA 分支（**正确，别改**）。
- `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py:53` 注释明确：`on_gfx9()` 有意排除 gfx906/gfx908。
- `CMakeLists.txt:1512-1540` 只把 `gfx1250` 从 skinny GEMM 源里排除；gfx906-only 构建仍会编译并注册 `wvSplitK*`，但运行时不可用。
- `csrc/mamba/gfx906_decode_kernels.cu` 已加入构建列表（`CMakeLists.txt:374`），含 13 个 torch op。

## 做法

### 1. 先做单文件试编（**只编译，绝不运行产物**）

```bash
source /opt/dtk/env.sh && source tools/dcu/env.sh
mkdir -p /mnt/game/port/probe && cd /mnt/game/port/probe
for f in csrc/mamba/gfx906_decode_kernels.cu csrc/layernorm_kernels.cu \
         csrc/quantization/gguf/gguf_kernel.cu csrc/rocm/attention.cu; do
  echo "=== $f"; hipcc -std=c++17 --offload-arch=gfx906 -c \
    -I$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/include \
    -I$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/include/torch/csrc/api/include \
    -Icsrc -I. /mnt/game/port/gfx906-vllm/$f -o /tmp/probe.o 2>&1 | tail -5
done
```

> ⚠️ 禁止把 `-o` 改成可执行文件并运行。2026-10-07 因运行自编译 HIP 程序触发 hydcu 驱动死锁，机器被迫重启。

### 2. CMake 排除 gfx906 的 skinny GEMM

在 `CMakeLists.txt` 的 `_rocm_C` 段（约 L1512-1545）把 gfx906 与 gfx1250 一同排除：

```cmake
  set(VLLM_SKINNY_ARCHES ${VLLM_GPU_ARCHES})
  list(FILTER VLLM_SKINNY_ARCHES EXCLUDE REGEX "gfx1250|gfx906")
  if(VLLM_SKINNY_ARCHES)
    ...
  else()
    message(STATUS "Skipping skinny_gemms (gfx1250/gfx906 only)")
    list(APPEND VLLM_ROCM_EXT_FLAGS "-DVLLM_SKIP_SKINNY_GEMMS")
  endif()
```

同时把运行期判据对齐（否则会调用未注册算子）：`vllm/model_executor/kernels/linear/scaled_mm/rocm.py` 里现有的 `gfx1250` 短路，扩展为 `gfx1250 或 gfx906`，理由文案区分二者。

`VLLM_ROCM_USE_SKINNY_GEMM=0`（`tools/dcu/env.sh` 已默认）保留为运行期双保险。

### 3. 逐文件修复

对试编失败的文件，按以下顺序处置（**逐文件、可回退**）：

1. DTK clang 版本不支持的内建 → 局部替换为等价写法（例如手工 bit 运算替代缺指令内建）；
2. 仅影响非关键路径 → 从 CMake 源列表移除并 `-DVLLM_SKIP_*` 关掉注册；
3. 不可修复 → 记录到 `docs/hygon_z100.md`，让对应路径走 Triton/torch 回退。

### 4. 验收 `_rocm_C`

```bash
python - <<'PY'
import torch, vllm._rocm_C  # noqa
ns = torch.ops._rocm_C
for op in ("LLMM1", "wvSplitK", "wvSplitKQ", "paged_attention"):
    print(op, hasattr(ns, op))
PY
```

## 验收（门禁）

| 判据 | 命令 |
|---|---|
| gfx906 专用测试通过 | `python -m pytest tests/kernels/attention/test_triton_unified_attention_gfx906.py tests/kernels/mamba/test_fused_recurrent_packed_decode.py tests/kernels/mamba/test_fused_sigmoid_gating.py tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py tests/model_executor/test_qwen3_5_gdn.py -v` |
| 目标码只含 gfx906 | `dccobjdump --offloading <so> \| grep -o gfx[0-9]* \| sort -u` |
| 未注册 skinny 算子且运行期已短路 | 上表 `hasattr(torch.ops._rocm_C,"wvSplitK")` 允许为 False，但**不得**在日志里出现 `_rocm_C has no wvSplitK` 之类的运行期 AttributeError |

## 失败处置

| 现象 | 处置 |
|---|---|
| `VLLM_DISPATCH_HALF_TYPES` 相关"not implemented for BFloat16" | 预期：gfx906 的 MTP/GDN 内核仅 fp16；用 fp16 跑，或转 [S10](z100-10-phase-b-features.md) 决定是否扩展派发 |
| `invalid device function` | 通常是目标码里混入了非 gfx906（检查 `PYTORCH_ROCM_ARCH`）或误用了 gfx928 预编译库（flash-attn/hipBLASLt） |
| GGUF 相关内核编译失败 | GGUF 属阶段 B，可先跳过不影响阶段 A；在 [S10](z100-10-phase-b-features.md) 处理 |

## 参考

- 前序计划 §4 Phase 2A 的移植优先级清单（csrc 优先）
- `artifacts/fork-vs-upstream031-inventory.md`（76 个含 gfx906 的文件清单）
