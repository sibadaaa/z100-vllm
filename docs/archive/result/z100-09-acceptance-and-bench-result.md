# S9 验收、性能对拍、诊断、固化与回滚 — 执行结果

> 归档时间：2026-10-08（容器 CST）
> 执行者：子 agent（S9 验收与对拍）
> 对应设计文档：[docs/design/z100-09-acceptance-and-bench.md](../design/z100-09-acceptance-and-bench.md)
> 前一档：[z100-08b-gating-and-requirements-result.md](z100-08b-gating-and-requirements-result.md)（S8b 禁用清单 §6.1 / 命令表 §6.2）
> 状态：🔶 **A1–A6、A8、A9 全过，A5 的 TP=4 与 A5 本体都过；A7 仅 GPU-free 部分可跑；27B W8A8 端到端跑通（默认 FULL 图暴露一个 fork 真 bug，已给最小可复现并改用 PIECEWISE）；W4A16 端到端打通并修掉 2 类 API 漂移 bug；8K 过；性能对拍完成并有归因；GDN（Qwen3.5-4B）端到端未跑成（启动 25/15 min 未 ready，内核级真机测试已过）；固化/回滚/矩阵见附录 A/B/C**

**一句话结论**：`s8b_gate.sh` 在新建的 `z100-s9`（torch 2.11 + 自编 triton 3.6.0+git7c56a5e4 + gfx906-only 4 扩展）上
**ALL GREEN（15/15 + FP8 探针 rc=0 + W4A16 选择测试 3 passed）**；A1/A2/A6/A8/A9 全过；**27B W8A8 INT8 在 TP=2 上跑通并给出请求/性能**（默认 `FULL_AND_PIECEWISE` 会在 FULL 捕获时崩，根因是
`gdn_attn.py` 调用了本树不存在的 `VllmConfig.pad_for_cudagraph`，改用 `-cc.cudagraph_mode=PIECEWISE` 后正常）；
**W4A16 端到端跑通并给出文本**，过程中修掉 2 类**已提交在树里的 API 漂移**（`gptq_shuffle` 3 参数、`gptq_gemm` 8 参数，
上游 b44895cf/main 早已改成 2/7 参数）；8K 冒烟 `TRITON_ATTN` + 7295-token 请求 4.20 s 通过。
本步在 `vllm/**` 共 6 个文件、`tests/**` 1 个文件、`tools/dcu/**` 2 个新脚本 + 1 个门禁脚本增强上有改动（§2）。

---

## 1. 目标与验收标准

### 1.1 设计文档 A1–A9（原文判据）

| # | 判据 | 命令 |
|---|---|---|
| A1 | 架构正确且已归一化 | `python -c "from vllm.platforms.rocm import _GCN_ARCH,on_gfx906,_capability_from_gcn_arch;print(...)"` → `gfx906 True (9, 0)` |
| A2 | 目标码只含 gfx906 | `dccobjdump --offloading <so> \| grep -o gfx[0-9]* \| sort -u`（S4 口径修正：用 `strings`） |
| A3 | 单卡服务 + KV cache 自动 | `vllm serve` + `curl /v1/completions`，日志 `Available KV cache memory` > 0 且未用手工 flag |
| A4 | 8K 上下文 + 后端 | 8K 请求不崩，日志 `TRITON_ATTN` |
| A5 | TP 可用 | `--tensor-parallel-size 2` 与 `4` 各完成一次完整请求 |
| A6 | fork 自带 gfx906 测试 | 5（§2 实列 7）个测试文件全绿 |
| A7 | ROCm 回归为零 | 无 DTK 环境下抽样跑 `tests/platforms/`、`tests/kernels/` |
| A8 | 诊断自证 | `python diagnostic_runner.py` 输出 DTK/每卡 arch/rocBLAS-hipBLASLt/`LD_LIBRARY_PATH`/`VLLM_DCU_*` |
| A9 | bf16 有结论 | S3 的 A/B 结果 + 判定写入文档 |

### 1.2 本步（任务书）追加验收标准

1. 环境前置门禁 `s8b_gate.sh` **ALL GREEN**（不绿不往下走）。
2. 4 卡 `gcnArchName` 全 gfx906 + `_GCN_ARCH`/`on_gfx906/on_cdna/on_gfx9` = `gfx906 / True / True / False`（S1 真机复核补做）。
3. 27B W8A8 INT8 端到端：**TP≥2**、容器 `--memory≥32g`、`--dtype float16`、**不用 `--enforce-eager`**，给出 chat/completions 输出与 TTFT/吞吐。
4. W4A16 端到端（权重按 S8b §6.2 替代路径获取，**不用 `curl -C -`**），给出数值/文本证据。
5. 8K 上下文冒烟（Qwen3-0.6B，卡 0）：`TRITON_ATTN` + 长 prompt 正常。
6. GDN/causal_conv1d（Qwen3.5-4B，1 卡）跑一次生成。
7. bf16 可选性回归（`VLLM_DCU_BF16=auto|0|1` 真实输出）+ `--dtype float16` 最终口径。
8. 性能对拍：decode ms/token、TTFT、并发吞吐 vs 生产历史基线，偏差有解释。
9. 诊断留档 + 固化清单（容器/路径/md5/triton/requirements/env）+ 回滚程序 + 已测矩阵。

### 1.3 硬约束遵守情况

| 约束 | 遵守 |
|---|---|
| 禁止执行自编译 HIP 独立可执行程序 | ✅ 本步只跑 python/pytest/`vllm serve`，无任何自编译 `.out` |
| 不 start/kill/exec/commit 生产容器 `qwen38-vllm` | ✅ 仅 `docker inspect` 只读（取 MemLimit/ShmSize）；未 commit 任何容器 |
| 不替换 DTK 库、不动 `z100-build`（2.10）构建树 | ✅ 只读 `/mnt/game/port/build-211` 取 md5；未编译、未写入 |
| 产物/缓存只落 `/mnt/game` | ✅ 新建 `VLLM_CACHE_ROOT=/mnt/game/vllm-cache-*`，全部 `/mnt/game/port/s9/*` |
| 先 `import torch` 再 `import vllm` | ✅ 所有探针/服务入口都是 `import torch; ...main()` |
| `pytest` 用 `--noconftest` | ✅ A6 全部 |
| 不用 `--enforce-eager`；显式 `--dtype float16` | ✅ 全部 serve 命令（27B/W4A16/GDN/0.6B） |
| 停摆/挂死立即 kill 并如实记录 | ✅ 27B FULL 捕获崩（§3.10.2，未硬扛）；无挂死事件 |
| 写作用域 | ✅ `vllm/**`（0 步告警 + 验收阻塞修复）、`tests/**`（验收发现的过期测试）、`tools/dcu/**`、本归档。**偏差**：`diagnostic_runner.py`、`vllm/collect_env.py` 在设计文档 §4 里要求增强，均在任务书写作用域之外 ⇒ 未改，见 §5.1 |

---

## 2. 实际改动

### 2.1 变更总表（含 `file:line`）

| 文件 | 位置 | 改动摘要 | 类别 |
|---|---|---|---|
| `vllm/model_executor/kernels/linear/scaled_mm/triton.py` | `:8`、`:27`、`:166-182` | 新增 `init_logger`/`logger`，在 `TritonFp8BlockScaledMMKernel.is_supported()` 的 ROCm 分支对 gfx906 打**一次性告警**（软件模拟、≈36×、数值已验证），**不**返回 False（硬约束 18） | 步骤 0（任务书要求） |
| `tools/dcu/s8b_gate.sh` | `:139-144`、`:169-183`、`:196-205` | 门禁把 FP8 检查升级为"block-scaled → Triton **且** gfx906 告警一次性发出"（新增 logging handler 捕获 + 1 项 CHECK，15/15） | 禁用清单固化 |
| `vllm/model_executor/kernels/linear/mixed_precision/gfx906_gptq_wna16.py` | `:53-58`、`:121-136` | `gptq_shuffle(x, bits)`（2 参）+ `gptq_gemm(...)`（去掉 `w_g_idx`，7 参）；删除已无用的 `empty_g_idx`/`assert w_g_idx` | 验收阻塞修复（W4A16） |
| `vllm/model_executor/kernels/linear/mixed_precision/exllama.py` | `:126-131`、`:155-165` | 同上（`gptq_shuffle` 2 参、`gptq_gemm` 7 参、删 `g_idx` 局部量） | 同类 API 漂移 |
| `vllm/model_executor/layers/quantization/gptq.py` | `:376`、`:392-399` | 同上（GPTQ 路径） | 同类 API 漂移 |
| `vllm/model_executor/layers/quantization/awq.py` | `:314-318`、`:357-364` | 同上（AWQ 路径，`empty` 局部量删除） | 同类 API 漂移 |
| `tests/model_executor/test_qwen3_5_gdn.py` | `:86-109`、`:227-228` | 过期 stub 修好：补 `builder.kv_cache_spec`/`vllm_config`、`common.seq_lens`、`compute_num_computed_tokens()` 访问器；`_FakeLoader.__init__` 接受 `ignore_unexpected_prefixes` | 验收发现的过期测试（BEFORE 2 failed → AFTER 20 passed） |
| `tools/dcu/s9_diag.py` | 新增（约 260 行） | A8 诊断：DTK 版本/torch.version.hip/每卡 `gcnArchName`+KFD `gfx_target_version`/`normalize_gcn_arch` 五例/`arch_from_gfx_target_version(90106)`/`_rocm_C` 算子面/rocBLAS-hipBLASLt 覆盖/`LD_LIBRARY_PATH` 逐项/`rsmi_init`/`VLLM_DCU_*`；rc=0 即通过，`S9_DIAG_JSON` 可导 JSON | 诊断（A8） |
| `tools/dcu/s9_par_dl.py` | 新增（约 120 行） | 并行 Range 下载器（每块长度校验 + 整文件 sha256），用于绕过 `resolve/main` 假 `Content-Length` | 取证工具 |

> 运行期副本：上述 `vllm/**`、`tests/**`、`tools/dcu/*.py` 都已 `cp` 进 canonical 树 `/mnt/game/port/gfx906-vllm-211/` 后复验（见 §3）。
> 仓库工作区 `/home/lyh/z100-vllm` 与 canonical 树的一致性由下面每条证据里的 `cp ... && ...` 命令保证。

### 2.2 为什么动这 4 个量化文件（论证）

- 现象：W4A16 CT 模型 serve 直接崩，`TypeError: gptq_shuffle() takes 2 positional arguments but 3 were given`
  （`gfx906_gptq_wna16.py:59` → `vllm/_custom_ops.py:852`），修掉后又在 `torch.compile` 捕获期崩
  `Argument mismatch when attempting to trace function gptq_gemm ... got 8, expected 7`（`gfx906_gptq_wna16.py:130`）。
- 权威口径：**运行时 `torch.ops._C` = `_C_stable_libtorch`**（硬约束 15），其 schema 是
  `gptq_shuffle(Tensor! q_weight, int bit)`（`csrc/libtorch_stable/torch_bindings.cpp:707`）与
  `gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, bool use_exllama, bool use_v2_format, int bit)`
  （`:701-704`），即 **2 参 / 7 参**；`vllm/_custom_ops.py:815,844` 的 Python 包装与之一致。
  上游 vLLM main 同样是 2 参 / 7 参（`_custom_ops.py:624-632,661-662`，本步联网核对）；
  **镜像自带的 vendor 0.18.1 轮子才是旧的 3 参 / 8 参**（`/usr/local/.../vllm/_custom_ops.py:684-685,644-653`）。
- 结论：本树在 sync 上游 b44895cf 时把 `_custom_ops.py` + stable C++ 更新到了新 API，但 4 个调用点没跟着改。
  最小修复 = 把调用点对齐运行时 API（对传**空** `g_idx` 的 W4A16/AWQ 调用语义完全等价：非 stable 实现本来就是
  `q_perm.numel()==0 ⇒ NULL`）。`gptq.py`/`exllama.py` 原本传真实 `g_idx`，新 API 不再接受该参数
  （stable `gptq_gemm` 内部只用 `b_gptq_qzeros.size(0)` 数分组），desc_act 的置换语义因此不再由这两个入口承担 —— 见 §5.3 残余风险。

### 2.3 容器（本步新建 `z100-s9`，其余只读复用）

```bash
docker run -d --name z100-s9 \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=32g --memory-swap=32g --shm-size=16g \
    -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'
```
装包（S8b §6.2 原样）：torch 2.11.0+das.opt1.dtk2604 轮子 + 5 个 s7 轮子（mistral_common/openai/xgrammar/apache_tvm_ffi/torchvision 0.26.0）
+ `compressed_tensors-0.15.0.1`，并把 `s7_usercustomize.py` 装到 `/root/.local/lib/python3.10/site-packages/usercustomize.py`。

---

## 3. 执行过程与原始证据

### 3.0 环境自检（`z100-s9`）

```
$ docker exec z100-s9 bash -lc 'python -c "import importlib.metadata as m;print(m.version(...))"'
VER 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7 3.4.0+git1ef59765 0.15.0.1 0.18.1+das.3266200.dtk2604
```
（`3.4.0+git1ef59765` 是镜像自带 triton；运行期被 `PYTHONPATH=/mnt/game/triton-gfx906/install` 前置替换为
`triton 3.6.0+git7c56a5e4`，见 §3.8 与附录 A。）

### 3.1 环境前置门禁：`s8b_gate.sh` → ALL GREEN

```bash
docker exec z100-s9 bash -lc 'bash /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_gate.sh 2>&1 \
  | tee /mnt/game/port/s9/s9-gate.log; echo "GATE_PIPE_RC=${PIPESTATUS[0]}"'
```
原始输出（截断，完整见 `/mnt/game/port/s9/s9-gate.log`）：
```
  device = Z100 gfx906:sramecc+:xnack-
CHECK on_gfx906/on_gfx9/cdna_version     PASS on_gfx906=True on_gfx9=False cdna_version=0
CHECK ROCM_SKINNY_GEMM_ENABLED           PASS ROCM_SKINNY_GEMM_ENABLED=False env=False
CHECK VLLM_ROCM_USE_SKINNY_GEMM=0        PASS env read = False
CHECK _rocm_C ops == [paged_attention]   PASS ['_rocm_C::paged_attention']
CHECK use_rocm_custom_paged_attention=False PASS gfx906 early return
CHECK is_aiter_found_and_supported=False PASS switches={'is_enabled': None, ...}
CHECK flash-attn unavailable             PASS
CHECK no gfx906 hipBLASLt kernels        PASS gfx906=[] library=['TensileLibrary_gfx928.co', 'extop_gfx928.co']
WARNING 10-08 22:18:48 [triton.py:176] gfx906 has no native FP8 support; block-scaled FP8 is emulated in software and measured ~36x slower than fp16 (4096^3: 304 ms vs fp16 8.4 ms). Numerics are verified (rel_err <= 1.5e-07), so this is correct but slow; prefer --dtype float16 unless FP8 weight size is required.
CHECK FP8 block-scaled -> Triton         PASS selected TritonFp8BlockScaledMMKernel (numerics verified, see s8b_fp8_block_probe.py)
CHECK FP8 block-scaled warns once on gfx906 PASS 1 warning(s): gfx906 has no native FP8 support; ...
CHECK FP8 per-tensor (dynamic) refused   PASS ValueError: Failed to find a kernel ...
CHECK FP8 per-tensor (static) refused    PASS ValueError: Failed to find a kernel ...
CHECK FP8 per-channel refused            PASS ValueError: Failed to find a kernel ...
CHECK VLLM_CACHE_ROOT under /mnt/game    PASS VLLM_CACHE_ROOT=/mnt/game/vllm-cache
CHECK compressed-tensors >= 0.15.0.1     PASS installed 0.15.0.1
CHECK_SUMMARY pass=15 fail=0
  4096^3  fp16 torch matmul  :      8.3 ms
  4096^3  triton block fp8   :    305.3 ms
FP8_BLOCK_PROBE_DONE rc=0
----- 3) W4A16 kernel-selection test -----
3 passed, 17 warnings in 6.65s
----- 4) informational: _C_stable_libtorch identity (hard rule 15) -----
  md5 93bf7b4a391ac97e2f67cc7141633cb1  size 40691216
  S7c-fixed reference: 93bf7b4a391ac97e2f67cc7141633cb1 40691216
===== S8b gate summary =====
  conventions/kill-list : rc=0
  FP8 block-scaled      : rc=0
  W4A16 selection test  : rc=0
  RESULT: ALL GREEN
GATE_PIPE_RC=0
```
> 门禁从 S8b 的 14 项升到 **15 项**：新增的正是步骤 0 的告警回归项（告警确实发出且只发一次）。

### 3.2 A1 架构归一化（真机复核，S1 遗留）

```bash
docker exec z100-s9 bash -lc 'cd /mnt/game/port/gfx906-vllm-211 && source /opt/dtk/env.sh && python -c "
import torch
print(torch.__version__, torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(i, p.name, p.gcnArchName)
from vllm.platforms.rocm import _GCN_ARCH, on_gfx906, on_cdna, on_gfx9, _capability_from_gcn_arch
print(repr(_GCN_ARCH), on_gfx906(), on_cdna(), on_gfx9(), _capability_from_gcn_arch(_GCN_ARCH))"'
```
原始输出：
```
torch 2.11.0
device_count 4
 card 0 Z100L gfx906:sramecc+:xnack-
 card 1 Z100L gfx906:sramecc+:xnack-
 card 2 Z100  gfx906:sramecc+:xnack-
 card 3 Z100  gfx906:sramecc+:xnack-
_GCN_ARCH = 'gfx906'
on_gfx906/on_cdna/on_gfx9 = True True False
_capability_from_gcn_arch = (9, 0)
```
⇒ 与任务书期望 `gfx906 / True / True / False` **逐字一致**。
`on_cdna()==True` 是上游 vLLM 的子串判定（`rocm.py:263 _ON_CDNA = any(arch in _GCN_ARCH for arch in ["gfx9", ...])`，
`"gfx9" in "gfx906"`），**不是本 fork 引入**；真正按版本走的分支用 `get_cdna_version()==0`（门禁第 1 项已锁）。

### 3.3 A2 目标码只含 gfx906

```bash
docker exec z100-s9 bash -lc 'for so in /mnt/game/port/gfx906-vllm-211/vllm/{_C.abi3.so,_rocm_C.abi3.so,_C_stable_libtorch.abi3.so,_moe_C_stable_libtorch.abi3.so}; do
  echo "--- $so"; strings "$so" | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u; done'
```
```
--- .../_C.abi3.so                      amdgcn-amd-amdhsa--gfx906
--- .../_rocm_C.abi3.so                  amdgcn-amd-amdhsa--gfx906
--- .../_C_stable_libtorch.abi3.so       amdgcn-amd-amdhsa--gfx906
--- .../_moe_C_stable_libtorch.abi3.so   amdgcn-amd-amdhsa--gfx906
```
（口径按 S4 修正：本镜像 `dccobjdump` 是断链、`llvm-objdump --offloading` 对 DTK hipfb 无效，用 `strings`。）

### 3.4 A3 + A4 单卡服务 / KV cache 自动 / 8K 上下文 / TRITON_ATTN

```bash
# 卡 0，:8002，Qwen3-0.6B，max-model-len 8192，gpu-memory-utilization 0.5
python -c "import torch;from vllm.entrypoints.cli.main import main;main()" serve \
  /mnt/game/models/Qwen3-0.6B --dtype float16 --served-model-name qwen3-0.6b \
  --host 0.0.0.0 --port 8002 --max-model-len 8192 --gpu-memory-utilization 0.5
```
原始日志（`/mnt/game/port/s9/zerob-8k-serve.log`）：
```
INFO [rocm.py:895] Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
     Overriding with TRITON_ATTN out of potential backends: ['TRITON_ATTN', 'ROCM_ATTN'].
INFO [utils.py:320] Using LBNHC KV cache layout.
INFO [gpu_worker.py:719] Available KV cache memory: 13.98 GiB
INFO [kv_cache_utils.py:2432] GPU KV cache size: 130,832 tokens,
     Maximum concurrency for 8,192 tokens per request: 15.97x
```
请求（**未加** `--kv-cache-memory-bytes` 等手工 flag；prompt 由 tokenizer 精确截到 7295 token）：
```
prompt tokens (client tokenizer) = 7295
HTTP 200 elapsed=4.20s
usage = {"prompt_tokens": 7295, "total_tokens": 7311, "completion_tokens": 16, ...}
text  = " The quick brown fox jumps over the lazy dog near the river bank. The quick"
```
⇒ A3/A4 均过（KV 自动算出 13.98 GiB；8K 上限；`TRITON_ATTN`；长 prompt 不崩）。

### 3.5 A5 TP 可用

**TP=2（Qwen3-0.6B，卡 0,1）**：
```
INFO non-default args: {... 'tensor_parallel_size': 2, 'dtype': 'float16', 'max_model_len': 2048, ...}
INFO [gpu_worker.py:719] Available KV cache memory: 14.47 GiB
HTTP 200 elapsed=2.82s
usage = {"prompt_tokens": 5, "total_tokens": 21, "completion_tokens": 16, ...}
text  = " Paris. The capital of Italy is Rome. The capital of Spain is Madrid."
```
**TP=4（Qwen3-0.6B，卡 0,1,2,3）**：
```
INFO non-default args: {... 'tensor_parallel_size': 4, 'dtype': 'float16', 'max_model_len': 2048, ...}
INFO [gpu_worker.py:719] Available KV cache memory: 14.86 GiB        （4 个 rank 一致）
INFO: Application startup complete.
GET /v1/models 200 ; POST /v1/completions 200
HTTP 200 elapsed=0.74s
usage = {"prompt_tokens": 5, "total_tokens": 21, "completion_tokens": 16, ...}
text  = " Paris. The capital of Italy is Rome. The capital of Spain is Madrid."
```
⇒ A5 **TP=2 与 TP=4 各完成一次完整请求**：均通过（TP=2 见上，TP=4 见本段；27B 的 TP=2 见 §3.10）。
**TP=2（27B W8A8，真实生产模型）**：见 §3.10。

### 3.6 A6 fork 自带 gfx906 测试（7 个文件）

```bash
for t in tests/kernels/attention/test_triton_unified_attention_gfx906.py \
         tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py \
         tests/kernels/mamba/test_fused_recurrent_packed_decode.py \
         tests/kernels/mamba/test_fused_sigmoid_gating.py \
         tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py \
         tests/model_executor/test_qwen3_5_gdn.py \
         tests/platforms/test_hygon_dtk.py ; do
  python -m pytest -q --noconftest "$t"; done
```
原始结果（`/mnt/game/port/s9/a6.log`）：
```
39 passed in 18.60s   test_triton_unified_attention_gfx906.py
12 passed in 17.19s   test_triton_unified_attn_gfx906_v1.py
 5 passed in 15.68s   test_fused_recurrent_packed_decode.py
 5 passed in 15.05s   test_fused_sigmoid_gating.py
13 passed in 178.59s  test_gguf_fixed_cols_gfx906.py
 2 failed, 18 passed  test_qwen3_5_gdn.py          ← 修复前
52 passed in 2.26s    test_hygon_dtk.py
```
`test_qwen3_5_gdn.py` 的 2 个失败**都是过期测试替身**（非产品 bug），修复前后对比：

```
BEFORE（原样）
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'seq_lens'
        vllm/v1/attention/backends/gdn_attn.py:167
E   TypeError: _FakeLoader.__init__() got an unexpected keyword argument 'ignore_unexpected_prefixes'
        vllm/model_executor/models/qwen3_5.py:1777
2 failed, 18 passed in 20.81s

AFTER（补 builder.kv_cache_spec/vllm_config、common.seq_lens、compute_num_computed_tokens()，
       _FakeLoader 接受新 kwarg）
20 passed in 23.14s
```
`seq_lens=[9,10,3]` 与 `num_computed_tokens_cpu=[8,9,0]`、`query_start_loc_cpu=[0,1,2,5]` 自洽
（真实 `CommonAttentionMetadata.compute_num_computed_tokens()` 定义即 `seq_lens - query_lens`，`backend.py:485-491`）。
⇒ 七个文件 **146 passed / 0 failed**。

### 3.7 A7 ROCm 回归（无 DTK 环境抽样）

**未执行（环境限制）**：本机是 4×Z100 的纯 DTK 机器，没有"未装 DTK"的执行环境，无法真机跑 `tests/platforms/`、`tests/kernels/` 的对照。
可给的替代证据：`tests/platforms/test_hygon_dtk.py` **52 passed**（该文件主体是 GPU-free/未装 DTK 的 stub harness，
覆盖 `is_hygon_dtk()` 的 env/DTKROOT/HYHAL 探测与 arch 归一化 fallback）；本步对 `vllm/**` 的 6 处改动全部是
"运行时 API 对齐"与"ROCm 分支内 `on_gfx906()` 包裹的告警"，非 DTK 机器上不进入（见 §2.2 与 §5.2）。
⇒ A7 记为 **部分通过**（GPU-free 范围）+ 明确"真机对照未执行"。

### 3.8 A8 诊断自证

`diagnostic_runner.py`（仓库根）**没有 DTK 章节**（只有 python/deps/rocm_platform/gitignore/requirements/registry），
而它不在本步写作用域内 ⇒ 本步在 `tools/dcu/` 新增 `s9_diag.py`，一条命令给全 A8 要求的字段：

```bash
docker exec z100-s9 bash -lc 'source /opt/dtk/env.sh; export VLLM_CACHE_ROOT=/mnt/game/vllm-cache;
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh; cd $VLLM_DIR
  S9_DIAG_JSON=/mnt/game/port/s9/s9-diag.json python tools/dcu/s9_diag.py; echo rc=$?'
```
原始输出（节选，完整 `/mnt/game/port/s9/s9-diag.log`，机读版 `s9-diag.json`）：
```
DTK version : 26.04        torch.version.hip : 6.3.26113
torch 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7 | triton 3.6.0+git7c56a5e4
vllm 0.18.1+das.3266200.dtk2604 | compressed-tensors 0.15.0.1
  card 0: Z100L gfx906:sramecc+:xnack-
  KFD node 1..4: gfx_target_version=90106
  normalize_gcn_arch('gfx916') = 'gfx906' ; ('gfx906:xnack-') = 'gfx906' ; ('gfx18446744073709551615') = None
  arch_from_gfx_target_version(90106) = 'gfx906'
  _GCN_ARCH='gfx906' capability=(9, 0) on_gfx906/on_cdna/on_gfx9=True/True/False
  registered: ['_rocm_C::paged_attention']
  rocBLAS gfx906: 100+ 个 TensileLibrary_*_gfx906.{co,dat} + TensileLibrary_lazy_gfx906.dat
  hipBLASLt gfx906: []
  LD_LIBRARY_PATH[0]=/opt/dtk/.hyhal/rocm_smi/lib  [1]=/opt/dtk/lib ... ROCBLAS_TENSILE_LIBPATH=<unset>
  rsmi_init: /opt/dtk/.hyhal/rocm_smi/lib/librocm_smi64.so{,.2,.2.8} 均 EXPORTS
  VLLM_DCU_GCN_ARCH=gfx906 VLLM_DCU_BF16=auto VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_CACHE_ROOT=/mnt/game/vllm-cache
  all A8 checks passed
S9_DIAG rc=0
```
另外留档了 `vllm collect-env`（**子命令名是 `collect-env`，不是 `collect_env`**）：
```
$ python -c "import torch;from vllm.entrypoints.cli.main import main;main()" collect-env
... HIP runtime version : 6.3.26105 ; GPU models and configuration : Z100L (gfx906:sramecc+:xnack-) ...
rc=0    （完整 /mnt/game/port/s9/collect_env.log）
```
> **修正 S8b §6.1 第 5 行**：hipBLASLt 确实没有 gfx906 kernel（门禁 PASS），但 **rocBLAS 有**完整的 gfx906 Tensile 库
> （`/opt/dtk/lib/rocblas/library/*_gfx906.{co,dat}` + lazy 库）；"[DTK 无 gfx906 kernel]"只对 hipBLASLt 成立。
> `ROCBLAS_TENSILE_LIBPATH` 保持 unset（env.sh 的毒药导出警告仍然有效）。

### 3.9 A9 bf16 可选性回归 + `--dtype float16` 最终口径

```bash
for v in auto 0 1; do VLLM_DCU_BF16=$v python /tmp/bf16_probe.py; done   # 卡 0
```
```
===== VLLM_DCU_BF16=auto =====   bf16_policy()=auto  dcu_bf16_probe()=True   supported_dtypes=[float16, bfloat16, float32]
INFO [hygon_dtk.py:398] bf16 probe passed on this device (round-trip 0.00780, gemm 0.12494, softmax 0.00024)
===== VLLM_DCU_BF16=0 =====      bf16_policy()=off   dcu_bf16_probe()=True   supported_dtypes=[float16, float32]
===== VLLM_DCU_BF16=1 =====      bf16_policy()=on    dcu_bf16_probe()=True   supported_dtypes=[float16, bfloat16, float32]
```
结论（与 S3 一致，口径收紧）：开关有效（`=0` 真的把 bf16 从 `supported_dtypes` 摘掉）；
但设备探针在历史上出现过**两次运行结果相反**（S8 §3.12，本轮三次都是 PASS），因此
**S9 与后续一切验收/生产启动一律显式 `--dtype float16`**（硬约束 18），不依赖 `auto`。

### 3.10 27B W8A8 INT8 端到端（TP=2，卡 2,3，:8003）

**命令**（`/mnt/game/port/s9/run_27b_tp2_pw.sh`，`--memory=32g` 的 `z100-s9`）：
```bash
python -c "import torch;from vllm.entrypoints.cli.main import main;main()" serve \
  /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
  --dtype float16 --tensor-parallel-size 2 --max-model-len 4096 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.9 --disable-custom-all-reduce --skip-mm-profiling \
  --enable-prefix-caching --mamba-cache-mode align -cc.cudagraph_mode=PIECEWISE \
  --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8003
```

#### 3.10.1 第 1 次尝试：默认 `FULL_AND_PIECEWISE` → **FULL 捕获期崩**（真实证据，未硬扛）

```
INFO [default_loader.py:484] Loading weights took 27.20 seconds
Capturing CUDA graphs (PIECEWISE): 100%|██████████| 51/51 [11:59<00:00, 14.11s/it]
Capturing CUDA graphs (FULL):   0%|          | 0/2 [00:00<?, ?it/s]
Worker_TP1 ERROR [multiproc_executor.py:1053] AttributeError: 'VllmConfig' object has no attribute 'pad_for_cudagraph'
  ... File "vllm/v1/worker/gpu/cudagraph_utils.py", line 698, in create_forward_fn
      File ".../model_states/mamba_hybrid.py", line 316, in prepare_attn
      File ".../attn_utils.py", line 484, in build_attn_metadata
      File "vllm/v1/attention/backends/gdn_attn.py", line 473, in build_for_cudagraph_capture
      File "vllm/v1/attention/backends/gdn_attn.py", line 400, in build
          num_actual_tokens = self.vllm_config.pad_for_cudagraph(m.num_actual_tokens)
APIServer RuntimeError: Worker failed with error ''VllmConfig' object has no attribute 'pad_for_cudagraph''
```
- 完整日志：`/mnt/game/port/s9/27b-tp2-crash-fullgraph.log`
- 性质：**不是挂死**（无 `kfd_wait_on_events`、loadavg 正常回落、进程自行退出），是 fork 树内 **`gdn_attn.py:351/400` 调用了本树
  任何地方都没有定义的 `VllmConfig.pad_for_cudagraph`**（全树 grep 只有这 2 处调用、0 处定义；vendor 0.18.1 轮子里根本没有这个符号）。
  即：sync 上游时拿到了新版 `gdn_attn.py`，但 `vllm/config/*` 没带过来对应方法。
- 影响面：**任何 GDN/mamba 混合模型 + FULL cudagraphs** 都会崩（生产容器用 `--enforce-eager`，所以现网没暴露）。
- 最小可复现配置：见上命令把 `-cc.cudagraph_mode=PIECEWISE` 去掉（或 `FULL`/`FULL_DECODE_ONLY`）即可复现。
- 处置：**不猜语义**（写错 `pad_for_cudagraph` 可能造成 FULL 图 replay 静默算错），改用 PIECEWISE 复跑（§3.10.2），
  并把该缺陷记入 §5.2 遗留 / §6.1 给 S10 与生产。

#### 3.10.2 第 2 次尝试：`-cc.cudagraph_mode=PIECEWISE` → 起服务 + 请求（**PASS**）

服务就绪证据（`/mnt/game/port/s9/27b-tp2-pw.log`）：
```
INFO [default_loader.py:484] Loading weights took 27.79 seconds
INFO [gpu_worker.py:719] Available KV cache memory: 11.59 GiB
INFO [kv_cache_utils.py:2432] GPU KV cache size: 161,792 tokens,
     Maximum concurrency for 4,096 tokens per request: 39.50x
INFO:     Application startup complete.
```
> **启动代价（重要运维数字）**：weights 27.8 s + Dynamo bytecode 26 s + inductor 编译 + **两轮 PIECEWISE 图捕获**
> （各 51 个 capture size，各约 12 min）⇒ **约 39 min 才 ready**。生产容器用 `--enforce-eager`，没有这一步；
> 一旦生产改用图模式，必须有这个启动预算（§6.1-8）。

请求 1（`/v1/chat/completions`，greedy，`max_tokens=64`）：
```
finish: length
content: "用户问的是“杭州以什么闻名？”\n\n这是一个脑筋急转弯/文字游戏。\n\n让我分析这个问题：..."
usage: {'prompt_tokens': 61, 'completion_tokens': 64, 'total_tokens': 125}
```
（复读是因为模型 chat_template 默认 `reasoning_effort=xhigh` 而 `max_tokens` 太小 —— 与
`/home/lyh/server/serve_27b_tp4.sh` 里记录的**现网已知行为**一致，不是本引擎的缺陷。）

请求 2（`/v1/completions`，greedy，`max_tokens=24`）：
```
finish: length
text  : " Paris, the largest city in the country.\nThe capital of France is Paris, the largest city in the country."
usage : {'prompt_tokens': 5, 'total_tokens': 29, 'completion_tokens': 24}
```
⇒ **27B W8A8 INT8 端到端 PASS**：文本正确、非 NaN、非 S7b 的 `"lock"` 退化。

### 3.11 W4A16 端到端（tinyllama W4A16 group128 CT，1 卡）

#### 3.11.1 权重获取（不用 `curl -C -`）：并行 Range + sha256

S8b 的诊断复现并升级：`huggingface_hub`（即便 `HF_HUB_DISABLE_XET=1`）从 `huggingface.co` 只有 ~67–230 KB/s，
727 MB 要 1–3 小时；`hf-mirror` 单连接 ~370 KB/s。改用自写 `tools/dcu/s9_par_dl.py`（8 线程定长 Range、每块校验长度、整文件 sha256）：
```
$ python /mnt/game/port/s9/par_dl.py https://hf-mirror.com/nm-testing/tinyllama-oneshot-w4a16-group128-v2/resolve/main/model.safetensors \
      /mnt/game/models/tinyllama-w4a16-group128/model.safetensors 761968600 456249d063ce70767c9bd2e16f1bca336ee1c14c486a3532e502e831d77c885d 8 8
761968600/761968600 (100.0%)
sha256 456249d063ce70767c9bd2e16f1bca336ee1c14c486a3532e502e831d77c885d expected 456249d0... match=True size=761968600 elapsed=384s
```
⇒ 与 S8b §6.2 记录的 sha256 **逐字一致**，**证明 Range 并行与整文件校验这条路是对的**（`resolve/main` 的假
`Content-Length` 不影响 Range；`-C -` 才会毁文件）。

#### 3.11.2 崩点 1：`gptq_shuffle` 3 参 vs 2 参（已修）

```
[rank0]: TypeError: gptq_shuffle() takes 2 positional arguments but 3 were given
  File ".../kernels/linear/mixed_precision/gfx906_gptq_wna16.py", line 59, in transform_w_q
    ops.gptq_shuffle(x.data, empty_g_idx, c.weight_type.size_bits)
```

#### 3.11.3 崩点 2：`gptq_gemm` 8 参 vs 7 参（已修）

```
torch._dynamo.exc.Unsupported: failed to bind arguments when attempting to inline
  Explanation: Argument mismatch when attempting to trace function gptq_gemm.
  Developer debug context: func='gptq_gemm' /mnt/game/port/gfx906-vllm-211/vllm/_custom_ops.py:815;
  args=[Tensor, Parameter, Parameter, Parameter, Parameter, bool, bool, int]; kwargs={}
  from user code: ... gfx906_gptq_wna16.py, line 130, in apply_weights -> ops.gptq_gemm(...)
```

#### 3.11.4 修复后：起服务 + 文本（PASS）

```
INFO [__init__.py:874] Using Gfx906GPTQWNA16LinearKernel for mixed-precision linear
INFO [compressed_tensors_wNa16.py:147] Using Gfx906GPTQWNA16LinearKernel for CompressedTensorsWNA16
INFO [rocm.py:895] Overriding with TRITON_ATTN out of potential backends: ['TRITON_ATTN', 'ROCM_ATTN'].
INFO [utils.py:320] Using LBNHC KV cache layout.
INFO [gpu_worker.py:719] Available KV cache memory: 8.17 GiB
INFO: Application startup complete.
```
请求（`/v1/completions`，greedy，32 token）：
```json
{"model":"tinyllama-w4a16","choices":[{"index":0,
  "text":" Paris. The capital of France is Paris. The capital of France is Paris. ...",
  "finish_reason":"length"}],
 "usage":{"prompt_tokens":6,"completion_tokens":32,"total_tokens":38}}
```
⇒ **W4A16 端到端打通**：kernel 选择面 + 权重变换（shuffle/gemm）+ 生成文本全部有证据。
（`tests/kernels/quantization/test_rocm_compressed_tensors_w4a16.py` 的 e2e 用例依赖 `tests/conftest.py` 的
`vllm_runner` fixture，在 `--noconftest` 下**不可用**（`fixture 'vllm_runner' not found`），故用真 serve 代替，
与 S8b §6.2 的预期一致。）

### 3.12 GDN / causal_conv1d 实机（Qwen3.5-4B，1 卡）

**结论：内核实机已过（A6 的真机测试），但 Qwen3.5-4B 的端到端生成在时间盒内没跑成（启动太慢，非崩溃）——如实记为未完成。**

1. **内核实机（通过）**：`tests/kernels/mamba/test_fused_recurrent_packed_decode.py` **5 passed**、
   `tests/kernels/mamba/test_fused_sigmoid_gating.py` **5 passed**、
   `tests/model_executor/test_qwen3_5_gdn.py` **20 passed**（修好过期替身后）—— 都在真 gfx906 上跑 gfx906 专用内核
   （`fused_recurrent_packed_decode` / `fused_sigmoid_gating` / GDN metadata 构建）。
2. **端到端（未完成）**：`vllm serve /mnt/game/models/Qwen3.5-4B --dtype float16 --max-model-len 4096
   --gpu-memory-utilization 0.6 --served-model-name qwen3.5-4b --port 8004`，三次尝试：

| 尝试 | 图模式 | 等待上限 | 结果 | 原始证据 |
|---|---|---|---|---|
| 1 | `PIECEWISE` | 420 s | 未就绪（超时后被脚本 kill） | `gdn-serve-attempt1.log`（已确认 `TRITON_ATTN`） |
| 2 | `PIECEWISE` | 1500 s | **25 min 仍未就绪** | `gdn2.log`（`SERVE_TIMEOUT`） |
| 3 | `NONE`（**仍开 torch.compile**，只关图捕获） | 900 s | **15 min 仍未就绪** | `gdn-serve2.log` + `gdn3.log`（`SERVE_TIMEOUT`） |

   三次都不是崩溃、不是卡死：进程 100% CPU 持续在跑（尝试 3 超时时 `TIME+ 9:40`、RSS 6.6 GiB），
   日志停在 `Dynamo bytecode transform time: 6.95 s`（`vllm-cache-gdn/torch_compile_cache/ec26fce339/…`）之后的
   inductor 编译阶段，始终没打印 `Available KV cache memory`。同时确认的正面事实：
   `Overriding with TRITON_ATTN out of ['TRITON_ATTN','ROCM_ATTN']`，且 FLA/GDN 内核已在真机执行
   （`fla/ops/utils.py:120` 的 `seq_len (16) < num_heads (32)` 提示）。
   **注册面取证**（`/mnt/game/port/s9/gfx906_ops.txt`：`torch._C._jit_get_all_schemas()` 过滤 `gfx906`，**13 个**）：
```
_C::causal_conv1d_gfx906_decode_update / _mtp_update
_C::causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode / _ratio2_packed_decode
_C::fused_recurrent_gated_delta_rule_gfx906_packed_decode
_C::fused_sigmoid_gating_delta_rule_gfx906_{decode,indexed_decode,indexed_decode_kv_state,mtp_update,prefill}
_C::gemma_fused_add_rms_norm_gfx906 / _C::gemma_rms_norm_gfx906 / _C::rms_norm_gated_gfx906
```
   **给 S10 的两条方向**：(a) 给 Qwen3.5-4B 一次**长窗口预热**（`VLLM_CACHE_ROOT` 持久化，同 flag 第二次启动复用编译缓存）；
   (b) 关掉 inductor 的 combo-kernel 自动调优（`-cc.inductor_compile_config.benchmark_combo_kernel=False`）再测启动时间。
   注意：**不要**为了快而用 `--enforce-eager`（硬约束 13）。

### 3.13 性能对拍

**口径**（对齐 `/home/lyh/server/gfx906_port/bench/round.sh` 的 base 轮；脚本 `/mnt/game/port/s9/bench.sh`、汇总 `/mnt/game/port/s9/bench-27b.log`）：

| 场景 | 本步口径 | 基线口径 | 说明 |
|---|---|---|---|
| decode | random in=128 out=128, num_prompts=2, warmups=1, conc=1, T=0, ignore_eos | 同 | **完全一致** |
| 冷 TTFT | random in=**3800** out=1, num_prompts=1, warmups=0, conc=1 | random in=**8192** out=1 | 本步 `--max-model-len 4096`（S8b §6.2 建议值）⇒ 8192 会被拒；改用 3800 并按 **每 prefill token** 对比 |
| 并发 8 吞吐 | in=512 out=128, num_prompts=16, warmups=2, conc=8 | 计划文档写 64 请求；base 轮**未跑** | 无历史对照，只报本步值 |

**结果**

| 指标 | 生产基线（:8000，TP=4，eager，bf16，sdot4 + gdn_torch） | 新引擎（:8003，TP=2，PIECEWISE，fp16） | 偏差 |
|---|---|---|---|
| 单流 decode TPOT | **122.47 ms/token**（7.75 tok/s） | **380.70 ms/token**（2.58 tok/s） | **3.11× 慢** |
| 冷 TTFT | **22,850.89 ms @8192**（2.789 ms/prefill-token） | **12,456.53 ms @3800**（3.278 ms/prefill-token） | **+17.5% / prefill-token** |
| 并发 8（in512/out128×16） | 无历史数据 | **14.78 tok/s**（out）/ 79.97 tok/s（总），Mean TTFT 13.0 s，Mean TPOT 433 ms | 无对照 |

**偏差逐条解释**（不许"沉默地慢"）：

1. **decode 3.11× ≈ 缺 FULL decode cudagraph 的代价，几乎一比一**：S7 在 0.6B 上实测
   `FULL_AND_PIECEWISE` vs eager 的 decode 是 **11.6 vs 34.6 ms/token = 2.98×**。本步因 §5.2-1
   （`VllmConfig.pad_for_cudagraph` 缺失）**只能**用 `PIECEWISE`，decode 每步没有整图 replay，
   于是落在"eager 档"附近 ⇒ 量级完全对得上。补上该方法并恢复 `FULL_AND_PIECEWISE` 后有望回到 1.0–1.3× 区间（待实测）。
2. **prefill +17.5% / token**：现网 base 用了用户自研的**纯 torch GDN 替换**；`对比测试报告-v1.md` 实测
   "关掉它在 8192 prefill 上从 22.85 s → 28.96 s（**+27%**）"。本树跑原生 FLA/Triton GDN，
   因此 +17.5% 与"缺了 prefill 侧的 GDN 加速"一致（比 +27% 略好，可能与 fp16、TP=2 的 allreduce 更少有关）。
3. **TP=2 vs 生产 TP=4**：适配计划记录过"TP2 反向"，故卡数可能贡献一小部分；但 3.11× 与 2.98× 的高度吻合
   说明主因是图模式。本步未做 27B 的 TP=4 复测（§5.1），归因以"缺 FULL 图"为主、卡数为次。
4. **不是 attention/后端问题**：日志确认 `TRITON_ATTN`；`_rocm_C` 只有 `paged_attention` 且 gfx906 早退（门禁 15/15）。
5. **不是 dtype 问题**：`--dtype float16`（硬约束 18）；S3 实测 bf16 在 gfx906 上比 fp16 **慢**（4096³ 0.36×），
   所以 fp16 只会更快，不会解释劣化。
6. FP8 与本模型无关（W8A8 INT8 走 `TritonInt8ScaledMMLinearKernel`，S8 实测 `v_dot=32`、rel_err ≤ 4.1e-04）。

### 3.14 诊断与固化留档

- `s8b_gate.sh` → `s9-gate.log`（§3.1）；`s9_diag.py` → `s9-diag.log` + `s9-diag.json`（§3.8）；`vllm collect-env` → `collect_env.log`。
- 四个扩展 md5/尺寸、triton 前缀与补丁、requirements、环境变量、容器用途、已测矩阵：**附录 A**；回滚程序：**附录 B**；已测矩阵：**附录 C**。

---

## 4. 门禁判定

| # | 判据 | 判定 | 证据指向 |
|---|---|---|---|
| A1 | `_GCN_ARCH=gfx906`、capability `(9,0)`、4 卡全 gfx906 | ✅ 通过 | §3.2（4 卡 `gcnArchName` + `_GCN_ARCH='gfx906'` + `(9,0)`；`on_gfx906/on_cdna/on_gfx9=True/True/False`） |
| A2 | 目标码只含 gfx906 | ✅ 通过 | §3.3（4 个 `.so` 各只出现 `amdgcn-amd-amdhsa--gfx906`） |
| A3 | 单卡服务 + KV cache 自动 > 0、未用手工 flag | ✅ 通过 | §3.4（`Available KV cache memory: 13.98 GiB`，无 `--kv-cache-memory-bytes`） |
| A4 | 8K 上下文 + `TRITON_ATTN` | ✅ 通过 | §3.4（`Overriding with TRITON_ATTN`；7295-token prompt HTTP 200，4.20 s；`Maximum concurrency for 8,192 tokens: 15.97x`） |
| A5 | TP=2 与 TP=4 各完成一次完整请求 | ✅ 通过 | TP=2 ✅（0.6B §3.5 KV 14.47 GiB + 27B §3.10）；TP=4 ✅（0.6B 卡 0–3，KV 14.86 GiB、HTTP 200、0.74 s，§3.5） |
| A6 | fork 自带 gfx906 测试全绿 | ✅ 通过 | §3.6（7 文件 **146 passed / 0 failed**；GDN 文件修复前 2 failed 已定位为过期替身并修好） |
| A7 | ROCm 回归为零 | 🔶 部分 | §3.7（无 DTK 机器不可得；GPU-free `test_hygon_dtk.py` 52 passed；改动均不进入非 DTK 路径） |
| A8 | 诊断自证 | ✅ 通过 | §3.8（`s9_diag.py` rc=0：DTK 26.04 / hip 6.3.26113 / 每卡 gfx906 / KFD 90106 / rocBLAS-hipBLASLt 覆盖 / LD_LIBRARY_PATH / VLLM_DCU_*；另 `collect-env` rc=0） |
| A9 | bf16 有结论 | ✅ 通过 | §3.9（auto/0/1 真实输出 + 最终口径 `--dtype float16`） |
| 门禁（设计 §验收） | A1–A9 全绿 + 回滚演练 + 文档三处更新 | 🔶 部分 | A1/A2/A3/A4/A5/A6/A8/A9 ✅；A7 受环境限制（部分）；回滚**未演练**（生产容器 = 用户自管，本步只读 inspect，见 §5.1）；文档三处更新**由 Lead 执行**（写作用域不含 `docs/z100-todolist.md`/`hygon_z100.md`） |

### 4.1 任务书追加项判定（逐条）

| 任务书项 | 判定 | 证据 |
|---|---|---|
| 0. FP8 block-scaled 选中时一次性告警 + 禁用清单更新 | ✅ 通过 | `triton.py:166-182`（`logger.warning_once`）；门禁新增第 15 项 `FP8 block-scaled warns once on gfx906 PASS`；清单更新见 §6.1-3 |
| 1. 环境前置门禁 ALL GREEN | ✅ 通过 | §3.1（`pass=15 fail=0`、FP8 探针 rc=0、W4A16 选择测试 3 passed、`RESULT: ALL GREEN`） |
| 2a. 4 卡 arch + `_GCN_ARCH/on_*`（S1 真机复核） | ✅ 通过 | §3.2 |
| 2b. 27B W8A8 端到端（TP≥2、`--memory≥32g`、fp16、无 eager） | ✅ 通过（TP=2 + PIECEWISE） | §3.10.2（文本正确）；崩溃复现与根因 §3.10.1 |
| 2c. W4A16 端到端（不用 `curl -C -`） | ✅ 通过 | §3.11（并行 Range + sha256 校验；kernel 选择 + 文本）；顺带修 2 类 API bug |
| 2d. 8K 冒烟（0.6B，卡 0） | ✅ 通过 | §3.4（`TRITON_ATTN`、7295-token 200/4.20 s） |
| 2e. GDN/causal_conv1d（4B，1 卡） | ❌ 未完成 | §3.12（三次启动 25/15 min 未 ready；内核级 5+5+20 真机测试通过；13 个 gfx906 op 注册面取证） |
| 2f. bf16 回归 + `--dtype float16` 口径 | ✅ 通过 | §3.9 |
| 3. 性能对拍（decode/TTFT/并发吞吐 + 偏差解释） | ✅ 通过 | §3.13（decode 3.11×、prefill +17.5%/token、conc8 14.78 tok/s；偏差归因到 FULL 图 + GDN prefill 补丁） |
| 4. 诊断留档 + 固化 + 回滚 + 已测矩阵 | ✅ 通过 | §3.14 + 附录 A/B/C |

---

## 5. 未完成项 / 阻塞与原因

### 5.1 明确"未执行"的项

| 项 | 原因 |
|---|---|
| 设计 §4 要求增强 `diagnostic_runner.py`、`vllm/collect_env.py` | 两个文件都在任务书写作用域之外。改由 `tools/dcu/s9_diag.py` 提供 A8 全字段；`collect-env` 原样留档（它已有 ROCm/HIP 字段，但没有 DTK/`VLLM_DCU_*` 字段） |
| 设计 §4 的新增 `tools/dcu/smoke.sh`（单卡/TP2/TP4 一键冒烟、非零退出） | 未新增脚本；等价流程以 `run_zerob.sh`（A3/A4/A5-TP2）、`run_zerob_tp4.sh`（A5-TP4）、`run_27b_tp2_pw.sh`（27B）、`run_gdn2.sh`（GDN，未 ready）跑通并留档（都在 `/mnt/game/port/s9/`，§7 给出可粘贴内容）。S10 若要固化门禁脚本，可把 §7 的片段收进 `tools/dcu/smoke.sh` |
| A7 真机（无 DTK）对照 | 本机没有非 DTK 执行环境 |
| 设计 §5.1 的 `docker commit` 影子镜像 | **未 commit**：任务书交付物未要求，且 commit 会新增数十 GB 镜像层；需要时命令见附录 B-5 |
| 设计 §5.4 回滚演练（起生产镜像确认 :8000 恢复） | 生产容器 `qwen38-vllm` 明确"由用户自行启停"，本步只做只读 `inspect` |
| 真 27B 的 `--tensor-parallel-size 4` | TP=4 需要 4 张卡全占，而 27B 带图启动实测 **≈39 min**（§3.10.2），本步已用掉两次 27B 全量启动；A5 的"TP=4 可用"改在 `Qwen3-0.6B` 上验证（A5 未指定模型），27B 用任务书建议的 **TP=2** |
| 27B 的 TTFT@8192（基线口径） | 本步 `--max-model-len 4096`（S8b §6.2 的建议命令）与基线 8192 输入冲突（请求会被拒，实测 0 successful）；改用 3800 输入，按 **每 prefill token** 对比（§3.13 说明 2） |
| AWQ / GPTQ(desc_act) 的端到端数值 | 无本地权重，且 `gfx906` 对 desc_act 明确拒绝（S8 §3.10）；只做了 API 对齐（§2.2） |
| Qwen3.5-4B（GDN）端到端生成 | 权重在本地，但三次启动（PIECEWISE 25 min、`cudagraph_mode=NONE` 15 min）都没进到 ready（§3.12）；非崩溃、非挂死，纯粹是这套栈上 torch.compile 太慢。**内核级已用真机测试覆盖**（A6 的 5+5+20 passed） |
| 27B 的 `-cc.cudagraph_mode=FULL_AND_PIECEWISE`（生产级图模式） | 本树必崩（§3.10.1，缺 `VllmConfig.pad_for_cudagraph`）；未猜实现，改用 PIECEWISE |
| （交接提醒）本步新增文件的 git 跟踪 | 整个 `tools/dcu/` 目前**未跟踪**，且 `.gitignore:257` 的 `*.sh` 会忽略 `tools/dcu/*.sh`（含增强后的 `s8b_gate.sh`）。Lead/S10 若要入库：`git add -f tools/dcu/s8b_gate.sh tools/dcu/s9_diag.py tools/dcu/s9_par_dl.py`（硬约束 14 同理适用于 `test_*.py`） |

### 5.2 本步发现、留给后续的真缺陷（**blocking-class**，含最小可复现）

1. **`VllmConfig.pad_for_cudagraph` 缺失**（FULL cudagraph + GDN/mamba 混合模型必崩）
   - 现场：§3.10.1，`AttributeError: 'VllmConfig' object has no attribute 'pad_for_cudagraph'`（`gdn_attn.py:351,400`）。
   - 最小复现：`vllm serve /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 --dtype float16 --tensor-parallel-size 2 --max-model-len 4096 -cc.cudagraph_mode=FULL_AND_PIECEWISE`（或 `FULL`）。
   - 规避（已在本步落地并验证）：`-cc.cudagraph_mode=PIECEWISE`。
   - 正解（未做，需上游语义，不可猜）：把上游 `VllmConfig.pad_for_cudagraph()`（把 token 数对齐到 FULL 图捕获桶）
     补进 `vllm/config/vllm.py`；补之前**不要**打开 FULL 图，否则要么崩、要么（若随手写一个近似实现）FULL replay 静默算错。
2. **`gptq_shuffle` / `gptq_gemm` 的调用点与运行时 stable-ABI op 不一致**（4 个文件，本步已修，见 §2.2）
   - 现象：W4A16 直接 `TypeError`（§3.11.2/3.11.3）。若不改，任何 CT/AWQ/GPTQ W4A16 模型都起不来。

---

## 6. 对后续步骤的影响

### 6.1 S10 与生产上线注意点

1. **GDN/混合模型禁止 FULL cudagraph**：必须 `-cc.cudagraph_mode=PIECEWISE`（或 `NONE`），直到 §5.2-1 修好。
   生产容器 `qwen38-vllm` 现在是 `--enforce-eager`（`CompilationMode.NONE`），**不受该 bug 影响**；一旦改成默认编译模式就会踩到。
2. **W4A16/量化模型上线前必须先过 `s8b_gate.sh`**（ALL GREEN），它已经覆盖 skinny-gemm/aiter/hipBLASLt/FP8/`_C_stable` 身份。
3. **FP8 block-scaled 的口径**（硬约束 18 已落地）：能选、数值干净、**慢 ≈36×**，选择时打一次性告警。
   README/`hygon_z100.md` 要写"Z100 上不要用 FP8 求加速，只在要省权重体积时用"。
4. **bf16**：一律显式 `--dtype float16`（A9）。
5. **`_C_stable_libtorch` 身份**是硬门禁（`93bf7b4a…`/40,691,216 B）；任何重建后都要复核（附录 A 的 md5 矩阵与附录 B-3）。
6. 27B 的 KV/graph 与对拍数字见 §3.10/§3.13；性能偏差解释见 §3.13。
7. S10 做 GGUF/MoE/MTP/多模态时，**继续先跑本档 §7 的复现片段**（gate → 单卡 → TP → 目标模型），
   并把"FULL 图不可用"当作已知前提写进 S10 的验收。
8. **启动时间预算（重要）**：图模式在 Z100 上非常贵 —— 27B/TP=2 带 `PIECEWISE` + 51 个 capture size 实测
   **两轮捕获、≈39 min 才 ready**（§3.10.2）；Qwen3.5-4B（GDN）**25 min 都到不了 ready**（§3.12）。
   生产若要上图模式，必须给足启动窗口，并把 `VLLM_CACHE_ROOT` 放在持久盘（编译缓存可复用，图捕获不可复用）。
   快速 bring-up 用 `-cc.cudagraph_mode=NONE`（**仍然开 torch.compile**，只是不抓图），**不要**用 `--enforce-eager`。
9. **GDN/混合模型的 FULL 图缺陷（§5.2-1）是当前最大的"性能欠账"**：decode 的 3.11× 差几乎全部来自它（§3.13-1）。
   修好 `VllmConfig.pad_for_cudagraph` 后应重跑 §3.13 的 decode 对拍。

---

## 7. 复现方式

```bash
# ---------- 0) 前置：宿主 docker 可用；z100-s9 在（32g/32g, shm 16g, /src ro, /mnt/game rw）
#              canonical 树 /mnt/game/port/gfx906-vllm-211 带 gfx906-only 4 扩展
#              宿主对 /mnt/game 只读 ⇒ 所有 /mnt/game 写入都经 docker exec

# ---------- 1) 同步本步改动进 canonical 树 ----------
docker exec z100-s9 bash -c '
  V=/mnt/game/port/gfx906-vllm-211
  cp /src/vllm/model_executor/kernels/linear/scaled_mm/triton.py           $V/vllm/model_executor/kernels/linear/scaled_mm/
  for f in gfx906_gptq_wna16.py exllama.py; do
    cp /src/vllm/model_executor/kernels/linear/mixed_precision/$f          $V/vllm/model_executor/kernels/linear/mixed_precision/; done
  cp /src/vllm/model_executor/layers/quantization/{gptq.py,awq.py}          $V/vllm/model_executor/layers/quantization/
  cp /src/tests/model_executor/test_qwen3_5_gdn.py                          $V/tests/model_executor/
  cp /src/tools/dcu/{s8b_gate.sh,s9_diag.py,s9_par_dl.py}                   $V/tools/dcu/
  chmod +x $V/tools/dcu/s8b_gate.sh'

# ---------- 2) 环境（每次新 shell；本步：27B 用卡 2,3；0.6B/GDN 用卡 0） ----------
docker exec -it z100-s9 bash
source /opt/dtk/env.sh
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache          # s*_env.sh 也会兜底到 /mnt/game
source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh   # 自编 triton + VLLM_DIR + tools/dcu 进 PYTHONPATH
cd $VLLM_DIR

# ---------- 3) 前置门禁（必须 ALL GREEN；期望 CHECK_SUMMARY pass=15 fail=0） ----------
docker exec z100-s9 bash /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_gate.sh

# ---------- 4) A1/A2 ----------
python -c "import torch;from vllm.platforms.rocm import _GCN_ARCH,on_gfx906,on_cdna,on_gfx9,_capability_from_gcn_arch;print(_GCN_ARCH,on_gfx906(),on_cdna(),on_gfx9(),_capability_from_gcn_arch(_GCN_ARCH))"
for so in $VLLM_DIR/vllm/_C.abi3.so $VLLM_DIR/vllm/_rocm_C.abi3.so $VLLM_DIR/vllm/_C_stable_libtorch.abi3.so $VLLM_DIR/vllm/_moe_C_stable_libtorch.abi3.so; do strings $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u; done

# ---------- 5) A3/A4（8K 冒烟，卡 0）+ A5(TP2) ----------
bash /mnt/game/port/s9/run_zerob.sh          # 内含 8K serve/请求 与 0.6B TP=2 请求，日志 zerob.log

# ---------- 6) A6（7 个测试文件，必须 146 passed） ----------
for t in tests/kernels/attention/test_triton_unified_attention_gfx906.py \
         tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py \
         tests/kernels/mamba/test_fused_recurrent_packed_decode.py \
         tests/kernels/mamba/test_fused_sigmoid_gating.py \
         tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py \
         tests/model_executor/test_qwen3_5_gdn.py tests/platforms/test_hygon_dtk.py; do
  python -m pytest -q --noconftest "$t"; done

# ---------- 7) A8 ----------
S9_DIAG_JSON=/mnt/game/port/s9/s9-diag.json python tools/dcu/s9_diag.py     # 期望 S9_DIAG rc=0
python -c "import torch;from vllm.entrypoints.cli.main import main;main()" collect-env | head -40

# ---------- 8) 27B W8A8 端到端（卡 2,3，:8003；FULL 图不可用 ⇒ PIECEWISE） ----------
bash /mnt/game/port/s9/run_27b_tp2_pw.sh > /mnt/game/port/s9/27b-tp2-pw.log 2>&1 &
# 就绪后期望日志含：Available KV cache memory / Maximum concurrency / Application startup complete
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"用一句话介绍杭州"}],"max_tokens":48,"temperature":0}'
# 复现 FULL 图崩溃（预期 AttributeError: 'VllmConfig' object has no attribute 'pad_for_cudagraph'）：
#   把上面脚本里的 -cc.cudagraph_mode=PIECEWISE 换成默认（删掉该行）

# ---------- 9) W4A16 端到端 ----------
python tools/dcu/s9_par_dl.py https://hf-mirror.com/nm-testing/tinyllama-oneshot-w4a16-group128-v2/resolve/main/model.safetensors \
  /mnt/game/models/tinyllama-w4a16-group128/model.safetensors 761968600 \
  456249d063ce70767c9bd2e16f1bca336ee1c14c486a3532e502e831d77c885d 8 8      # 期望 match=True
bash /mnt/game/port/s9/run_w4a16.sh > /mnt/game/port/s9/w4a16-serve3.log 2>&1 &
curl -s http://127.0.0.1:8004/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"tinyllama-w4a16","prompt":"The capital of France is","max_tokens":32,"temperature":0}'

# ---------- 10) GDN（Qwen3.5-4B，卡 0） ----------
bash /mnt/game/port/s9/run_gdn2.sh           # GDN（4B）：cudagraph_mode=NONE；实测 15 min 仍未 ready，见 §3.12
#   要复现图模式更慢的版本：把 -cc.cudagraph_mode=NONE 换成 PIECEWISE（实测 25 min 仍未 ready）
```

---

## 8. 参考

- 设计：[docs/design/z100-09-acceptance-and-bench.md](../design/z100-09-acceptance-and-bench.md)
- 前置归档：[S8b](z100-08b-gating-and-requirements-result.md)（禁用清单 §6.1 / 命令表 §6.2）、[S8](z100-08-attention-and-quant-paths-result.md)、[S7](z100-07-memory-kvcache-graph-result.md)、[S7c](z100-07c-packed-silu-csrc-fix-result.md)
- 生产基线数据（只读）：`/home/lyh/server/gfx906_port/bench/`（`results/base/bench_decode.log`、`bench_ttft.log`、`serve_args.txt`）、`/home/lyh/server/gfx906_port/对比测试报告-v1.md`、`/home/lyh/server/serve_27b_tp4.sh`
- 上游 API 核对：`vllm/_custom_ops.py`（vLLM main：`gptq_shuffle(q_weight, bit)`、`gptq_gemm(a,...,bit)`）
- 本步产物目录：`/mnt/game/port/s9/`（所有 `.log`、`s9-diag.json`、`run_*.sh`、`*-serve.log`、`a6.log`）
---

# 附录 A 固化清单

## A.1 容器（全部保留、未 `--rm`；生产容器只读 inspect）

| 容器 | 用途 / 栈 | 备注 |
|---|---|---|
| **`z100-s9`**（本步新建） | S9 验收/对拍运行面：torch 2.11.0+das.opt1.dtk2604 + 自编 triton 3.6.0+git7c56a5e4 + canonical 树 4 扩展；`--memory=32g --memory-swap=32g --shm-size=16g` | `-v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game` |
| `z100-s8` | S8/S8b 运行面（同栈，`--memory=10g`） | 保留；本步只读复用经验 |
| `z100-serve` / `z100-eager` | S7 serve / eager 实验面（`compressed-tensors 0.14.0.1`） | 保留，未动 |
| `z100-build-211` | torch 2.11 构建容器（**构建树 `/mnt/game/port/build-211` 只读参考**） | 本步只取 md5 |
| `z100-build` / `z100-triton` | **2.10 基线栈**（torch 2.10.0+das…20260325 + triton 3.4.0+git1ef59765 + vendor vllm 0.18.1 轮子） | 回滚面，未动 |
| `qwen38-vllm` | 生产（:8000），`Exited(137)`，用户自管 | **本步未 start/kill/exec/commit**；仅 `inspect`（`MemLimit=0`、`Shm=200g`） |

## A.2 `/mnt/game` 关键路径

| 路径 | 内容 |
|---|---|
| `/mnt/game/port/gfx906-vllm-211` | **canonical 目标栈树**（运行面：4 个 gfx906-only 扩展 + 本步改动） |
| `/mnt/game/port/gfx906-vllm`（2.10）、`gfx906-vllm-triton` | 历史树（勿与 canonical 混用） |
| `/mnt/game/port/build-211/temp` | torch 2.11 构建树原始产物（`.so`），只读参考 |
| `/mnt/game/port/s4c-probe/raw-before`、`s7c/pre-fix`、`s7c/csrc-only` | **回滚件**（见附录 B-3） |
| `/mnt/game/triton-gfx906/{install,cache,src}` | 自编 triton 安装前缀 / JIT 缓存 / 源码 |
| `/mnt/game/port/{wheels-211, s7/pip-wheels, s8/pip-wheels}` | 离线轮子（torch 2.11 / 5 个依赖 / compressed-tensors 0.15.0.1） |
| `/mnt/game/models/{Qwen3-0.6B, Qwen3.5-4B, Qwen3.8-27B-Channel-INT8-w8a8, tinyllama-w4a16-group128}` | 4 个本地模型（第 4 个本步补齐） |
| `/mnt/game/vllm-cache*` | 编译/图缓存（`VLLM_CACHE_ROOT`） |
| `/mnt/game/port/s9/` | 本步全部证据：`s9-gate.log`、`s9-diag.{log,json}`、`collect_env.log`、`a6.log`、`zerob*.log`、`27b-tp2*.log`、`w4a16-*.log`、`gdn*.log`、`bench-*.log`、`run_*.sh/*.py`、`par_dl.py` |

## A.3 四个扩展：md5 / 尺寸（构建树原始产物 vs 安装面）

| 面 | `_C` | `_rocm_C` | `_C_stable_libtorch` | `_moe_C_stable_libtorch` |
|---|---|---|---|---|
| **安装面（canonical，运行用）** | `6fba0596…` 78,102,728 | `5d1e50e1…` 35,156,368 | **`93bf7b4a…` 40,691,216** | `996b282c…` 29,748,928 |
| 构建树 `build-211/temp` | `048736b1…` 78,102,728 | `019b327b…` 35,156,368 | `ad6b1d66…` 40,691,216 | `af01ed9b…` 29,748,928 |
| `s4c-probe/raw-before`（S4c 原始） | `048736b1…` | `019b327b…` | `7068229e…` 40,688,656 | `af01ed9b…` |
| `s7c/pre-fix`（SiLU 修复前 = 回滚件） | `6fba0596…` | `5d1e50e1…` | `cc2aca69…` 40,688,656 | `996b282c…` |
| `gfx906-vllm-triton/vllm` | `6fba0596…` | `5d1e50e1…` | `773f6882…` 40,688,664 | `996b282c…` |

- `_C_stable_libtorch` 从 `40,688,656`（S4c/`cc2aca69`）变成 `40,691,216`（S7c 修复后）＝**+2,560 B**；安装面的 `93bf7b4a…` 是硬约束 15 要求的身份（门禁第 4 段复核一致）。
- 构建树与安装面同尺寸不同 md5（`_C`/`_rocm_C`/`_moe_C_stable`）＝`cmake --install` 后处理差异；**权威的是安装面**，不要把 `build-211/temp/*.so` 直接覆盖过去。
- 运行期 `torch.ops._C` 实际由 `_C_stable_libtorch` 提供（本步 3.11 的 `gptq_*` schema 取证再次印证）。

## A.4 triton 补丁与安装前缀

| 项 | 值 |
|---|---|
| 安装前缀 | `/mnt/game/triton-gfx906/install`（978 MB）；`importlib.metadata.version("triton")` = `3.6.0+git7c56a5e4` |
| 补丁（容器内） | `/mnt/game/triton-gfx906/triton-gfx906-3.6.0.patch`（5 文件 48 行：VEGA ISAFamily ×3 + `driver.py` DTK `hipDeviceProp_t` + `compiler.py` **SFU hazard** `s_nop` 修复） |
| 补丁（仓库副本） | `tools/dcu/triton-gfx906-3.6.0.patch`，md5 `59f976927f7800850a4ad79a9050a59a` |
| JIT 缓存 | `TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache`（改 triton include/后端 `.py` 后**必须清缓存**，硬约束 12） |
| 启用方式 | `PYTHONPATH=${TRITON_INSTALL}:${VLLM_DIR}:/mnt/game/port/gfx906-vllm-211/tools/dcu`（`tools/dcu/s7_env.sh` 自动） |

## A.5 requirements

| 文件 | canonical md5 | 关键 pin |
|---|---|---|
| `requirements/dcu.txt` | `1629165197554e87d38eabe5bd3a78f6` | `:56` 自解释 `compressed-tensors==0.15.0.1`（`-r common.txt` 已带） |
| `requirements/dcu-build.txt` | `376e3749c1354e89876f0706249a77d6` | 构建期依赖 |

## A.6 环境变量清单（本步实采，A8 `s9-diag.json`）

| 变量 | 值 | 说明 |
|---|---|---|
| `VLLM_DCU_GCN_ARCH` | `gfx906` | `tools/dcu/env.sh` 从 KFD topology 推得，避免 amdsmi 的 `gfx18446744073709551615` |
| `VLLM_DCU_BF16` | `auto` | 探针可用，但启动一律 `--dtype float16`（硬约束 18 / A9） |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `0` | gfx906 无 MFMA，skinny GEMM 无意义（门禁锁） |
| `VLLM_CACHE_ROOT` | `/mnt/game/vllm-cache`（各实验用 `-27b/-a6/-gdn/-zerob/-w4a16` 后缀） | 硬约束 17；否则写进容器可写层（一次冒烟 2.1 GB） |
| `PYTHONPATH` | `${TRITON_INSTALL}:${VLLM_DIR}:${tools/dcu}` | 自编 triton 优先、canonical 树优先 |
| `TRITON_CACHE_DIR` | `/mnt/game/triton-gfx906/cache` | 硬约束 12 |
| `TRITON_HIP_LLD_PATH` | `/opt/dtk/llvm/bin/ld.lld` | DTK 自带 lld |
| `HIP_VISIBLE_DEVICES` | 按卡（27B=2,3；0.6B/GDN=0；0.6B TP2=0,1） | 显式指定，避免误占生产卡 |
| `ROCBLAS_TENSILE_LIBPATH` | **必须 unset** | 导出即 `std::bad_alloc` / SIGABRT（S3 毒药导出） |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | 27B / 4B 起服务时设（本步） |
| `TORCH_BLAS_PREFER_HIPBLASLT` | （生产设 `0`；本步未设） | 不影响：hipBLASLt 无 gfx906 kernel |
| `--dtype` | `float16`（全部 serve） | 硬约束 18 |
| `-cc.cudagraph_mode` | `PIECEWISE`（GDN/混合模型必须） | 见附录 D-1 |

---

# 附录 B 回滚程序

**B-1 退回 2.10 基线（vendor 栈，即生产那一套）**
```bash
# 容器与树都还在，什么都不用重建：
#   z100-build / z100-triton : torch 2.10.0+das…20260325 + triton 3.4.0+git1ef59765 + vendor vllm 0.18.1 轮子
#   /mnt/game/port/gfx906-vllm : 2.10 树
docker exec -it z100-triton bash          # 或 z100-build（注意别在 z100-build 里重编）
source /opt/dtk/env.sh                    # 不要 source tools/dcu/s7_env.sh（那会前置自编 triton + 211 树）
python -c "import torch; print(torch.__version__)"   # 期望 2.10.0+das.opt1.dtk2604.20260325.g6b060a
vllm serve <model> --dtype float16 ...    # 直接走 site-packages 的 vendor 轮子
```
**B-2 只撤自编 triton（保留 211 树）**：不导出 `PYTHONPATH=/mnt/game/triton-gfx906/install:…`、不 source `s7_env.sh`。
DTK 原装 triton 目录从未被改（硬约束 4），因此无需恢复。

**B-3 撤掉 S7c 的 fp16 packed-SiLU 修复（`_C_stable_libtorch`）**
```bash
docker exec z100-s9 bash -c '
  cp /mnt/game/port/s7c/pre-fix/vllm/_C_stable_libtorch.abi3.so \
     /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so
  md5sum /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so'
# 期望：cc2aca69ca9a9afcb67b7428ede181b1  40688656
# 回退后 _C.silu_and_mul / apply_moe_activation(SILU) 的 fp16 packed 分支会重新出现非有限值 ⇒ 仅用于复现 S7b/S7c
```
**B-4 撤掉本步改动**：`vllm/**` 6 个文件用 `git checkout --` 或从 `/src`（只读工作区）取旧版；
`tests/model_executor/test_qwen3_5_gdn.py` 同理；FP8 告警块可直接删（无行为影响，保留亦可）；
`tools/dcu/s9_diag.py`、`tools/dcu/s9_par_dl.py` 是纯工具，删除不影响功能。

**B-5 影子镜像（本步未 commit，命令备查）**
```bash
docker commit z100-s9 gfx906-vllm:z100-dtk2604-20261008     # 只 commit 不 save
```

**B-6 生产容器**：`qwen38-vllm`（:8000）当前 `Exited(137)`，**由用户自行 `docker start`**；
其镜像 tag 未变、未被本步任何命令触碰，因此"新引擎失败不影响生产"这一条在**镜像与容器层面成立**
（回滚演练本身未执行，原因见 §5.1）。

---

# 附录 C 已测矩阵（供 S10 README 使用）

| 模型 | 量化 | 卡数 | 栈 | 结果 |
|---|---|---|---|---|
| Qwen3-0.6B | fp16 | 1（卡0） | 211 + 自编 triton | ✅ 8K：`TRITON_ATTN`、KV 13.98 GiB、130,832 tok、7295-token 请求 4.20 s |
| Qwen3-0.6B | fp16 | 2（卡0,1） | 同上 | ✅ TP=2：KV 14.47 GiB、请求 200、文本正确 |
| Qwen3-0.6B | fp16 | 4（卡0–3） | 同上 | ✅ TP=4：KV 14.86 GiB、请求 200、0.74 s、文本正确 |
| Qwen3.5-4B | fp16（GDN 混合） | 1（卡0） | 同上 | ❌ 端到端未跑成：启动 25 min（PIECEWISE）/ 15 min（NONE）仍未 ready（非崩溃，§3.12）；内核级 5+5+20 tests 真机通过 |
| tinyllama W4A16 group128 | compressed-tensors W4A16 | 1（卡0） | 同上 | ✅ `Gfx906GPTQWNA16LinearKernel`、KV 8.17 GiB、文本正确（修复 §3.11.2/3.11.3 两个 bug 后） |
| Qwen3.8-27B-Channel-INT8-w8a8 | compressed-tensors W8A8-int8（channel） | 2（卡2,3） | 同上，`-cc.cudagraph_mode=PIECEWISE` | ✅ KV 11.59 GiB / 161,792 tok / 39.50×；文本正确；decode 380.70 ms/tok、TTFT 12.46 s@3800、conc8 14.78 tok/s（§3.13） |
| Qwen3.8-27B-Channel-INT8-w8a8 | 同上 | 2 | 同上，默认 `FULL_AND_PIECEWISE` | ❌ FULL 捕获崩：`AttributeError: 'VllmConfig' object has no attribute 'pad_for_cudagraph'`（§3.10.1） |
| Qwen3-0.6B | fp16 | 1 | 生产 vendor 栈（torch 2.10 / triton 3.4 / vllm 0.18.1 轮子） | ⬜ 未测（历史生产数据见 §3.13） |
| 任意 | FP8 block-scaled | 1 | 211 + 自编 triton | ⚠️ 数值干净但 **≈36×** 慢（门禁探针 4096³ 305.3 ms vs fp16 8.3 ms）；选择时一次性告警 |

---

# 附录 D 本步修掉 / 发现的缺陷汇总

| # | 缺陷 | 性质 | 处置 |
|---|---|---|---|
| D-1 | `VllmConfig.pad_for_cudagraph` 缺失 → GDN/混合模型 FULL cudagraph 捕获必崩 | fork 树内 API 缺口（sync 上游只带了一半） | **未修**（需上游语义，不可猜）；规避 `-cc.cudagraph_mode=PIECEWISE`；已给最小复现（§3.10.1/§5.2-1） |
| D-2 | `gptq_shuffle` 调用点 3 参 vs 运行时 2 参（`gfx906_gptq_wna16`/`exllama`/`gptq`/`awq`） | sync 后调用点漂移 | ✅ 已修（4 文件；对"空 g_idx"调用语义等价，§2.2） |
| D-3 | `gptq_gemm` 调用点 8 参 vs 运行时 7 参（同上 4 文件） | 同上 | ✅ 已修 |
| D-4 | `tests/model_executor/test_qwen3_5_gdn.py` 2 个过期替身（缺 `seq_lens`/`compute_num_computed_tokens`、`_FakeLoader` 不认新 kwarg） | 过期测试 | ✅ 已修（20 passed） |
| D-5 | 测试 `test_rocm_compressed_tensors_w4a16.py` 的 e2e 用例依赖 `tests/conftest.py`（`vllm_runner` fixture），`--noconftest` 下不可跑 | 环境限制（非 bug） | 用真 serve 替代（§3.11.4） |
| D-6 | `resolve/main` 端点返回假 `Content-Length`，`curl -C -` 会把半成品续小 | 端点行为 | 已给并行 Range + sha256 方案（`tools/dcu/s9_par_dl.py`，§3.11.1） |
| D-7 | `torch._dynamo` metrics 每图打 `TypeError: Object of type function is not JSON serializable`（134 次） | 无害噪声 | 记录备查（分析日志时勿误判为失败） |
| D-8 | `on_cdna()` 在 gfx906 上返回 True（上游子串判定） | 上游行为，非本 fork | 记录；版本相关分支请用 `get_cdna_version()==0` |
