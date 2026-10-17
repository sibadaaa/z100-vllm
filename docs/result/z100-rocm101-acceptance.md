# Z100：ROCm 10.1 + torch 2.16 新栈适配验收报告（执行中，2026-10-18）

> 计划书：`docs/z100-torch216-rocm101-adapt-plan.md`（D1–D7、P0–P5、L-1→L5）。
> 本文是**执行结果**，与计划分离；所有数字均来自本次真机执行，证据路径可复查。

## 1. 当前结论（一句话）

新栈**基础设施全部打通并可运行真实模型推理**：torch 2.16 + ROCm 10.1 + triton 3.6.0(rocm101)
+ SFU 补丁 clang 重编的 vLLM 扩展、gfx906 RCCL（TP>1 通信）、独立运行容器与 env 契约均完成；
DTK 回滚通道的 4 个 merge 回归已修复且同树基线已冻结。**剩余阻塞是 Triton 内核在
serving 小序列形状下的数值错误（F14）与 bf16 模型级 NaN（F21）**——正确性验证用
`VLLM_GFX906_ATTN_EAGER_FALLBACK=1` + fp16 可完成，性能门禁（L4/L5）需等 triton 侧修复。

## 2. 交付身份

| 项 | 值 |
|---|---|
| 分支 | `rocm101-torch216`（从 `sync/upstream-20261016-v2` @ `6067e19a6` 拉出） |
| 代码面 HEAD | `29d605c44`（L-1 冻结时）+ 后续 tools/docs/tests 提交；当前 HEAD 见 `git log` |
| torch | `2.16.0.dev20261016+rocm10.1`（自编 wheel-sfu，sha256 `46a5f447…`，gfx906 code objects，SFU 补丁 clang 24） |
| Triton | `3.6.0`（`/mnt/game/triton-z100-rocm101/install`，LLVM 22.0.0git pin `f6ded0be`，gfx906 路由 + Z100 overlay） |
| ROCm | TheRock `10.1`，`ROCM_PATH=/opt/rocm/core-10.1`，HIP `7.16.26385` |
| RCCL | 自编 gfx906：`/mnt/game/rccl-gfx906/lib/librccl.so.1.0` md5 `4889939094649fef320ce9e28e687206`（源码 `rocm-systems@7da43026`） |
| vLLM 扩展（新栈） | `_C 9302ece8` / `_C_stable_libtorch ffc176af` / `_moe_C_stable 7b4185c6` / `_rocm_C a250420b`（`/mnt/game/port/gfx906-vllm-rocm101/vllm/`） |
| DTK 基线 .so | `_C d44117ae` / `_C_stable_libtorch e303b0b2`（F1 修复）/ `_moe 50af36b9` / `_rocm_C f205be8f` |
| 运行容器 | `z100-vllm-rocm101-rt`（image `z100-rocm10.1:24.04-r642`） |
| env 契约 | `tools/dcu/env-rocm101.sh`；栈切换 `tools/dcu/use_stack.sh` |

## 3. 门禁状态

| 门 | 状态 | 证据 / 关键数字 |
|---|---|---|
| **L-1 DTK 同树基线** | ✅ PASS | `/mnt/game/z100-baseline/`（REPORT.md + 13 产物）；canonical 13 = 579 items → 509 passed / 70 failed；G9 4B/0.6B finite PASS；PERF1 `B0_anchor`：step p50 **123.68 ms**、decode **23.92 t/s**、TTFT **1607.7 ms**、prefill **280.55 t/s**、tok/step 3.048；0.6B 两次采集 token 流一致 |
| **L0 环境/栈自检** | ✅ PASS | `L0-RESULTS.md`：torch 身份/4 卡、verify-gpu `ALL_PASS`、verify-numerics `NUMERIC_GATE_PASS`（softmax 8.4e-9 / tanh 7.2e-8 / layer_norm 4.0e-7 / decoder_block 2.78e-4）、triton G0–G7 8/8、SFU 变异门禁（正向 0 mismatch、负向 100% lane 8–11）、import 顺序；**L0.6 4 卡 all_reduce 见 P0-R** |
| **L1 扩展加载/注册** | ✅ PASS | 4 扩展 import OK；`_rocm_C` 8 个算子 + `_C` gfx906 算子注册；`ROCM_SKINNY_GEMM_ENABLED=False`；平台属性 rocm/(9,0)/bf16 列表按 F21 修正 |
| **P0-R RCCL/TP>1** | ✅ PASS | 4 卡 all_reduce 5120+1MiB 全 rank PASS（`librccl.so.1.0` 加载确认、NCCL_DEBUG 真实 kernel collective）；证据 `/mnt/game/port/rccl-gfx906-build/23-smoke-result.md` |
| **L2 kernel 数值** | ✅ PASS（子集判定） | 新栈 13 文件 340 passed / **69 failed ⊆ DTK 70**（66 causal_conv1d + 3 qwen3_5_gdn，均平台无关 expected-fail）；4 个原 DTK-only 文件门禁已改为 stack-agnostic，补跑 **169 passed / 1 failed**（该 1 项 DTK 也失败） |
| **L3 G8/G9** | ⚠️ 部分（0.6B PASS / 4B blocked F22） | G9：DTK 基线已冻结；新栈 fp16+attention fallback 采集见 §4。G8：DTK `UNRUNNABLE_AS_CLEAN_GATE`（环境 fla stub + 默认 Triton 链编译崩溃），新栈同法记录 |
| **L4 精度/perf** | ⛔ 阻塞（F14） | 性能测量必须走真实 attention kernel；当前 serving 需 eager fallback（性能不可用）。同树 B0 数字已备，待 triton 修复后测 |
| **L5 TP=4/稳定性** | ⛔ 部分 | 通信通路已 PASS；端到端 TP=4 服务正确性依赖 F14/F21 修复 |

## 4. 新栈 serving 冒烟与 G9（L3）

- **P2.3 冒烟 PASS**：Qwen3-0.6B（fp16，`--enforce-eager`，`VLLM_GFX906_ATTN_EAGER_FALLBACK=1`）→
  `' Paris. The capital of France is also'`，与 HF CPU 参考逐 token 一致。
- **0.6B G9（dense 对照）**：新栈 fp16 + fallback，`finite_check=PASS, nonfinite=0`；
  文本 ` Paris. The capital of France is also the capital of the` / ` 2, 2 + 1 = 3,` /
  `______，是世界上最早的城市，也是世界上最大的城市`。
  与 DTK fp16 对照（`dtk-g9-tokens-0.6b-fp16.json`）逐 token：2/3 prompt **12/12 完全一致**；
  英文 prompt 在第 5 token 分歧（控制侧 top1-top2 margin = 0.0156 nat，略高于 0.01 的 near-tie 阈值）——
  归因于 eager attention fallback 与 kernel 路径的数值差异（两者 top-2 都是合理续写）。
- **4B G9（GDN 模型）：阻塞（F22）**。权重加载 + KV cache 初始化完成后，进程被 hycu 驱动 kill
  （`hycu: killed process … comm:python3`）；`VLLM_SKIP_KERNEL_WARMUP=1` 也在同一位置被 kill。
  0.6B（dense）同栈正常，故问题域锁定在 GDN/FLA 内核的 GPU 访问，需单独修复。
- **bf16 不可用（F21）**：Qwen3-0.6B bf16 logprobs 全 NaN → HTTP 400；fp16 正常。
  Q1 的"开放 bf16"决策回退为 DTK-only；DTK 侧 fp16 对照已采（4B fp16 与 bf16 token 流 12/12 一致），
  L3 对比用 fp16-vs-fp16。

## 5. 执行期发现（F1–F21 摘要，详表见计划书 §9.2）

| # | 一句话 | 状态 |
|---|---|---|
| F1 | DTK redo merge 丢失 S7c fp16 packed-activation 标量守卫 → 已恢复到新路径 `csrc/activation_kernels.cu` | 修复并验证（packed-silu 3/3、hygon_dtk 52/52） |
| F2 | sync 文档 known-fail 列表过时 | 由 L-1 实测替代 |
| F3 | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 会让首次 GPU 分配永久挂起（drm syncobj 无限等待） | env 契约 unset；verify-gpu PASS |
| F4 | `rocm_sdk_devel` wheel 需手工解包 `_devel.tar` 才有 RUNPATH/amdsmi | P0 记录 |
| F5 | RCCL 源码在 `rocm-systems` monorepo pin `7da43026`；SDK librccl 无任何 device code（NOBITS） | P0-R 重建依据 |
| F6 | clang24 拒绝 `static constexpr` 调用 `__device__ constexpr get_warp_size()` | 已修（`29d605c44`） |
| F7 | DTK `_C_stable_libtorch` 最终 md5 `e303b0b2…`（HEAD `29d605c44`） | L-1 身份口径 |
| F8 | qwen3_5 import 链硬依赖 torchvision（无匹配 wheel） | import-only shim `tools/dcu/install_torchvision_shim.sh` |
| F9 | InprocClient 缺 `engine_ranks_managed`/`resources` → serve 启动崩 | 已修 `fd0ba331b` |
| F10 | qwen3_5 `reduce_results` kwarg/属性缺失 → 4B/27B serve 崩 | 已修 `4f284dfdf` |
| F11 | V1 runner `uses_xdrope_dim` 未定义（另有其他陈旧 API） | 已补入口；V1 仍非可用路径 |
| F12 | `chunk_gated_delta_rule` 重复 CustomOp 注册 | 已修 `73be7d098`（fork 侧改名 `_z100`） |
| F13 | 另有 4 对潜在重复注册 | 未触发，登记 |
| F14 | **Triton attention 在 serving 小序列形状下输出错误**（单测大形状全过；eager 参考与 HF 逐 token 一致） | 现行阻塞；`VLLM_GFX906_ATTN_EAGER_FALLBACK=1` 验证通道 |
| F15 | DTK 同树 G9 基线正常（4B Paris/2/北京；finite PASS） | L-1 交付 |
| F16 | `torchrun`/elastic rendezvous 在本 torch wheel 上不可用（libuv） | RCCL 冒烟用 env launcher；vLLM 正常路径不受影响 |
| F17 | 4 个 gfx906 内核测试门禁 DTK-only → 新栈 170 items skip | 已改 stack-agnostic 并补跑 |
| F18 | RCCL 重建 + 4 卡 all_reduce PASS | P0-R 关闭 |
| F19 | attention 单测（大形状）与 serving 行为矛盾 | 归入 F14 |
| F20 | MIOpen 缺 `libMIOpenCKGroupedConv_gfx906.so` → grouped conv1d `hipErrorInvalidKernelFile` | `MIOPEN_DEBUG_AMD_ROCM_CK_GROUPED_CONV_FWD/BWD=0` 写进 env |
| F21 | 新栈 bf16 模型级 NaN（fp16 正常） | Q1 回退：bf16 仍 DTK-only |

## 6. 性能基线（同树 DTK `B0_anchor`，供 L4 使用）

详见 `/mnt/game/z100-baseline/dtk-perf1-anchor-summary.txt`（配方/repeat/交付日志齐全）。
新栈性能**未测**：F14 未修复前 attention 必须走 eager fallback，测出的数字不代表新栈能力。

## 7. 回滚通道

- DTK 栈与 `z100-regress` 未改动；`/home/lyh/z100-vllm/vllm/*.so` 仍指向 DTK 构建产物，4 个 merge 回归修复后 **DTK serve 实测可用**（4B/0.6B/27B TP4 均启动成功，G9 finite）。
- 栈切换：`tools/dcu/use_stack.sh status|dtk|rocm101`（两棵独立树，互不影响）。
- 回滚演练：见 §8（P5 待执行）。

## 8. 复现/交付脚本

| 脚本 | 用途 |
|---|---|
| `tools/dcu/env-rocm101.sh` | 新栈环境契约（含 F3/F4/F20 修正） |
| `tools/dcu/use_stack.sh` | 两栈 .so 切换 + 身份打印 |
| `tools/dcu/rccl-gfx906-build.sh` / `rccl_allreduce_smoke.py` | RCCL 重建与 4 卡验收 |
| `tools/dcu/g9_token_capture.py` | G9 token 捕获/对比（near-tie 规则） |
| `tools/dcu/smoke_rocm101.py` | 无 curl 容器的最小 OpenAI 冒烟客户端 |
| `tools/dcu/install_torchvision_shim.sh` | torchvision import shim（F8） |
| `/mnt/game/port/{setenv,configure,build,install}-rocm101.sh` | 新栈扩展构建链 |

## 9. 待办（P5/P4）

1. **F14 修复**（triton/LLVM 侧或 serving 小序列 kernel 路线）：修复后移除 eager fallback、重跑 L2（attention 小形状用例）、L4 性能、L5 TP4。
2. F21 归因（bf16 模型级 NaN）：修好后恢复 Q1；否则保持 DTK-only。
3. P5 固化：`serve_rocm101.sh` 生产配方脚本（fp16 + fallback 之外的 final 形态）、容器可重建文档、验收报告定稿、回滚演练记录。
