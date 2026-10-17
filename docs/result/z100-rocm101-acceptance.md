# Z100：ROCm 10.1 + torch 2.16 新栈适配验收报告

> 计划书：`docs/z100-torch216-rocm101-adapt-plan.md`（D1–D7、P0–P5、L-1→L5）。
> 本文是执行结果，所有数字来自本次真机执行，证据路径可复查。
> 最后更新：2026-10-18（夜间执行轮，含 triton 根因修复后的 L2/L3/L4 复验）。

## 1. 结论

新栈**已打通到可生产验证的形态**：torch 2.16 + ROCm 10.1 +（修复后的）triton 3.6.0
+ SFU 补丁 clang 重编的 vLLM 扩展 + gfx906 RCCL（TP>1 通信）+ 独立运行容器/env 契约。

- **L-1 DTK 同树基线**冻结完成（含 F1 修复）；回滚通道实测可用。
- **L0/L1/L2/L3/L4 全部通过**（L4 在 Z100L 卡上）：kernel 数值子集判定 PASS；
  0.6B/4B G9 与 DTK fp16 基线 **token 流 CONFIRMED**；L4 12 prompt × 64 token
  **12/12 CONFIRMED**，性能与 DTK 差异 ≤1.1%。
- **F14（attention NaN）根因已定位并修复**：ROCm 10.1 triton 源码中 fork 的
  gfx906/VEGA20 补丁被误当作 CDNA（`isCDNA()` 含 VEGA20；`reduce()` 丢了 VEGA
  的 DPP-broadcast 分支而落入 RDNA 的 `permlanex16`），修正后重建 triton，
  attention 单测 57/57、serving 无 fallback 正确。
- **F22（4B GDN serve 失败）解除**：真实原因是 MIOpen 在 gfx906 上运行时 JIT
  grouped-conv 不稳定（F20 残留），把 gfx906 causal_conv1d fallback 的 conv 改为
  ATen 原生实现后 4B serve 稳定（ready ~52s，G9 2/2 次一致）。
- **F24（新发现，需决策/厂商支持）**：同一份 HSACO 在 DTK 运行时 4 卡全对，在
  ROCm 10.1 运行时 **Z100（卡 2/3）** 上 NaN、只写出 72/10240 个输出
  （dmesg: `hycu: sq_intr: error … err_type 2` + vmfault）；Z100L（卡 0/1）正常。
  当前新栈验证/服务使用卡 0/1，TP=4（跨 Z100+Z100L）被此项阻塞。
- **F26（TP=2 挂起）与 F21（bf16 NaN）均已解除**：TP=2 现已稳定（请求正确、TTFT 690 ms、
  decode 20.2 t/s），此前挂起是冷 Triton cache 首跑编译（4B GDN 首跑约 6 min）+ 残留进程；
  bf16 在 Z100L 上与 fp16 token 流逐一致，0.6B/4B 的 bf16 G9 均与 DTK bf16 基线
  CONFIRMED——此前的 NaN 是 F24（Z100 卡）误判，Q1（bf16 开放）已恢复。
- 仍开放：**F24**（Z100 卡在 ROCm 10.1 runtime 下异常，TP=4 阻塞，需厂商/runtime 修复；
  厂商报告 `f23-evidence/F24-vendor-report.md`）；F16 `torchrun` 不可用；F13 潜在重复注册；
  V1 runner 不可用；27B-FP8 生产配方待评估。

## 2. 交付身份

| 项 | 值 |
|---|---|
| 分支 | `rocm101-torch216`（从 `sync/upstream-20261016-v2` @ `6067e19a6` 拉出） |
| 代码面 HEAD | `8ac4433bd`（本报告对应提交；L-1 冻结时 HEAD `29d605c44`） |
| torch | `2.16.0.dev20261016+rocm10.1`（自编 wheel-sfu sha256 `46a5f447…`，gfx906 code objects，SFU 补丁 clang 24） |
| Triton | `3.6.0`（`/mnt/game/triton-z100-rocm101/install`），**含本次两处 gfx906/VEGA20 修复**：libtriton md5 `35b826d9f3ffdd5bab1e6ca7081dac07`，wheel sha256 `efe61036751902c27073f184e15afe8d702cb8b2532be77e998165f269bb7ee1`（342,202,319 B）；LLVM pin 22.0.0git `f6ded0be` |
| Triton 修复来源 | `triton-z100/patches/apply_z100_overlay.py` 步骤 **D**（`isCDNA()` 去掉 VEGA20 → release/acquire fence lowering）与 **E**（`reduce()` 恢复 VEGA20 的 GCN DPP-broadcast 路径） |
| ROCm | TheRock `10.1`，`ROCM_PATH=/opt/rocm/core-10.1`，HIP `7.16.26385` |
| RCCL | 自编 gfx906：`/mnt/game/rccl-gfx906/lib/librccl.so.1.0` md5 `4889939094649fef320ce9e28e687206`（源码 `rocm-systems@7da43026`） |
| vLLM 扩展（新栈） | `_C 9302ece8` / `_C_stable_libtorch ffc176af` / `_moe_C_stable 7b4185c6` / `_rocm_C a250420b` |
| DTK 基线 .so | `_C d44117ae` / `_C_stable_libtorch e303b0b2`（F1 修复）/ `_moe 50af36b9` / `_rocm_C f205be8f` |
| 运行容器 | `z100-vllm-rocm101-rt`（image `z100-rocm10.1:24.04-r642`） |
| env 契约 | `tools/dcu/env-rocm101.sh`（Triton 修复后 `VLLM_GFX906_ATTN_EAGER_FALLBACK` 默认 **0**；含 F3/F4/F20 修正与 F24 卡型说明） |
| 栈切换 | `tools/dcu/use_stack.sh status|dtk|rocm101` |

## 3. 门禁矩阵

| 门 | 状态 | 证据 / 关键数字 |
|---|---|---|
| **L-1 DTK 同树基线** | ✅ PASS | `/mnt/game/z100-baseline/REPORT.md`；canonical 579 → 509 passed/70 failed；G9 4B/0.6B finite PASS + fp16 对照（4B fp16≡bf16 12/12）；PERF1 `B0_anchor`：step p50 **123.68 ms**、decode **23.92 t/s**、TTFT **1607.7 ms**、prefill **280.55 t/s**；0.6B 两次采集 token 流一致 |
| **L0 环境/栈自检** | ✅ PASS | `L0-RESULTS.md`：torch/4 卡身份、verify-gpu ALL_PASS、verify-numerics NUMERIC_GATE_PASS、triton G0–G7 8/8、SFU 变异门禁正/负向、import 顺序 |
| **L1 扩展加载/注册** | ✅ PASS | 4 扩展 import OK；`_rocm_C` 8 算子 + `_C` gfx906 算子注册；`ROCM_SKINNY_GEMM_ENABLED=False` |
| **P0-R RCCL/TP>1** | ✅ PASS | 4 卡 all_reduce 5120+1MiB 全 rank PASS（`librccl.so.1.0` 加载确认、NCCL_DEBUG 真实 kernel collective）；证据 `/mnt/game/port/rccl-gfx906-build/23-smoke-result.md` |
| **L2 kernel 数值** | ✅ PASS（子集判定） | 新栈 13 文件 340 passed / **69 failed ⊆ DTK 70**（66 causal_conv1d + 3 qwen3_5_gdn，均平台无关 expected-fail）；4 个原 DTK-only 文件改 stack-agnostic 后补跑 **169 passed / 1 failed**（该 1 项 DTK 也失败）；attention 单测修复后在本轮复验 **57/57 passed** |
| **L3 G8/G9** | ✅ PASS（G9；G8 见注） | **0.6B**：新栈 fp16 无 fallback，`finite_check PASS`，与 DTK fp16 对照 3/3 prompt **12/12 ids 一致**（修复后英文输出与 DTK 完全相同 ` Paris. The capital of Italy is Rome`）。**4B**：`finite_check PASS`，与 `dtk-g9-tokens-4b-fp16.json` **CONFIRMED 3/3 prompts（12/12 ids）**，两次独立采集输出一致。**bf16**：0.6B/4B bf16 G9 均与 DTK bf16 基线 **CONFIRMED 3/3（12/12 ids）**、`finite_check PASS`。G8：DTK 侧 `UNRUNNABLE_AS_CLEAN_GATE`（环境 fla stub + 默认 FLA Triton 链编译崩溃），本轮以 GDN kernel 套件 + G9 作为 GDN 证据（与计划 §7 判读一致） |
| **L4 精度/perf** | ✅ PASS（Z100L 卡） | **L4.1 parity**：4B TP1、12 prompts × 64 greedy tokens、top-5 logprobs，新栈 vs DTK **12/12 prompts、64/64 token ids CONFIRMED**。**L4.5 perf**（4B TP1，单流 64 tok，3 repeats，同配方两栈）：见 §4。 |
| **L5 TP=4/稳定性** | ⚠️ 部分（TP1/TP2 PASS；TP4 受 F24 阻塞） | **TP=1**：单卡全流程 PASS（0.6B/4B serve + G9 + L4）。**TP=2（Z100L 卡 0/1）**：serve ready、请求正确 ` Paris.\nA. True`、perf TTFT **690.3 ms** / decode **20.20 t/s** / step **49.61 ms**（3 repeats），warmup 3.5 s（cache 热）。**TP=4**：worker-2 在模型初始化阶段死亡，dmesg 在 Z100 卡（BDF 38/3b）报 `sq_intr err_type 2` + vmfault → 受 F24 阻塞。RCCL 通信层本身已 4 卡 PASS（P0-R） |

## 4. L4 结果（4B TP1，DCU0=Z100L，单流 64 tokens，3 repeats）

| 指标 | 新栈（ROCm 10.1 + repaired triton） | DTK 栈 | 差异 |
|---|---|---|---|
| TTFT p50 | **1134.3 ms** | 1147.2 ms | −1.1% |
| decode | **21.653 t/s** | 21.672 t/s | −0.09% |
| step p50 | **46.18 ms** | 46.67 ms | −1.06% |

原始数据：`/mnt/game/z100-rocm101-validation/l4/perf-rocm101.json` / `perf-dtk.json`；
token parity：`l4/l4-rocm101.json` / `l4/l4-dtk.json`（compare `RESULT: CONFIRMED (12/12 prompts)`）。

## 5. 执行期发现（F1–F27）

完整表见计划书 §9.2；本报告只列对结论有直接影响的：

| # | 一句话 | 状态 |
|---|---|---|
| F1 | DTK redo merge 丢失 S7c fp16 packed-activation 标量守卫 | 已修复并验证（packed-silu 3/3、hygon_dtk 52/52） |
| F3 | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 使首次 GPU 分配永久挂起 | env 契约 unset |
| F9–F12 | merge 跨文件回归：InprocClient 属性、`reduce_results` kwarg、V1 runner 属性、重复 CustomOp 注册 | 已修复（`fd0ba331b`/`4f284dfdf`/`73be7d098`），DTK 与新栈 serve 均实测 |
| F14/F23 | **attention NaN 根因**：rocm101 triton 源码中 gfx906/VEGA20 被误当 CDNA（`isCDNA` 含 VEGA20 → 用 `s_waitcnt` 代替 release/acquire fence；`reduce()` 丢 VEGA 分支 → gfx906 走 RDNA `permlanex16`，LLVM 无法选择直接 abort） | **已修复**（overlay D+E，重建 triton，libtriton md5 见 §2）；证据 `/mnt/game/z100-rocm101-validation/f23-evidence/` |
| F20/F25 | MIOpen 缺 gfx906 CK grouped-conv solver、运行时 JIT 不稳定（`hipErrorInvalidKernelFile` 被抛在后续 op） | env 关 CK solver；4B fallback 的 conv 改 ATen 原生实现（`causal_conv1d.py`） |
| F21 | 新栈 bf16 曾报真实模型 NaN | **已解除 = F24 误判**：Z100L 上 bf16 与 fp16 token 流一致（0.6B/4B bf16 G9 对 DTK bf16 基线 CONFIRMED 3/3、12/12 ids），Q1 恢复 |
| F22 | 4B GDN serve 曾随机失败 | **已解除**（F25 修复后 stable） |
| F24 | Z100 卡（2/3）在 ROCm 10.1 运行时下异常（同 HSACO 在 DTK/4 卡正常） | **限制**：验证/服务用卡 0/1；TP=4 阻塞，需厂商/runtime 修复 |
| F26 | TP=2 曾启动挂起（shm_broadcast 60s 超时） | **已解除**：冷 Triton cache 首跑编译（4B GDN ~6 min，bf16 首跑 370 s）+ 残留进程；cache 热后 TP2 ready ~52 s、请求/perf 正常 |
| F27 | 首跑/首 dtype 的 JIT 编译成本（cache 冷时 4B 首启 370 s、TP2 首启 >6 min） | 运维注意：部署需预热 `TRITON_CACHE_DIR`；cache 热后 4B/TP2 均 ~52 s |

## 6. 已知限制

1. **F24**：新栈生产/验证固定 `HIP_VISIBLE_DEVICES=0,1`（Z100L）；TP=4 需要 Z100 卡，当前被阻塞。`hy-smi` 4 卡均识别为 `gfx906:sramecc+:xnack-`，DTK 正常，说明是 ROCm 10.1 runtime/driver 对 Z100 SKU 的问题（dmesg `sq_intr err_type 2`）。
2. **首次启动/首次 dtype 的 JIT 成本（F27）**：cache 冷时 4B GDN 首启约 6 min（bf16 首跑 370 s），TP2 冷 cache 首启 >6 min；cache 热后 4B/TP2 均 ~52 s。部署时需预热 `TRITON_CACHE_DIR`。
3. **F16 torchrun**：elastic rendezvous 在本 wheel 不可用（libuv）；vLLM 正常 `python -m vllm` 路径不受影响。
4. **F13**：4 对潜在重复 CustomOp 注册未触发（`rms_norm`/`silu_and_mul`/`fused_rms_norm_gated`/`sparse_attn_indexer_kpool`），保留监控。
5. V1 model runner 仍非可用路径（F11 只补了入口）。
6. 27B-FP8 TP4 生产配方（DTK FP8 模拟 kernel）尚未在新栈验证；新栈性能口径以 4B TP1/TP2 为准。

## 7. 回滚通道

- DTK 栈（`z100-regress` 容器 + `/home/lyh/z100-vllm/vllm/*.so`）未改动；F9–F12 修复后 DTK 4B/0.6B/27B TP4 均实测可启动，G9 finite PASS。
- `tools/dcu/use_stack.sh status|dtk|rocm101` 在两棵独立树之间切换 `.so`，互不影响。
- 回滚演练步骤与本轮验证一致：`use_stack.sh dtk` → `smoke`/`G9` 复跑（基线产物在 `/mnt/game/z100-baseline/`）。

## 8. 复现脚本与证据路径

| 用途 | 路径 |
|---|---|
| 新栈 env 契约 | `tools/dcu/env-rocm101.sh` |
| 栈切换 | `tools/dcu/use_stack.sh` |
| RCCL 重建/验收 | `tools/dcu/rccl-gfx906-build.sh`、`rccl_allreduce_smoke.py` |
| G9 token 捕获/对比 | `tools/dcu/g9_token_capture.py` |
| attention serving-call 捕获/重放 | `/mnt/game/z100-rocm101-validation/{capture_attn_args.py,replay_attn.py}` |
| triton 修复证据 | `/mnt/game/z100-rocm101-validation/f23-evidence/`（README + overlay + TargetUtils/TargetInfo 快照 + hsaco/llir 对比） |
| L2/L3/L4 原始数据 | `/mnt/game/z100-rocm101-validation/{l2,l2b,g9,l4,post-fix2}/` |
| DTK 基线 | `/mnt/game/z100-baseline/`（README.md/REPORT.md + dtk-* 产物） |
| triton 重建日志 | `/home/lyh/triton-z100/logs/build-rocm101-20261017_194608.log`（首建，含 D）与 `…_202702.log`（含 E，最终产物） |

## 9. 下一步（按优先级）

1. **F24**：拿到厂商/ROCm 侧对 Z100 SKU 的说明或修复；之后用卡 2/3 复跑 attention replay（脚本现成）与 TP=4。
2. **TP=4**：待 F24 修复后重跑（现成命令：`HIP_VISIBLE_DEVICES=0,1,2,3` + TP=4），并补 4B TP 曲线（TP1/2/4）。
3. 27B-FP8 TP4 生产配方在新栈的启用性评估（FP8 模拟 kernel 依赖 DTK 专属实现，需评估新栈替代路径）。
4. bf16 生产启用演练（Q1 已恢复，建议先在 4B/0.6B 上跑一轮 perf + 质量电池）。
