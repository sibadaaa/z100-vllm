# S9 验收、性能对拍、诊断、固化与回滚

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md)
> 前置：S5–S8 · 状态：⬜ 待做

## 目的

把"能跑"变成"可交付"：跑完 A1–A9 判据、与生产基线对拍、把诊断能力补齐、固化镜像与回滚路径。

## 1. 验收清单（A1–A9）

| # | 判据 | 命令 |
|---|---|---|
| A1 | 架构正确且已归一化 | `python -c "from vllm.platforms.rocm import _GCN_ARCH,on_gfx906,_capability_from_gcn_arch;print(_GCN_ARCH,on_gfx906(),_capability_from_gcn_arch(_GCN_ARCH))"` → `gfx906 True (9, 0)` |
| A2 | 目标码只含 gfx906 | `dccobjdump --offloading <so> \| grep -o gfx[0-9]* \| sort -u` |
| A3 | 单卡服务 + KV cache 自动 | `vllm serve <model> --dtype float16 --max-model-len 4096`，`curl /v1/completions`，日志 `Available KV cache memory` > 0 且未用手工 flag |
| A4 | 8K 上下文 + 后端 | 8K 请求不崩，日志 `TRITON_ATTN` |
| A5 | TP 可用 | `--tensor-parallel-size 2` 与 `4` 各完成一次完整请求 |
| A6 | fork 自带 gfx906 测试 | 5 个测试文件全绿（见 §2） |
| A7 | ROCm 回归为零 | 无 DTK 环境下抽样跑 `tests/platforms/`、`tests/kernels/`，行为与改动前一致 |
| A8 | 诊断自证 | `python diagnostic_runner.py` 输出 DTK 版本、每卡 arch（torch+KFD 双源）、rocBLAS/hipBLASLt 覆盖、`LD_LIBRARY_PATH` 顺序、`VLLM_DCU_*` 是否设置 |
| A9 | bf16 有结论 | [S3](z100-03-bf16-verification.md) 的 A/B 结果与判定写入文档 |

## 2. 必跑测试（fork 自带，直接当验收用例）

```bash
source tools/dcu/env.sh
python -m pytest -v \
  tests/kernels/attention/test_triton_unified_attention_gfx906.py \
  tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py \
  tests/kernels/mamba/test_fused_recurrent_packed_decode.py \
  tests/kernels/mamba/test_fused_sigmoid_gating.py \
  tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py \
  tests/model_executor/test_qwen3_5_gdn.py \
  tests/platforms/test_hygon_dtk.py
```

## 3. 性能对拍

对拍口径必须与 [S0](z100-00-baseline-and-prereq.md) 采集的基线一致（同模型、同 prompt、同并发）：

| 指标 | 基线（生产 :8000） | 新引擎（:8001） | 结论 |
|---|---|---|---|
| decode ms/token | | | |
| 冷 prefill | | | |
| TTFT | | | |
| 并发 8 吞吐 | | | |

规则：
- 任一项劣化 > 20% ⇒ 先定位（attention backend / `_rocm_C` 是否生效 / bf16 vs fp16 / graph）；
- 漂移必须在文档中解释清楚，不允许"沉默地慢"。

## 4. 诊断增强

- `diagnostic_runner.py`：增加 DTK 章节（DTK 版本、`torch.version.hip`、每卡 `gcnArchName`、KFD `gfx_target_version`、`normalize_gcn_arch` 结果、`_rocm_C` 算子清单、rocBLAS/hipBLASLt 覆盖、`LD_LIBRARY_PATH` 顺序、`rsmi_init` 可解析性）。
- `vllm/collect_env.py`：加入 DTK 字段与 `tools/dcu/env.sh` 关键环境变量。
- 新增 `tools/dcu/smoke.sh`：起服务 → `/v1/completions` 与 `/v1/chat/completions` → 打印 KV cache 行与 backend 名 → 单卡/TP2/TP4 三种模式，非零退出即失败。

## 5. 固化与回滚

1. `docker commit` 新影子容器为 `gfx906-vllm:z100-dtk2604-<date>`（只 commit 不 save）。
2. 归档本次所有补丁：`triton-gfx906-3.6.0.patch`、`rocm.py` 改动 diff、`CMakeLists.txt` diff、requirements。
3. 更新文档：本方案 §4「已实施进展」、[hygon_z100.md](../hygon_z100.md)、[Todolist](../z100-todolist.md) 状态列。
4. **回滚演练**：从 `gfx906-vllm:prod-0.18.1-das-<date>` 起生产容器，确认 :8000 服务恢复、现网行为不变（即"新引擎失败不影响生产"）。

## 验收（门禁）

A1–A9 全绿 + 回滚演练成功 + 文档三处已更新。

## 失败处置

| 现象 | 处置 |
|---|---|
| A5 TP 卡死 | `--disable-custom-all-reduce` 复测；再看 RCCL 版本与 `/dev/kfd`、`/dev/mkfd` 挂载 |
| A7 回归不通过 | 定位是否新增分支未被 `is_hygon_dtk()` 包裹；单文件回滚 |
| 性能劣化 > 20% | 依次排查：backend 是否 TRITON_ATTN、`_rocm_C` 是否被调用、dtype（bf16/fp16）、是否误开 graph |
| `ps`/`hy-smi` 变慢 | 可能有卡死进程：`fix/install_reaper.sh`，必要时重启（见 [S0](z100-00-baseline-and-prereq.md)） |

## 参考

- 前序计划 §4 Phase 4（性能验收 + 固化 + 回滚）
- `/home/lyh/server/gfx906_port/bench/`（可复现脚本）
