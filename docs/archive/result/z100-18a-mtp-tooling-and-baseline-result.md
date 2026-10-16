# S18a MTP 适配：工具链与基线（S18-0）— 执行结果

> 归档时间：2026-10-11
> 执行者：Lead（本步为工具/环境/门禁准备，非子 agent 执行）
> 对应设计文档：docs/design/z100-18-mtp-optimization.md
> 前一档：无
> 状态：🔶 部分完成（工具与单测基线已交付；G0 的"单测全绿"按**基线口径**重定义为"相对本档零新增失败"）

## 1. 目标与验收标准

设计文档 G0 原文：**"单测集合全绿（含新增）；探针能复现旧归档的指标口径。"**

本次执行前的用户决策（2026-10-11）：目标槽 = **FP8 block-scaled**；**停生产、全力配合适配**；不引入 GGUF 正对照。

因此本步的验收标准实际落地为：

1. 停掉生产并留可回滚配方，4 卡与宿主内存归零；
2. 建立独立实验容器（不占生产容器身份、无内存上限、端口可达、工作区可见）；
3. 交付"一条臂端到端"的工具链（起服务 → 健康 → 接受率探针 → S9 协议基准 → 杀干净）；
4. 取得 **HEAD 原样**的 MTP 相关单测基线（后续任何改动都以此对照）。

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/s18_mtp_arm.sh` | 新增（129 行） | 单臂启动器：`<SLOT> <TP> <PORT> <K>`，K=0 为对照臂；按 SLOT 设量化环境（fp8 → `VLLM_DCU_USE_FP8_EMU=1`；int8 → `VLLM_DCU_USE_INT8_SDOT4=1`）；默认 `--enforce-eager`、`mamba-cache-mode align`、`HSA_FORCE_FINE_GRAIN_PCIE=1`；K>0 时默认打开 `VLLM_QWEN35_MTP_{CONV,GDN}_DEBUG`；把 args/日志落到 `/mnt/game/s18` |
| `tools/dcu/s18_bench.sh` | 新增（52 行） | S9/S11l/S14 同协议基准（decode/ttft/ttft8k/conc8），模型与 served-name 参数化 |
| `tools/dcu/s18_mtp_probe.py` | 新增（196 行） | 接受率探针：读 `vllm:spec_decode_*`（含 per-position），跑 5 条固定 prompt，输出 drafts/accepted/accept_rate/`mean_acceptance_length`/tok/s 到 JSON；`--mode greedy|prod`（prod = temperature 0.7 / rep 1.05，接受率对 penalty 敏感） |
| `tools/dcu/s18_tests.sh` | 新增（52 行） | MTP 相关 8 个测试文件的基线跑法（沿用 `s11l_tests.sh` 的 `--noconftest` 约定） |
| `tools/dcu/s18_one_arm.sh` | 新增（120 行） | 一条臂端到端驱动：杀残留 → 起臂 → 等 `/health` → 探针 → 基准 → 杀干净 → 写 `summary-<label>.md` |
| `docs/design/z100-18-mtp-optimization.md` | 新增 | S18 方案（含两处对既有文档的更正与两处对本方案自身的二次更正） |
| `docs/z100-todolist.md` | 执行总览表 | 新增 S18 行（指向设计文档） |

**容器**：新建 `z100-s18`（镜像 `z100-s9-snap-s14`，`--device /dev/kfd,/dev/mkfd,/dev/dri`、`--group-add video`、`--ipc=host`、`--security-opt label=disable`、`--shm-size 16g`、`-p 8001:8001`、`--memory=0`、`/home/lyh/z100-vllm:/src:ro`、`/mnt/game:/mnt/game`）。**未改动**生产容器 `z100-vllm`、镜像、DTK 库、`_C*` 产物。

## 3. 执行过程与原始证据

### 3.1 停生产（用户授权）

```
$ ss -ltn | grep :8000            # 停之前：LISTEN 0.0.0.0:8000
$ docker exec z100-vllm bash -lc 'kill -TERM 13266 13260'   # api server + 包装 bash
   after wait: 只剩 13266/13334 与 3 个 Worker_TP
$ docker exec z100-vllm bash -lc 'pkill -9 -x VLLM; kill -9 13334 13266 13260'
   remaining: (空)
$ curl -s -m 4 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health
000                                # 不再响应
$ hy-smi --showmeminfo vram | grep "Used Memory"
DCU[0..3]: 0 MiB                   # 四卡全 0
$ free -g
Mem: total 31  used 2  free 22  available 27
```

- 残留的 `VLLM::Worker_TP`/`python` 在 `ps -o stat` 下全部为 **`Z`（defunct）**，不占显存/内存；
- `:8000` 上仍有 `docker-proxy`（pid 3746/3753）监听，这是**容器端口映射**本身，不是引擎；`:8000` 连接被 reset ⇒ 生产确实已停。
- **回滚配方（原样记录）**：`docker exec -d z100-vllm bash /mnt/game/port/serve_fp8_8000.sh`（该脚本 `exec`s `tools/dcu/s11f_e2e_fp8.sh /mnt/game/models/Qwen3.8-27B-FP8 4 8000 ...`，其余 flags 见脚本原文）。

### 3.2 HEAD 原样的 MTP 单测基线（`tools/dcu/s18_tests.sh`，GPU 0）

| 测试文件 | 结果 | 归类 |
|---|---|---|
| `tests/kernels/mamba/test_causal_conv1d.py` | **66 failed / 147 passed** | 既有：65× `test_causal_conv1d_update_with_batch_gather` + 1× fp32 state 用例 |
| `tests/kernels/mamba/test_causal_conv1d.py -k gfx906_causal_conv1d_mtp_update` | **48 passed** | ✅ gfx906 卷积 MTP 更新 op 数值正确，**fp16 与 bf16 参数化都在内**（19.76s） |
| `tests/kernels/mamba/test_fused_sigmoid_gating.py` | 5 passed | gfx906 非 MTP 路径 |
| `tests/kernels/mamba/test_gdn_fused_mtp.py` | 1 skipped | 仅 CC≥8.0 |
| `tests/model_executor/test_qwen3_5_gdn.py` | **20 passed** | 含 `test_gfx906_mtp_gdn_uses_custom_update` |
| `tests/model_executor/model_loader/test_mtp_validation.py` | 1 passed | |
| `tests/models/test_qwen3_5_mtp_config.py` | 4 passed | |
| `tests/v1/spec_decode/test_mtp.py` | 1 passed / **2 failed** | 失败 = `ImportError`：`aiter/jit/core.py` 在 gfx906 上不可用（平台阻塞，非 MTP 逻辑） |
| `tests/model_executor/test_qwen3_5_mtp_gguf.py` | **1 collection error** | 既有：import 期实例化 CustomOp 需要 `default_vllm_config` fixture（`AssertionError: Current vLLM config is not set`） |

环境要点（会复用到后续所有步）：

- 该镜像**缺 `tblib`** ⇒ 用 conftest 必须 `pip install tblib`（已在 `z100-s18` 安装 3.2.2）；即便如此 `tests/conftest.py` 才能加载，`test_glm_mtp_defers_lm_head` 由 error 变 passed；
- `--noconftest` 是 `s11l_tests.sh` 的既有约定，但它会让需要 fixture 的用例报 "fixture not found" ——**判读时必须区分**"约定引入的 error"与真失败；
- aiter 相关用例在本平台**结构性不可跑**（S8 结论一致）。

### 3.3 两条机器证据（推翻/替代既有文档结论）

1. **block-scaled FP8 仿真已是活路径**：A0（FP8 生产配方）启动日志
   `(Worker_TP0) INFO ... [__init__.py:743] Selected Gfx906Fp8BlockScaledMMLinearKernel for Fp8LinearMethod`
   ⇒ "block-scaled 只能走 36× Triton"（S8b 状态）已被 S11f 的 `dac6861d5` 取代；`csrc/rocm/fp8_emu_gemm.cu:68` 的 `kMsMax=8` 表明 **GEMV 覆盖 M≤8**，MTP 的 `M=1+k` 落在其中。
2. **bf16 在 gfx906 MTP op 上可用**：`test_gfx906_causal_conv1d_mtp_update` 的 bf16 参数化 **48/48 passed**；配合 `csrc/dispatch_utils.h:22-27`（`VLLM_DISPATCH_HALF_TYPES` 含 BFloat16）⇒ `docs/hygon_z100.md:91` 与 `z100-10-phase-b-features.md:22` 的"bf16 + MTP 必报 `not implemented for BFloat16`"**不成立**。

### 3.4 一条高价值新线索（留给 S18-2）

65 个 `test_causal_conv1d_update_with_batch_gather` 失败中，参考实现与内核输出的**卷积状态整体错位一格**：

```
E  where False = torch.equal(
     tensor([[[ 0.3020, -0.3337], [ 0.1102, -4.2301], [-2.2140, -1.2717], ...]]),   # 参考
     tensor([[[ 0.1808,  0.3020], [-0.5523,  0.1102], [ 0.9238, -2.2140], ...]]))  # gfx906 内核
```

即"**带 padding 的 batch-gather decode 下卷积状态滚窗错位**"（width-3 state 被按错误轴/偏移滚动），与 fork 在 GDN 侧修过的 `7026983a8`（shared-core spec-token 的 ssm state 转置）属同一族缺陷。MTP verify 的批次**天然带 padding**（`pad_slot_id`），因此这是 S18-2 的候选主因之一（E1 位置依赖 + 状态错位的直接对照）。

## 4. 门禁判定

| G0 子项 | 判定 | 证据 |
|---|---|---|
| 停生产、卡/内存归零、有回滚配方 | ✅ | §3.1 |
| 独立实验容器可用 | ✅ | `z100-s18` 起来、`/src` 可见（183 个 tools/dcu 文件）、DTKROOT 正常 |
| 工具链交付（起/测/杀一条龙） | ✅ | §2 五个脚本，`bash -n` 与 `ast.parse` 通过；实际用于 A0 |
| 探针口径对齐旧归档 | ✅ | 复用 `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}{,_per_pos}_total`，`mean_acceptance_length = 1 + accepted/drafts`，与 `vllm/v1/spec_decode/metrics.py:107-129` 及旧 `gfx906_port/bench/probe_real.py:89-94` 一致 |
| "单测集合全绿" | 🔶 **按基线口径重定义** | HEAD 原样已有 66 个失败 + 2 个 aiter 阻塞失败 + 1 个 collection error；本步**不修改**任何 vllm 源码，故这些全部是既有状态。G0 改为：**后续改动相对本档零新增失败** |

## 5. 未完成项 / 阻塞与原因

1. `fused_sigmoid_gating_delta_rule_gfx906_mtp_update` 的**直接单测仍未写**（finding G-c）——排入 S18-4，并顺带补 `tools/dcu/s8_path_probe.py` 的存在性探针（finding G-d）；
2. `tests/v1/spec_decode/test_mtp.py` 的 2 个 aiter 用例在本平台不可跑（**结构性**，不修）；
3. `test_qwen3_5_mtp_gguf.py` 的 collection error 为既有测试设计问题（import 期建 CustomOp），未修；
4. 真机臂 A0/A1/A2 **尚未产出**（A0 正在启动，见 S18-1 归档）。

## 6. 对后续步骤的影响

- S18-1 可以开始：容器/工具/口径已就绪；**底座用 FP8 + `--enforce-eager`**（生产配方同一套 flags），首轮 `maxlen=8192` 快速闭环；
- 设计文档已按本节 3.3 的两条证据**二次修正**（block-scaled 不再是前置、bf16 不是阻塞项）；
- §3.4 的状态错位线索**提升为 S18-2 的首查项**：若 MTP 的 verify 批次在 padding 行上滚窗错误，低接受率就有了机制解释；
- 后续所有测试判读必须带"与本节基线对照"，禁止把既有的 66/2/1 失败记到新改动头上。

## 7. 复现方式

```bash
# 1) 停生产（用户授权后）
ss -ltn | grep :8000
docker exec z100-vllm bash -lc 'kill -TERM <api_pid>; sleep 15; pkill -9 -x VLLM; kill -9 <worker_pids>'
curl -s -m 4 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health   # 期望 000
hy-smi --showmeminfo vram | grep "Used Memory"                              # 期望 4×0

# 2) 实验容器（本档已建，重做为幂等保留）
docker run -d --name z100-s18 --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  --group-add video --ipc=host --security-opt label=disable --shm-size 16g -p 8001:8001 \
  -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
  --entrypoint bash z100-s9-snap-s14 -c 'sleep infinity'
docker exec z100-s18 pip install -q tblib

# 3) 单测基线
docker exec -d z100-s18 bash /src/tools/dcu/s18_tests.sh /mnt/game/s18/tests
docker exec z100-s18 bash -lc 'cd /src && export PYTHONPATH=/src:/mnt/game/triton-gfx906/install HIP_VISIBLE_DEVICES=0 && \
  python -m pytest -q --noconftest -p no:cacheprovider tests/kernels/mamba/test_causal_conv1d.py -k gfx906_causal_conv1d_mtp_update'

# 4) 一条臂端到端（示例：FP8 对照臂）
docker exec -d z100-s18 bash -lc 'S18_BENCH_MODES=decode bash /src/tools/dcu/s18_one_arm.sh fp8 4 0 8192'
```

## 8. 参考

- 设计文档：`docs/design/z100-18-mtp-optimization.md`
- 旧实测（2026-10-07）：`/home/lyh/server/gfx906_port/对比测试报告-v1.md`
- 既有归档：`docs/result/z100-11k-sdot4-compile-integration-result.md`、`z100-14e-reproducibility-boundary-result.md`、`z100-17-p2p-finegrain-e2e-result.md`
- 测试计数原文：`/mnt/game/s18/tests/*.log`
