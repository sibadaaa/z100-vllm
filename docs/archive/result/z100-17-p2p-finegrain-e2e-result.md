# Z100 27B TP=4 端到端 A/B：`HSA_FORCE_FINE_GRAIN_PCIE` 打开 RCCL P2P（S17）

> 承 [S16](z100-16-p2p-finegrain-pcie-result.md)：S16 在集合通信层证明「缺 `HSA_FORCE_FINE_GRAIN_PCIE=1`
> 是 RCCL 全走 SHM 的真因」。本步把它放到**真实 `vllm serve`**（27B W8A8 INT8，TP=4）上做两臂 A/B：
> **唯一差异是这个环境变量**。

## 0. 结论

**开 `HSA_FORCE_FINE_GRAIN_PCIE=1` 后，引擎内 RCCL 通道从 `via SHM=112 / P2P=0` 变为 `via P2P/IPC=112 / SHM=0`，
端到端吞吐与延迟全面改善，且数值逐位不变：**

| 指标（graph，PIECEWISE + 4 桶） | A：`=0`（SHM） | B：`=1`（P2P） | B 相对 A |
|---|---|---|---|
| decode TPOT（128 in / 128 out，median） | 49.56 ms | 47.83 ms | **快 3.6%** |
| decode TTFT（median） | 654.45 ms | 403.21 ms | **快 1.62×** |
| prefill 3800 in TTFT（mean） | 8877.34 ms | 4124.77 ms | **快 2.15×** |
| prefill 8100 in TTFT | 13052.58 ms | 6918.25 ms | **快 1.89×** |
| conc8 输出吞吐 | 39.52 tok/s | 54.42 tok/s | **快 1.38×** |
| conc8 E2EL（mean） | 25626 ms | 18722 ms | **快 1.37×** |

**正确性（eager 臂，可逐位复现口径）**：4 条 prompt × 128 token = **512/512 token 逐位一致、
`max|Δlogprob| = 0.000e+00`、文本 sha256 全等** ⇒ 换传输不改变数值。

## 1. 方法与配置

* 宿主 Z100 ×4（gfx906，DTK 26.04 容器 / 宿主 DTK 24.04），内核 5.15.0-191、`iommu=pt`、ACS 已清零；
  容器 `z100-s14`（`/src` = 本仓库、`/mnt/game` 共享）。
* 模型 `/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8`（29 GB），`--dtype float16`、TP=4、PP=1、
  `--max-model-len 8192`、`--max-num-batched-tokens 2048`、`--gpu-memory-utilization 0.85`、
  `--disable-custom-all-reduce`、`-cc.cudagraph_mode=PIECEWISE --cudagraph-capture-sizes 1 2 4 8`。
  **maxlen 取 8192（不是生产的 262144）**：集合通信消息尺寸由 hidden×token 决定，与 maxlen 无关；
  262144 会让宿主 pinned 内存翻 32 倍（本机 31 GB RAM，见 §4）。
* 启动器 [`tools/dcu/s16_arm.sh`](../../tools/dcu/s16_arm.sh)：只把 `HSA_FORCE_FINE_GRAIN_PCIE=<0|1>`
  导出后再调 [`tools/dcu/serve_z100_tp4.sh`](../../tools/dcu/serve_z100_tp4.sh)。
* 压测 [`tools/dcu/s14_bench.sh`](../../tools/dcu/s14_bench.sh)（与 S13/S14 同口径）：
  `decode`(128/128, n=5)、`ttft`(3800/1, n=4)、`ttft8k`(8100/1, n=1)、`conc8`(512/128, np=16, conc=8)。
* 两臂共用同一编译缓存 `/mnt/game/serve/vllm-cache`，日志显示两臂都
  `Directly load AOT compilation from .../torch_aot_compile/dee20878…`（各 4 次）⇒ **同一份 AOT 产物，key 不受该环境变量影响**。

## 2. 引擎内 RCCL 证据（决定性）

| 臂 | `NCCL_DEBUG=INFO` 通道统计 | `Missing "HSA_FORCE_FINE_GRAIN_PCIE=1"` 警告 | `hipDeviceAttributeFineGrainSupport` |
|---|---|---|---|
| A（`=0`） | `via SHM/direct/direct` **112**，`via P2P` **0** | 4 个 worker 全部告警 | 0 |
| B（`=1`） | `via P2P/IPC` **112**，`via SHM` **0** | **0 次** | 1 |

（112 = 4 rank × 28 条通道；与 S16 的 4 rank all-reduce 探测一致。原始日志：`/mnt/game/s16/rccl-{A,B,A2,B2}.*.log`。）

## 3. 完整压测结果

graph 臂（A/B）：

| 模式 | 指标 | A（SHM） | B（P2P） | 比 |
|---|---|---|---|---|
| decode | Mean/Median TTFT | 654.80 / 654.45 ms | 406.40 / 403.21 ms | 1.61× |
| decode | Mean/Median TPOT | 49.60 / 49.56 ms | 47.86 / 47.83 ms | 1.036× |
| decode | Mean E2EL | 6953.78 ms | 6484.46 ms | 1.07× |
| ttft (3800) | Mean/Median TTFT | 8877.34 / 8867.98 ms | 4124.77 / 4122.72 ms | 2.15× |
| ttft8k (8100) | TTFT | 13052.58 ms（1.611 ms/prefill-token） | 6918.25 ms（0.854 ms/prefill-token） | 1.89× |
| conc8 | Mean/Median TPOT | 137.46 / 122.24 ms | 113.61 / 111.41 ms | 1.21× |
| conc8 | Mean/Median TTFT | 8169.28 / 10416.72 ms | 4293.75 / 4664.32 ms | 1.90× |
| conc8 | 输出吞吐 / E2EL | 39.52 tok/s / 25626 ms | 54.42 tok/s / 18722 ms | 1.38× / 1.37× |

（decode median TPOT 49.56 ms 与 S14 归档的 TP=4 graph `49.66 ms/token` 吻合 ⇒ 本臂代表性成立。）

**解读**：收益集中在**大消息**（prefill 8K 1.89×、conc8 1.38×），**decode 小消息仅 3.6%** ——
与 S16 的尺寸扫描（≤64 KiB 两态持平、≥256 KiB 8–14×）和 S13「10 KB 消息延迟受限」完全一致。

## 4. 正确性（eager 臂逐位对照）

graph 在两跑之间**本身不可复现**（S14e：同配置重跑 0/8 逐位一致、mean|Δlogprob| 0.3–0.4 nat），
所以 token 流对照必须在 eager 下做（S14d：eager 1024 token 8/8 flip、重复 Δ=0.000）。

* A2 `=0`、B2 `=1`，`--enforce-eager`，其余同 A/B；两臂各 `via SHM=112` / `via P2P=112` 确认生效。
* 工具 [`tools/dcu/s14b_greedy.py`](../../tools/dcu/s14b_greedy.py) + `/mnt/game/s14b/prompts.txt`（4 条中文/英文 prompt），
  `temperature=0`、`max_tokens=128`、`logprobs + top_logprobs=5`。

| prompt | token 数 | token_ids 逐位一致 | `max\|Δlogprob\|` | `mean\|Δlogprob\|` | 文本 sha256 一致 |
|---|---|---|---|---|---|
| s14_greedy_long | 128 | ✅ | 0.000e+00 | 0.000e+00 | ✅ |
| p2_zh_slip | 128 | ✅ | 0.000e+00 | 0.000e+00 | ✅ |
| p2_en_batch | 128 | ✅ | 0.000e+00 | 0.000e+00 | ✅ |
| p2_zh_code | 128 | ✅ | 0.000e+00 | 0.000e+00 | ✅ |

## 5. 运维发现（必须记住）

1. **TP=4 在这台 31 GB 宿主上必须开 swap，否则启动期被 host global OOM 杀 worker。**
   首次尝试（`/swap.img` 4 GB 且已用满）在 22:27 触发 `global_oom`，
   `dmesg`：`Out of memory: Killed process … (VLLM::Worker_TP) anon-rss:6.2 GB`
   → 引擎 `Worker proc VllmWorker-2 died unexpectedly`，但 **`/health` 仍返回 200**（假健康）。
   启用既有 `swapon /mnt/game/swapfile`（16 GB）+ `swapon /mnt/game/serve-swapfile`（6 GB）后，
   四臂全部启动成功（启动期 Swap 用 2–3 GB）。测试结束已 `swapoff` 还原（两个文件仍在 /mnt/game）。
2. **生产容器 `z100-vllm` 当前是"假活"**：`/health` 200 但四卡 VRAM 0、
   `VLLM::Worker_TP` 早已被 OOM 杀（dmesg 14:08:36）⇒ 需要重启才真正可用（本轮未动它）。
3. 启动耗时（AOT 缓存热）：graph A 211 s / B 196 s；eager A2 140 s / B2 160 s。

## 6. 未做 / 边界

* **maxlen 262144 的生产配方未做两臂 A/B**（见 §1 的内存理由）；生产启用前建议先用 262144 起一次确认
  宿主内存与 swap（本步已证明 8192 下 swap 2–3 GB，262144 需按 S11f 的 pinned 分析预留 ~8 GB 以上）。
* 未做 n≥4 的统计重复（每模式只跑一轮）；但 S16 的集合通信层已给出 2 次独立运行的区间，且本轮
  decode/ttft 的差异（1.6–2.2×）远超测量抖动。
* 未跑 KV 复用/长上下文（maxlen 8192）与多并发 >8 的形态。

## 7. 复现命令

```bash
# 需要先给宿主加 swap（否则启动期会被 host OOM 杀 worker）
sudo swapon /mnt/game/swapfile && sudo swapon /mnt/game/serve-swapfile

# A 臂（SHM）：HSA=0
bash tools/dcu/s16_arm.sh A 0 8001 8192 0
for m in decode ttft ttft8k conc8; do
  sudo docker exec z100-s14 bash -lc "S14_BENCH_OUT=/mnt/game/s16 \
    bash /src/tools/dcu/s14_bench.sh 8001 A $m"; done
sudo docker exec z100-s14 bash -lc 'pkill -f "entrypoints[.]cli[.]main"'

# B 臂（P2P）：HSA=1
bash tools/dcu/s16_arm.sh B 1 8002 8192 0
for m in decode ttft ttft8k conc8; do
  sudo docker exec z100-s14 bash -lc "S14_BENCH_OUT=/mnt/game/s16 \
    bash /src/tools/dcu/s14_bench.sh 8002 B $m"; done
sudo docker exec z100-s14 bash -lc 'pkill -f "entrypoints[.]cli[.]main"'

# 正确性（eager，逐位对照）
bash tools/dcu/s16_arm.sh A2 0 8003 8192 1   # 然后
sudo docker exec z100-s14 bash -lc 'python3 /src/tools/dcu/s14b_greedy.py --port 8003 \
  --label A2 --out /mnt/game/s16/greedy-A2.jsonl --prompts-file /mnt/game/s14b/prompts.txt --max-tokens 128'
bash tools/dcu/s16_arm.sh B2 1 8004 8192 1   # 然后同上（--port 8004 --label B2 ... greedy-B2.jsonl）

# 收尾：还原 swap
sudo swapoff /mnt/game/serve-swapfile && sudo swapoff /mnt/game/swapfile
```

产物：`/mnt/game/s16/`（`serve-{A,B,A2,B2}.log`、`rccl-*.log`、`bench-*-*.log`、`greedy-{A2,B2}.jsonl`）。

## 8. 生产槽位重启 + P2P 上线（2026-10-11 23:16，用户授权）

容器 `z100-vllm` 的 `:8000` 服务此前是**假活**（`/health` 200 但四卡 VRAM 0，worker 早在 14:08:36 被 host OOM 杀掉）。
按生产配方重启（`/mnt/game/port/serve_fp8_8000.sh` → [`tools/dcu/s11f_e2e_fp8.sh`](../../tools/dcu/s11f_e2e_fp8.sh)，
模型 `/mnt/game/models/Qwen3.8-27B-FP8`，TP=4，maxlen 262144，eager），并开启 P2P。

**改动（都已落地）**

| 文件 | 改动 | 回滚 |
|---|---|---|
| [`tools/dcu/s11f_e2e_fp8.sh`](../../tools/dcu/s11f_e2e_fp8.sh) | 显式 `export HSA_FORCE_FINE_GRAIN_PCIE="${S11F_P2P:-${HSA_FORCE_FINE_GRAIN_PCIE:-1}}"` + 启动日志打印 | `S11F_P2P=0` 或 `HSA_FORCE_FINE_GRAIN_PCIE=0` |
| `/home/lyh/server/serve_27b_tp4.sh`（INT8 回滚槽，仓库外） | 生成的内层脚本加 `export HSA_FORCE_FINE_GRAIN_PCIE=${P2P:-1}`（备份 `serve_27b_tp4.sh.bak-20261011-s17`） | `P2P=0 bash serve_27b_tp4.sh` |
| 宿主 swap | `/mnt/game/swapfile`(16 GB) + `serve-swapfile`(6 GB) **保持开启**（maxlen 262144 需要；`/swap.img` 4 GB 不够） | `sudo swapoff /mnt/game/serve-swapfile /mnt/game/swapfile` |

**验证（运行中实例）**：`via P2P/IPC` **112**、`via SHM` 0；`Missing "HSA_FORCE_FINE_GRAIN_PCIE=1"` **0 次**；
`/health` 200；真实 chat 请求返回正常；四卡 VRAM **86%**；启动 ~120–150 s（AOT 缓存热）。

**生产槽位自身的两臂 A/B**（同一 FP8 配方，仅 `S11F_P2P` 不同）：

| 指标 | P0（`HSA=0`，SHM） | P1（`HSA=1`，P2P） | 提升 |
|---|---|---|---|
| 8K prefill TTFT（seed 777） | 41147.98 ms | 31918.16 ms | **1.29×** |
| 8K prefill TTFT（seed 888） | 40927.96 ms | 31706.34 ms | **1.29×** |
| decode median TTFT | 1086.24 ms | 888.93 ms | 1.22× |
| decode median TPOT | 125.04 ms | 123.36 ms | 1.4% |
| decode 输出吞吐 | 7.54 tok/s | 7.73 tok/s | 1.03× |
| RCCL 通道 | 112 SHM | 112 P2P | — |

**注意（口径）**：FP8 槽的绝对 prefill 明显慢于归档的 INT8 生产栈（`2.789 ms/prefill-token`）——
P0 41.1 s/8100 tok = **5.08 ms/prefill-token**，P1 31.9 s = **3.94 ms**。差在**量化路径**
（`VLLM_DCU_USE_FP8_EMU=1` 的 LDS 查表 + `v_dot2_f32_f16` 仿真内核）而非传输，
所以 P2P 在该槽只拿到 1.29×，而 INT8 + graph 臂（§2）拿到 1.89× —— 通信占比越低，P2P 收益越小。
机器空闲时 FP8/eager 槽的稳态 decode = 123–125 ms/token，与归档 INT8/eager 生产基线（121–122 ms/token）同量级。

