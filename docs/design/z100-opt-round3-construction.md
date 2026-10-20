# Z100（gfx906）优化第 3 轮 · 施工方案（可执行版）

> 版本：v1 · 2026-10-20 · 适用栈：`/home/lyh/z100-vllm`（vLLM 0.18.1 + DTK 26.04 + torch 2.11，gfx906 ×4）
> 依据：[第 2 轮实测](../result/) · [CUDA graph 可行性调研](./../result/z100-cudagraph-feasibility-20261020.md) · [S13/S16/S17 P2P 归档](../archive/z100-todolist.md) · 2026-10-20 套件 A/B（`runs/live-suite` vs `runs/host-regression-20261018-232832`）
> 读者：执行本轮的 agent。**每张 WP 卡片可独立分派**，卡片内命令可逐条复制执行。

---

## 0. 怎么用本文

1. 先读 §1 硬约束、§2 公共命令，**不可跳读**。
2. 按 §3 总表接单；每接一单，在 `.r3/STATUS.md` 把该 WP 置 `running` 并写 `owner`。
3. 每个 WP 必须产出 §6 规定的目录结构与 `RESULT.md`；**没有证据文件的结果一律不计**。
4. 遇到 §7 的升级条件立刻停下并上报，不要自行扩大范围。

---

## 1. 硬约束（违反任一 → 立即回滚并报告）

| # | 约束 | 校验方式 |
|---|---|---|
| C1 | **生产服务优先**：任何占卡实验前必须停 `z100-vllm-host`（user unit），并确认四卡 VRAM 合计 < 4096 MiB | `systemctl --user is-active z100-vllm-host` = inactive；`hy-smi --showmeminfo vram` |
| C2 | **实验结束必须恢复生产**：`systemctl --user start z100-vllm-host` → `/health=200`、`/v1/models` 有 `qwen3.8-27b`、发一条真实 chat 请求成功 | §2.7 |
| C3 | **单卡所有者**：同一时刻只允许一个 WP 占卡；用 `.r3/LOCK` + `flock` 串行化 | §2.1 模板 |
| C4 | **GPU 必须经 `systemd-run --user` 出沙箱**；直接在工作区跑 HIP 会 `Unable to open /dev/kfd`（假失败） | §2.2 |
| C5 | **必须 source 环境**：`DTK_ROOT=/mnt/game/host-runtime/dtk2604` + `tools/dcu/env.sh`（含 `HSA_FORCE_FINE_GRAIN_PCIE=1`，RCCL P2P 的门禁） | §2.0 |
| C6 | **缓存根固定**：`VLLM_CACHE_ROOT=/mnt/game/vllm-cache-host`、`TRITON_CACHE_DIR=/mnt/game/host-serve/triton-cache`；否则首启编译 10+ min 且跨臂不可比 | `serve.log` 首行 env dump |
| C7 | **不重建不交付**：改 `csrc/` 必须备份旧 `.so` md5 → 用 `build-sync.sh` 重建 → 记录新 md5；禁止手改 `/mnt/game/port/build-sync/temp` 里的产物 | §2.4 |
| C8 | **单变量**：每臂只允许一个差异；A/B/A 交错 ≥3 轮；单次差值 **<2% 视为噪声**，不作结论 | `COMPARE.md` 的 A/A drift 行 |
| C9 | **不碰生产脚本**：`tools/dcu/serve_host_dflash2.sh` 本轮只读；实验用 `.r3/scripts/arm_serve.sh` 副本 | `git status` |
| C10 | **DP4X 约束**：卡 2/3（Z100）有 DP4X 静默错算缺陷，仅 Triton 内核有运行时 guard；**新增自研 kernel 必须走 guard，微基准优先用 `DEVICES=0,1`** | kernel 内 `guard` 分支 |
| C11 | **量化/编译类改动必须做数值对拍**：eager 臂 4 prompt × 128 token，`max|Δlogprob|` 与文本 sha256 | §6.3 |
| C12 | **不许外推**：历史数字（S7/S9b/S14 的 2.44×）不得作为本轮预期，只作参考 | 报告里注明出处与差异 |

---

## 2. 公共环境与命令

### 2.0 环境装载（每个 shell 开头）

```bash
export REPO=/home/lyh/z100-vllm
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
set +u; . "$REPO/tools/dcu/env.sh"; set -u
export PYTHONPATH="$REPO:/mnt/game/triton-gfx906/install"
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-host
export TRITON_CACHE_DIR=/mnt/game/host-serve/triton-cache
```

### 2.1 取卡（所有 WP 的统一开头/结尾）

```bash
mkdir -p "$REPO/.r3"/{logs,scripts,so-backup}
exec 9>"$REPO/.r3/LOCK"; flock -n 9 || { echo "FATAL: another WP holds the GPUs"; exit 1; }
bash "$REPO/.r3/scripts/free_gpu.sh"      # 停生产 + 确认 VRAM 释放
trap 'bash "$REPO/.r3/scripts/restore_prod.sh"' EXIT
```

`.r3/scripts/free_gpu.sh`（首次创建）：

```bash
#!/usr/bin/env bash
set -u
systemctl --user stop z100-vllm-host 2>/dev/null || true
for i in $(seq 1 60); do
  used=$(hy-smi --showmeminfo vram 2>/dev/null \
    | sed -n 's/.*vram Total Used Memory (MiB): *\([0-9][0-9]*\).*/\1/p' \
    | awk '{s+=$1} END{print s+0}')
  [ "${used:-99999}" -lt 4096 ] && { echo "[free_gpu] OK used=${used} MiB"; exit 0; }
  sleep 5
done
echo "[free_gpu] FATAL: VRAM still in use"; exit 1
```

`.r3/scripts/restore_prod.sh`：

```bash
#!/usr/bin/env bash
set -u
systemctl --user start z100-vllm-host
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 \
    -H "Authorization: Bearer $(cat /mnt/game/host-serve/api-key 2>/dev/null)" \
    http://127.0.0.1:8000/health || echo 000)
  [ "$code" = 200 ] && { echo "[restore_prod] healthy"; exit 0; }
  sleep 5
done
echo "[restore_prod] WARN: not healthy"; exit 1
```

### 2.2 出沙箱跑 GPU 负载

```bash
systemd-run --user --unit=z100-wpX --collect \
  -p WorkingDirectory="$REPO/.r3" \
  -p Environment=DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
  /bin/bash -lc 'cd /home/lyh/z100-vllm && . .r3/env.sh && <命令>'
journalctl --user -u z100-wpX -f          # 实时看输出
```

### 2.3 起一条实验臂（复用第 2 轮已验证的 harness）

```bash
cp /home/lyh/vllm-test/vllm-docker-probe/runs/hipopt-20261019/scripts/arm_serve.sh "$REPO/.r3/scripts/"
ARM_NAME=e0 OVERLAY= EXTRA_ENV="" PORT=8000 MBT=4096 \
  bash "$REPO/.r3/scripts/arm_serve.sh" start
bash "$REPO/.r3/scripts/arm_serve.sh" wait      # /health 轮询，WAIT_S=600
bash "$REPO/.r3/scripts/arm_serve.sh" stop
```

> `arm_serve.sh` 已内建：拒绝在生产 unit 存活时启动、按 VRAM<4096 MiB 判 `GPU FREE`、把 `OVERLAY`/`EXTRA_ENV` 叠加在生产配方之上、日志落 `logs/arm_<name>.log`。

### 2.4 重建 `_rocm_C`（只有 WP2/WP3/WP7 改 csrc 时）

```bash
md5sum "$REPO/vllm/_rocm_C.abi3.so" | tee "$REPO/.r3/so-backup/before.md5"
docker exec -e MAX_JOBS=5 z100-regress bash -lc 'bash /mnt/game/port/build-sync.sh _rocm_C'
md5sum "$REPO/vllm/_rocm_C.abi3.so" | tee "$REPO/.r3/so-backup/after.md5"
```

- `build-sync.sh` ≈ 100 s（增量）；**不要**用 `build-sync2.sh`（强删 stamp 会全量重 hipify）。
- 重建后先 `tools/dcu/check_rocm_c_ops.py` 自检，再起服务。

### 2.5 跑标准化套件（跨臂可比）

```bash
cd /home/lyh/vllm-test/vllm-docker-probe
export VLLM_API_KEY="$(cat /mnt/game/host-serve/api-key 2>/dev/null)"
RUNTIME=host PORT=8000 RUN_DIR="$REPO/.r3/runs/<arm>/suite" \
  bash scripts/run_suite.sh
python3 -m probe compare --a "$REPO/.r3/runs/e0/suite/suite_results.json" \
                         --b "$REPO/.r3/runs/e2/suite/suite_results.json"
```

### 2.6 抓 trace / step 明细

```bash
ITER_DETAILS=1 bash "$REPO/.r3/scripts/arm_serve.sh" start   # 带 iteration details
# 用 probe 的 /start_profile 限步抓取（避免 36K 全量 OOM，主机仅 31 GiB）
```

### 2.7 收尾自检（每个 WP 必做）

```bash
systemctl --user is-active z100-vllm-host                 # active
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $(cat /mnt/game/host-serve/api-key)" http://127.0.0.1:8000/health   # 200
curl -s -H "Authorization: Bearer $(cat /mnt/game/host-serve/api-key)" http://127.0.0.1:8000/v1/models | head -c 200
hy-smi | head -12                                          # 四卡无残留进程
```

---

## 3. 工作包总表

| ID | 工作包 | 依赖 | GPU 时长 | 人日 | 主产出 | 一句话停止线 |
|---|---|---|---|---|---|---|
| **WP0** | CUDA Graph 三臂（E0/E1/E2） | H1 | 6–8 h | 2.0 | `WP0/RESULT.md` + 三份 suite | 修完 LUT 后仍有第二处 Dynamo break 且 4 h 内无解 → 转路线 B 或关闭 |
| **WP1** | SOL 天花板标定 | — | 1 h | 0.5 | `WP1/sol.md` | 基准 CV>3% 则重跑；不设停止线（必做） |
| **WP2** | custom all-reduce（借 nlzy 参考） | WP1 | 3–4 h | 1.5 | `WP2/RESULT.md` | ①AR 时间被最慢卡主导 ②≤64 KB 段 <20% 改善 ③hang → 关闭 |
| **WP3** | 草稿塔 weight-only 量化 | WP1 | 6 h | 3.0 | `WP3/RESULT.md` + 量化权重 | 接受长度降 >0.15 或 kernel 收益 <5% → 关闭 |
| **WP4** | RCCL 参数矩阵 | — | 2 h | 0.5 | `WP4/RESULT.md` | 全矩阵 <2% → 关闭 |
| **WP5** | c8 尾延迟（ITL p99 4.5 s） | — | 3 h | 1.0 | `WP5/RESULT.md` | 无法定位到可修点 → 转为调度器提案，不落地 |
| **WP6** | 套件口径修正 + 单变量归因矩阵 | WP0/2/3 | 4 h | 1.0 | `WP6/*` | — |
| **WP7** | GDN prefill `.item()` 一行修复 | — | 2 h | 0.5 | `WP7/RESULT.md` | 单测红或无 TTFT 收益 → 只留 default-inert |

**人工前置（H1，无 sudo 需用户执行）**

```bash
sudo swapon /mnt/game/swapfile                 # 16 GiB，WP0 graph 臂必需（S14c：净 +3.5–4.3 GiB）
sudo sysctl vm.swappiness=200                  # 窗口内；收尾回 60
```

> 若 H1 未完成，WP0 **只能**跑到 E1（仅编译），不得捕获图。

---

## 4. WP 详细卡片

### WP0 · CUDA Graph 三臂（最高优先）

**目标**：判定"编译 / 编译+PIECEWISE 图"在**当前生产配方**（27B-FP8 + DFlash k=7 + GDN + TP=4）上的真实收益，产出可回滚的配置结论。

**已知阻塞**（已实测，勿重复发现）：`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py:163` 的 `if inf_codes.any():` 是 tensor 条件，Dynamo 判 `Data-dependent branching`，4/4 worker 在捕获前退出。

**步骤**

1. **修 LUT 分支**（唯一必要的代码改动）。生成 patch，实体改到工作树：

   ```bash
   mkdir -p "$REPO/.r3/WP0/patches"
   # 改：if inf_codes.any():  ->  if fp8_dtype == torch.float8_e5m2:
   # （e4m3fn 的 inf_codes 恒为全零，静态化后逐位等价）
   git -C "$REPO" diff -- vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py \
     > "$REPO/.r3/WP0/patches/0001-lut-static-branch.patch"
   ```

2. **单测**（必须绿）：

   ```bash
   cd "$REPO" && .venv/bin/python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q
   ```

3. **编译探针（E1，仅编译不捕获图）**：

   ```bash
   ARM_NAME=e1 EXTRA_ENV="VLLM_ENABLE_V1_MULTIPROCESSING=1" \
     bash "$REPO/.r3/scripts/arm_serve.sh" start   # 内部用 --compilation-config '{"cudagraph_mode":"none"}'
   grep -c 'Data-dependent branching' "$REPO/.r3/logs/arm_e1.log"   # 期望 0
   ```
   非 0 → 记下新栈顶，回到步骤 1 继续消 break（**每个 break 一个 patch，单独提交到 `.r3/WP0/patches/`**），累计 4 h 无解即触发停止线。

4. **E2（编译 + PIECEWISE）**：`EXTRA_ENV` 追加

   ```bash
   --compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16]}' --cudagraph-metrics
   ```
   记录 `Graph capturing finished … took X GiB`、`MemAvailable` 最低值（`/proc/meminfo`）、`hy-smi` VRAM。

5. **三臂数据**：E0(`--enforce-eager`，现网) / E1 / E2，**交错 A/B/A**，每臂 3 次套件 + 1 次 iteration-details 窗口。

**验收（全绿才算通过）**

| 判据 | 阈值 |
|---|---|
| G-0 | 4/4 worker 日志 `Data-dependent branching` = 0；`/health` 200 |
| G-1 | c1 TPOT p50 改善 **≥10%**（E2 vs E0，3 次中位数） |
| G-2 | E1 vs E0 单独量化"编译"贡献（用于归因，不设阈值） |
| G-3 | c8 吞吐 ≥ E0−2%；s04/s07 TTFT ≥ E0−3% |
| G-4 | 0 个 NaN/ERROR/RCCL 报错；accept length 变化 ≤0.15 |
| G-5 | MemAvailable 低水位 ≥1.5 GiB；E2 VRAM ≤ E0+1 GiB |

**回滚**：`git -C $REPO apply -R .r3/WP0/patches/*.patch`；生产脚本未改动，`systemctl --user start z100-vllm-host` 即恢复。

**产出**：`WP0/{RESULT.md, patches/, arms/{e0,e1,e2}/suite/, trace/}`。

---

### WP1 · SOL 天花板标定（先做，给后续 WP 定量尺）

**目标**：拿到本机 fp16/bf16/int8 算力与带宽天花板，把主力 kernel 折算成 SOL%，决定哪些 kernel 线还值得投入。

**步骤**

1. 把博客 [373](https://blog.muwinds.com/archives/373/) 的 `dcu_benchmark.py` 落盘为 `$REPO/tools/dcu/dcu_benchmark.py`（正文已给完整代码）。
2. 空闲态跑（`DEVICES=0,1` 优先，避开 DP4X 卡）：

   ```bash
   systemd-run --user --unit=z100-wp1 --collect /bin/bash -lc \
     'cd /home/lyh/z100-vllm && . .r3/env.sh && .venv/bin/python tools/dcu/dcu_benchmark.py \
        --out-dir .r3/WP1/bench --seconds 20 --warmup 2'
   ```

3. 折算 SOL（写进 `sol.md`）：

   | 对象 | 形状/口径 | SOL 公式 |
   |---|---|---|
   | fp8_emu block GEMM | 5 真实形状 × M=7920（第 2 轮 census） | `2·M·N·K / t / 峰值_bf16` |
   | unified_attention 2D/3D | 36864 ctx，M=3520 | `4·S·D·L / t / 峰值` |
   | 草稿塔 bf16 GEMM | 每 step 18% 那份 | 同上（bf16） |
   | KV 读写 | copy 带宽 | `bytes / t / 带宽` |

**判据**：每个 benchmark 3 次，CV<3%；输出一张"占比 × SOL%"表。

**判据后的处置**：任一 kernel SOL ≥60% → 该 kernel 线关闭（不再做 tile/微调）；SOL <20% 且占比 >10% → 升级为下一轮候选。

---

### WP2 · custom all-reduce（借参考仓库，禁止直接"开开关"）

**前置事实（已核实，勿重复调研）**

- 参考仓库 `nlzy/vllm-gfx906`（已克隆到 `$REPO/.refs/vllm-gfx906`）**未改门禁**：`use_custom_allreduce()` 仍只认 gfx94/95；`world_size>2` 且非 fully-connected 仍禁用；`custom_all_reduce.cu` 仍用 `hipDeviceMallocUncached`。它提供的是**可编译的 kernel + 独立测量程序**，不是使能补丁。
- 本机拓扑（今日实测）：`hy-smi --showtopo` → 4 卡两两 `PCIE`、`Hops=2` ⇒ `is_fully_connected()` 必 False。
- `hipDeviceMallocUncached`(0x3) 在本平台恒 `rc=2`（S16 实测，开 `HSA_FORCE_FINE_GRAIN_PCIE=1` 后仍失败）；`hipDeviceMallocFinegrained`(0x1) 可用，且 RCCL 自己就用它。
- 旧"收益 0"结论作废：S13 的 A/B 实际是 RCCL vs RCCL。

**步骤（顺序不可颠倒）**

1. **先判"值不值得"**（30 min，零代码）：跑 WP5 的 iteration-details 窗口（c1、`--enable-logging-iteration-details`），从 trace 里取每 rank 的 `ncclDevKernel_AllReduce*` 时长与到达时刻。
   - 若 AR 墙钟被**最慢 rank 的到达延迟**主导（skew >50%）→ **直接关闭 WP2**，结论记为"通信受卡间算力不对称限制（硬件线，已关闭）"。
2. **alloc 能力复核**（10 min）：`systemd-run` 跑 `tools/dcu/p2p_capability_probe.py`，确认 Finegrained `rc=0`、Uncached `rc=2`，并记录 `hipDeviceCanAccessPeer` 与 `hipMemcpyPeer` 带宽。
3. **改 alloc**（唯一 csrc 改动）：`csrc/custom_all_reduce.cu:168-171` 改成

   ```cuda
   #if defined(HIP_UNCACHED_MEMORY)
     hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached);
   #else
     hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocFinegrained);
   #endif
   ```
   并加 `VLLM_DCU_CUSTOM_AR_ALLOC=uncached|finegrained|default`（默认 `default`=原行为），重建 `_rocm_C`（§2.4）。
4. **门禁**（Python，全部 env 后置、默认关）：`VLLM_DCU_CUSTOM_AR=1` 时
   - `vllm/platforms/rocm.py:1354-1356` 追加 `or "gfx906" in _GCN_ARCH`（仅当 env=1）；
   - `custom_all_reduce.py` 的 `world_size>2 and not fully_connected` 分支改为"env=1 时仅告警不禁用"。
5. **独立测量**：新写 `tools/dcu/car_bench.py`（4 rank，`torch.distributed` NCCL 对照 + `CustomAllreduce` 实测），尺寸 {10 KB, 40 KB, 160 KB, 1 MB, 4 MB}，**必须校验归约结果与 NCCL 一致**，每格 ≥50 样本。超时 60 s 视为 hang。
6. **e2e**：`EXTRA_ENV="VLLM_DCU_CUSTOM_AR=1 VLLM_DCU_CUSTOM_AR_ALLOC=finegrained"` 起臂，跑套件。

**判据**

| 判据 | 阈值 |
|---|---|
| 步骤 1 | 若 skew>50% → 关闭（不算失败，是证据） |
| 步骤 5 | ≤64 KB 段延迟 **≥20% 优于 RCCL**；≥1 MB 不回退；结果逐元素一致 |
| 步骤 6 | c1 TPOT **≥3%**；c8 ITL p99 不恶化；0 hang / 0 NaN |

**回滚**：env 默认关；`.so` 用 `so-backup/before.md5` 对应的备份还原（`git apply -R` csrc patch + 重建）。

**预期（写进报告，防外推）**：RCCL 在 ≤64 KB 已接近 PCIe 极限（S16：持平约 80 µs），in-engine 411 µs/call 主要是等最慢卡 ⇒ 合理预期 **1–4% step**，不是 10%+。

---

### WP3 · 草稿塔 weight-only 量化

**目标**：把 DFlash2 草稿塔（`/mnt/game/models/Qwen3.8-27B-DFlash2-ms`，bf16，decode step 占 ~18%）降到 W8A16，复用已落地的 `fp8_emu_gemm_block_w8a16` 路线。

**步骤**

1. 离线量化脚本放 `$REPO/.r3/WP3/quant/`：per-channel int8 权重 + fp16 激活；校准集用 512 条真实业务 prompt（从生产日志取，去敏）。
2. 权重落 `/mnt/game/models/Qwen3.8-27B-DFlash2-ms-w8a16/`（**不改原目录**）。
3. kernel 级：`mprobe` 对比 bf16 GEMV vs W8A16 tile，5 个真实形状（草稿塔），要求加权 **≥5%**。
4. 接受率对拍：同 64 条 prompt，比较 `mean_acceptance_length`、逐位置接受率直方图。
5. e2e 三臂：原草稿 / 量化草稿 / 原草稿（A/B/A）。

**判据**：kernel ≥5%；accept length 变化 ≤0.15（且第 7 位不塌）；c1 TPOT ≥5%；0 NaN。

**停止线**：accept 长度降 >0.15，或 kernel <5% → 关闭并归档量化脚本（供后续用）。

---

### WP4 · RCCL 参数矩阵（便宜、先跑）

**步骤**

1. 用 `tools/dcu/ar_bench.py` 先扫尺寸-延迟基线（10 KB / 40 KB / 160 KB / 1 MB）。
2. 单变量扫：`NCCL_ALGO`(Ring/Tree)、`NCCL_PROTO`(Simple/LL/LL128)、`NCCL_MIN_NCHANNELS`(1/2/4/8)、`NCCL_BUFFSIZE`、`NCCL_NTHREADS`。
3. **禁止**：`NCCL_SHM_DISABLE=1`（已证 7.3× 慢）、`NCCL_P2P_DISABLE=1`（已证无益）。
4. 最优 1–2 组进 e2e 套件（c1/c8/s04）。

**判据**：microbench ≥5% 且 e2e c1 TPOT ≥2%，否则关闭。

---

### WP5 · c8 尾延迟治理（ITL p99 4.5 s）

**背景**：2026-10-20 套件 c8 `ITL p99 = 4506 ms`（A 轮 3134 ms，**恶化 44%**），而 TPOT p50 正常（97 ms）、preemptions=0、KV 峰值 70.7% ⇒ 是调度停顿而非内存压力。生产 `MBT=4096`（注意不是 8192）。

**步骤**

1. `ITER_DETAILS=1` + 固定 c8 负载，导出 step 序列，找出 `ITL > 2 s` 的 step 及其 `num_prefill_tokens` / `num_decode_tokens`。
2. 判定形态：(a) prefill chunk 插入打停全批 decode；(b) 投机验证步过长；(c) 单请求 straggler。
3. 若 (a)：`MBT ∈ {2048, 4096, 8192}` × c8 单变量扫描（双指标：c8 ITL p99 + c1 TTFT）；仍不达标则提出 fork 调度器改动（decode 优先/每步保留 decode 预算），**只出提案 + 小 patch，不在本轮上线**。

**判据**：c8 ITL p99 <1.5 s 且 c1 TTFT 回退 ≤2%。

---

### WP6 · 套件口径修正 + 单变量归因矩阵

**已知口径缺陷（本轮修复项）**

| 缺陷 | 证据（2026-10-20） | 修法 |
|---|---|---|
| token 估算系统性偏低 ~45% | 标称 512/4096/16384/24576 → 实测 709/5771/23659/35797 | 用 tokenizer 精确构造并把场景名改为实测值 |
| s06 前缀缓存只命中 1760/3589 | 前两次 `cached=0` | 对齐 `block_size=880`，或显式预热 |
| s07 场景间前缀污染 | 首请求 `cached_tokens=22000` | 场景间清 KV / 用互不重叠语料 |
| 重复次数不足 | s04×2、s07×1 | 关键场景 ≥3 |
| 遥测粒度 | 30 s | 改 1 s（复用博客 373 采样器） |

**做法**：把 probe 仓库拷到 `$REPO/.r3/probe/` 内修改（工作区外不可写），用 `PYTHONPATH` 指向修改版跑套件；产出 patch 交人工合并。

**归因矩阵**（本轮的"总账"）：对 10-18→10-20 叠加上线的 4 项各做一次**单关**（O1/O2 fastpath、DECODE_CFG=35、S24 prefill tiles、W8A16/FP16_PREFILL），交错 A/B/A，输出各项贡献表。

---

### WP7 · GDN prefill `.item()` D2H 一行修复

**背景**（第 1 轮 P2 遗留，2026-10-20 已复核到当前行号）：gfx906 的 conv fallback 在
`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:243-252` 会**优先信任调用方给的 CPU 镜像**（`query_start_loc_cpu=`，长度一致才用），没有则逐序列 `.item()`（每层每 prefill chunk 2 次 D2H）。镜像本身已经在 `vllm/v1/attention/backends/gdn_attn.py:101,397` 算好（`non_spec_query_start_loc_cpu`），**但 prefill 调用点没传**：

- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1616-1626` 的 `causal_conv1d_fn(...)` 只传了 `query_start_loc=non_spec_query_start_loc`。

**步骤**

1. 在该调用点补一个参数（一行）：

   ```python
   query_start_loc_cpu=attn_metadata.non_spec_query_start_loc_cpu,
   ```

   注意 `causal_conv1d.py` 仅在长度与 `query_start_loc` 一致时采用镜像，因此即使 metadata 为 `None` 或切片不同也安全（自动回退原行为）。
2. 单测：`.venv/bin/python -m pytest tests/ -k gdn -q`（就近用例）。
3. 验证：36K cold TTFT（预期 <1%，GPU 98% busy）+ c8 ITL p99（host jitter 敏感，可能有感）。

**判据**：单测绿、无回退；有收益则默认开启，无收益则保留 default-inert patch。

---

## 5. 排期（GPU 窗口按可用时段滚动）

| 日 | 上午 | 下午 | 备注 |
|---|---|---|---|
| D0 | 人工前置 H1 + WP1（1 h） | WP4（2 h）+ WP5 诊断（2 h） | 全部在 `DEVICES=0,1` 或短窗口 |
| D1 | WP0 步骤 1–3（3 h） | WP0 步骤 4（2 h） | 需 swapfile |
| D2 | WP0 步骤 5（3 h）三臂 | WP0 判定 + 报告 | |
| D3 | WP2 步骤 1–4（3 h） | WP2 步骤 5–6（2 h） | 依赖 `.so` 重建 |
| D4 | WP3（6 h） | — | 最长单包 |
| D5 | WP6 归因矩阵（4 h） | WP7（2 h）+ 汇总报告 | |

**GPU 总时长** ≈ 22 h；每包结束必须恢复生产（C2）。

---

## 6. 证据与汇报规范

### 6.1 目录

```
$REPO/.r3/
├── STATUS.md                 # 总看板（每步更新）
├── LOCK                      # flock 锁
├── env.sh                    # §2.0 的环境
├── scripts/{free_gpu,restore_prod,arm_serve}.sh
├── so-backup/{before,after}.md5 + 对应 .so
├── runs/<arm>/{suite/,trace/,arm.json}
└── WP<N>/{RESULT.md,patches/,data/,logs/}
```

### 6.2 `run_meta.json`（每臂必写）

```json
{
  "wp": "WP0", "arm": "e2", "started_at": "2026-10-21T10:00:00+08:00",
  "git_head": "6d6af0d44", "so_md5": "f205be8f", "dirty": false,
  "hw": {"cards": ["0,1,2,3"], "pwr_cap": [280,280,250,250]},
  "env": {"VLLM_DCU_CUSTOM_AR": "0", "MBT": "4096", "MAXLEN": "262144"},
  "cmd": "…完整 cmdline…", "cache_root": "/mnt/game/vllm-cache-host"
}
```

### 6.3 `RESULT.md` 模板（缺章节视为未完成）

```markdown
# WP<N> · <名称> 结果

## 0 结论（一句话 + 判据表逐行 pass/fail）
## 1 变更清单（文件:行、patch 文件名、是否已回滚）
## 2 测量方法（臂、次数、交错顺序、单变量证明）
## 3 数据（表格：绝对值 + 相对值 + A/A drift）
## 4 数值/正确性对拍（若涉及 kernel/量化）
## 5 与历史结论的异同（引用具体文档与数字）
## 6 残留风险与建议下一步
## 7 复现命令（逐条可复制）
```

### 6.4 状态看板

```markdown
| WP | 状态 | owner | 开始 | 结束 | 判据结果 | 证据目录 |
|---|---|---|---|---|---|---|
| WP0 | running | agent-a | 10-21 10:00 | — | — | .r3/WP0 |
```

---

## 7. 升级/停止规则（遇到即停，不自行扩范围）

1. 出现 **hang / 卡死 / RCCL 报错** 超过 2 次 → 停 WP，恢复生产，上报。
2. **宿主 `global_oom`**（历史 S17 教训：worker 被杀但 `/health` 仍 200 假活）→ 立即 `hy-smi --showpids` 核实、恢复生产并上报。
3. 单次 A/B 差值 <2% 却想"再跑一次看看" → 先做 A/A drift 对照，否则不许宣称收益。
4. 需要改生产脚本 / 改门禁默认值 / 改 `=1` 默认 → 一律上报，不得自行决定。
5. 任何"顺手优化"（重构、清理、格式化）→ 不做。

---

## 8. 本轮禁止清单（历史已证否，重复即浪费）

| 项 | 结论 | 出处 |
|---|---|---|
| GPU2/3 功率墙/时钟不对称 | 硬件（Z100L 280 W vs Z100 250 W），**已关闭** | 2026-10-20 遥测 |
| prefill fp8 GEMM tile/kFBk 调优 | cfg10 已近上限，无 ≥5% 候选 | 第 2 轮 |
| GDN decode kernel 微调 | 上限 0.73–1.03% | 第 1 轮 |
| 短算子融合 | 无单一 >5% 项 | 第 1 轮 |
| spec k 值扫描 | k=7 已最优 | optmatrix |
| `NCCL_SHM_DISABLE=1` | 慢 7.3× | S11f |
| 去 `--disable-custom-all-reduce` 但不改代码 | 门禁吃掉，A/B 实为 RCCL vs RCCL | S13（本轮 WP2 已改为有条件重开） |
| 把历史 2.44× 当本轮预期 | 跨栈/跨模型，不可外推 | graph 调研 §7 |

---

## 附录 A · WP 卡片机器可读版

```yaml
wp:
  - id: WP0
    name: cuda-graph-three-arm
    depends: [H1]
    gpu_hours: 7
    owner_role: graph
    deliverable: .r3/WP0/RESULT.md
    gates: [trace-complete, tpot-10pct, c8-2pct, mem-1.5gib]
    kill: second-dynamo-break-unfixed-4h
  - id: WP1
    name: sol-ceiling
    depends: []
    gpu_hours: 1
    owner_role: bench
    deliverable: .r3/WP1/sol.md
    gates: [cv-under-3pct]
  - id: WP2
    name: custom-allreduce
    depends: [WP1]
    gpu_hours: 4
    owner_role: comms
    deliverable: .r3/WP2/RESULT.md
    gates: [straggler-diag, le64kb-20pct, e2e-3pct]
    kill: skew-over-50pct-or-hang
  - id: WP3
    name: draft-tower-w8a16
    depends: [WP1]
    gpu_hours: 6
    owner_role: quant
    deliverable: .r3/WP3/RESULT.md
    gates: [kernel-5pct, accept-delta-0.15]
  - id: WP4
    name: rccl-env-matrix
    depends: []
    gpu_hours: 2
    owner_role: comms
  - id: WP5
    name: c8-tail-latency
    depends: []
    gpu_hours: 3
    owner_role: scheduler
    gates: [itl-p99-1.5s]
  - id: WP6
    name: suite-hygiene-and-attribution
    depends: [WP0, WP2, WP3]
    gpu_hours: 4
    owner_role: lead
  - id: WP7
    name: gdn-d2h-one-liner
    depends: []
    gpu_hours: 2
    owner_role: mamba
```

## 附录 B · 首日 30 分钟自检清单

```bash
- [ ] .r3/{logs,scripts,so-backup} 已建；env.sh 可 source
- [ ] free_gpu.sh / restore_prod.sh 可执行且 end-to-end 验过一次
- [ ] 生产 unit 状态记录进 STATUS.md
- [ ] 当前 .so md5 已备份
- [ ] 宿主 MemAvailable / swap 状态已记录（决定 WP0 能否到 E2）
- [ ] hy-smi --showtopo 快照存档（WP2 的前置证据）
```
