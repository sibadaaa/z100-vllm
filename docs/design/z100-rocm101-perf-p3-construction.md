# Z100 ROCm 10.1 P3 施工图（任务卡版）

> **读者**：没有前置对话上下文的执行 agent。本文档假定你没读过上游判读，也能独立开工。
> **上游判读（why）**：[`z100-rocm101-perf-p3-optimization-plan.md`](z100-rocm101-perf-p3-optimization-plan.md)（本文只讲"怎么做"）
> **输入报告**：`/mnt/game/port/perf-runs/20261024-053333-E2C/TUNE_REPORT.md`
> **配套文件（本机资产，见下方说明）**：
> - 臂定义：`/home/lyh/z100-vllm/tools/dcu/p3_arms/*.env`（可 `source`，一个文件 = 一个臂）
> - 判读器：`/home/lyh/z100-vllm/tools/dcu/p3_score.py`（**唯一允许的指标口径**）
>
> ⚠️ **`tools/dcu/` 整个目录被 `.gitignore` 排除**（`.gitignore:313`，这是仓库既有选择，
> `s23_arms/`、`s24_arms/`、`perf1_*` 同样如此）⇒ 上面两个是**本机资产，不在 git 里**。
> 为了让任何环境都能重建，**§3.4 给出了每个臂的逐字内容**——臂文件丢了就照着 §3.4 重建。
> **体例**：沿用 [`z100-rocm101-perf-construction.md`](z100-rocm101-perf-construction.md)（WP 编号、纪律、G0–G8 门）。

---

## 0. 三十秒看懂

**要做的事**：把现网候选（E2C）从

| 指标 | 现值 | Must | 判定 |
|---|---:|---:|---|
| TTFT 512c1 | 2731 ms | ≤2500 | ✗ |
| decode step | 240.9 ms | ≤210 | ✗ |
| TPOT | 46.0 ms | ≤45 | ✗ |
| 接受长度 / 率 | 5.36 / 62.3% | ≥4.0 / ≥45% | ✓ |

推到 **Must 全绿**。判读已定位 4 个可回收点，合计理论可回收 ~150 ms/步 + ~900 ms TTFT。

**执行顺序（严格）**：

```
WP-P0.0 基线复现（两把尺子）
   │
   ├─► WP-A1  GDN prefill chunked      ← 最大单项，预期 TTFT −600ms
   ├─► WP-A2  GDN decode triton        ← 预期 step −80~93ms（有反证，必须先测）
   ├─► WP-A4  CPU 快路 O1+O2
   ├─► WP-A5  S24 DYNK
   ├─► WP-A6  BF16_GEMV                ← 用户决策项，只测不默认开
   ├─► WP-A7  attn prefill 3D          ← 只影响长 ctx
   ├─► WP-A3  decode tile cfg 11–19    ← **禁止用 35**
   └─► WP-A8  conv1d triton            ← **有 GPU wedge 风险，仅短 ctx**
            │
            └─► WP-COMBO（A1+A2+A4 叠加）→ WP-P3.1 graph route A → WP-P4 验收

并行（不占 GPU，可同时派给别的 agent）：
   WP-P1.1 修 trace · WP-P1.2 修遥测 · WP-P1.3 补微基准 · WP-P1.4 补 k=8 覆盖
```

**三条铁律**（违反任何一条，本轮数据作废）：

1. **一次只改一个变量。** 每次只 `source` 一个 `a*.env`。唯一例外是 `a_combo.env`，且必须在其三个单臂**各自单独过门之后**才允许。
2. **4 张卡是互斥资源。** 同一时刻只能有一个实例在 8100 上。开工前必须过 §2.1 的预检，收工必须过 §2.2。
3. **指标只能用 `tools/dcu/p3_score.py`。** 不要自己算，不要引用别人的 `case_metrics.json`。

---

## 1. 只读事实（**不要重新调研**，直接用）

### 1.1 栈与模型

| 项 | 值 |
|---|---|
| 运行时 | `host`（裸跑，非 docker） |
| 源码树 | `/mnt/game/port/gfx906-vllm-rocm101-perf`（= `dtk/dtk` 已 merge 的 rocm101 树） |
| Python | `/home/lyh/rocm101-host/venv/bin/python`（3.12.14） |
| torch / HIP | `2.16.0.dev20261016+rocm10.1` / `7.16.26385` |
| triton | `3.6.0`（`/mnt/game/triton-z100-rocm101/install`） |
| 目标模型 | `/mnt/game/models/Qwen3.8-27B-FP8`（Qwen3.5 架构，64 层 = 48 linear_attention + 16 full_attention；hidden 5120；24 q / 4 kv × head_dim 256；FFN 17408；vocab 248320；fp8 block 128） |
| 草稿模型 | `/mnt/game/models/Qwen3.8-27B-DFlash2-ms`（5 层，**bf16**，k=7，block_size 8） |
| 显存 | **单卡已用到 92–93%** ⇒ 任何"多缓存一份权重"的方案都放不下 |

### 1.2 实测 token ≠ 标称 token（**先除这个坑**）

探针的 `prompt_tokens` 是**估算**，系统性偏低 ~38%：

| 标称 case | 实际 prompt token |
|---|---:|
| `512c1` / `512c4` | **709** |
| `709c1` / `709c2` | **979** |
| `256c4` | 379（首请求） |
| `64c1` | 123 |

⇒ 历史"B1 709 tok"与本节 `512c1` 同长可直接比；**不要把 `512c1` 当 512 tok 去比别的记录。**

### 1.3 基线（E2C = E2 生产候选，`p3_score.py` 口径）

| case | tok | c | TTFT ms | step ms | TPOT ms | accLen | acc% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64c1 | 123 | 1 | 641.04 | 239.98 | 64.00 | 3.771 | 39.6 |
| 256c4 | 379 | 4 | 3879.62 | 440.86 | 99.57 | 4.690 | 52.7 |
| 512c1 | 709 | 1 | **2731.26** | **243.33** | **45.98** | 5.462 | 63.7 |
| 512c4 | 709 | 4 | 6798.05 | 540.85 | 112.85 | 5.269 | 61.0 |
| 709c1 | 979 | 1 | 3741.49 | 243.03 | 57.40 | 4.516 | 50.2 |
| 709c2 | 979 | 2 | 4805.58 | 382.09 | 91.33 | 4.516 | 50.2 |

> **口径偏移备忘**：本表 step 是 inter-token-latency 的**均值**，历史 `case_metrics.json` 给的
> `decode_step_p50_ms`=240.9 是 **p50**；accLen 本表含 bonus token，历史值略低（5.36 vs 5.462）。
> 差 ≤2%，**不改变任何 gate 判定**。跨臂比较一律用本表口径。

### 1.4 屋顶（判"还值不值得做"的唯一依据）

| 量 | 值 | 出处 |
|---|---|---|
| D2D 拷贝（读+写总量） | **721.5 GiB/s = 775 GB/s** | 本 run `mb_memcpy_d2d_1g` |
| 纯读流天花板（保守） | **407 GB/s** | `.r3/REPORT-round3.md:30-33` |
| fp16 GEMM | **18.2**（Z100L 4096³）/ **19.61** TFLOP/s | `vllm/platforms/rocm.py:288-290` / `.r3:30-33` |
| bf16 GEMM（仿真） | 8.2 / 8.91 TFLOP/s | 同上 |
| 实测 fp8 W8A16 decode（M=8） | 2.80 TFLOP/s、**175 GB/s** | 本 run |
| 实测 bf16 GEMM（M=8） | **0.163 TFLOP/s、20.4 GB/s** | 本 run |

### 1.5 时间去向（判读结论，直接用）

**decode 每步 240.9 ms**（48 个 GDN 层 / 256 次 GEMM，已交叉校验）：

| 项 | ms/步 | 归属 |
|---|---:|---|
| GDN 线性注意力 `fused_sigmoid_gating_delta_rule_gfx906_mtp_update` 48×2.040 | **97.9** | R1 |
| W8A16 decode GEMM 256 次（150–175 GB/s） | 46.1 | R2 |
| TP allreduce ≈128–143 次 × 260 µs | 37–43 | — |
| rocBLAS 非量化 GEMM（草稿塔 + lm_head） | 36.9 | R4 |

**prefill 2717 ms**：

| 项 | ms | 归属 |
|---|---:|---|
| 物化后 `aten::mm`（fp16，256 次，5.06 TFLOP/s） | 1893 | R5 |
| `fused_sigmoid_gating_delta_rule_gfx906_prefill` 48×14.112 | **677** | R1 |
| 其余（含 conv1d 的 149 次 host 同步） | ~146 | R6 |

### 1.6 本文档撰写时的机器状态（**必读**）

- 生产 unit `z100-vllm-host.service`：**inactive**（可以自由用卡）。
- **但 8100 端口被一个陈旧实例占着**：PID `197832`，就是 E2C 那次 run 起的实例，
  显存仍 ~93%。**任何新臂都会因端口冲突 + 显存不足而失败。**
  它属于别人的会话，**本文档不替你 kill**；请按 §2.1 第 2 条自行处置并记录。
- 栈身份（E2C 冻结值，用于核对）：`_C=1442bb12…`、`_C_stable_libtorch=a94a4d25…`、
  `_moe_C_stable_libtorch=b4974b2e…`、`_rocm_C=38bb1abb…`。

---

## 2. 硬件互斥：预检与收尾（**每个臂都要做，不许跳**）

### 2.1 开工前预检（5 项全过才许起服务）

```bash
cd /home/lyh/vllm-test/llm-engine-probe
RUNS=/mnt/game/port/perf-runs

# 1) 生产 unit 必须 inactive
systemctl is-active z100-vllm-host.service     # 期望输出：inactive（非 0 退出码也算通过）

# 2) 8100 必须没有实例（关键！）
pgrep -af "port 8100" | grep -v grep           # 期望：无输出
curl -s -m 3 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8100/health   # 期望：000（连接失败）
#    若有输出/返回 200 ⇒ 存在陈旧实例。处置（三选一，按顺序试）：
#      bash hosts/vllm-host.sh down vendor/hygon
#      kill "$(cat $RUNS/<上一个run>/serve.pid)"   # 先看 serve.pgid，必要时 kill -- -PGID
#      # 仍不退时再考虑 kill -9，但必须先确认那不是别人的在用实例

# 3) 显存必须接近空闲（陈旧实例没退干净的典型症状）
hy-smi --showmeminfo vram 2>/dev/null | head    # 期望每卡占用 <5%

# 4) 栈身份必须对（错栈 ⇒ 全部数字作废）
/home/lyh/z100-vllm/tools/dcu/use_stack.sh status | head -20

# 5) 记下环境基线（本轮所有结论都假设这四项没变）
date -u; hy-smi --showtemp 2>/dev/null | head -4; nproc; free -g | head -2
```

**锁文件**（防止两个 agent 同时上机）：

```bash
LOCK=/mnt/game/port/perf-runs/.p3.lock
if ! ( set -o noclobber; echo "ARM=$ARM_ID pid=$$ host=$(hostname) at=$(date -u +%FT%TZ)" > "$LOCK" ) 2>/dev/null; then
  echo "❌ 已被占用："; cat "$LOCK"; exit 1
fi
trap 'rm -f "$LOCK"' EXIT
```

### 2.2 收工后（**先收尾再写报告**）

```bash
bash hosts/vllm-host.sh down vendor/hygon
sleep 5; pgrep -af "port 8100" | grep -v grep || echo "OK: 8100 已释放"
hy-smi --showmeminfo vram 2>/dev/null | head   # 期望回落
```

必须归档进 run 目录（缺一项视为本臂未完成）：
`serve.log`、`serve_launch.sh`、`stack_identity.json`、`host_check.json`、`run_meta.json`、
`probe-*/http_summary.json`、`probe-*/metrics_diff.json`、`p3_score.json`。

---

## 3. 入口：怎么起、怎么探、怎么收

### 3.1 机制（一段话）

探针工具箱把配置按 **`.env` → 环境变量 → CLI** 三级合并。`hosts/vllm-host.sh` 负责把
`hosts/<类别>/<名字>.env` 模板链叠上去；**显式导出的环境变量优先级最高**。
本施工图把每个臂做成一个**可 `source` 的 env 文件**，它只覆盖 3 个键：

| 键 | 作用 | 为什么必须显式给 |
|---|---|---|
| `EXTRA_ENV` | 追加/覆盖引擎进程的 `VLLM_*` 变量 | 模板链里为空，臂的全部差异都在这里 |
| `MAX_NUM_SEQS` | 并发上限 | 模板 `tp4-prod` 给 64，**A/B 纪律要求 4** |
| `EXTRA_VLLM_ARGS` | 追加型参数 | 模板带 `--enable-prefix-caching`，**A/B 要关** |

> ⚠️ `EXTRA_VLLM_ARGS` 是**追加型**键：显式导出会**整体替换**模板的追加（`env_overlay.sh` 会打 ⚠️ 提示）。
> 这是我们要的效果（丢掉 `--enable-prefix-caching`），但看到 ⚠️ 不要以为是错误。

**建图时已实测验证过的部分**（2026-10-24，用 `a1_gdn_chunked.env` 走 §3.2 ①' 的 dry-run，exit=0）：

| 检查项 | 实测结果 |
|---|---|
| `--max-num-seqs` | **4** ✓（模板默认 64，已被 `_common.env` 覆盖） |
| `--enable-prefix-caching` | **不存在** ✓（已被显式 `EXTRA_VLLM_ARGS` 丢掉） |
| `--disable-custom-all-reduce --skip-mm-profiling` | 存在 ✓ |
| `--enforce-eager` / `--tensor-parallel-size 4` / `--max-num-batched-tokens 8192` | 存在 ✓ |
| `export VLLM_*` | 5 个基线键 + `VLLM_GFX906_GDN_CHUNKED_PREFILL=1` ✓ |

**全部 10 个臂都做过同样的 dry-run**，逐个确认「`--max-num-seqs 4` + 无 `--enable-prefix-caching`
+ 本臂新增键确实出现在 `export VLLM_*` 里」：

| 臂 | 本臂新增（dry-run 里亲眼看到） |
|---|---|
| `base_e2c` | 无（= E2C 原文 5 键） |
| `a1` | `VLLM_GFX906_GDN_CHUNKED_PREFILL` |
| `a2` | `VLLM_QWEN35_MTP_GDN_TRITON` |
| `a3` | `VLLM_DCU_FP8_DECODE_CFG` |
| `a4` | `VLLM_DCU_FP8_EMU_FASTPATH` + `…_QUANT_FASTPATH` |
| `a5` | `VLLM_S24_DYNK_{LO,HI,EMA}` |
| `a6` | `VLLM_DCU_USE_BF16_GEMV` |
| `a7` | `VLLM_GFX906_TRITON_ATTN_PREFILL_3D` + `…_3D_MIN_CTX` |
| `a8` | `VLLM_GFX906_TRITON_CAUSAL_CONV1D`（+ `…_UPDATE`） |
| `a_combo` | 上述 A1 + A2 + A4 三键 |

> **未实测**：`up-kernel` 本身（建图的环境里 8100 被一个陈旧实例占着，且显存已 93%，起不来）。
> 所以**第一次使用请务必先跑 ①' dry-run**，再去起服务。

### 3.2 一个臂的三条命令（逐字，可直接执行）

```bash
set -euo pipefail
export PROBE=/home/lyh/vllm-test/llm-engine-probe
export P3=/home/lyh/z100-vllm/tools/dcu/p3_arms
export RUNS_DIR=/mnt/game/port/perf-runs
export RUNTIME=host
ARM_FILE=$P3/a1_gdn_chunked.env            # ← 换成你要跑的臂
export TAG=$(basename "$ARM_FILE" .env)    # 产出目录名后缀

# 过 §2.1 预检 + 拿锁之后：

# ①' 先 dry-run 验证拼装（不启动、不占卡，20 秒；能省掉一整天）
( set -a; . "$ARM_FILE"; set +a
  bash hosts/vllm-host.sh dry-run vendor/hygon scenario/tp4-prod spec/dflash ) | tee /tmp/dry-$TAG.txt
#    必须看到：--max-num-seqs 4 / 没有 --enable-prefix-caching / EXTRA_ENV 里有本臂新增的那个键
#    看到两条 ⚠️「显式 … 优先于模板：模板的追加将被丢弃」是**预期的**，不是错误
#    ⚠️ 若看到 `mkdir: cannot create directory '…/runs/…': Permission denied`
#       ⇒ 你的执行环境不能写该目录，RUNS_DIR 必须显式指到 /mnt/game/port/perf-runs

# ① 起服务（kernel 档 = 带 torch profiler + 逐层 NVTX，与 E2C 同档）
( set -a; . "$ARM_FILE"; set +a
  cd "$PROBE" && bash hosts/vllm-host.sh up-kernel vendor/hygon scenario/tp4-prod spec/dflash )

# ② 找到本次 run 目录（最新一个）
RUN=$(ls -1dt "$RUNS_DIR"/*-"$TAG" | head -1); echo "RUN=$RUN"

# ③ 四组探测（顺序固定，见 §4.0）
cd "$PROBE"
bash scripts/probe_http.sh --prompt-tokens 512 --max-tokens 128 --concurrency 1 --repeat 2 \
     --endpoint chat --run-dir "$RUN/probe-512c1"
bash scripts/probe_http.sh --prompt-tokens 709 --max-tokens 128 --concurrency 1 --repeat 2 \
     --endpoint chat --run-dir "$RUN/probe-709c1"
bash scripts/probe_http.sh --prompt-tokens 512 --max-tokens 128 --concurrency 4 --repeat 1 \
     --endpoint chat --run-dir "$RUN/probe-512c4"
bash scripts/probe_http.sh --prompt-tokens 64  --max-tokens 128 --concurrency 1 --repeat 2 \
     --endpoint chat --run-dir "$RUN/probe-64c1"

# ④ 判读（唯一口径）
/home/lyh/z100-vllm/.venv/bin/python /home/lyh/z100-vllm/tools/dcu/p3_score.py \
    "$RUN" --json "$RUN/p3_score.json" --baseline /mnt/game/port/perf-runs/20261024-053333-E2C

# ⑤ 收尾（§2.2）
bash hosts/vllm-host.sh down vendor/hygon
```

**长上下文档（只有 A7 / A5 需要）**：把 `--prompt-tokens 512` 换成 `8192`，
`--run-dir` 换成 `"$RUN/probe-8k1"`，`--max-tokens 32`。
注意实 token 会是 ~11723，单次要跑几分钟。

### 3.3 一次起服务、四组探测

**不要每个 case 重启一次服务**（历史纪律已禁：重启带来的编译/缓存差异会污染 A/B）。
`up-kernel` 一次，然后 4 个 `probe_http.sh` 顺序打完再 `down`。

### 3.4 臂的逐字内容（`tools/dcu/` 不入 git，丢了照这里重建）

**基线 5 键**（= E2C `serve_launch.sh` 原文，一个都不能增减）：

```
EXTRA_ENV="VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 VLLM_DCU_FP8_W8A16=1 \
VLLM_DCU_FP16_PREFILL=1 VLLM_DCU_BF16=1"
```

**公共覆盖**（每个臂都要，等价于 `tools/dcu/p3_arms/_common.env`）：

```
MAX_NUM_SEQS=4
EXTRA_VLLM_ARGS="--disable-custom-all-reduce --skip-mm-profiling"
```

**各臂增量**（追加到基线 5 键之后）：

| 臂文件 | ID | 在基线 5 键之后**追加**的键 |
|---|---|---|
| `a1_gdn_chunked.env` | A1 | `VLLM_GFX906_GDN_CHUNKED_PREFILL=1` |
| `a2_gdn_triton.env` | A2 | `VLLM_QWEN35_MTP_GDN_TRITON=1` |
| `a3_cfg_template.env` | A3 | `VLLM_DCU_FP8_DECODE_CFG=12` |
| `a4_cpu_fastpath.env` | A4 | `VLLM_DCU_FP8_EMU_FASTPATH=1` `VLLM_DCU_FP8_QUANT_FASTPATH=1` |
| `a5_dynk.env` | A5 | `VLLM_S24_DYNK_LO=16384` `VLLM_S24_DYNK_HI=65536` `VLLM_S24_DYNK_EMA=0.1` |
| `a6_bf16gemv.env` | A6 | `VLLM_DCU_USE_BF16_GEMV=1` |
| `a7_attn3d.env` | A7 | `VLLM_GFX906_TRITON_ATTN_PREFILL_3D=1` `VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX=16384` |
| `a8_conv1d_triton.env` | A8 | `VLLM_GFX906_TRITON_CAUSAL_CONV1D=1` `VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE=1` |
| `a_combo.env` | COMBO | `VLLM_GFX906_GDN_CHUNKED_PREFILL=1` `VLLM_QWEN35_MTP_GDN_TRITON=1` `VLLM_DCU_FP8_EMU_FASTPATH=1` `VLLM_DCU_FP8_QUANT_FASTPATH=1` |
| `base_e2c.env` | P0.0 | （无，= 基线 5 键） |

重建方式（示例，A1）：

```bash
mkdir -p /home/lyh/z100-vllm/tools/dcu/p3_arms
cat > /home/lyh/z100-vllm/tools/dcu/p3_arms/a1_gdn_chunked.env <<'EOF'
ARM_ID="A1"
ARM_DESC="GDN prefill 走 FLA chunked"
MAX_NUM_SEQS=4
EXTRA_VLLM_ARGS="--disable-custom-all-reduce --skip-mm-profiling"
EXTRA_ENV="VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 VLLM_DCU_FP8_W8A16=1 VLLM_DCU_FP16_PREFILL=1 VLLM_DCU_BF16=1 VLLM_GFX906_GDN_CHUNKED_PREFILL=1"
EOF
```

---

## 4. 任务卡

> 每张卡的 **判据** 都用 §3.2 ④ 的 `p3_score.py` 输出。**baseline 固定为
> `/mnt/game/port/perf-runs/20261024-053333-E2C`**（P0.0 复现成功后可用新基线）。
> **停止线**：单臂相对基线 <3%（TTFT/step 任一）即判该臂无效，保留原配置，写清结论收工。

### 4.0 WP-P0.0 — 基线复现（**阻塞全部其它卡**）

| 项 | 内容 |
|---|---|
| 目的 | 证明"尺子没变"：用与 E2C 逐字相同的配置复现出 2731 ms / 243 ms，偏差 ≤3% |
| 臂文件 | `tools/dcu/p3_arms/base_e2c.env` |
| 依赖 | 无（第一张卡） |
| 交付物 | 1 个 run 目录（4 组 probe）+ `p3_score.json` + 一页 `BASELINE.md` |
| 通过判据 | `512c1` TTFT ∈ [2650, 2812]、step ∈ [236, 251]；`route_evidence.materialization=4`；`W8A16 engaged=True` |
| 不通过 | **停下来报告**。偏差 >3% ⇒ 栈/机器/配方已变，本轮所有历史数字不可比，需重新定基线 |
| 工时 | 0.5 天（含冷启动、编译缓存预热） |

### 4.1 WP-A1 — GDN prefill 走 FLA chunked（**收益最大**）

| 项 | 内容 |
|---|---|
| 假设 | prefill 里 `fused_..._gfx906_prefill` 48×14.112 = 677 ms 是回退路径；chunked 版单层 1.062 ms |
| 臂文件 | `a1_gdn_chunked.env`（`+VLLM_GFX906_GDN_CHUNKED_PREFILL=1`） |
| 依赖 | P0.0 |
| 交付物 | run 目录 + `REPORT.md`（§5 模板）+ **numerical parity 记录** |
| 通过判据 | `512c1` TTFT ≤ 2200 ms（预期 ~2100）；`step` 不回退 >2%；`gdn_chunked_lines > 0` |
| 硬前置 | **先跑数值对齐**：`tools/dcu/perf1_gdn_chunked_parity.py`（历史 6/6 通过，阈值 `max\|dout\|≤1e-3`、`max\|dstate\|≤6e-4`）。不通过 ⇒ **禁止启用**，不是"再调调" |
| 已知风险 | 该链历史上"曾被误编译而在层级别绕过"；本栈 triton 3.6.0 与 DTK 自编 triton 不同 ⇒ 必须重新做 parity |
| 停止线 | TTFT 改善 <3% 或 parity 不过 |
| 工时 | 0.5 天 |

### 4.2 WP-A2 — GDN decode 走 Triton（**收益第二，但反证最强**）

| 项 | 内容 |
|---|---|
| 假设 | `..._mtp_update` 48×2.040 = 97.9 ms/步 是**布局**导致的：状态按 `[slot][head][v][k]` 分配，相邻 lane 差 512 B，每次 warp 载入触 32 条 cache line（`gfx906_gdn_mtp_triton.py:1-40` 有实测：714 µs → 48 µs） |
| 臂文件 | `a2_gdn_triton.env`（`+VLLM_QWEN35_MTP_GDN_TRITON=1`） |
| 依赖 | P0.0 |
| **必读反证** | 同一个开关历史上有**两个相反结论**：GPU-bound 窗口 −97.7 ms/步（`tools/dcu/DFLASH2_OPT_RESULT.md`）；CPU-launch 主导窗口 **+9.4 ms/步**（`tools/dcu/perf1_gdn_report.md`，因 Triton 侧 host 成本 +190~215 µs/call × 48）。**本臂必须先量方向** |
| 通过判据 | `512c1` step ≤ 220 ms（预期 150–205）；`256c4`/`512c4` 不回退 >5%；`gdn_triton_lines > 0` |
| 判定技巧 | 先开 `VLLM_QWEN35_MTP_GDN_TRITON_PROBE=1` 看 host/gpu 计时；若 host ≥150 µs/call，先别整轮跑 |
| 已知风险 | kernel 文档声明"最多 4 个投机 token"，**本配方 T=8**，覆盖度需补证（→ WP-P1.4） |
| 停止线 | step 改善 <3% |
| 工时 | 0.5 天（含方向判定） |

### 4.3 WP-A3 — decode tile cfg 扫描（**最容易照抄错的一张卡**）

| 项 | 内容 |
|---|---|
| 背景 | `VLLM_DCU_FP8_DECODE_CFG` 是 **W8A8 时代**的旋钮。W8A16 路径经 `w8a16_cfg()`（`csrc/rocm/fp8_emu_gemm.cu:1786`）映射：**cfg 32–40 → 固定成员（35→cfg2）**；只有 **cfg 11–19** 保留 `(M ≤ kMsMax) ? 5 : …` 语义 |
| 现状 | 未设该变量 ⇒ auto 规则 ⇒ M≤8 取 **cfg 5**（`BM=8,BN=64`，profile 里抓到的是 `<64,8,64,1,2>`） |
| **禁止** | **不要设 `=35`**。它会把 M≤8 的 BM=8 tile 换成 BM=32（A 瓦片 75% 浪费），c=1 档很可能回退；O3 的 −7.34% 是在 **W8A8** 上量的 |
| 臂文件 | `a3_cfg_template.env`（用 `ARM_CFG=<n>` 选 cfg） |
| 依赖 | P0.0；建议排在 A1/A2 之后（它们收益更确定） |
| 扫法 | `for cfg in 11 12 13 15 19; do ARM_CFG=$cfg ...; done`，每个 cfg 只跑 `512c1` + `256c4` |
| 通过判据 | `512c1` step −3% 以上，且 `256c4` 不回退 |
| 停止线 | 全部 cfg 都 <3% ⇒ 记为"auto 规则已最优"，关掉此卡 |
| 工时 | 0.5 天（扫描，每个 cfg 只需 2 组 probe） |

### 4.4 WP-A4 — CPU 快路 O1+O2

| 项 | 内容 |
|---|---|
| 假设 | `FP8_EMU_FASTPATH`（eager 下绕过 `torch.ops` 直调 `_rocm_C`）与 `FP8_QUANT_FASTPATH` 合计省 ~17 ms **CPU**/步 |
| 臂文件 | `a4_cpu_fastpath.env` |
| 依赖 | P0.0 |
| 通过判据 | `512c1` step 不回退，**且** CPU 时间下降（看 `kernel_op_tables` 的 `Self CPU time total`）；c=4 不回退 >5% |
| 已知风险 | O1 历史在 c=4 **+6.6%**、c=16 +3.55%，c=1 持平 ⇒ **必须同时看 c=1 与 c=4 两档** |
| 停止线 | c=1 与 c=4 都不能同时不为负 ⇒ 关闭 |
| 工时 | 0.5 天 |

### 4.5 WP-A5 — S24 投机验证长度截断

| 项 | 内容 |
|---|---|
| 假设 | DTK 生产用 `S24_DYNK_LO=16384 HI=65536` 按累计 token 截断验证长度；rocm101 未带该优化 |
| 臂文件 | `a5_dynk.env` |
| 依赖 | P0.0 |
| 通过判据 | **长 ctx 档**（`8k1`）step −10%，且 `512c1` 接受长度不回退 >0.3 |
| 注意 | 只在长上下文才触发；**512/64 档预期无变化**，别因为 512 档没变化就判失败 |
| 停止线 | 长 ctx 无改善，或接受长度掉 >0.3 |
| 工时 | 0.5 天 |

### 4.6 WP-A6 — `VLLM_DCU_USE_BF16_GEMV`（**用户决策项，只测不默认开**）

| 项 | 内容 |
|---|---|
| 假设 | rocBLAS 在 M=8 的 bf16 GEMM 只有 0.163 TFLOP/s / 20.4 GB/s（本 run 实测），草稿塔+lm_head 占 36.9 ms/步；自研 GEMV 可能更好 |
| 臂文件 | `a6_bf16gemv.env` |
| 依赖 | P0.0 |
| **必须先读** | `z100-w8a16-rocm101-port-plan.md:184` 记"**不设 `VLLM_DCU_USE_BF16_GEMV`（用户要求：关闭）**"；另有记录"草稿塔 rocBLAS 比自研 triton/emu **快 2–3×**" |
| 处置 | 只做**一次**单臂；`step` −3% 以上才提交给用户决策，否则回滚并记录失败 |
| 停止线 | <3% |
| 工时 | 0.5 天 |

### 4.7 WP-A7 — 长上下文 prefill split-KV 3D

| 项 | 内容 |
|---|---|
| 臂文件 | `a7_attn3d.env`（`+VLLM_GFX906_TRITON_ATTN_PREFILL_3D=1`，`MIN_CTX=16384`） |
| 依赖 | P0.0 |
| 通过判据 | **只测 8k/32k 档**：TTFT −5%；**`512c1`/`709c1` 回退必须 ≤2%** |
| 已知风险 | 短 ctx 大亏（历史 1429-token prefill 2612→3376 ms），故必须靠 `MIN_CTX` 门挡住 |
| 工时 | 0.3 天 |

### 4.8 WP-A8 — conv1d Triton 路（**唯一带硬件风险的卡**）

| 项 | 内容 |
|---|---|
| 假设 | prefill 里 149 次 `_local_scalar_dense`（2.389 s CPU 阻塞）来自 `causal_conv1d` 的 torch 回退；Triton 路可去掉 |
| 臂文件 | `a8_conv1d_triton.env` |
| 依赖 | P0.0；**建议最后做** |
| **⚠️ 硬风险** | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py:42-52` 明写：fully-async Triton conv 在**长分块 prefill** 下会**确定性地把 GPU compute ring 卡死**（ring 100% 空转，只能 kill 进程恢复）；缓解手段是每次 conv 后整条流同步 |
| 强制约束 | **只允许 `64c1` / `512c1` 两档**；**禁止** 8k/32k 档；旁边必须挂超时监控：<br>`timeout 300 bash scripts/probe_http.sh … \|\| echo "⚠️ 超时，检查 ring"`；一旦出现超时或 `hy-smi` 显示 ring 100% 空转，**立刻** `kill -9` 并记录 |
| 通过判据 | `512c1` TTFT −5%，且无 wedge |
| 停止线 | 任何一次 wedge ⇒ **永久关闭此卡**，写进 §7 死路表 |
| 工时 | 0.3 天（高风险，不要为它排时间预算） |

### 4.9 WP-COMBO — 叠加（**只在单臂各自过门后**）

| 项 | 内容 |
|---|---|
| 组成 | `A1 + A2 + A4`（只叠**已单独过门**的臂；A3/A5/A6/A7/A8 不参与） |
| 臂文件 | `a_combo.env` |
| 通过判据 | **Must 三项全绿**：TTFT ≤2500、step ≤210、TPOT ≤45；接受长度 ≥4、率 ≥45% |
| 若未全绿 | 不要继续叠更多旋钮。转入 WP-P2 的 kernel 工作，或按 §6 判读树处理 |
| 工时 | 0.5 天 |

### 4.10 WP-P1.x — 测量修复（**不占 GPU，可并行派给别的 agent**）

| 卡 | 做什么 | 交付物 | 通过判据 |
|---|---|---|---|
| **P1.1** | 修 kernel 级 trace：`traces/*.pt.trace.json.gz` 现在全部解压为 `[]`；查 `--profiler-config` 的落盘路径与 rocm101 下 kineto/rocprof 行为 | 一个**非空** `*.pt.trace.json`，且能列出 decode 步的 kernel 时间线 | trace 里能读到 ≥10 个 kernel 事件 |
| **P1.2** | 修遥测聚合：`TUNE_REPORT.md` 的 `vram_pct_max=0` 是取错字段（真值 92–93%）；补 `sclk/mclk/throttle` | `probe/telemetry` 的字段修补 + 一页说明 | 报告里 VRAM/时钟字段非 0 且与 `hy-smi` 一致 |
| **P1.3** | 补两格微基准：① **fp16**（不是 bf16）`torch.mm` 的 M 扫描；② `fp8_emu_gemm_block_w8a16` 在 **M=709/1024** 的效率 | `tools/dcu/kernel_microbench.py` 的补丁 + 一次新 run 的微基准表 | 新表里出现 `mb_gemm_fp16_m{512,2048,8192}` 与 `mb_w8a16_m{709,1024}` |
| **P1.4** | 补 k=8 覆盖：`gfx906_gdn_mtp_triton.py` 文档只声明"最多 4 个投机 token"，本配方 T=8 ⇒ 用 `perf1_gdn_parity.py` 在 **T=8** 下补 parity | parity 报告 | T=8 下 `max\|dout\| ≤ 1e-3` |

### 4.11 WP-P2.x — kernel（**必须在 COMBO 之后**，否则基线被低估）

| 卡 | 目标 | 做法 | 通过判据 / 停止线 |
|---|---|---|---|
| **P2.1** | W8A16 decode GEMV：175 GB/s → 375+ GB/s | 三个独立子假设：**C1 网格不足**（`o`/`down` N=5120 只有 80 block / 60 CU）→ split-K 或 BN 64→32；**C2 流水太浅**（`BK=64` 每级一次 `__syncthreads`，K=17408 要 272 次）→ 3–4 级缓冲；**C3 流式 GEMV 在 M≥5 被 44 KiB LDS 门挡住** → K 分块流式化 | 微基准 `N=5120,K∈{6144,17408},M=8` ≥300 GB/s；e2e step −3%。**停止线**：微基准 <10% 或 e2e <3% |
| **P2.2** | 并发档 M∈[9,32] 的重量 tile | `709c2`(M=16) step 382 ms vs `512c1`(M=8) 243 ms，**权重流量相同**；`GEMV_MAX_M` 在 `gfx906_fp8.py:83` 被 clamp 到 [0,8]，M>8 必落 `MT64x32x8` 类 tile。⇒ 补一个"权重只读一次"的 T 行 tile | `256c4`/`709c2`/`512c4` step −20%。停止线 −5% |
| **P2.3** | prefill fp16 GEMM 5.06 → 10 TFLOP/s | `D1` 打印 `torch.mm` 实际后端 + `ROCBLAS_TENSILE_LIBPATH` / hipBLASLt 可用性；`D2` 单 role 扫 `VLLM_DCU_FP16_PREFILL_MIN_M`；`D3` 用 P1.3 的新微基准定界 | TTFT −10%。停止线：单点 <10% 即接受现状 |

### 4.12 WP-P3.1 — CUDA graph 路线 A（**COMBO 之后才做**）

| 项 | 内容 |
|---|---|
| 依据 | DTK 实测 decode **127.5→118.4 ms（−7.1%）**、TTFT 持平、greedy 8/8 逐 token 一致；前置修复已在树里（`vllm::fp8_emu_mm_block_w8a16_auto` + LUT 独立 op） |
| 做法 | 去掉 `--enforce-eager`，加 `--compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16]}'`。**rocm101 栈上从未跑过 compile/graph 臂** |
| 通过判据 | decode ≤ −5%；TTFT 回退 ≤2%；c=4 不回退；greedy 8/8 一致 |
| 已知代价 | ready **~306 s**、宿主 +3 GiB、**KV −2.5%**、**图模式跨重启数值不可复现**（逐位一致率 2.4%）⇒ 图只能用于延迟/吞吐口径，**SLA 与质量必须留 eager** |
| **禁止** | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` —— 已判死：`tok/step=1.00`、logprobs NaN/Inf、c=4 **KERNEL VMFault** |

---

## 5. 交付物模板（每个臂一份 `REPORT.md`，字段名不许改）

```markdown
# ARM <ARM_ID> — <ARM_DESC>

## 0. 结论（一句话）
<通过/不通过 + 关键数字>

## 1. 环境
- run 目录：/mnt/game/port/perf-runs/<ts>-<tag>
- 起服命令：见 serve_launch.sh（原样贴出 EXTRA_ENV 那几行）
- 栈身份：p3_score.py 输出的 stack_identity 段（torch/triton/四个 .so md5）
- 时钟/温度/功耗（开工时 + 收工时各一次）：<hy-smi 原始输出关键行>

## 2. 路由证据（证明开关真的生效了）
- W8A16 engaged        : <grep 结果原文>
- materialization hits : <数字，期望 4>
- gdn_chunked / gdn_triton / conv1d lines : <数字>

## 3. 指标（p3_score.py 原样粘贴，不要手改）
<表格>

## 4. 与基线对比
| case | TTFT Δ | step Δ | 判定 |
...

## 5. 数值/质量门（仅数值相关臂需要）
<parity 工具与阈值、max|dout|、max|dstate|、负对照结果>
<greedy 逐 token 对拍：n/8 一致>

## 6. 判定
- 通过判据：<施工图 §4 对应条目原文>
- 实测：<数字>
- 结论：PASS / FAIL / 需用户决策
- 若不通过：为什么（机理，不是"没效果"）

## 7. 副作用与残留
- 显存、KV token 数、ready 时间、宿主内存
- 是否引入新风险（wedge / 数值漂移 / 长度不可复现）

## 8. 回滚
<一条命令或一个 env 键>
```

---

## 6. 判读树（做完每张卡照着走，不要自己发挥）

```
P0.0 基线复现
  ├─ 偏差 >3% ──────────────► 停：报告"栈/配方已变"，重新定基线，本轮作废
  └─ 通过
       ├─ A1 TTFT ≤2200 且 parity 过 ──► 保留
       │    └─ 否则 ─────────────────► 关闭 A1，记录 parity 结论
       ├─ A2 step ≤220 且 c≥2 不回退 ──► 保留
       │    └─ 否则（含 +9.4ms 那种反例）► 关闭 A2，把方向写进 §7
       ├─ A4 c=1/c=4 皆不劣 ─────────► 保留
       ├─ A1+A2+A4 = COMBO ──────────► 跑 Must 三项
       │    ├─ 全绿 ─────────────────► 进 P3.1（graph），再进 P4
       │    └─ 未全绿 ───────────────► 进 P2.1（GEMV 带宽）→ P2.2（并发 tile）
       └─ A3/A5/A6/A7/A8 各自按 §4 停止线独立判定，不进 COMBO
```

**汇总板**（每个 agent 领一张卡后在这里改状态，避免重复劳动）：

| WP | 依赖 | 占 GPU | 工期 | 状态 | 结论 |
|---|---|---:|---:|---|---|
| P0.0 | — | 是 | 0.5d | ☑ | PASS：512c1 2777.9/244.7（+1.7%/+0.6% ≤3%）；mat=4、W8A16=True（run 20261024-070027） |
| A1 | P0.0 | 是 | 0.5d | ⚠ | 边缘：parity 6/6 过、TTFT −20.1%（2218.6，门 ≤2200 差 0.7% 稳定超）；**需用户拍板**（Must 依赖它） |
| A2 | P0.0 | 是 | 0.5d | ☑ | PASS：方向测定→step 165.34（−32.4% ≤220）；c4 改善；gdn_triton=1（run 20261024-071951） |
| A4 | P0.0 | 是 | 0.5d | ☑ | PASS：c1 step −0.4%、c4 +2.7% 门内、Self CPU −4.8%（15.61→14.87s/rank 同流程对照） |
| A3 | P0.0 | 是 | 0.5d | ✗ | 关闭：cfg 11/12/13/15/19 全部 <3%，auto 规则已最优（A3_SCAN_REPORT.md） |
| A5 | P0.0 | 是(长ctx) | 0.5d | ✗ | 关闭：8k1 −1.3%（ctx 11755<LO=16384 机制惰性，判据与臂参数矛盾）；**附加发现**：12k1(ctx>LO) step −23%，调 LO 需用户拍板 |
| A6 | P0.0 | 是 | 0.5d | ✗ | 关闭：本栈 cuteDSL 不可用→ll_bf16_gemm disabled，臂 no-op（c1 −0.8% <3%） |
| A7 | P0.0 | 是(长ctx) | 0.3d | ☑ | PASS：12k1 TTFT −5.3%（MIN_CTX=16384 下唯一可达档）；短档 0 回退；8k1 结构性 no-op |
| A8 | P0.0 | 是 | 0.3d | ✗ | **永久关闭**：Triton conv1d 首用即编译崩溃（`USE_PAD_SLOT is not defined`，树内 bug 非 wedge）；64c1 500×2 |
| COMBO | A1,A2,A4 | 是 | 0.5d | ☑ | **Must 三项全绿**（A1+A2+A4）：2214.75/167.42/31.16、accLen 5.48、acc% 64.0；收益可加无负交互 |
| P1.1 | — | 否 | 1d | ⚠ | 根因定位：kineto 导出层丢 device 事件（裸 torch.profiler 复现，vLLM 无关）；表路径 Self CUDA 可用；修复归 torch 构建侧 |
| P1.2 | — | 否 | 0.5d | ☑ | PASS：telemetry.py 补 sclk/mclk（已验证）；vram_pct_max=0 的生成器系会话私有脚本、两仓无源，raw jsonl 数据完好 |
| P1.3 | — | 是(空闲卡) | 0.5d | ☑ | PASS：kernel_microbench_p13.py 新格齐（fp16 mm m8192=13.0 TFLOP/s vs 活体 5.06 ⇒ P2.3 抓手）；w8a16 m709=4.7 |
| P1.4 | — | 否 | 0.5d | ☑ | PASS：--t-max 补丁，T=1..8 全覆盖 max\|dout\|≤3.05e-5（余量 33×）；A2 的 T=8 风险正式关闭 |
| P2.1 | COMBO | 是 | 2d | ☐ | （COMBO 已全绿，按 §4.11 P2.x 仅在 COMBO 未绿时进入；本轮未触发） |
| P2.2 | COMBO | 是 | 2d | ☐ | （同上） |
| P2.3 | COMBO | 是 | 1d | ☐ | （同上；P1.3 已给出 D1/D3 定界抓手） |
| P3.1 | COMBO | 是 | 1.5d | ☑ | PASS 全门：step 139.38（−16.7%）、TTFT −1.4%、c4 不回退、greedy 8/8 一致；KV −3.5%、ready 281s；**需用户拍板**（SLA 留 eager） |

> **GPU 是串行队列**：`占 GPU=是` 的卡不能同时跑。可以并行的是"写 env 文件、分析产物、P1.1/P1.2/P1.4"。

### 6.1 本轮执行补充（2026-10-24，P0.0→P3.1 全部跑完后回填）

1. **§3.2 逐字命令有一个环境陷阱（首次执行即中招）**：探针工具箱 `.env:24` 的
   `SRC_DIR=/mnt/game/port/gfx906-vllm-rocm101` 指向**非 -perf 树**。起服前必须显式
   `export SRC_DIR=/mnt/game/port/gfx906-vllm-rocm101-perf`，否则 serve_launch.sh 的
   `cd`/`PYTHONPATH` 全错、栈身份不匹配（§2.1 第 4 条判定作废）。`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`
   由 kernel 档自动发射，无需手动。run 目录建议带 `-- --tag <arm名>`（§3.2 ② 的 `*-$TAG` 才能命中）。
2. **P3.1 必须用普通 `up` 档**：kernel 档（up-kernel）自动追加 `--enforce-eager`
   （start_vllm.sh:228，NVTX 与图互斥）；`--compilation-config` 的 JSON 必须无空格
   （`$EXTRA_VLLM_ARGS` 按空白切分）。
3. **A1 的 2200 门与 Must 的关系**：A1 实测 2214–2219（超门 0.7% 且稳定），但 COMBO 全绿
   依赖 A1 的 TTFT 收益（A2/A4 不动 TTFT）——保留/关闭 A1 交用户拍板。
4. **A5/A7 的"8k 档"措辞错位**：两卡的 MIN 阈值（LO=16384 / MIN_CTX=16384）都高于 8k 探测
   的实际 token（~11755），8k 档按机制必为 no-op；触发区实测见各自 REPORT（A5 12k1 step
   −23%，A7 12k1 TTFT −5.3%）。32k nominal 探测会超 max_model_len=32768，不可用。
5. **A8 的死因是树内代码 bug**（`causal_conv1d.py` 第二内核变体缺 `USE_PAD_SLOT: tl.constexpr`），
   不是预告的 ring-wedge；timeout 护栏对编译崩溃无效，500 监控才是该臂的正确护栏。
6. **CPU/长档对照协议**（本轮沉淀，跨臂可比）：`POST /start_profile` → 单次 512c1 probe →
   `POST /stop_profile` 读 `traces/rank*.profiler_out.txt` 的 Self CPU total（stop 导出 ~20s，
   curl 需放宽超时）；长档对照基线 run：`20261024-074625-base_e2c_longctx`（8k1+12k1）。
7. P1.2/P1.3/P1.4 的补丁位置：`llm-engine-probe/probe/telemetry.py`（备份 .p12bak）、
   本仓 `tools/dcu/kernel_microbench_p13.py`（生产树零改动）、本仓 `tools/dcu/perf1_gdn_parity.py`
   （+`--t-max`）。详见 `/mnt/game/port/perf-runs/20261024-p1x-findings/P1_FINDINGS.md`。
8. **验收（P4）**：eager 口径 COMBO（A1+A2+A4 五键）为 Must 全绿配置
   （run `20261024-083026-a_combo`）；P3.1 图模式四门全过（run `20261024-083554-p31_graph`）
   但按 §8.3 属"需用户决策"，SLA/质量留 eager。**未上线任何变更**——全部臂仅以 env 键存在，
   生产配方回滚 = 不 source 对应 arm 文件。

---

## 7. 明确不做（**任何人不得重新提议**）

| 项 | 结论 |
|---|---|
| custom all-reduce（`VLLM_DCU_CUSTOM_AR`） | 判死：小消息 +21.7%~+53.2% 变慢，e2e c1 TPOT +1.9~+6.3% |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | 判死：NaN/Inf、accept 0、c=4 KERNEL VMFault |
| 草稿塔改 fp16 | 判死：4/4 worker kv dtype 冲突 + fp16 溢出 MLP 激活（`config/speculative.py:429-435`） |
| 草稿塔 weight-only 量化 | 阻塞：草稿塔必须 bf16 激活，gfx906 fp8 内核无 bf16 激活 datapath |
| spec k 值扫描 | 已证 k=7 最优 |
| GDN C++ kernel 微调（`__launch_bounds__`/block） | 上限 0.73–1.03%，要动就走布局（R1） |
| O5 量化融合（`FP8_FUSE_QUANT`） | kernel 逐 bit 正确但 e2e **+54.0%** |
| O4 stream 路由（`VLLM_DCU_FP8_STREAM_MAX_M`） | parity 8 条 5 条分歧 |
| fp8 prefill tile/kFBk 调优 | cfg10 已近上限，无 ≥5% 候选 |
| 短算子融合 | 无单一 >5% 项 |
| `VLLM_DCU_FP8_CVT` | 死旋钮（全仓无消费者） |
| `VLLM_DCU_FP8_DECODE_CFG=35` | **W8A8 值**，会换掉 M≤8 的 BM=8 tile（见 A3） |
| `NCCL_SHM_DISABLE=1` / `NCCL_P2P_DISABLE=1` | 慢 7.3× / 无益 |
| 4 卡时钟不对称 | 本 run 4 rank 差 <1%，不构成瓶颈 |

---

## 8. 附录

### 8.1 路径索引

| 用途 | 路径 |
|---|---|
| 探针工具箱 | `/home/lyh/vllm-test/llm-engine-probe` |
| 起服模板链 | `vendor/hygon` → `scenario/tp4-prod` → `spec/dflash` |
| run 产物根 | `/mnt/game/port/perf-runs/<YYYYmmdd-HHMMSS>-<tag>/` |
| 臂定义 | `/home/lyh/z100-vllm/tools/dcu/p3_arms/*.env` |
| 判读器 | `/home/lyh/z100-vllm/tools/dcu/p3_score.py` |
| 栈切换 | `/home/lyh/z100-vllm/tools/dcu/use_stack.sh {status\|dtk\|rocm101}` |
| GDN parity | `tools/dcu/perf1_gdn_chunked_parity.py`、`tools/dcu/perf1_gdn_parity.py` |
| 历史谱系 | `tools/dcu/DFLASH2_OPT_RESULT.md`（GDN 双开关）、`tools/dcu/perf1_gdn_report.md`（A-B-A 反证）、`docs/design/z100-rocm101-perf-p3-optimization-plan.md`（判读） |

### 8.2 复算命令

```bash
# 任取一个 run 判读（唯一口径）
/home/lyh/z100-vllm/.venv/bin/python /home/lyh/z100-vllm/tools/dcu/p3_score.py \
  /mnt/game/port/perf-runs/<run> --json /tmp/x.json

# 两个 run 对比（打印 Δ）
/home/lyh/z100-vllm/.venv/bin/python /home/lyh/z100-vllm/tools/dcu/p3_score.py \
  /mnt/game/port/perf-runs/<new> --baseline /mnt/game/port/perf-runs/<old>

# 路由证据（快速自查）
grep -c "first materialization" <run>/serve.log     # 期望 4
grep "W8A16 engaged" <run>/serve.log
grep -iE "chunked|gdn_mtp_triton|CAUSAL_CONV1D" <run>/serve.log | head

# 栈身份
/home/lyh/z100-vllm/tools/dcu/use_stack.sh status

# 臂文件自检（source 不报错、三个键齐全）
for f in /home/lyh/z100-vllm/tools/dcu/p3_arms/a*.env; do
  ( set -a; . "$f"; set +a; printf '%-24s %-6s seqs=%s env_len=%s\n' \
      "$(basename $f)" "$ARM_ID" "$MAX_NUM_SEQS" "${#EXTRA_ENV}" )
done
```

### 8.3 交接约定

- 每张卡**必须**留下：run 目录 + `REPORT.md`（§5 模板）+ 汇总板状态更新。
- **不要**修改 `/mnt/game/port/gfx906-vllm-rocm101-perf`（生产树）里的代码；
  需要改代码的卡（P2.x / P1.x）走 `dtk` 分支 + `use_stack.sh`，并在 `REPORT.md` 里写 md5 变化。
- 结论为"需用户决策"的卡（A6、P3.1 的图模式）**不要自行上线**，交给用户拍板。
