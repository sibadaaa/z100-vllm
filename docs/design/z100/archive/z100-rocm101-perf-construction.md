# Z100 ROCm 10.1 性能调优施工图（压缩版）

> **依据**：[`docs/result/z100-rocm101-perf-integration-investigation.md`](../../../result/z100-rocm101-perf-integration-investigation.md)
> **目标树**：`$DATA_ROOT/port/gfx906-vllm-rocm101`（T2，ROCm 10.1 + torch 2.16）
> **参照树**：`$WORKDIR`（T1，DTK 26.04 + torch 2.11，分支 `dtk`）
> **本版约束（用户指定）**：
> 1. **不重跑基线**，直接信任 13:57 / 15:01 / 16:47 三份现有数据；
> 2. 不再做 A0–A8 九臂 × 多次启动；压缩为 **3 次必跑启动 + 1 次可选启动**，
>    每次启动内**一次跑完多组探测**（512/709/8k/c4）。
> **编写日期**：2026-10-24

---

## 0. 三十秒看懂

现状不是“w8a16 没开”，而是：T2 是 S23 + 9 个手工文件 + 2 个新文件的拼装快照，
S23 之后的调优没有合并；15:01 的 1595 ms 是 DTK；真实 rocm101 的 512-token prefill
是 **2393.7 ms**、decode **238–265 ms/step**。

本版只做 **3 次必跑启动 + 1 次可选启动**：

| 启动 | 树 | 关键开关 | 目的 | 与谁比 |
|---|---|---|---|---|
| **E1** | T2 当前 | `W8A16=0`，`FP16_PREFILL=1` | W8A16 是否拖慢 decode；验证 Plan B | 信任基线 16:47 / 13:57 |
| **E2** | T2 merged | `W8A16=1`，`FP16_PREFILL=1` | 代码合并是否改善；同时做 G1–G4 | E1 / 信任基线 |
| **E3** | T2 merged | `W8A16=0`，`FP16_PREFILL=1` | 最可能的生产候选 | E2 / E1 |
| **E4（可选）** | T2 merged | `W8A16=1` + O2 + attention | 只在 E2/E3 decode 仍 >210 ms 时做 | E2/E3 |

每次启动只起一次服务，然后**顺序跑 4 组探测**（512/128 c1、709/128 c1、8k/32 c1、512/128 c4），
不再为每个负载重启。预计 3 次必跑启动合计 **1.5–2.5 小时**（含模型加载），而不是原来的 9 臂 × 3 次。

---

## 1. 信任的基线（不再重跑）

### 1.1 基线表

| 编号 | 运行 | 栈 | prompt/out | TTFT / prefill | decode step p50 | TPOT p50 | 接受率 |
|---|---|---|---|---|---|---|---|
| **B1** | `20261023-135729` | rocm101，无 w8a16 | 709 / 128 | 4937 ms / 4930 ms | **192.2 ms** | 37.1 ms | 57.5% |
| **B2** | `20261023-150120` | **DTK**，w8a16 + FP16 | 709 / 128 | **1594.9 ms** / 1594 ms | **200.5 ms** | 39.9 ms | 50.6% |
| **B3** | `20261023-164704` | rocm101，w8a16 + FP16 | 手动 512 | **2393.7 ms** | **238–265 ms** | — | 15–43% |
| **B4** | `20261023-164058` | rocm101，w8a16 + FP16 | 手动 193 | 2032.8 ms | 237–257 ms | — | ~35% |

使用规则：

- **B1** 是 rocm101 无 w8a16 的 decode 底线（192 ms），用来判断“W8A16 是否拖慢”。
- **B2** 是 DTK 的性能参照（1594 ms / 200.5 ms），只作目标，不与 rocm101 直接做显著性比较。
- **B3/B4** 是现有 rocm101 “功能已生效”的锚点；B3 是手动请求、带 tool parser，接受率不能当基线，
  只用于确认“E1/E2/E3 的 decode 与现状同量级”。
- **不再启动 DTK 服务、不再重跑 B1–B4**；需要引用时直接引用上表与 run 目录。

### 1.2 目标

| 指标 | Must | Target | Stretch |
|---|---|---|---|
| TTFT p50，512 prompt | ≤ 2.5 s | **≤ 1.8 s** | ≤ 1.1 s |
| TTFT p50，709 prompt | ≤ 2.5 s | **≤ 1.8 s** | ≤ 1.6 s |
| decode step p50 | ≤ 210 ms | **≤ 200 ms** | ≤ 192 ms |
| TPOT p50 | ≤ 45 ms | **≤ 40 ms** | ≤ 37 ms |
| 接受长度 | ≥ 4.0 | **≥ 4.5** | ≥ 5.0 |
| 接受率 | ≥ 45% | **≥ 50%** | ≥ 55% |

目标不是“必须超过 DTK”，而是**至少回到 rocm101 无 w8a16 的 192 ms decode 底线，
同时保留 FP16 物化的 TTFT 收益**；能到 DTK 的 200 ms/1.6 s 更好。

### 1.3 压缩后的统计口径

因为不再做“每臂 3 次独立启动”，改用：

- **每个负载 2 次重复**（c1）；c4 与 8k 各 1 次；
- **每个负载先 warmup 2 次**，不计入统计；
- 同一启动内按固定顺序跑 4 组探测，所有臂顺序一致；
- 两个臂的 p50 差异 <5% 且区间重叠 → 判“不可判定”；
- 只有“不可判定”或结果与信任基线矛盾时，才**只重跑该臂一次**，不整套重跑。

代价是统计置信度下降；因此本版把“方向性结论”和“生产线”分开：
方向性结论（W8A16 是否拖慢、merge 是否改善）允许用一次启动；
生产候选（E3）在 P4 再单独跑一次完整验收。

---

## 2. 阶段与工作包

| 阶段 | WP | 内容 | 工期 | 退出条件 |
|---|---|---|---|---|
| **P0** | WP0.1 | 基线冻结（只归档，不跑） | 0.5 天 | `baseline-freeze.md` 完成；run_meta 身份核对通过 |
| **P0** | WP0.2 | T2 入 git + 栈身份门禁 | 0.5 天 | snapshot commit；`stack_identity` exit 0 |
| **P0** | WP0.3 | **E1**：T2 + `W8A16=0`，一次启动多探测 | 0.5 小时 | E1 的 4 组 summary 完整 |
| **P1** | WP1.1 | merge `dtk/dtk` + 冲突表 | 0.5 天 | `git diff dtk/dtk` 只剩 rocm101 专属 |
| **P1** | WP1.2 | 全量重编四个扩展 | 0.5–1 天 | 4 个 `.so` rc=0，op 注册通过 |
| **P1** | WP1.3 | **E2**：merged + `W8A16=1`，G1–G4 + 多探测 | 0.5 小时 | G1–G4 绿；E2 的 4 组 summary 完整 |
| **P2** | WP2.1 | **E3**：merged + `W8A16=0`，一次启动多探测 | 0.5 小时 | E3 的 4 组 summary 完整 |
| **P2** | WP2.2 | 判读 + 可选 **E4**（O2+attention 合并臂） | 0.5 天 | 一页 verdict；E4 仅在触发时跑 |
| **P3** | WP3.1–3.4 | 定点优化（W8A16 / rocBLAS / attention / 接受率） | 1–2 天 | 至少一项 e2e ≥5%，或证明栈上限 |
| **P4** | WP4.1–4.4 | 最终验收、稳定性、回滚、生产 `.env` | 0.5–1 天 | Must 全绿，回滚通过 |

**总工期：3.5–6 天**（原版 5.5–10 天）。

---

## 3. 通用纪律

### 3.1 本版禁止

- **禁止**重跑 DTK 基线（B2）或 rocm101 无 w8a16 基线（B1）；直接引用 §1.1。
- **禁止**再做 A0–A8 九臂矩阵；只允许 E1/E2/E3 + 可选 E4。
- **禁止**为每个负载重启一次服务；一次启动内跑完 4 组探测。
- **禁止**再把文件从 T1 `cp` 到 T2 作为集成；必须 merge/cherry-pick。
- **禁止**一次改两个变量；E4 是唯一允许的“组合臂”，且只在触发条件下做。
- **禁止**在未跑 `stack_identity` 的情况下把 DTK 数字写成 rocm101。

### 3.2 本版必须

- 每个臂保留 `stack_identity.json`、`serve.log`、4 组 `http_summary.json`。
- 每个臂的 4 组探测顺序固定：**512 c1 → 709 c1 → 8k c1 → 512 c4**。
- 每组探测前 warmup 2 次；c1 每组 repeat 2；c4/8k repeat 1。
- 调 prefill 的 A/B 统一**关闭 prefix caching**；E3 最终生产复验再打开。
- 所有臂使用同一个 `MAX_NUM_SEQS=4`、TP=4、maxlen=32768、MBT=8192、`--enforce-eager`。

### 3.3 环境文件

```bash
# .env.rocm101（E1，T2 基线树）
RUNTIME=host
HOST_PYTHON=$HOME/rocm101-host/venv/bin/python
HOST_PRELUDE_FILES="$HOME/rocm101-host/env-host.sh"
SRC_DIR=$DATA_ROOT/port/gfx906-vllm-rocm101
TRITON_CACHE_DIR=$DATA_ROOT/triton-z100-rocm101/cache/jit
TRITON_INSTALL_DIR=$DATA_ROOT/triton-z100-rocm101/install
VLLM_CACHE_ROOT=$DATA_ROOT/vllm-cache-rocm10
PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
MAX_NUM_SEQS=4
PROBE_LEVEL=steps

# .env.rocm101.merged（E2/E3/E4；只改 SRC_DIR）
SRC_DIR=$DATA_ROOT/port/gfx906-vllm-rocm101-perf
```

不再创建 `.env.dtk`；DTK 基线直接引用 B2，不启动。

---

## 4. P0：冻结基线 + 入 git + E1

### 4.1 WP0.1 — 基线冻结（只归档，不跑）

新增 `docs/result/z100-rocm101-perf-baseline-freeze.md`，内容就是 §1.1 的表 +
每份 run 的 `run_meta.json` 片段 + 四个 `.so` md5。**不需要启动任何服务**。

只做只读核对：

```bash
for r in 20261023-135729-serve 20261023-150120-serve \
         20261023-164058-serve 20261023-164704-serve; do
  echo "== $r"
  $WORKDIR/.venv/bin/python -c "import json;m=json.load(open('$HOME/vllm-test/llm-engine-probe/runs/$r/run_meta.json'));print(m.get('host'))"
done
```

确认 B2 的 `torch_version` 是 2.11.0、B1/B3/B4 是 2.16.0；把结果写进冻结文档。
**退出条件**：冻结文档完成，B1–B4 的栈身份清楚；不产生新的性能数字。

### 4.2 WP0.2 — T2 入 git + 栈身份门禁

```bash
# 0) 备份当前四个 .so（回滚用）
cp -a $DATA_ROOT/port/gfx906-vllm-rocm101/installed-so \
      $DATA_ROOT/port/gfx906-vllm-rocm101/installed-so.baseline-20261023

# 1) 从 S23 建 worktree
cd $WORKDIR
git worktree add -b rocm101-perf $DATA_ROOT/port/gfx906-vllm-rocm101-perf f7547bac0

# 2) 把 T2 当前源码覆盖进去（排除构建产物/临时目录）
cd $DATA_ROOT/port/gfx906-vllm-rocm101-perf
rsync -a \
  --exclude='.git' --exclude='*.so' --exclude='installed-so*' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.mypy_cache' \
  --exclude='.ruff_cache' --exclude='.bench' --exclude='runs' \
  --exclude='.s[0-9]*' --exclude='backups' --exclude='vllm/_version.py' \
  --exclude='build*' --exclude='.rocprofv3' \
  $DATA_ROOT/port/gfx906-vllm-rocm101/ ./

# 3) 预期 11 个文件（9 改 + 2 新）
git status --short
git add -A
git commit -m "snapshot(z100): rocm101 tree as deployed 2026-10-23"
```

栈身份门禁 `tools/dcu/stack_identity.py` 沿用原要求：校验 `vllm.__file__`、torch/triton 版本、
`sys.executable`、四个 `.so` md5。**E1 之前必须 exit 0。**

### 4.3 WP0.3 — E1：T2 + `W8A16=0`（一次启动，多探测）

E1 的目的只有一个：**验证“W8A16 是否拖慢 decode”**。它不改代码，只改环境。

```bash
cd $HOME/vllm-test/llm-engine-probe
export RUNS_DIR=$DATA_ROOT/port/perf-runs
RUN=$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-E1
mkdir -p "$RUN"

ENV_FILE=$DATA_ROOT/port/perf-env/.env.rocm101 \
EXTRA_ENV="VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 \
VLLM_DCU_FP8_W8A16=0 VLLM_DCU_FP16_PREFILL=1 VLLM_DCU_BF16=1" \
EXTRA_VLLM_ARGS="--disable-custom-all-reduce --skip-mm-profiling" \
  bash scripts/start_vllm.sh --level steps --run-dir "$RUN"

# 见 §6.2 的 4 组探测；跑完后 stop 一次
bash scripts/stop_vllm.sh --run-dir "$RUN"
```

> `EXTRA_VLLM_ARGS` 去掉了 `--enable-prefix-caching`：A/B 阶段统一关闭前缀缓存，
> 保证 4 组探测不互相命中。E3 的最终生产复验再打开。

**退出条件**：E1 的 4 组 summary 完整；E1 decode 与 B3 的 238–265 ms 可直接比较。
如果 E1 decode 回到 ~192 ms，就已经可以确定 Plan B；不需要额外跑“W8A16=1”的 E0。

---

## 5. P1：代码收敛 + E2

### 5.1 WP1.1 — merge `dtk/dtk`

```bash
cd $DATA_ROOT/port/gfx906-vllm-rocm101-perf

git remote add dtk $WORKDIR 2>/dev/null || \
  git remote set-url dtk $WORKDIR
git fetch dtk dtk
git merge --no-ff dtk/dtk -m "merge(z100): bring the dtk tuning lineage onto the rocm101 tree"
```

### 5.2 冲突解决表（只列必须人工决策的文件）

| 文件 | 取值 | 额外动作 |
|---|---|---|
| `csrc/rocm/fp8_emu_gemm.cu` | `dtk/dtk` | 删掉 fused quant tile 里未使用的 `K4`/`w4`（rocm101 `-Werror`），等价 `4638418a3` |
| `csrc/rocm/fp8_emu_gemm.h`、`torch_bindings.cpp`、`fp8_dequant.cuh` | `dtk/dtk` | 无 |
| `gfx906_fp8.py`、`gfx906_fp16_prefill.py` | `dtk/dtk` | 无 |
| `BlockScaledMMLinearKernel.py`、`fp8_utils.py` | `dtk/dtk` | 恢复 S23 O2/O5 host 路径 |
| `triton_unified_attention.py` | `dtk/dtk` | 恢复 S23 T4 默认值与 S24 decode overrides |
| `scheduler.py` | `dtk/dtk` | 恢复 S24 DYNK（默认仍关） |
| `config/model.py`、`config/vllm.py` | `dtk/dtk` | bf16 coercion / bf16 草稿 KV |
| `platforms/rocm.py`、`hygon_dtk.py` | `dtk/dtk` | 保留 rocm101 探测差异 |
| `envs.py` | `dtk/dtk` | 补齐 S23 O2 / S24 DYNK 旋钮 |
| `vllm/v1/engine/core_client.py` | **保留 T2** | rocm101 InprocClient |
| `tools/dcu/env-rocm101.sh`、`platforms/hygon_dp4x.py` | **保留 T2** | rocm101 专属 |
| `pyproject.toml`、`requirements/` | **保留 T2** | torch/triton pin |

**diff 门**：

```bash
git diff --stat dtk/dtk -- \
  csrc/rocm vllm/model_executor/kernels/linear/scaled_mm \
  vllm/v1/attention vllm/v1/core vllm/config vllm/platforms

git grep -n 'VLLM_S24_DYNK_LO' -- vllm/envs.py vllm/v1/core/sched/scheduler.py
git grep -n 'GFX906_DECODE_BLOCK_M_OVERRIDE' -- vllm/v1/attention/ops/triton_unified_attention.py
git grep -n 'gfx906_coerce_model_dtype' -- vllm/platforms/rocm.py vllm/config/model.py
git grep -n 'fp8_emu_gemm_block_w8a16' -- csrc/rocm
```

### 5.3 WP1.2 — 全量重编四个扩展

```bash
docker run --rm --name z100-rocm101-perf-build \
  -v $DATA_ROOT:$DATA_ROOT -v $WORKDIR:/src-dtk:ro \
  --shm-size 8g z100-rocm101-vllm:20261023 bash -c '
set -e
export REPO_DIR=$DATA_ROOT/port/gfx906-vllm-rocm101-perf
export BUILD_DIR=$DATA_ROOT/port/build-rocm101-perf
export LOG=$DATA_ROOT/port/build-rocm101-perf.log
. $DATA_ROOT/port/setenv-rocm101.sh
bash $DATA_ROOT/port/configure-rocm101.sh
MAX_JOBS=2 bash $DATA_ROOT/port/build-rocm101.sh \
  _C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C
bash $DATA_ROOT/port/install-rocm101.sh \
  _C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C
'
```

安装后把四个 `.so` 同步到 `installed-so` 并 relink（若 `install-rocm101.sh` 已写入
`vllm/`，按下面逻辑处理）：

```bash
B=$DATA_ROOT/port/gfx906-vllm-rocm101-perf
mkdir -p $B/installed-so
for m in _C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C; do
  [ -L "$B/vllm/$m.abi3.so" ] || cp -f "$B/vllm/$m.abi3.so" "$B/installed-so/$m.abi3.so"
  ln -sfn "$B/installed-so/$m.abi3.so" "$B/vllm/$m.abi3.so"
done
md5sum $B/installed-so/*.so | tee $DATA_ROOT/port/perf-env/so.md5
strings $B/installed-so/_rocm_C.abi3.so | grep -c w8a16   # >0
```

**门禁**：构建 rc=0；4 个 `.so` 时间戳为本次；w8a16 符号 >0；`check_rocm_c_ops.py` exit 0；
`stack_identity` 的 `so_md5` 更新。

### 5.4 WP1.3 — E2：merged + `W8A16=1`（G1–G4 + 多探测共用一次启动）

E2 同时承担“功能门”和“代码合并效果”两个职责，**只启动一次**：

```bash
cd $HOME/vllm-test/llm-engine-probe
export RUNS_DIR=$DATA_ROOT/port/perf-runs
RUN=$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-E2
mkdir -p "$RUN"

ENV_FILE=$DATA_ROOT/port/perf-env/.env.rocm101.merged \
EXTRA_ENV="VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 \
VLLM_DCU_FP8_W8A16=1 VLLM_DCU_FP16_PREFILL=1 VLLM_DCU_BF16=1" \
EXTRA_VLLM_ARGS="--disable-custom-all-reduce --skip-mm-profiling" \
  bash scripts/start_vllm.sh --level steps --run-dir "$RUN"
```

**G1 在启动前跑**（`check_rocm_c_ops.py` 会初始化 GPU，避免与服务抢显存）；启动后只做 G2–G4：

```bash
# 启动前（见 §5.3）
( cd $DATA_ROOT/port/gfx906-vllm-rocm101-perf && \
  source $DATA_ROOT/port/setenv-rocm101.sh && \
  python3 tools/dcu/check_rocm_c_ops.py )

# 启动后
grep -E 'W8A16 engaged=True|first materialization route engaged' "$RUN/serve.log"
curl -s http://127.0.0.1:8100/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  | $WORKDIR/.venv/bin/python -m json.tool
```

然后直接跑 §6.2 的 4 组探测，最后 stop 一次。**E2 不再单独重启做性能。**

**退出条件**：G1–G4 绿；E2 的 4 组 summary 完整；否则回退冲突文件，不计入 A/B。

---

## 6. P2：E3 + 判读（+ 可选 E4）

### 6.1 E3：merged + `W8A16=0`（最可能的生产候选）

```bash
cd $HOME/vllm-test/llm-engine-probe
export RUNS_DIR=$DATA_ROOT/port/perf-runs
RUN=$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-E3
mkdir -p "$RUN"

ENV_FILE=$DATA_ROOT/port/perf-env/.env.rocm101.merged \
EXTRA_ENV="VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 \
VLLM_DCU_FP8_W8A16=0 VLLM_DCU_FP16_PREFILL=1 VLLM_DCU_BF16=1" \
EXTRA_VLLM_ARGS="--disable-custom-all-reduce --skip-mm-profiling" \
  bash scripts/start_vllm.sh --level steps --run-dir "$RUN"
```

跑 §6.2 的 4 组探测，stop。E3 是“代码收敛 + Plan B”的组合，若它达标，P3 可以不做 kernel 优化。

### 6.2 一次启动跑完的 4 组探测

固定顺序，每组一个子目录，互不覆盖：

```bash
RUN=...   # E1/E2/E3 的 run 目录

run_case() {
  local name=$1 p=$2 m=$3 c=$4 r=$5
  local d="$RUN/probe-$name"
  mkdir -p "$d"
  cp -f "$RUN/run_meta.json" "$d/" 2>/dev/null || true
  cp -f "$RUN/serve.log" "$d/" 2>/dev/null || true
  bash scripts/probe_http.sh \
    --run-dir "$d" --prompt-tokens "$p" --max-tokens "$m" \
    --concurrency "$c" --repeat "$r" --warmup 2
  python3 -m probe report --outdir "$d" --title "$name" || true
}

run_case 512c1  512  128 1 2
run_case 709c1  709  128 1 2
run_case 8k1    8192 32  1 1
run_case 512c4  512  128 4 1
```

产物：

| 子目录 | 用途 | 主要指标 |
|---|---|---|
| `probe-512c1` | 主性能 | TTFT、step、TPOT、接受率 |
| `probe-709c1` | 与 DTK 1594 ms 对齐 | TTFT、prefill |
| `probe-8k1` | 长上下文 decode | step、TPOT、KV |
| `probe-512c4` | 并发 | 吞吐、TPOT、接受率 |

> 4 组都在同一个服务生命周期内完成；模型只加载一次。
> 所有臂都关闭 prefix caching，避免 512/709 两组互相命中。

### 6.3 可选 E4：O2 + attention 合并臂

**只在** E2/E3 的 `probe-512c1` decode 仍 >210 ms，或 TTFT 仍 >2.0 s 时做，且只做一次：

```bash
cd $HOME/vllm-test/llm-engine-probe
export RUNS_DIR=$DATA_ROOT/port/perf-runs
RUN=$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-E4
mkdir -p "$RUN"

ENV_FILE=$DATA_ROOT/port/perf-env/.env.rocm101.merged \
EXTRA_ENV="VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 \
VLLM_DCU_FP8_W8A16=1 VLLM_DCU_FP16_PREFILL=1 VLLM_DCU_BF16=1 \
VLLM_DCU_FP8_QUANT_FASTPATH=1 VLLM_GFX906_ATTN_DECODE_BLOCK_M=16" \
EXTRA_VLLM_ARGS="--disable-custom-all-reduce --skip-mm-profiling" \
  bash scripts/start_vllm.sh --level steps --run-dir "$RUN"
```

跑同样 4 组探测。E4 是唯一允许的组合臂；若它没有 ≥5% 改善，不再拆成更多臂。

### 6.4 判读树（只看 E1/E2/E3，不重跑基线）

```text
E1 decode ≤200 ms，E3 decode ≤200 ms
  → W8A16=0 有效；E3 是生产候选
  → 若 E3 TTFT ≤2.0 s：直接进 P4
  → 若 E3 TTFT >2.0 s：只做 P3.2 rocBLAS，不动 decode

E1 decode >220 ms，E3 decode >220 ms
  → 不是 W8A16；看 E2
  → E2 ≤210 ms：代码合并有效，继续优化 E2
  → E2 也 >220 ms：栈/attention 问题，走 P3.3

E1 ≤200 ms，E3 >220 ms
  → merge 引入了回归
  → 用 git diff dtk/dtk 找 suspect 文件；先回退最近一个性能文件，再跑 E3

E2 ≤210 ms，E3 ≤200 ms
  → 两条都可用；优先 E3（关闭 W8A16，风险更低）

E2/E3 TTFT >2.2 s，decode 已 ≤210 ms
  → prefill 是剩余主项；只做 P3.2，不碰 decode

E4 比 E3 改善 <5%
  → 停止组合调参，回到 E3；把 O2/attention 留作后续可选
```

### 6.5 出口条件

- E1/E2/E3 各 4 组 summary 完整；
- 一页 `VERDICT.md`：明确“decode 主因 / prefill 主因 / 生产候选是 E2 还是 E3”；
- 只有在 verdict 指向 kernel 时才做 P3.1；否则直接进 P4。

---

## 7. P3：定点优化（只做一件事）

### 7.1 WP3.1 — W8A16 kernel（仅当 E1 已证明 W8A16 拖慢）

1. 生产缓解：`W8A16=0 + FP16_PREFILL=1`（E3）先上线；这是默认 Plan B。
2. 若仍想保留 W8A16：对比 T2/DTK 的 `fp8_emu_gemm_block_w8a16_kernel` code object，
   看 VGPR/SGPR/LDS/occupancy；只改一个参数（`__launch_bounds__`/`MINB`/block size）。
3. 每次改完只跑 **512c1 + 709c1** 两组探测（8k/c4 留给 P4），不再跑全套四组。
4. 微基准改善 <10% 或 e2e <3% → 停止，保留 E3。

### 7.2 WP3.2 — prefill / rocBLAS（仅当 E3 TTFT >2.0 s）

1. 检查 `torch.mm` 实际后端与 `ROCBLAS_TENSILE_LIBPATH` / hipBLASLt 可用性。
2. 用 `VLLM_DCU_FP16_PREFILL_MIN_M` 只调一个 role，跑 512c1/709c1 两组。
3. 若 rocBLAS 无解：接受 2.4 s，Must 仍 ≤2.5 s；**不要为 TTFT 回退 decode**。

### 7.3 WP3.3 — attention / host（仅当 E2/E3 decode >210 ms 且 trace 指向 attention/host）

1. 先确认 `VLLM_GFX906_ATTN_VERIFY_3D=1` 真的在跑（与 `=0` 对比一次）。
2. 只试 E4 已包含的 `BLOCK_M=16`；不要展开 segments/warps 全扫。
3. O2 只在 E4 中试一次；不单独再开臂。

### 7.4 WP3.4 — 接受率（最后做）

1. 用 E3 的 `probe-512c1` 看 per-position acceptance；
2. 只改一个变量（`VLLM_DCU_BF16` 或 `SPEC_KV_DTYPE`），跑一次 512c1；
3. 不调温度/采样伪装接受率。

---

## 8. P4：验收与固化

### 8.1 最终验收（只跑一次完整启动）

以 E3（或 E2）为候选，**打开 prefix caching**，在同一个服务生命周期内跑：

```bash
cd $HOME/vllm-test/llm-engine-probe
export RUNS_DIR=$DATA_ROOT/port/perf-runs
RUN=$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-FINAL
mkdir -p "$RUN"

ENV_FILE=$DATA_ROOT/port/perf-env/.env.rocm101.merged \
EXTRA_ENV="VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 \
VLLM_DCU_FP8_W8A16=0 VLLM_DCU_FP16_PREFILL=1 VLLM_DCU_BF16=1" \
EXTRA_VLLM_ARGS="--disable-custom-all-reduce --skip-mm-profiling --enable-prefix-caching" \
  bash scripts/start_vllm.sh --level steps --run-dir "$RUN"
# 若最终候选是 E2，把上面的 W8A16=0 改成 1

# 先 G1–G4，再按 §6.2 跑 512c1、709c1、8k1、512c4；最后 stop
bash scripts/stop_vllm.sh --run-dir "$RUN"
```

门禁：

| 门 | 证据 | Must |
|---|---|---|
| G0 栈身份 | `stack_identity --expect rocm101` | exit 0 |
| G1 op | `check_rocm_c_ops.py` | exit 0；w8a16 在注册表 |
| G2 路由 | `W8A16 engaged`（若 E3 则 `False` 也要记录） | 与最终开关一致 |
| G3 物化 | `first materialization route engaged` | 出现 |
| G4 数值 | `' Paris.'` | 语义一致 |
| G5 性能 | `probe-512c1` / `probe-709c1` | TTFT ≤2.5 s；step ≤210 ms；TPOT ≤45 ms |
| G6 投机 | `probe-512c1` | 接受长度 ≥4；接受率 ≥45% |
| G7 稳定性 | 2 次重启 + 15 min c1/c4 soak | 0 crash/OOM/NaN |
| G8 回滚 | `use_stack.sh dtk` + smoke | 可回 DTK；再回 rocm101 |

### 8.2 稳定性（压缩）

- 2 次冷启动（不是 3 次），记录 ready 时间与首次 TTFT；
- 15 min soak：c1 与 c4 各 7–8 min；
- 8k/32 与 32k/8 各一次，确认 decode 不塌；
- prefix caching 开/关各一次，确认命中率与 TTFT。

### 8.3 回滚

```bash
[ -d $DATA_ROOT/port/gfx906-vllm-rocm101/installed-so.baseline-20261023 ] || \
  cp -a $DATA_ROOT/port/gfx906-vllm-rocm101/installed-so \
        $DATA_ROOT/port/gfx906-vllm-rocm101/installed-so.baseline-20261023

cd $WORKDIR
tools/dcu/use_stack.sh dtk

ROCM101_SO_DIR=$DATA_ROOT/port/gfx906-vllm-rocm101/installed-so.baseline-20261023 \
  tools/dcu/use_stack.sh rocm101
```

### 8.4 生产切换

优先把 `SRC_DIR` 指向 `$DATA_ROOT/port/gfx906-vllm-rocm101-perf`（保留 git 历史）。
若必须保留 T2 原路径，用 `git archive <commit> | tar -x -C T2` 导出后按 §5.3 重建，
**不要手工 `cp` 文件**。

---

## 9. 工期、分工、风险

### 9.1 工期

| 阶段 | 时间 | 说明 |
|---|---|---|
| P0 冻结 + 入 git + E1 | 0.5–1 天 | E1 一次启动多探测约 20–30 min |
| P1 merge + 构建 + E2 | 1–1.5 天 | 构建占大头（30–120 min） |
| P2 E3 + 判读（+E4） | 0.5 天 | E3 一次启动；E4 仅在触发时 |
| P3 定点优化 | 1–2 天 | 可能只做 Plan B，不修 kernel |
| P4 验收 | 0.5–1 天 | 最终一次启动 + 2 次重启 |
| **合计** | **3.5–6 天** | 不含排队等卡 |

### 9.2 分工

| 角色 | 负责 |
|---|---|
| 执行 agent | P0–P2、E1/E2/E3/E4、探测与判读 |
| kernel 工程师 | P3.1/P3.2（仅触发时） |
| 验收人 | P4 的 G0–G8、回滚 |
| 用户/运维 | 是否接受 Plan B（`W8A16=0`）与生产切换窗口 |

### 9.3 风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 一次启动的统计置信度不足 | 高 | 中 | 每组 repeat 2；差异 <5% 判不可判定；只重跑该臂一次 |
| prefix cache 关闭与生产不一致 | 中 | 低 | P4 用生产参数复验一次 |
| merge 冲突引入回归 | 中 | 高 | 冲突表 + E2/E3 对比 + G1–G4 |
| W8A16 在 ROCm 10.1 就是慢 | 中 | 高 | E3 Plan B，保留 TTFT 收益 |
| prefill 是 rocBLAS 上限 | 中 | 中 | 接受 2.4 s，Must ≤2.5 s |
| F24 复现（卡 2/3） | 低 | 高 | 立即停，记录 dmesg；改用 0/1 TP2 |
| 构建 OOM/超时 | 中 | 中 | `-j2`、`--shm-size 8g`、串行 |

---

## 10. 最终验收清单

- [ ] 基线冻结文档完成；**没有重跑 B1–B4**
- [ ] T2 入 git；`rocm101-perf` 分支存在
- [ ] `stack_identity --expect rocm101` exit 0
- [ ] 四个 `.so` 同源，md5 记录
- [ ] E1/E2/E3 各 4 组 summary 完整；E4 仅在触发时存在
- [ ] G1–G4 全绿
- [ ] 512/128 c1：TTFT ≤2.5 s；step ≤210 ms；TPOT ≤45 ms
- [ ] 709/128 c1：TTFT ≤2.5 s
- [ ] 接受长度 ≥4，接受率 ≥45%
- [ ] 8k/32、32k/8 无 decode 塌陷
- [ ] 2 次重启 + 15 min soak：0 crash/OOM/NaN
- [ ] 回滚演练通过
- [ ] 最终 `.env` 与报告归档

---

## 附录 A：为什么是这 3+1 次启动

| 问题 | 需要的臂 | 不需要的臂 |
|---|---|---|
| W8A16 是否拖慢 decode？ | E1（W8A16=0） vs 信任基线 B3（W8A16=1） | 不需要再跑 W8A16=1 的 E0 |
| 代码合并是否改善？ | E2（merged, W8A16=1） vs B3/E1 | 不需要重跑 DTK B2 |
| 生产候选是什么？ | E3（merged, W8A16=0） | 不需要单独跑 FP16_PREFILL=0 的臂 |
| O2/attention 是否有额外收益？ | E4（仅当 E2/E3 未达标） | 不需要逐项拆 4 个臂 |
| 长上下文/并发是否回归？ | 每个臂内的 8k/c4 探测 | 不需要单独起服务 |

## 附录 B：命令速查

```bash
# 0) 只读核对基线身份
for r in 20261023-135729-serve 20261023-150120-serve \
         20261023-164058-serve 20261023-164704-serve; do
  $WORKDIR/.venv/bin/python -c "import json;m=json.load(open('$HOME/vllm-test/llm-engine-probe/runs/$r/run_meta.json'));print('$r',m.get('host'))"
done

# 1) E1 启动 + 4 组探测 + stop（完整流程）
cd $HOME/vllm-test/llm-engine-probe
export RUNS_DIR=$DATA_ROOT/port/perf-runs
RUN=$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-E1; mkdir -p "$RUN"

ENV_FILE=$DATA_ROOT/port/perf-env/.env.rocm101 \
EXTRA_ENV="VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 \
VLLM_DCU_FP8_W8A16=0 VLLM_DCU_FP16_PREFILL=1 VLLM_DCU_BF16=1" \
EXTRA_VLLM_ARGS="--disable-custom-all-reduce --skip-mm-profiling" \
  bash scripts/start_vllm.sh --level steps --run-dir "$RUN"

for spec in "512c1 512 128 1 2" "709c1 709 128 1 2" \
            "8k1 8192 32 1 1" "512c4 512 128 4 1"; do
  set -- $spec; name=$1; p=$2; m=$3; c=$4; r=$5
  d="$RUN/probe-$name"; mkdir -p "$d"
  cp -f "$RUN/run_meta.json" "$d/" 2>/dev/null || true
  bash scripts/probe_http.sh --run-dir "$d" --prompt-tokens "$p" \
    --max-tokens "$m" --concurrency "$c" --repeat "$r" --warmup 2
done

bash scripts/stop_vllm.sh --run-dir "$RUN"

# 2) E2/E3 只改 SRC_DIR 与 W8A16，重复上面 4 组
# 3) E4 仅在触发时加：
#    VLLM_DCU_FP8_QUANT_FASTPATH=1 VLLM_GFX906_ATTN_DECODE_BLOCK_M=16
```

---

**一句话执行顺序**：P0 冻结基线 + 入 git + 跑 E1（`W8A16=0`）；如果 E1 decode 回到 ~192 ms，
P1 合并后直接跑 E3（`merged + W8A16=0`）作为生产候选；只有 E2/E3 仍 >210 ms 时才跑 E4 或修 kernel。
**不重跑 B1–B4，不做九臂矩阵。**
