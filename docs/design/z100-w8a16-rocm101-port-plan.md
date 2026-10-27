# 施工图：w8a16 三件套 → rocm101 树移植（gfx906 / Z100L）

> **读者**：执行本移植的 agent。本文件是唯一施工依据，冲突时以本文件为准。
> **背景**（不必重读，结论已并入本文）：
> - 原 W8A16 设计与 DTK 树实现：`/home/lyh/z100-vllm/docs/design/z100-w8a16-prefill-fp16-construction.md`（数学契约见其 §1，**移植不改变数学**）
> - DTK host 栈的实测收益（2026-10-23）：TTFT 4937→1595 ms（fp16 物化），decode TPOT 39.9 ms、DFlash2 接受率 50.6%
> - rocm101 栈现状（无 w8a16）：TTFT 4937 ms / TPOT 37.1 ms / E2E 9652 ms（512/128/c1，`runs/20261023-135729-serve`）
>
> **事后调查（2026-10-24）**：本次只带入了 w8a16 三件套；目标树 99.8% 的源文件仍停在 S23，
> 且 2026-10-23 15:01 的 1595 ms 数据实际来自 DTK。原因与后续 merge 方案见
> [`docs/result/z100-rocm101-perf-integration-investigation.md`](../result/z100-rocm101-perf-integration-investigation.md)。

---

## 0. 三十秒看懂

把 **w8a16 三件套**（① `fp8_emu_gemm_block_w8a16` emu 内核 ② `gfx906_fp16_prefill` 大 M fp16 物化路由 ③ 4 个 env 旋钮）从 DTK 树移植到 rocm101 树：

```
DTK 树（源，功能超集）                 rocm101 树（目标，待移植）
/home/lyh/z100-vllm                    /mnt/game/port/gfx906-vllm-rocm101
  csrc/rocm/fp8_emu_gemm.cu   ──拷贝──▶  同路径（已验证 DTK 为严格超集）
  csrc/rocm/fp8_emu_gemm.h    ──拷贝──▶  同路径
  csrc/rocm/fp8_dequant.cuh   ──新增──▶  同路径（rocm101 缺失）
  csrc/rocm/torch_bindings.cpp ─hunk──▶  仅追加 w8a16 注册（S2 有 diff 门）
  vllm/.../gfx906_fp8.py      ──拷贝──▶  同路径（DTK 为超集，S3 有 diff 门）
  vllm/.../gfx906_fp16_prefill.py ─新增─▶ 同路径
  vllm/envs.py                ──打点──▶  仅插入 4 个旋钮（两树此文件分歧大，禁止整拷）
然后容器内重编 _rocm_C → 六道验收门 → .env 切换
```

**非目标**：不移植 s24 动态 k、调度器改动、DTK 树的 env.sh、platforms、`v1/engine/core_client.py`
（rocm101 树已带 InprocClient 修补，别覆盖）。

---

## 1. 已验证的前置事实（不要重新论证，直接引用）

1. `fp8_emu_gemm.cu` 函数名级 diff：**rocm101 侧 0 个独有函数**（DTK 侧新增 27 个：
   `w8a16_load/store/fold_block_scale`、`fp8_dequant_to_fp16_out`、`fp8_gemm_{load,compute,store}_wide`、
   `launch_gemm_block_wide2`、`launch/dispatch_gemm_block_w8a16`、`check/run_block_w8a16`、
   `fp8_emu_gemm_block_w8a16(+_meta)`、`fp8_emu_gemm_block_quant(+_meta/_x)`、
   `fp8_emu_quant_a_kernel`、`launch_quant_a`、`run_block_quant`、`dispatch_block_quant` 及
   `w8a16_cfg/w8a16_gemv_max_m/fused_decode_cfg/stream_max_m` 等配置位）→ **整文件搬运**。
2. `gfx906_fp8.py` 符号级 diff 同向（DTK 独有：`ops_available_block_w8a16/quant`、`_W8A16_MODE`、
   `_PREFILL_ROUTE`、`_FUSE_QUANT`、`_GEMV_MS_MAX`、`_QUANT_EPS`、`_lut_for`、`_fp8...` 等）→ **整文件搬运**。
3. `_custom_ops.py` 两树 0 差异；`gfx906_fp16_prefill.py` 依赖仅 4 项：
   `envs`、`vllm.logger`、`.BlockScaledMMLinearKernel.FP8BlockParams`、`.gfx906_fp8._device_lut`
   ——前三者 rocm101 树已有，`_device_lut` 随 S3 一起到位。
4. rocm101 树的 `.so` 是符号链接 → `installed-so/`；原始构建命令形态见 `/mnt/game/port/build.log`
   （`cmake --build . -j=2 --target=...`）。rocm101 树**没有**现成 build 目录，重编为全量。
5. rocm101 树已带（不要动）：`vllm/v1/engine/core_client.py` 的 `profile_async` 修补、
   `vllm/platforms/hygon_dp4x.py`、`tools/dcu/env-rocm101.sh`。本机 4 卡现均为 Z100L/DP2X
   （`dp4x_status=0`），DP4X guard 自动不触发。
6. rocm101 树对「fp16 目标 + bf16 草稿 KV」**无** DTK 树那条 `_reject_mismatched_float_kv_cache`
   强校验（10-23 实测 dflash bf16 KV 正常），不需要 config 补丁。

---

## 2. 前置条件与约束

- GPU 空闲（`hy-smi` 显存 0%）：构建不碰卡，但 S6 验收需要独占 4 卡。
- 构建容器：`z100-rocm101-vllm:20261023`（内含 ROCm 10.1 + torch 2.16 + build-essential/cmake/ninja）。
  也可重建自镜像 `z100-rocm10.1:24.04-r642` + 装轮子，但直接用 commit 镜像省事。
- **构建串行**：同一时刻只允许一个 agent 编译（内存 31 G，`-j` ≤ 2，见 build.log 先例）。
- 每个被替换文件先备份到 `<repo>/backups/w8a16-port-20261023/`（保留原路径结构），构建后记录 md5。
- 主机 31 G 内存：构建用容器加 `--shm-size 8g`，`-j2`。

---

## 3. 步骤

### S1 — 备份与基线留痕（~2 min）

```bash
B=/mnt/game/port/gfx906-vllm-rocm101
mkdir -p $B/backups/w8a16-port-20261023/csrc/rocm $B/backups/w8a16-port-20261023/scaled_mm \
         $B/backups/w8a16-port-20261023/installed-so
cp -a $B/csrc/rocm/fp8_emu_gemm.{cu,h} $B/csrc/rocm/torch_bindings.cpp \
      $B/backups/w8a16-port-20261023/csrc/rocm/
cp -a $B/vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py \
      $B/backups/w8a16-port-20261023/scaled_mm/
cp -a $B/vllm/envs.py $B/backups/w8a16-port-20261023/
cp -a $B/installed-so/_rocm_C.abi3.so $B/backups/w8a16-port-20261023/installed-so/   # 实体备份
md5sum $B/installed-so/_rocm_C.abi3.so | tee $B/backups/w8a16-port-20261023/rocm_C.baseline.md5
```

### S2 — csrc 搬运（~5 min）

```bash
A=/home/lyh/z100-vllm; B=/mnt/game/port/gfx906-vllm-rocm101
cp -a $A/csrc/rocm/fp8_emu_gemm.cu $A/csrc/rocm/fp8_emu_gemm.h $B/csrc/rocm/
cp -a $A/csrc/rocm/fp8_dequant.cuh $B/csrc/rocm/            # 新文件
cp -a $A/csrc/rocm/torch_bindings.cpp $B/csrc/rocm/
```

**diff 门（必做）**：`diff $B/backups/w8a16-port-20261023/csrc/rocm/torch_bindings.cpp $B/csrc/rocm/torch_bindings.cpp`
——预期差异**只有** `fp8_emu_gemm_block_w8a16`/`_quant*`/`fp8_emu_quant_a` 的注册块与注释。
若发现 rocm101 独有的绑定行被删（例如 rocm101 树后来加过的注册项），把它们手工合回并记录。
`fp8_emu_gemm.cu/.h` 同理复核一遍（预期：纯新增，无删除）。

### S3 — python 路由搬运（~5 min）

```bash
S=$B/vllm/model_executor/kernels/linear/scaled_mm
cp -a $A/vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py $S/
cp -a $A/vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py $S/   # 新文件
```

**diff 门（必做）**：对 `gfx906_fp8.py` 重复 S2 式复核，预期差异全部是 DTK 侧新增
（W8A16/quant/PREFILL_ROUTE/fastpath 等）。特别注意两点：
1. rocm101 树若在 `gfx906_common.py`/`gfx906_fp8.py` 有 BF16 路由相关行被覆盖，手工合回；
2. `gfx906_fp16_prefill.py` 引用 `envs.VLLM_DCU_FP16_PREFILL{,_MIN_M}` 与
   `envs.VLLM_DCU_FP16_WS_BYTES` ——由 S4 提供，缺一会在 import 时报错（这是好事，算自检）。

### S4 — envs.py 打点（~10 min，禁止整文件拷贝）

两树 `envs.py` 分歧大，只做**两处插入**（以 DTK 树 `vllm/envs.py` 为蓝本，锚点按内容找）：

1. 字段区：在 rocm101 树 `VLLM_DCU_FP8_GEMV_MAX_M` 字段定义之后，插入 4 个字段（注释连同拷贝）：
   - `VLLM_DCU_FP8_W8A16: str = "auto"`
   - `VLLM_DCU_FP16_PREFILL: str = "0"`
   - `VLLM_DCU_FP16_PREFILL_MIN_M: int = 0`
   - `VLLM_DCU_FP16_WS_BYTES: int = 0`
2. lambda 解析区：在 rocm101 树 `"VLLM_DCU_FP8_GEMV_MAX_M"` lambda 之后，插入同名 4 条
   lambda（从 DTK 树 envs.py 对应段落原样拷贝）。

**验收**：`python3 -c "from vllm import envs; print(envs.VLLM_DCU_FP8_W8A16, envs.VLLM_DCU_FP16_PREFILL)"`
在源环境（S6 前）应输出 `auto 0` 且无异常。

### S5 — 容器内重编 _rocm_C（~30–60 min，全量）

```bash
docker run --rm --name z100-w8a16-build \
  -v /mnt/game:/mnt/game -v /home/lyh/z100-vllm:/src-dtk:ro \
  --shm-size 8g z100-rocm101-vllm:20261023 bash -c '
set -e; . /src/tools/dcu/env-rocm101.sh
cd /mnt/game/port/gfx906-vllm-rocm101
python3 -m pip install -e . --no-build-isolation -j 2 2>&1 | tee /mnt/game/port/w8a16-port-build.log'
```

- 全量构建（repo 无 build 目录），预计 30–60 min；构建日志必须留档。
- 构建完成后核验（宿主即可）：

```bash
strings /mnt/game/port/gfx906-vllm-rocm101/installed-so/_rocm_C.abi3.so | grep -c w8a16   # 预期 >0
md5sum /mnt/game/port/gfx906-vllm-rocm101/installed-so/_rocm_C.abi3.so
```

> 若 `pip install -e` 因 CMake 找不到目标而失败，回退到显式两步：
> `python3 setup.py build_ext --inplace -j 2`（同一容器），产物落 `vllm/` 下；两种方式的
> 产物都必须通过 S6-G1。

### S6 — 验收六门（顺序执行，任何一门不过即停）

| 门 | 命令 | 通过标准 |
|---|---|---|
| G1 op 注册 | 容器内 `python3 tools/dcu/check_rocm_c_ops.py`（先 `. /src/tools/dcu/env-rocm101.sh`） | exit 0；`fp8_emu_gemm_block_w8a16` 在注册表 |
| G2 路由日志 | `bash scripts/start_vllm.sh`（.env 见 S7）后 grep `W8A16 engaged=True (built op True)` | 出现且 `GEMV_MAX_M=8` |
| G3 物化路由 | 同日志 grep `fp16 prefill materialization: VLLM_DCU_FP16_PREFILL=1, role min_m` | 出现且 role 表非空（gate_up_proj:128 等） |
| G4 数值 | `curl /v1/completions` 贪心 `"The capital of France is"` | 输出与 `' Paris.\nThe capital of Germany is'` 语义一致（≤ 1 token 偏差可接受，记录实际值） |
| G5 性能 | `bash scripts/probe_all.sh --level steps --run-dir <新run目录>` | TTFT p50 **≤ 2.5 s**（对照无 w8a16 的 4937 ms）；TPOT p50 ≤ 45 ms |
| G6 投机解码 | 同 G5 的 `投机解码` 段 | 平均接受长度 ≥ 4，接受率 ≥ 45% |

> G5/G6 期间 GPU 独占；kernel 档/probe 插件保持关闭（`.env` 现值即 steps/插件关）。

### S7 — .env 切换块（验收全过后执行）

在 `.env` 的 rocm101 形态（模板：`.env.bak-rocm101-20261023`）上追加/修改：

```bash
IMAGE=z100-rocm101-vllm:20261023
MODE=run
SRC_DIR=/mnt/game/port/gfx906-vllm-rocm101
DTK_ENV_FILE=                                # 空：跳过 DTK prelude
DCU_ENV_FILE=/src/tools/dcu/env-rocm101.sh
TRITON_CACHE_DIR=/mnt/game/triton-z100-rocm101/cache/jit
TRITON_INSTALL_DIR=/mnt/game/triton-z100-rocm101/install
VLLM_CACHE_ROOT=/mnt/game/vllm-cache-rocm10
MAX_NUM_SEQS=4                               # 用户要求：本地部署并发上限 4
EXTRA_ENV="VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 VLLM_DCU_FP8_W8A16=1 VLLM_DCU_FP16_PREFILL=1 VLLM_DCU_BF16=1"
# 注意：rocm101 树无 VLLM_DCU_BF16 强校验问题；VLLM_DCU_BF16=1 保持草稿走仿真 bf16
# （env-rocm101.sh 默认 auto 亦可，但显式 1 与 DTK 栈行为对齐）
# 不设 VLLM_DCU_USE_BF16_GEMV（用户要求：关闭）
```

---

## 4. 验收矩阵汇总

| 项 | 判定 | 证据落盘 |
|---|---|---|
| G1 op | check_rocm_c_ops exit 0 | 容器输出 |
| G2/G3 路由 | 两行日志 | `<run>/serve.log` |
| G4 数值 | 冒烟一致 | 命令输出 |
| G5 性能 | TTFT ≤2.5s | `<run>/http_summary.json` |
| G6 投机 | accept ≥45% | `<run>/steps_summary.json` |
| 构建 | w8a16 符号>0 + md5 | `w8a16-port-build.log`、md5 文件 |

全部通过后：把本文件 §7 的 .env 差异与最终数字回填到
`/home/lyh/vllm-test/llm-engine-probe/runs/20261023-140045-serve/OPTIMIZATION-PLAN.md`。

## 5. 风险与已知坑

1. **envs.py 行号漂移**：两树行号不同，只按锚点内容插入，勿按行号。
2. **gfx906_fp8.py 若存在 rocm101 独有行**（函数级 diff 未发现，但行级未逐一复核）：
   diff 门发现即手工合回；拿不准的行保 rocm101 版本并记录。
3. **构建 OOM/慢**：`-j2` 上限；首次全量无缓存，不要中途 kill（半成品 .so 会污染 installed-so）。
4. **`installed-so/` 覆盖不可逆**：S1 已备份 md5 与原 .so? ——原 .so 本体在 installed-so/，
   备份命令应包含 `cp -a installed-so/_rocm_C.abi3.so backups/.../`（S1 已列 md5，**务必补实体备份**）。
5. **torch 2.16 py3.12 编译器差异**：DTK 树的 .cu 在 torch 2.11/py3.10 编译；若 rocm10.1 clang
   对某些 `__forceinline__` 模板报错，优先局部 `#if` 降级而不是改语义。
6. **勿动**：`vllm/v1/engine/core_client.py`、`vllm/platforms/`、`tools/dcu/env-rocm101.sh`、
   `hosts/`、`scripts/`（本仓库白名单守卫）。

## 6. 回滚

```bash
B=/mnt/game/port/gfx906-vllm-rocm101
cp -a $B/backups/w8a16-port-20261023/csrc/rocm/* $B/csrc/rocm/
cp -a $B/backups/w8a16-port-20261023/scaled_mm/gfx906_fp8.py \
      $B/vllm/model_executor/kernels/linear/scaled_mm/
rm -f $B/vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py
cp -a $B/backups/w8a16-port-20261023/envs.py $B/vllm/envs.py
cp -a $B/backups/w8a16-port-20261023/_rocm_C.abi3.so $B/installed-so/   # 实体备份恢复
# 重新构建（同 S5）或直接用恢复的 .so；.env 用 .env.bak-rocm101-20261023
```

## 7. 预期终态

rocm101 栈 + w8a16 后：TTFT ~1.6–2.5 s（fp16 物化主导，两栈同走 rocBLAS），
TPOT ~37–40 ms（fp8_emu_gemv 与 DTK 相当），DFlash2 接受率 ~50%，
即 **DTK 栈的 TTFT 优势 + rocm101 栈的既有 decode 能力**。若 G5 的 TTFT 显著劣于 2.5 s，
优先排查 fp16 物化路由是否生效（G3）与 rocBLAS 在 rocm10.1 的 fp16 GEMM 形状选择，
而不是回滚。
