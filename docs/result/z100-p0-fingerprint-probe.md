# Z100 P0 测量基座：策略指纹（WP0.4）+ 真实文档探针（WP0.5）

> 设计依据：[`z100-decode-prefill-optimization-design.md`](../design/z100-decode-prefill-optimization-design.md)（DESIGN）§5、[`z100-decode-prefill-optimization-construction.md`](../design/z100-decode-prefill-optimization-construction.md)（BUILD）§2/§4/§12。
> 属主：B。工作性质：**零卡**（本报告全部内容未启动/未占用任何 GPU 服务；Lead 独占 GPU）。
> 树：`/home/lyh/z100-vllm`，分支 `dtk`，树 SHA `7078b5fe137feeab3750ee243daeaf172dc439d8`（脏树：`git status --porcelain` 非空，见 §WP0.4-3）。
> 产物：
> - `tools/dcu/dcu_policy_fingerprint.py`（WP0.4，新增）
> - `tools/dcu/s20_single_probe.py`（WP0.5，加法式改动）
> - 本地证据目录 `.s27/b-fingerprint/`（`.gitignore` 覆盖，不提交）

| 工作包 | 状态 | 门禁 |
|---|---|---|
| WP0.4 策略指纹 | **完成**（`--selftest` 41/41） | G3 可打印且与报告一致 |
| WP0.5 探针 `--prompt-file` | **完成**（8 KB 自检通过） | WP1.3 前置 |

---

## WP0.4 —— 策略指纹 `tools/dcu/dcu_policy_fingerprint.py`

### WP0.4-1. 一句话结论

新增零卡指纹工具：把 109 个 `VLLM_DCU_*` / `VLLM_GFX906_*` / `VLLM_QWEN35_*` / `VLLM_S24_*`（本树无 `VLLM_SPEC_*`）旋钮解析成**实际生效值**（默认值、三态回落、主开关组合、编译期 clamp 全部落地），输出 `FP <sha256[:16]>` + `FP_FULL key=value…` 两行；同一 env 连跑两次指纹一致，改有效旋钮指纹必变，`VLLM_DCU_FP8_GEMV_MAX_M=16` 时 WARN `requested=16 effective=8` 且 `FP_FULL` 与 `=8` 完全相同（只差在 requested 入哈希与 WARN）；工具 0.06 s / 21 MB RSS，**不 import torch / 不初始化 HIP**，生产 env 指纹为 `7e70fc2c130b9191`。

### WP0.4-2. A/B 表

| 臂 | 唯一变量 | 策略指纹 | `FP8_GEMV_MAX_M` 生效值 | WARN | 说明 |
|---|---|---|---|---|---|
| `bare`（未 source 生产 env） | — | `6eab564bf1ccdb91` | 8 | 无 | 默认策略基线；两次运行逐字节一致 |
| `prod`（`serve_host_dflash2.sh` 的 19 个 export + `env.sh`） | — | `7e70fc2c130b9191` | 8 | 无 | 生产臂，见 §WP0.4-3 env 全文 |
| `prod-gemv1` | `VLLM_DCU_FP8_GEMV_MAX_M=1` | `e472fc80881fd4fe` | 1 | 无 | 指纹改变 ⇒ 旋钮被真实采集（WP0.2 的臂 2） |
| `prod-gemv16` | `VLLM_DCU_FP8_GEMV_MAX_M=16` | `3b9c580127579c29` | **8** | `requested=16 effective=8 (clamped to _GEMV_MS_MAX=8)` | `FP_FULL` 与 `prod` 逐字节相同；哈希因 requested 逐字入账而不同 |

> 哈希口径：`sha256(json(schema, effective 全部生效值, requested 全部 env 原文, caps))[:16]`。
> 即**同一 env ⇒ 同一哈希**（C3 逐字钉死）；**任何 env 改动 ⇒ 哈希变**（含被 clamp 的 `=16`）；被 clamp 前后的**行为等价**由 `FP_FULL` 相同来证明。C4 出处要求的树 SHA / 脏树标记在 `--json` 里（`git_sha` / `git_dirty`），不进哈希（否则每个 commit 都换指纹、无法跨臂比对策略）。

### WP0.4-3. env 全文 + 策略指纹 + 树 SHA

树 SHA：`7078b5fe137feeab3750ee243daeaf172dc439d8`（分支 `dtk`；`git_dirty=true`，因为本会话同时有 Lead 的其他改动，属于预期）。

生产臂 env 全文（`env | grep -E '^(VLLM_(DCU|GFX906|QWEN35|S24)_[A-Z0-9_]+|DTK_ROOT)=' | sort`；即 `tools/dcu/serve_host_dflash2.sh` 头部的 19 个旋钮 + `env.sh` 的 `VLLM_DCU_GCN_ARCH`）：

```text
DTK_ROOT=/mnt/game/host-runtime/dtk2604
VLLM_DCU_BF16=1
VLLM_DCU_FP16_PREFILL=auto
VLLM_DCU_FP8_DECODE_CFG=35
VLLM_DCU_FP8_EMU_FASTPATH=1
VLLM_DCU_FP8_GEMV_MAX_M=8
VLLM_DCU_FP8_QUANT_FASTPATH=1
VLLM_DCU_FP8_W8A16=1
VLLM_DCU_GCN_ARCH=gfx906
VLLM_DCU_USE_BF16_GEMV=1
VLLM_DCU_USE_FP8_EMU=1
VLLM_GFX906_ATTN_FORCE_CAUSAL=0
VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX=16384
VLLM_GFX906_ATTN_PREFILL_SEGMENTS=4
VLLM_GFX906_ATTN_VERIFY_3D=1
VLLM_GFX906_GDN_CHUNKED_PREFILL=1
VLLM_GFX906_TRITON_ATTN_PREFILL_3D=1
VLLM_QWEN35_MTP_GDN_TRITON=1
VLLM_S24_DYNK_EMA=0.1
VLLM_S24_DYNK_HI=65536
VLLM_S24_DYNK_LO=16384
```

`prod` 臂完整两行（`FP_FULL` 共 111 个生效键：109 旋钮 + 2 派生 + 6 编译期 caps；此处完整贴出，方便与后续任何 A/B 的指纹逐字对照）：

```text
FP 7e70fc2c130b9191
FP_FULL FP16_PREFILL_DEFAULT_MIN_M=4096 FP8_GEMV_MAX_M_W8A16=8 VLLM_DCU_BF16=on VLLM_DCU_BF16_ROUTE=auto VLLM_DCU_CUSTOM_AR=0 VLLM_DCU_CUSTOM_AR_ALLOC=- VLLM_DCU_FP16_PREFILL=auto VLLM_DCU_FP16_PREFILL_MIN_M=0 VLLM_DCU_FP16_WS_BYTES=0 VLLM_DCU_FP8_CVT=lut VLLM_DCU_FP8_DECODE_CFG=35 VLLM_DCU_FP8_EMU_FASTPATH=1 VLLM_DCU_FP8_FUSE_QUANT=0 VLLM_DCU_FP8_GEMV_MAX_M=8 VLLM_DCU_FP8_GEMV_PREF=3 VLLM_DCU_FP8_GEMV_STREAM=1 VLLM_DCU_FP8_PREFILL_CFG=0 VLLM_DCU_FP8_QUANT_FASTPATH=1 VLLM_DCU_FP8_STREAM_MAX_M=0 VLLM_DCU_FP8_W8A16=1 VLLM_DCU_GCN_ARCH=gfx906 VLLM_DCU_INT8_GEMV_MAX_M=8 VLLM_DCU_SDOT4_INNER=sdot4 VLLM_DCU_SMALLM_GEMM=off VLLM_DCU_USE_BF16_GEMV=1 VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_USE_GFX906_EMU=0 VLLM_DCU_USE_INT8_SDOT4=0 VLLM_GFX906_ATTN_DECODE_BLOCK_M=0 VLLM_GFX906_ATTN_DECODE_SEGMENTS=0 VLLM_GFX906_ATTN_DECODE_WARPS=0 VLLM_GFX906_ATTN_EAGER_FALLBACK=0 VLLM_GFX906_ATTN_FORCE_CAUSAL=0 VLLM_GFX906_ATTN_MULTI_QUERY_3D=1 VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX=16384 VLLM_GFX906_ATTN_PREFILL_3D_STAGES=1 VLLM_GFX906_ATTN_PREFILL_3D_WARPS=4 VLLM_GFX906_ATTN_PREFILL_BLOCK_M=0 VLLM_GFX906_ATTN_PREFILL_SEGMENTS=4 VLLM_GFX906_ATTN_PREFILL_STAGES=1 VLLM_GFX906_ATTN_PREFILL_TILE=0 VLLM_GFX906_ATTN_PREFILL_WARPS=4 VLLM_GFX906_ATTN_VERIFY_3D=1 VLLM_GFX906_CORE_PROFILE=0 VLLM_GFX906_CORE_PROFILE_LIMIT=64 VLLM_GFX906_DP4X_DEBUG=0 VLLM_GFX906_DP4X_GUARD=auto VLLM_GFX906_DP4X_MIN_WARPS=8 VLLM_GFX906_DP4X_WATCH=1 VLLM_GFX906_GDN_CHUNKED_PREFILL=1 VLLM_GFX906_GDN_DEC_V2=1 VLLM_GFX906_GDN_DEC_V2_DEBUG=0 VLLM_GFX906_GDN_DEC_V2_TIME=0 VLLM_GFX906_GDN_MTP_V2=1 VLLM_GFX906_GDN_MTP_V2_DEBUG=0 VLLM_GFX906_GDN_MTP_V2_TIME=0 VLLM_GFX906_GDN_TRITON_PREFILL=1 VLLM_GFX906_RUNNER_PROFILE=0 VLLM_GFX906_RUNNER_PROFILE_LIMIT=32 VLLM_GFX906_TRITON_ATTN_PREFILL_3D=1 VLLM_GFX906_TRITON_ATTN_TUNING=1 VLLM_GFX906_TRITON_CAUSAL_CONV1D=0 VLLM_GFX906_TRITON_CAUSAL_CONV1D_SYNC=1 VLLM_GFX906_TRITON_MROPE=1 VLLM_GFX906_TRITON_RESHAPE_CACHE=1 VLLM_GFX906_TRITON_UNIFIED_ATTN=1 VLLM_QWEN35_CALL_ORDER=ba VLLM_QWEN35_CHUNK_DECODE=auto VLLM_QWEN35_COMBINED_PACKED_DECODE=1 VLLM_QWEN35_EMPTY_CORE_ATTN_OUT=0 VLLM_QWEN35_EMPTY_CORE_ATTN_SINGLE_TOKEN=1 VLLM_QWEN35_EXPAND_QK=1 VLLM_QWEN35_FORCE_BA_SPLIT=0 VLLM_QWEN35_FORCE_Z_ONES=0 VLLM_QWEN35_FUSED_QK_NORM_ROPE=1 VLLM_QWEN35_GGUF_ALLOW_FULL_CUDAGRAPH=0 VLLM_QWEN35_GGUF_ALLOW_PIECEWISE_CUDAGRAPH=0 VLLM_QWEN35_GGUF_DEBUG=0 VLLM_QWEN35_GGUF_DEBUG_FILE=- VLLM_QWEN35_GGUF_DEBUG_LAYER=model.layers.0.linear_attn VLLM_QWEN35_GROUPED_GGUF_BA_MMVQ=1 VLLM_QWEN35_GROUPED_GGUF_MMVQ=1 VLLM_QWEN35_LAYER_PROFILE=0 VLLM_QWEN35_LAYER_PROFILE_LIMIT=2 VLLM_QWEN35_LAYER_PROFILE_MAX_TOKENS=- VLLM_QWEN35_LINEAR_ATTN_PROFILE=0 VLLM_QWEN35_LINEAR_ATTN_PROFILE_LIMIT=4 VLLM_QWEN35_LINEAR_ATTN_PROFILE_MAX_TOKENS=- VLLM_QWEN35_LOAD_DEBUG_FILE=- VLLM_QWEN35_LOCAL_REC_DECODE=0 VLLM_QWEN35_MTP_CONV_CUSTOM=1 VLLM_QWEN35_MTP_CONV_DEBUG=0 VLLM_QWEN35_MTP_GDN_CUSTOM=1 VLLM_QWEN35_MTP_GDN_DEBUG=0 VLLM_QWEN35_MTP_GDN_TRITON=1 VLLM_QWEN35_MTP_GDN_TRITON_BV=4 VLLM_QWEN35_MTP_GDN_TRITON_PROBE=0 VLLM_QWEN35_MTP_GDN_TRITON_WARPS=1 VLLM_QWEN35_PACKED_DECODE=auto VLLM_QWEN35_PACKED_DECODE_TILED_QK=auto VLLM_QWEN35_QK_L2NORM=1 VLLM_QWEN35_RATIO2_FUSED_DECODE=1 VLLM_QWEN35_RATIO2_FUSED_DECODE_DEBUG=0 VLLM_QWEN35_REC_PREFILL=0 VLLM_QWEN35_REVERSE_BA_SHARDS=0 VLLM_QWEN35_RUNTIME_DEBUG_FILE=- VLLM_QWEN35_SHARED_GDN_CORE=0 VLLM_QWEN35_TILED_QK_EXPAND=auto VLLM_S24_DYNK_EMA=0.1 VLLM_S24_DYNK_HI=65536 VLLM_S24_DYNK_LO=16384
```

`prod` 相对 `bare` 的**生效值差异**（其余 97 键相同）：

| 键 | bare | prod | 来源 |
|---|---|---|---|
| `VLLM_DCU_BF16` | `off` | `on` | serve L70 |
| `VLLM_DCU_FP8_W8A16` | `auto` | `1` | serve L79 |
| `VLLM_DCU_FP16_PREFILL` | `0` | `auto` | serve L86 |
| `VLLM_DCU_FP8_DECODE_CFG` | `auto` | `35` | serve L92 |
| `VLLM_DCU_FP8_EMU_FASTPATH` | `0` | `1` | serve L96 |
| `VLLM_DCU_FP8_QUANT_FASTPATH` | `0` | `1` | serve L97 |
| `VLLM_DCU_USE_BF16_GEMV` | `0` | `1` | serve L113（经 `gfx906_emu_enabled` 组合） |
| `VLLM_DCU_USE_FP8_EMU` | `0` | `1` | serve L71 |
| `VLLM_DCU_GCN_ARCH` | `auto` | `gfx906` | `env.sh` L79 |
| `VLLM_GFX906_TRITON_ATTN_PREFILL_3D` | `0` | `1` | serve L122 |
| `VLLM_GFX906_GDN_CHUNKED_PREFILL` | `0` | `1` | serve L112 |
| `VLLM_QWEN35_MTP_GDN_TRITON` | `0` | `1` | serve L111 |
| `VLLM_S24_DYNK_LO` / `HI` | `0` / `0` | `16384` / `65536` | serve L106–107 |

### WP0.4-4. 判读（对照 WP0.4 阈值）

| 验收项（BUILD §4/WP0.4 + 任务板） | 结果 | 证据 |
|---|---|---|
| 同一 `.env` 连跑两次指纹一致 | **过** | `bare` 两次 `6eab564bf1ccdb91`，stdout `cmp` 相同（`.s27/b-fingerprint/fp-bare-run{1,2}.txt`）；selftest 亦覆盖 |
| 改任一旋钮指纹必须变 | **过** | `GEMV_MAX_M` 8→1：`7e70fc2c…`→`e472fc80…`；含未建模变量亦会变（`@raw` 入账） |
| clamp 可见 + WARN | **过** | `=16`：`VLLM_DCU_FP8_GEMV_MAX_M=8`、stderr `requested=16 effective=8`；`FP8_GEMV_MAX_M_W8A16` 派生键同步为 8 |
| 全文 `key=value`、按 key 排序、两行输出 | **过** | selftest `text output is exactly two lines` / `sha256[:16]` |
| `--json` / `--selftest` | **过** | 41/41 checks（`.s27/b-fingerprint/selftest.log`） |
| 零卡、不初始化 GPU | **过** | 只执行 `vllm/envs.py`（独立 `importlib` 加载，无 `vllm/__init__`、无 torch）；selftest 断言 `torch` 未进 `sys.modules` |
| 不碰 `vllm/`、不碰 `serve_host_dflash2.sh` | **过** | 唯一新增文件 + 本报告；serve 脚本未改（接入行交给 Lead） |

> **G2 静态检查**：本机 `.venv` **未安装 `pre-commit`**（无 `pre-commit` 二进制/模块；只有 `ruff 0.17.1`，hook 声明为 v0.14.0），因此用等价命令代替：`ruff check --no-cache` + `ruff format --check --no-cache` 对 `dcu_policy_fingerprint.py` 与 `s20_single_probe.py` 全绿。若需严格 `pre-commit run --files`，请在装有 pre-commit 的环境复核一次（预计零改动）。

**接入 `serve_host_dflash2.sh` 的具体一行**（建议插在 L124 `VLLM_GFX906_ATTN_PREFILL_SEGMENTS` 之后、`MODEL=` 之前；由 Lead 落盘）：

```bash
"${PY}" "${REPO}/tools/dcu/dcu_policy_fingerprint.py" --prefix "[host-serve] policy: " 2>&1 || true
```

启动日志里得到（`|| true` 保证指纹失败不阻塞服务；`2>&1` 把 clamp/dead-knob WARN 一并落日志）：

```text
[host-serve] policy: FP 7e70fc2c130b9191
[host-serve] policy: FP_FULL FP16_PREFILL_DEFAULT_MIN_M=4096 FP8_GEMV_MAX_M_W8A16=8 VLLM_DCU_BF16=on ...
```

**关键算法（评审要点）**：

1. **`vllm/envs.py` getter 优先、但零卡**：用 `importlib.util.spec_from_file_location` 单独执行 `vllm/envs.py`（14 ms，只依赖 stdlib），22 个 envs 内旋钮直接取 getter；`vllm/__init__.py` 会 `import torch` 并加载 DTK 库，因此**不 import `vllm` 包**。getter 不可用时退化为内置 replica（`--no-envs-module` 可强制），selftest 断言两者在 22 个键上逐个相等。
2. **不在 `envs.py` 的旋钮按消费者复刻**：83 个（含 ATTENTION/GDN/QWEN35/C++ `getenv`）逐条 `src=path:line` 标注解析位置；`_tri_state`、`std::atoi`、`!= "0"`、严格 `== "1"`、`int() if raw else default` 等语义分开建模（这些差异是有效值事故的常见来源）。
3. **主开关组合**：`VLLM_DCU_USE_INT8_SDOT4/FP8_EMU/BF16_GEMV` 的生效值是 `gfx906_common.gfx906_emu_enabled()` 的组合（显式三态优先，否则 `VLLM_DCU_USE_GFX906_EMU`），**不是** `envs.py` 的单键 bool——`USE_FP8_EMU=1` 在生产 env 中最终显示 `1` 即来自该组合。
4. **编译期 caps 入指纹**：`_GEMV_MS_MAX=8`、`kMsMax=8`（`w8a8_sdot4.cu`、`fp8_emu_gemm.cu`）、`GFX906_VERIFY_MAX_QUERY_LEN=8`、`GFX906_MAX_DECODE_SEGMENTS=128`、`_FORCE_MIN_M=16`。selftest 用正则回读源码，**cap 漂移直接 FAIL**（不是靠文档记）。
5. **模型 config 依赖的默认值显示 `auto`**：`VLLM_QWEN35_{CHUNK_DECODE,PACKED_DECODE,PACKED_DECODE_TILED_QK,TILED_QK_EXPAND}` 的默认由 `split_projections` 决定，启动前不可知，显示为 `auto` 并在 `--json` 的 `note` 写明；未设 env 时不猜值。

**给 Lead 的 4 个发现（均为源码事实，非性能结论）**：

1. `VLLM_DCU_FP8_CVT` 是**死旋钮**：全仓无消费者（`vllm/` + `csrc/` + `tools/dcu/` 均只有 `envs.py` 定义；`docs/design/z100-rocm101-perf-p3-*.md` 亦标注"死旋钮"）。设了不会改变任何行为，工具会 WARN。任何 A/B 若把它当变量，该臂作废。
2. `VLLM_DCU_INT8_GEMV_MAX_M > 8` 是**潜在启动失败**：python 侧不 clamp（`gfx906_sdot4.py:82`），而 kernel `TORCH_CHECK(M <= kMsMax=8)`（`w8a8_sdot4.cu:394`）——`=16` 时 M∈[9,16] 会被派到 GEMV 然后直接报错。工具保留 requested 值并 WARN（与 FP8 的 clamp 行为不同，勿混用）。
3. `VLLM_DCU_FP8_GEMV_MAX_M` 的**两条路由阈值并不完全同源**：python W8A8 派发是 `min(max(env,0),8)`（`=0` ⇒ 0），C++ W8A16 分支是 `env==null?8 : max(1,min(v,8))`（`=0` ⇒ 1）。`FP_FULL` 同时给出 `VLLM_DCU_FP8_GEMV_MAX_M`（W8A8 口径）与派生 `FP8_GEMV_MAX_M_W8A16`（W8A16 口径）；`=0` 时两者会不同，`=1/8` 时一致（生产用后者）。
4. 生产 env 的 `VLLM_DCU_GCN_ARCH=gfx906` 来自 `tools/dcu/env.sh:79` 而不是 `serve_host_dflash2.sh`。用 `env` 快照做 C4 出处时别漏掉 `env.sh` 注入的变量（本报告已含）。

### WP0.4-5. 反证条件（什么数据会推翻这个结论）

- 若**同一份 env 两次运行指纹不同** ⇒ 工具非确定性（当前 selftest/实测均相同；若出现，先查 `os.environ` 是否被 shell 在两次间改变）。
- 若**改了有效旋钮指纹不变** ⇒ 该旋钮未被采集或解析错误；覆盖扫描（`--selftest` 的 coverage check）会列出所有"源码有、registry 无"的名字。
- 若**两个 env 指纹相同但服务行为不同** ⇒ 说明差异落在五类前缀之外（例如 `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`、`VLLM_ROCM_*`、模型/`SPEC`/CLI 参数、`.so` 版本、或 `auto` 类模型 config 依赖项）。这是本工具的**已知边界**：指纹只覆盖约定的五个前缀 env + caps；必须与 `git_sha`、完整 env、启动命令行一起引用（`--json` 给 `git_sha`/`git_dirty`）。
- 若某旋钮**被 clamp/忽略但指纹仍显示 requested 值** ⇒ 说明 post 规则缺失。当前 `=16` 的 `FP_FULL` 与 `=8` 逐字节相同即为反证通过。
- 若源码 cap 常量被改 ⇒ `--selftest` 的 `compile-time caps match source` FAIL（正则回读），不会静默沿用旧数字。

### WP0.4-6. 回滚方式（一条命令）

```bash
rm tools/dcu/dcu_policy_fingerprint.py
```

serve 脚本回滚：删掉接入的那一行即可（本报告未改 `serve_host_dflash2.sh`）。`vllm/` 下零改动。

> **交付注意（给 Lead，需在提交前处理）**：`.gitignore:313` 的整目录规则 `tools/dcu` 会隐藏该目录下**新增**文件（`git check-ignore -v tools/dcu/dcu_policy_fingerprint.py` 命中；`git add` 需 `-f`）。后续 WP1.1 的 `tools/dcu/s27_graph_launch.sh` 也会被吞。建议二选一：提交时 `git add -f tools/dcu/dcu_policy_fingerprint.py`，或把 `.gitignore:313` 的 `tools/dcu` 改成 `tools/dcu/*`（保留既有 `!tools/dcu/**` 白名单意图，且不动已跟踪文件）。本属主按 C6 未改 `.gitignore`。

---

## WP0.5 —— 探针加 `--prompt-file`（`tools/dcu/s20_single_probe.py`）

### WP0.5-1. 一句话结论

`s20_single_probe.py` 已加 `--prompt-file PATH`（与 `--prompt-tokens` argparse 互斥）：读 UTF-8 文档、用**服务端模型的 tokenizer 离线副本**（`transformers.AutoTokenizer` 优先，纯客户端 shell 下自动退化为 Rust `tokenizers` 读同一份 `tokenizer.json`）计数，并把**实际 token 数**打到 stdout、写进 summary JSON 的 `prompt_tokens_actual`、逐 run 记录，且与 server `usage.prompt_tokens` 自动交叉校验；8 KiB 冻结切片 = **3165 tokens**（两个 backend、两次运行一致），7.1 KiB 真实文档 = **2783 tokens**；新增 `--truncate-tokens`（WP1.3 的截断扫描前置）与 `--dry-run`（零卡自检，完全不连服务）；**不传 `--prompt-file` 时 prompt 构造/请求体/summary 键集与改动前一致**（mock server 对照实测）。

### WP0.5-2. 自检矩阵（替代 A/B 表：本 WP 无数值性能臂，只有计数一致性）

输入均为冻结副本（sha256 见 §WP0.5-3），命令统一为
`.venv/bin/python tools/dcu/s20_single_probe.py --prompt-file <输入> --dry-run [--truncate-tokens N] [--tokenizer-backend B]`：

| 输入（bytes / sha256[:16]） | `--truncate-tokens` | backend | 实际 chars | `tokens_actual` | 两次一致 | 备注 |
|---|---:|---|---:|---:|---|---|
| `probe-8k.txt`（8192 / `ab72ea5df3a4d388`） | 0 | transformers | 5667 | **3165** | ✓ | BUILD 文档前 8192 字节（UTF-8 协商截断） |
| `probe-8k.txt` | 0 | tokenizers | 5667 | **3165** | ✓ | 与 transformers 逐 token 一致 |
| `probe-8k.txt` | 256 | tokenizers | 498 | **256** | — | 截断按 token id，重新编码闭合 |
| `probe-8k.txt` | 1024 | transformers / tokenizers | 1942 | **1024** | ✓ | 两 backend 一致 |
| `probe-8k.txt` | 2048 | tokenizers | 3687 | **2048** | — | WP1.3 的 2K 档可用 |
| `probe-8k.txt` | 4096 | tokenizers | 5667 | 3165（`truncated=0`） | — | 文档不足 4096，不截断 |
| `probe-7282b.md`（7282 / `70a2a8e24eb1b04a`） | 0 | transformers / tokenizers | 5601 | **2783** | ✓ | `docs/design/z100-p0-retest-plan.md` 的冻结副本 |
| `docs/design/z100-decode-prefill-optimization-construction.md`（20508 chars / `de0845c34b857685`） | 0 | auto（两种环境） | 20508 | **10936** | — | 活文档，按测量时 sha 记录 |

> 证据：`.s27/b-fingerprint/wp05-selftest.log`、`wp05-frozen-inputs.log`、`inputs-and-tokenizer.sha256`。

### WP0.5-3. env 全文 + 策略指纹 + 树 SHA

- 树 SHA：`7078b5fe137feeab3750ee243daeaf172dc439d8`（分支 `dtk`）。
- 被测策略指纹（本 WP 不启服务，取 WP0.4 的 bare env 指纹）：`FP 6eab564bf1ccdb91`（env 全文即"未 source 生产 env"的默认态；完整 `FP_FULL` 由 `dcu_policy_fingerprint.py` 可复现）。真正跑探针时的服务臂 env + 指纹由 Lead 在 A/B 报告中引用（WP0.1/WP1.x）。
- tokenizer 资产（服务端口径，`/mnt/game/models/Qwen3.8-27B-FP8`）：
  - `tokenizer.json` sha256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
  - `tokenizer_config.json` sha256 `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`，`model_max_length=262144`，`add_bos_token=false`/`add_eos_token=false`（故 `add_special_tokens=True/False` 计数相同）
- 工具链：Python `3.10.12`（`.venv/bin/python`）、`transformers 5.19.0`、`tokenizers 0.23.3`。
- 运行环境：`--dry-run` 在**裸 shell**（无 DTK 库）即可跑（`transformers` 会 `ImportError: libgalaxyhip.so.5`，自动退化 tokenizers）；要强制 transformers 口径时 `export DTK_ROOT=/mnt/game/host-runtime/dtk2604 && . tools/dcu/env.sh`。两种环境下 3165/2783 完全一致。
- **未启动、未连接任何 GPU 服务**：所有自检走 `--dry-run`，集成路径用 monkeypatch 的假 server（无真实 socket）；真实 `/v1/models`/`/metrics` 路径由 Lead 在 live 服务上验证。

### WP0.5-4. 判读（对照 WP0.5 阈值）

| 验收项（BUILD §4/WP0.5 + 任务板） | 结果 | 证据 |
|---|---|---|
| 给 8 KB 文本能报出 token 数 | **过** | 8192 B → `tokens_actual=3165`（stdout + `.log`） |
| 与离线计数一致 | **过** | `transformers.AutoTokenizer` 与 `tokenizers.Tokenizer`（同一 `tokenizer.json`）在 4 组输入上计数逐值相同；与 `/tokenize` 的服务端口径由**运行时 `usage.prompt_tokens` 自动校验**（不一致会逐 run WARN；mock 注入 9999 时确实 WARN） |
| 同一文件两次一致 | **过** | 3165/3165、2783/2783 |
| 实际 token 数写入 summary JSON | **过** | `summary.prompt_tokens_actual` + `summary.prompt_file`；每 run 也有 `prompt_tokens_actual`（mock 实测：3165） |
| max prompt token 不超过服务 max_model_len | **过** | `/v1/models` 取 `max_model_len`：`tokens>=max` **拒绝发送 exit 2**（`--allow-over-max-len` 才放行）；`tokens+max_tokens>max` WARN；端点不可达 WARN 并继续（mock 四种路径均实测） |
| 加法式、不改默认行为 | **过** | 不传新参数：prompt 仍等于 `build_prompt(--prompt-tokens)`（mock 逐字节比对通过）、summary 键集仍为原 10 个键、run 记录无新键、请求 body 字段未变 |
| 错误路径不 traceback | **过** | 互斥参数、`--dry-run` 无文件、缺文件、空文件、坏 tokenizer 全部 exit 2 + 一行 `[probe] ERROR`（实测） |
| 六段模板 | **过** | 本节 1–6 |

**用法**（服务由 Lead 启动后；8K 文档的完整 doc 现在约 10936 tokens）：

```bash
# 0) 零卡：只报 token 数（不连服务）
.venv/bin/python tools/dcu/s20_single_probe.py --prompt-file <doc> --dry-run
# 1) 真实文档整篇（WP0.5 判读）
.venv/bin/python tools/dcu/s20_single_probe.py --prompt-file <doc> \
    --max-tokens 96 --repeat 2 --tag real-20k --out .s27/p0
# 2) 同一文档截到 2K/4K/8K（WP1.3 的上下文扫描）
.venv/bin/python tools/dcu/s20_single_probe.py --prompt-file <doc> \
    --truncate-tokens 2048 --max-tokens 96 --repeat 2 --tag real-2k --out .s27/p0
# 3) 强制服务端口径的 tokenizer 实现（默认 auto 已优先 transformers）
.venv/bin/python tools/dcu/s20_single_probe.py --prompt-file <doc> \
    --dry-run --tokenizer-backend transformers
```

stdout 新增行只在 `--prompt-file` 模式出现：

```text
[probe] tokenizer backend=transformers path=/mnt/game/models/Qwen3.8-27B-FP8
[probe] prompt-file=<doc> chars=5601 tokens_actual=2783 truncated=0 tokenizer_max_len=262144
[probe] server max_model_len=262144
```

summary JSON（仅 `--prompt-file` 模式追加这两个键，其余不变）：

```json
"summary": {"prompt_tokens_actual": 2783, "prompt_file": "<doc>", "...": "..."}
```

**实现说明（评审要点）**：

1. token 数取自**服务端 tokenizer 的离线副本**（`--tokenizer` 默认 `/mnt/game/models/Qwen3.8-27B-FP8`），不假设 `chars/4`；`add_special_tokens=True` 与服务端 completion 路径一致（该模型 `add_bos/eos=false`，两种取值计数相同）。
2. `--truncate-tokens N` 是**按 token id 截断后 decode 回文本**，再重新计数；`tokens_actual` 永远是重编码后的实际值（当前 tokenizer 上 N≤3165 全部精确闭合：256/1024/2048 实测等于 N）。WP1.3 因此可直接用它做同源截断。
3. 与 server 的 token 一致性用**同一次请求的 `usage.prompt_tokens`** 校验（不需要额外打 `/tokenize`，也不会引入第二条请求扰动）；`--dry-run` 完全不发请求。
4. `--dry-run` 只加载 tokenizer（~2 s transformers / ~0.2 s tokenizers），退出码 0 可用于 G2 的零卡 smoke。

### WP0.5-5. 反证条件（什么数据会推翻这个结论）

- 若 live 服务出现 `[run i] WARN client tokens=X != server usage.prompt_tokens=Y` ⇒ 客户端 tokenizer 与服务端不一致（例如服务用了不同 tokenizer 资产或 `add_special_tokens` 语义不同）。此时**不得**用 `--truncate-tokens` 的数字做跨臂 acc 比较，先对齐 tokenizer。
- 若两个 backend 计数不同 ⇒ `tokenizer.json`（fast）与 `vocab.json`+`merges.txt`（slow）资产不一致；以 transformers（服务端口径）为准，并记录两者差异。
- 若 `--truncate-tokens N` 后 `tokens_actual != N` ⇒ decode→encode 不闭合（特殊 token/unicode 边界）。当前 4 组输入全闭合；一旦出现，应改为按 token id 直接发 prompt（需 vLLM 扩展），而不是继续用 decode 近似。
- 若 8 KB 的 token 数与其他环境不同 ⇒ 正常：数字绑定 tokenizer sha（§WP0.5-3）；跨 tokenizer 版本的 token 数不可比。
- 若真实文档跑出的 `usage.prompt_tokens` 与 `--dry-run` 数字一致但 acc 与合成 prompt 不可比 ⇒ 这正是 WP0.5 的目的（PROJ §1：acc 只在同一 prompt 源下横向可比），不是反证而是判读前提。

### WP0.5-6. 回滚方式（一条命令）

```bash
git checkout -- tools/dcu/s20_single_probe.py
```

行为改动是纯加法：新增 `--prompt-file`/`--tokenizer`/`--tokenizer-backend`/`--truncate-tokens`/`--dry-run`/`--allow-over-max-len` 参数与 4 个 helper、以及 `main()` 里 `tokens_actual is not None` 的分支；旧路径（`--prompt-tokens` 合成 prompt）在 mock 对照下 prompt、请求体、summary 键集逐项不变。
说明：本次同时用 `ruff format` 规范化了该文件**原有**的 15 行超长/格式差异（baseline 该文件本来就过不了 `ruff check`/`ruff format`，G2 要求全绿）；若要求零格式噪音，可只回滚格式 hunk，但 G2 会重新变红。

