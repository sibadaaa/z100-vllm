# Z100 gfx906 BF16 入口层 fp16 化:硬件验证与改造记录

> 日期:2026-10-20。
> 环境:4× Hygon DCU(2× Z100L + 2× Z100,全部 `gfx906:sramecc+:xnack-`),
> DTK 26.04 容器(`z100-triton`,torch 2.10.0+das / hip 6.3.26093),与
> [muwinds 博客 Z100 微基准测试](https://blog.muwinds.com/archives/373/)的环境同款。
> 背景:该博客把 "BF16 Matmul" 列为 Z100 的测试项并给出了结果,容易被读成
> "Z100 有原生 BF16 硬件"。本文先给硬件层验证结论,再记录按用户指示完成的
> **入口层 bf16 → fp16 改造**(不动 C++、不需要重编)。
> 关联文档:[`z100-fp8-w8a16-implementation.md`](z100-fp8-w8a16-implementation.md)、
> [`z100-fp8-emu-fp16-conversion-plan.md`](z100-fp8-emu-fp16-conversion-plan.md)。

---

## 0. 结论(三句话)

1. **gfx906 没有原生 BF16 硬件**:DTK 自带汇编器对 `v_pk_fma_bf16`、
   `v_mfma_*_bf16` 全部报 *"instruction not supported on this GPU"*,而
   `v_pk_fma_f16` / `v_dot2_f32_f16`(对照组)正常汇编。gfx906 连 MFMA
   矩阵单元都没有,FP16 的 2x 速率来自打包指令,不是张量核心。
2. **DTK 栈里能跑的 bf16 全是软件仿真**:rocBLAS 的 Tensile 库里 gfx906 的
   bf16 kernel 数量为 0,hipBLASLt 连 gfx906 的 tensile lib 都没有;实测
   bf16 GEMM 8.16 TFLOP/s,是三种精度里最慢的(fp32 10.52、fp16 18.16,
   4096³,Z100L)——原生 bf16 硬件(如 gfx908/MI100)应 ≥ fp16 速率。
   数值签名(对 fp32 参考最大相对误差 2.35e-3 ≈ bf16 输出半 ulp;1e4²×K=256
   累加 2.55e10 无 inf)表明它是 "bf16 存储 + fp32 数学" 契约。
3. **改造按"入口层"落地**:模型 dtype 在加载时强制转 fp16,FP8 走 W8A16,
   其余算子自然命中各自的 fp16 内核;内核里的 bf16 分支保留为 opt-in 死代码
   (`VLLM_DCU_BF16=1` 可回到仿真路径),因此**不需要重编 `_C`/`_rocm_C`**。

---

## 1. 硬件验证记录(2026-10-20 实测)

### 1.1 ISA 汇编器(DTK 26.04 clang,`--target=amdgcn-amd-amdhsa`)

| 指令 | gfx906 结果 | 说明 |
|---|---|---|
| `v_pk_fma_f16` | 汇编通过 | 原生 FP16 打包指令存在(对照) |
| `v_dot2_f32_f16` | 汇编通过 | 原生 FP16 dot2 存在(对照) |
| `v_pk_fma_bf16` | **error: instruction not supported on this GPU** | BF16 打包运算不存在 |
| `v_mfma_f32_16x16x16_bf16` | **error: instruction not supported on this GPU** | BF16 张量核心不存在 |
| `v_mfma_f32_16x16x16_f16` | **error: instruction not supported on this GPU** | gfx906 无任何 MFMA(对照:报错与 bf16 无关) |

注意报错是 *"not supported on this GPU"* 而非 *"unknown instruction"*:
助记符在 ISA 家族里存在(gfx908/MI100 一代引入 BF16),只是 gfx906 没有。

### 1.2 BLAS 库

- `/opt/dtk/lib/rocblas/library/`:gfx906 相关文件中 **bf16 命名 kernel 数量 0**,
  `TensileLibrary_lazy_gfx906.dat` 无 bf16 引用。
- hipBLASLt:**没有任何 gfx906 tensile lib**,每次 import torch 都会打印
  `rocblaslt warning: No paths matched /opt/dtk/lib/hipblaslt/library/*gfx906*co`。

### 1.3 torch 实测(cuda:0,Z100L,4096³,3s 稳态)

| 项 | 结果 |
|---|---|
| `torch.cuda.is_bf16_supported()` | `True`(软件栈口径:bf16 类型可用 = 仿真可用) |
| fp32 / fp16 / bf16 matmul | **10.522 / 18.161 / 8.156 TFLOP/s**(bf16 最慢 ⇒ 纯仿真) |
| bf16 存储回读 `1e20 / 3.0e38 / 1e-30` | 完全一致(指数域与 fp32 同宽) |
| `1e4 @ 1e4, K=256` | 2.55e10,0 inf / 0 nan(若走 fp16 累加早已溢出 ⇒ fp32 数学) |
| bf16 matmul vs fp32(同输入)最大相对误差 | 2.35e-3 ≈ bf16 输出舍入半 ulp(2⁻⁹)⇒ fp32 累加 + bf16 存储 |

两张非 L 版 Z100(HCU2/3)同样能跑通 bf16 matmul、数值正常;当时卡上有并发
负载,绝对吞吐不作为判据(ISA 结论是架构级的,Z100 与 Z100L 同为 gfx906 IP)。

### 1.4 代码佐证

`csrc/rocm/bf16_emu_gemv.cu` 头注释明确写着:bf16 lane 用 `v_lshlrev_b32 16`
精确加宽后走 fp32 FMA,即 "bf16 weights, fp32 math" 契约——这条 fork 就是因为
gfx906 无原生 bf16 才写了仿真 GEMV。

---

## 2. 改造内容(2026-10-20,全部 Python 入口层)

| # | 改动 | 位置 | 行为 |
|---|---|---|---|
| 1 | 新增 `gfx906_coerce_model_dtype()`,接入 `_get_and_verify_dtype` | `vllm/platforms/rocm.py`(`gfx906_coerce_model_dtype`),`vllm/config/model.py` | gfx906 上 bfloat16 请求(显式 `--dtype bfloat16` 或 bf16 checkpoint 的 auto)一律转 float16 + warning;权重加载时经 fp32 一次转换 |
| 2 | `VLLM_DCU_BF16` 默认 `auto` → **`off`**;空值/未知值也归 `off` | `vllm/envs.py`、`vllm/platforms/hygon_dtk.py`(`bf16_policy`) | 默认 fp16-only;`VLLM_DCU_BF16=1` opt-in 回到仿真 bf16;`=auto` 保留探针 |
| 3 | `VLLM_DCU_FP8_W8A16` 默认 `0` → **`auto`** | `vllm/envs.py` | **FP8 走 W8A16**:激活不量化、直接 fp16 进 kernel,权重保持 fp8;构建缺 w8a16 op 时自动降级 W8A8;`0`/`1` 可强制 |
| 4 | 其余算子走对应内核(无需代码) | — | 模型 dtype 已是 fp16:dense GEMV 命中 `bf16_emu_gemv` 的 fp16 dot2 路由、INT8 走 sdot4、conv1d/GDN/RMSNorm/rotary/quickreduce 全部命中各自 fp16 实例化 |
| 5 | GGUF 收口 | `vllm/model_executor/layers/quantization/gguf.py`(`get_supported_act_dtypes`) | gfx906(bf16 off)不再提供 bf16 激活 dtype;`VLLM_DCU_BF16=1` 时保留 |

设计取舍:按用户指示**只做入口层**,核内 bf16 分支(bf16_emu_gemv 的 bf16
route、`gfx906_decode_kernels.cu` 的 `at::BFloat16` 实例化、w8a8_sdot4 /
fp8_emu_gemm 的 BFloat16 分支、quickreduce bf16 codec)全部保留——默认配置下
不可达,但 `VLLM_DCU_BF16=1` 实验时仍然可用,且免去重编与回归。

同目录不属于 gfx906 的 bf16 代码未动:`skinny_gemms*.cu`(gfx906 构建已
`REMOVE_ITEM`)、`attention.cu` 的 MFMA bf16(gfx906 运行时不选)、
`q_gemm_rdna3*`(仅 gfx1100)。

---

## 3. 运行时行为矩阵

> 注:gfx906 的 FP8 仿真内核本身仍是 opt-in(`VLLM_DCU_USE_FP8_EMU=1` 或
> `VLLM_DCU_USE_GFX906_EMU=1`,仓库默认保持 off / 位稳定)。下表 "FP8 走
> W8A16" 指该内核启用后的路由默认由 `0` 改为了 `auto`。

| 配置 | bf16 checkpoint | FP8 checkpoint | fp16 checkpoint |
|---|---|---|---|
| 默认 | 加载转 fp16,全 fp16 内核 | FP8 内核启用时走 **W8A16**(激活不量化);未启用时保持原选择 | 原样 |
| `VLLM_DCU_BF16=1` | 仿真 bf16(GEMV 走 fp32 route,大 GEMM 走 rocBLAS 仿真,~8.2 TF) | bf16 层保持 W8A8 量化路径(fp8 emu 内核的 bf16 datapath) | 原样 |
| `VLLM_DCU_FP8_W8A16=0` | 同默认 | 回到 W8A8(激活量化 + LUT 加宽) | 原样 |
| `VLLM_DCU_FP8_W8A16=1` | 同默认 | 强制 W8A16,缺 op 或非 fp16 层直接报错 | 原样 |

---

## 4. 数值注意

- bf16 → fp16 加载转换是 torch 标准语义(经 fp32):|w| ≤ 65504 且 ≥ 2⁻¹⁴
  精确;更小的值落 fp16 subnormal;|w| > 65504 变 inf(LLM 权重实际不会出现)。
  这与业界 fp16 推理的加载口径一致,与"内核里把 bf16 按 fp16 重解释"
  (bf16_emu_gemv 的 `route=2` 无 range_safe)**不是一回事**,后者对 σ~1e-3 的
  权重分布会因 2⁻¹⁴ flush 损失约 6% 信号,勿混淆。
- `--dtype auto` 的首选仍是 fp16(`_rocm_supported_dtypes` 中 fp16 恒在首位);
  转换只影响"请求了 bf16"的场景。
- 博客口径修正:"BF16 Matmul 能跑、数值正确"为真,但那是软件仿真;把它读成
  "Z100 有原生 BF16 加速"是错的——bf16 GEMM 比 fp32 还慢 ~22%,比 fp16 慢 ~2.2x。

---

## 5. 验证记录

- `tests/platforms/test_hygon_dtk.py`:**53 passed**(含本次新增:默认 off、
  `gfx906_coerce_model_dtype` 三态、`("", probe)` 参数化改判 off)。
  唯一失败 `test_gcn_arch_falls_back_from_torch_to_kfd` 经 `git stash` 对比确认
  在改动前的 HEAD 上同样失败(容器 python 3.10 跑 3.11+ 源码树的预存环境问题,
  `typing.Never/Self/assert_never` 均不在 3.10),与本次改动无关。
- `ruff check` / `ruff format --diff`:新增行零问题(这批文件剩余的 lint 项均为
  预存;repo pin 的 pre-commit ruff 与本地 0.17.1 存在风格差异)。
- 全部改动 py_compile 通过;纯 Python,**重启进程即生效,无需重编扩展**。

---

## 6. 回滚

| 目标 | 操作 |
|---|---|
| 回到仿真 bf16(不转 fp16) | `VLLM_DCU_BF16=1` |
| FP8 回到 W8A8 量化路径 | `VLLM_DCU_FP8_W8A16=0` |
| 完全回退本次改动 | revert `vllm/envs.py`、`vllm/platforms/{rocm,hygon_dtk}.py`、`vllm/config/model.py`、`vllm/model_executor/layers/quantization/gguf.py`、`tests/platforms/test_hygon_dtk.py` 中本次 bf16/W8A16 相关 hunk |

---

## 7. 运维记录(2026-10-21):`VLLM_DCU_FP8_W8A16=1` 上线

### 7.1 落地

- `tools/dcu/serve_host_dflash2.sh` 增加常驻
  `export VLLM_DCU_FP8_W8A16="${VLLM_DCU_FP8_W8A16:-1}"`(调用方可覆盖)。
  目标模型 `--dtype float16`,W8A16=1 的两个硬前提(构建含
  `fp8_emu_gemm_block_w8a16` op、激活全 fp16)在启动前已验证。
- 服务:`z100-vllm-host.service`(transient unit,启动脚本
  `serve_host_dflash2.sh`,Qwen3.8-27B-FP8 + DFlash2 k=7,TP4,262144)。
  注:该 unit 带 `CollectMode=inactive-or-failed`,失败即被 systemd 回收,
  重建用 `tools/dcu/s24_post_l2v2.sh` 里的同款 `systemd-run` 命令。

### 7.2 启动 OOM 事件与定位(与 bf16→fp16 改造无关)

重启后 6 次启动全部失败:worker 在 `RequestState` 的
`UvaBuffer((256, 262144), int32)`(256 MB pinned)分配处
`hipErrorOutOfMemory`。排查结论:

| 排除项 | 证据 |
|---|---|
| W8A16=1 改动 | `=auto` 同样失败(二分);失败点在内核选择之前 |
| bf16→fp16 入口改造 | 失败路径不经过 dtype 解析;同代码 `07:05/07:38`(本地)刚成功启动过 |
| 代码差异 | 失败栈涉及的全部文件 mtime 早于上次成功启动;git 树仅多本篇文档相关提交 |
| GPU 显存 | HIP 视角 31.72/31.98 GiB free,单卡连续分配 8/16/20/24 GiB 全部成功 |
| 锁页/GTT | 独立进程 pin 1 GiB、精确复现 (256,262144) 形状均成功;GTT sysfs 全程 0;`Mlocked` 27 MB;无孤儿 GPU 进程 |
| RCCL | 4 rank TP4 NCCL 初始化 + 各 1 GiB pin 复现探针全部成功 |

结论:对长驻实例执行 SIGTERM(30 s)→SIGKILL 之后,rock 驱动层的
worker 可用 pinned 预算出现对计数器不可见的收缩;fresh 进程不受影响,
vLLM worker 的 (256, 262144) int32 pinned 分配确定性失败。
昨天 22:27-22:34 的 3 连败后成功与此为同一现象。

### 7.3 处置(workaround)

- `MAX_SEQS` 写入 `serve_host_dflash2.sh` 为默认 **4**(调用方仍可用
  `MAX_SEQS=n` 覆盖):`RequestState` 的 pinned 降为 (4, 262144) int32 =
  4 MB,worker 全部通过,服务完整起来:262144 上下文、KV cache
  806,753 tokens、W8A16=1 生效(=1 硬模式干净启动即代表全部 FP8 层
  走 W8A16;若 op 缺失或层非 fp16 会在启动期直接 raise)。冒烟请求正常。
  (中间曾以 64 验证过路径,后按操作员指示收敛到 4。)
- **代价**:`max_num_seqs` 256 → 4(服务端并发上限;对 262144 全长请求
  本就只有 ~3x KV 并发,主要影响短请求并发)。
- **恢复大并发**:需要 root 侧动作(hymgr/driver reload 或重启主机)后,
  用 `MAX_SEQS=n` 覆盖重新拉起即可;本次会话无 sudo,未执行。
