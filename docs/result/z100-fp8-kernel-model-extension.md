# 自研 FP8 内核（gfx906 / Z100）能拓展到哪些模型

> 日期：2026-10-19 · 范围：`/home/lyh/z100-vllm`（vLLM gfx906 fork）
> 方法：只读取证 + 预算测算（未改代码、未跑 GPU 负载；生产 27B-FP8 TP=4 服务全程未触碰）
> 取证附件：`.s25/fp8-kernel.md`（内核能力边界 338 行）、`.s25/fp8-llm-models.md`（LLM 目录 190 行）、`.s25/fp8-diffusion.md`（扩散/视频 337 行）
> 关联：`docs/result/z100-minimax-h3-quant-feasibility.md`、`docs/result/z100-fp8-emu-fixed-time-investigation.md`

## 0. 结论摘要（先看这 12 条）

1. **这套内核的能力边界很硬也很窄**：真 **W8A8**（A 必须 fp8，且与 W 同 dtype）、**e4m3fn**（`e5m2` 内核支持但主机侧未接线）、scale 支持 **per-tensor / per-channel / 128×128 block**、M 任意（GEMV ≤8 / tiled GEMM）、权重**零拷贝**消费 checkpoint 布局、out 支持 fp16/bf16/fp32（§1）。
2. **它能覆盖的"新增模型面"只有两类**：**稠密 LLM 的 FP8 checkerpoint**（27B 已验收，32B 是最高价值空白）与**扩散/视频的 Comfy `fp8_scaled` 权重**（H3 的契约逐条对齐）。**MoE / MLA / FP4 / W8A16 / e5m2 / SAGE 全部吃不到**（§3、§4）。
3. **速度现状（"还在优化"的准确口径）**：decode 已从"不可用"打到"比 fp16 rocBLAS 快"（M=1 **3.75×**；端到端 1984.8 → **123.0 ms/token = 16.1×** vs Triton 标量回退）；prefill 大 M 已到 **10.2 TFLOP/s/卡**（自测峰值 ~20；M=512 比原生 bf16 GEMM 快 1.5×）。但**每步 69–72% 的时间与 M 无关**，且 c=1 步进只有 38% 是 GEMM（§2）。
4. **FP8 的真增量是"容量 + 吃生态"，不是算力**：1 B/参把 TP=2 的上限从 ~27B 推到 ~32B（再往上顶)；同参数量 decode 只比 fp16 快 ≈1.3×。**"16.1×"是对 Triton 标量回退的倍数，不能当"比 fp16 快 16 倍"用**（§5）。
5. **31 项 FP8 量化入口里，8 条最终都汇到这两个 kernel**：`fp8` per-tensor/block-128、compressed-tensors W8A8 FP8、modelopt FP8、quark W8A8 FP8、online `fp8_per_{tensor,block,channel}`；`Gfx906Fp8{ScaledMM,BlockScaledMM}LinearKernel` 分别在 ROCm 候选表**第 0 位**（`vllm/model_executor/kernels/linear/__init__.py:476`、`:515`），开关 `VLLM_DCU_USE_FP8_EMU=1`（默认 OFF）。
6. **最有价值的下一个目标：Qwen3-32B-FP8**（TP=2 on 卡 0/1，16.0 GiB/卡权重 + 4.0 GiB KV@32k ≈ 20 GiB；131k 需 KV fp8）。这是"零内核工作、纯配置+加载"就能拿下的唯一高价值空白（§3.1）。
7. **MoE 是最大的"吃不到"**：`select_fp8_moe_backend` 在 gfx906 上 aiter 死、cutlass/deepgemm CUDA-only ⇒ 落 `Fp8MoeBackend.TRITON`，而 `supports_fp8()` 因 `on_cdna()` 匹配 `"gfx9"` 子串**意外返回 True**，把全部 fp8 组合放行给 Triton（同类路径实测退化 ~36×）。`fp8_emu_*` **没有 grouped-GEMM 变体**（§3.2、R2）。
8. **扩散/视频是本次最大的新增域**：H3 官方 `minimax_h3_fl2va_pruned_fp8_scaled`（20.96 GB）的量化契约 = **per-tensor weight scale + per-tensor 激活**，与本内核 per-tensor 分支**逐条对齐**（含零拷贝 `[N,K]` 连续、标量 scale 走 stride 0、epilogue 顺序一致）。字节级证据：int8 比 fp8 大 12,174,008 B = 4 B × 3,043,502 行 ⇒ int8 是 per-row scale，fp8 是 per-tensor（§4）。
9. **H3 用 FP8 的定位**：TP=2 每卡 21.3 GiB（DiT 9.76 + TE 6.10 + VAE 5.41），(c) 档临界 ⇒ 必须**分阶段装载**；算力 10.2 vs GGUF 7.3 TFLOP/s（1.4×），但**画质有官方门禁**（vLLM-Omni 已合并 PR：LPIPS 0.1156 ≤0.20、PSNR 23.63 dB、音频 cos 0.9589、峰值 −22%），且**消掉 GGUF 路线的两个高风险**（无元数据、ConvRot 坐标系未知）。⇒ 定位为**"FP8 主画质档 + GGUF 低显存档"并行**，不是替换（§4.3）。
10. **KV cache fp8 是"门禁放行但零实测"**：`_is_supported_kv_cache_dtype` 只查 capability≥89，gfx906 被算成 (9,0)→90 而通过；fork 的 attention op 只接受 per-tensor 档，`fp8_per_token_head` 会直接 `NotImplementedError`。它对**长上下文**是唯一让 32B 跑到 131k 的杠杆，但**不能当既得能力**（R3）。
11. **三个"低成本解锁"的接线缺口**（当前纯属没接线，不是不能做）：`e5m2` 权重（内核支持、主机侧恒 e4m3fn ⇒ 被 TORCH_CHECK 拒）、`FP8 embedding/lm_head`（`Fp8Config` 不处理，vocab 权重大约 1.2–2.5 GB 仍 fp16）、`PREFILL_CFG=auto` 默认化（N=3584:5120 实测快 1.40×）。
12. **不要把 FP8 与 INT8 sdot4 混为一谈**：两者都是 1 B/参、同档性能（INT8 上限更高：22.2 vs 10.2 TFLOP/s），但 **FP8 的独占价值是"直接吃生态里现成的 FP8 checkpoint"**（Qwen3 全系 / DeepSeek 原生 / RedHatAI-dynamic / ModelOpt / Quark / Comfy fp8_scaled），而 INT8 需要本地再量化 + 数值再验收，且 **sdot4 会改 greedy token**。

---

## 1. 内核能力矩阵（决定"能不能吃"的全部条件）

| 项 | 结论 | 证据 |
|---|---|---|
| 计算语义 | 真 **W8A8**：A 必须 fp8 且与 W 同 dtype；无 fp16 输入接口 | `csrc/rocm/fp8_emu_gemm.cu:1217-1220`、`:1349-1350` |
| fp8 码表 | kernel 级 **e4m3fn 与 e5m2** 都支持（256 项 LUT 无损加宽）；**主机侧只接线 e4m3fn** ⇒ e5m2 权重被 `TORCH_CHECK` 拒 | `csrc/rocm/gfx906_emu_common.cuh:55-73`；`vllm/platforms/rocm.py:1342-1351` |
| MAC/累加 | 无 fp8 指令：fp8 → fp16 走 256×fp16（512 B）LDS 表 + `v_dot2_f32_f16` fp32 累加 | `fp8_emu_gemm.cu:4-6`；`gfx906_emu_common.cuh:48-53` |
| scale 粒度 | ① per-tensor（a `[1]`/w `[1]`）② per-token × per-channel（a `[M]`/w `[N]`）③ block-scaled（a `[M,K/block_k]`、w `[cdiv(N,block_n),K/block_k]`，`block_k%32==0 && K%block_k==0`；**实测只覆盖 128×128**） | `fp8_emu_gemm.cu:1230-1245`、`:1366-1374`、`:1381-1387` |
| M 范围 | GEMV 入口 1..8（阈值 `VLLM_DCU_FP8_GEMV_MAX_M`）；GEMM 入口任意 M；block 路径 M=5..8 会改投 tiled GEMM | `scaled_mm/gfx906_fp8.py:65`、`fp8_emu_gemm.cu:1545-1546` |
| 形状/布局 | K%4==0 即可（GEMM 尾部有掩码）；N 任意；A 必须 contiguous；W 只要 `stride(1)==1`（padding 后 `stride(0)=K+256` **只有 block 路径支持**）；per-tensor 路径 W 必须完全 contiguous | `fp8_emu_gemm.cu:1225-1227`、`:1360-1361` |
| 精度出口 | out ∈ {fp16, bf16, fp32}；bias 必须与 out 同 dtype | `fp8_emu_gemm.cu:1234-1243` |
| 选核位置 | per-tensor/block 两个 kernel 都是 `_POSSIBLE_FP8{,_BLOCK}_KERNELS[ROCM]` **第 0 位** | `vllm/model_executor/kernels/linear/__init__.py:476`、`:515` |
| 开关 | `VLLM_DCU_USE_FP8_EMU`（显式优先）/ `VLLM_DCU_USE_GFX906_EMU`（总开关）；**默认 OFF**；OFF 时 per-tensor FP8 直接失败、block 回 Triton（实测慢 16.1×） | `scaled_mm/gfx906_common.py:39-64` |
| 不能吃 | W8A16 weight-only FP8（ROCm 候选表空）、MXFP8（退化成 bf16 反量化）、FP4/NVFP4 全家、FP8 MoE、FP8 attention | `.s25/fp8-llm-models.md:§2`、`.s25/fp8-kernel.md:§4` |

---

## 2. "速度还在优化"的定位（哪些优化是提速、哪些是解锁）

### 2.1 现状数字（全部来自既有实测，未新跑）

| 场景 | 数字 | 出处 |
|---|---|---|
| decode M=1（17408×5120） | 0.220 ms @ **405 GB/s** = **3.75×** fp16 rocBLAS | `docs/archive/result/z100-11f-fp8-bf16-emu-kernels-result.md:166` |
| 端到端 27B-FP8（Triton 回退 → 本内核） | 1984.8 → **123.0 ms/token（16.1×，0.50 → 8.1 tok/s）** | 同上 `§9.4` |
| prefill 大 M | M=512 **12.837 ms** vs 原生 bf16 GEMM 19.149 ms（1.5×）；等效 **~9–10 TFLOP/s/卡**（峰值 ~20） | `.s20/S20C_LOWCONC_OPT.md:§3,62` |
| DiT 形状（N=3584/K=5120/M=7920） | cfg10 **28.40 ms** = **10.23 TFLOP/s**（cfg5 39.69 ms） | `scaled_mm/gfx906_fp8.py:71-84` |
| 生产每步（c=1，256 次调用） | fp8-emu GEMM 42.8–44.9 ms，其中 **69–72% 与 M 无关** | `.s23/fp8_emu_kernel_fixed_times.md:235-256` |
| 每步 kernel 数 | **512 = 256 quant + 256 emu** | `.s23/S23_OPTIMIZATION_PLAN.md:§1` |
| host 派发税 | 裸 op 12–23 µs/次、custom_op 43–83 µs/次 ⇒ **11–19.7 ms/步** | `docs/result/z100-fp8-emu-fixed-time-investigation.md:§0.1` |
| c=1 步进构成 | 194–217 ms 里算子仅 ~38%，其余是 all-reduce 屏障 + 派发气泡 + rank 偏斜 | `.s20/S20_KERNEL_RESULT.md:59-83` |

### 2.2 优化项分类（这条对"拓展模型"很关键）

| 类别 | 项目 | 是否解锁新模型 |
|---|---|---|
| **纯提速** | O1 去 custom_op 包装税（−8~14 ms/步）、O2 量化 host 包装下沉（−8.5 ms）、O3 tile 拦截项（−14 ms）、O4 streaming 带宽塌缩修复（116 → 736 GB/s）、O5 量化+GEMM 融合（少 256 次 launch）、O7 CUDA graph 复核 | ❌ 只影响同参数量的 tok/s |
| **解锁新模型** | ① **fp8 grouped-GEMM（MoE）** ② **KV cache fp8 打通并实测** ③ **e5m2 接线** ④ **FP8 embedding/lm_head** ⑤ 大 M 的 cfg10/auto 默认化（DiT 场景） | ✅ 见 §3 |

> 一句话：**速度优化的收益是"现有 27B 更顺"，模型面的拓展要靠 ①–④ 这四项新能力**。①的工作量是内核级（tile kernel 上 expert 分段 + permute），②③④ 是接线+验证级。

---

## 3. 能拓展到的模型清单

判定口径：**TP=2 on 卡 0/1（仅这两张无 DP4X 缺陷）**，每卡 ≤26 GiB 舒适 / ≤30 GiB 极限；KV 公式 `L_attn × (n_kv/TP) × 2 × head_dim × 2 B × max_len`（已被 27B TP=4 实测 16.3 KiB/token/卡 反证）。

### 3.1 稠密 LLM：FP8 的真增量区

| 模型 | 量化形态 | FP8 体积 | 每卡权重 | +KV@32k | 判定 | 需要什么 |
|---|---|---:|---:|---:|---|---|
| **Qwen3.8-27B-FP8**（本地已验收） | fp8 block-128 + dyn 激活 | **28.75 GiB**（实测字节） | 14.38 | 15.4 | ✅ **已验收**（TTFT 8101.6 ms / 11.55 t/s） | 无（`VLLM_DCU_USE_FP8_EMU=1`、TP=2、eager） |
| **Qwen3-32B-FP8** | fp8 block-128 + dyn | ≈32.1 GiB | 16.0 | **20.0 ✅**（131k → 32.0 ❌） | ✅ **可直接跑（首选目标）** | 无内核工作；131k 需 KV fp8 |
| Mistral-Small-24B / Llama-3.1-8B 等 `*-FP8-dynamic` | compressed-tensors W8A8（per-channel + dyn token） | 23.1 / 8.0 GiB | 11.5 / 4.0 | 14.0 / 6.3 | ✅ 可跑 | 无（同一 kernel；建议一次 token-parity 实测） |
| Qwen3-14B/8B-FP8 | fp8 block/per-tensor | 14.5 / 8.0 GiB | 7.2 / 4.0 | 9.7 / 6.3 | 🔶 **增量小**（fp16 单卡本来也勉强能跑） | 无 |
| 40–50B 稠密 FP8 | — | 19–24 GiB/卡 | — | 需 KV fp8 或降 maxlen | 🔶 边界 | KV fp8 必须打通 |
| amd（Quark）/ NVIDIA（ModelOpt）FP8 | quark W8A8 / modelopt FP8（含 e8m0 scale） | 同参数量 | 同上 | 同上 | 🔶 代码通、**零实测** | 一次 token parity（`fp8_utils.py:1500-1535` 的 e8m0 分支未验证） |

### 3.2 吃不到的（写进"不可用"清单，别再反复评估）

| 类别 | 模型举例 | 阻塞点 |
|---|---|---|
| **FP8 MoE** | Qwen3-30B-A3B-FP8、Qwen3-Next-80B-A3B、GLM-4.5-Air/4.6、MiniMax-M2、Mixtral、ERNIE-MoE | 无 grouped-GEMM 仿真内核；落 Triton（同类实测 ~36×）。**显存够也没用**（30B-A3B 每卡 14.9 GiB） |
| **MLA** | DeepSeek-V2-Lite / V3 / R1、Kimi | `flashmla` 在 ROCm 被禁、aiter 不可用、`TRITON_MLA` 零实测；FP8 内核对此零帮助 |
| **70B+ 稠密** | Llama-3.3-70B-FP8（34.5 GiB/卡 @TP2） | TP=2 装不下；TP=4 撞 DP4X 卡（且 RCCL 248 在 rocm101 栈不可修） |
| **160B+ / 235B** | Qwen3-235B-A22B-FP8、DeepSeek-V3 671B | 体积直接出局 |
| FP4 / MXFP8 | GPT-OSS-20B(MXFP4)、NVFP4 系列、`*_mxfp8` | gfx906 无 FP4/无 dot_scaled；mxfp8 退化成 bf16 反量化（等于放弃省显存） |
| W8A16 weight-only FP8 | 部分 `*-FP8-WO` | ROCm 候选表为空（`kernels/linear/__init__.py:535-537`） |

### 3.3 扩散/视频：本次最大的新增域

| 权重 | 形态 | DiT 体积 | 每卡（TP=2，含 TE+VAE） | 判定 | 需要什么 |
|---|---|---:|---:|---|---|
| **MiniMax-H3 `fl2va_pruned_fp8_scaled`** | Comfy `fp8_e4m3` + **per-tensor** scale | 20.96 GB / 19.52 GiB | **21.3 GiB**（DiT 9.76 + GGUF-TE 6.10 + VAE 5.41） | ✅ **契约逐条对齐**，推荐作主画质档 | H3 模型骨架 + Comfy safetensors 加载器 + 非因果 attention；**量化侧零 kernel 工作** |
| H3 `int8_convrot` | INT8 + Hadamard-256 per-row | 20.97 GB | 21.3 | 🔶 需 ConvRot 旋转内核（gfx906 wave64） | 额外内核 + 坐标系自证 |
| H3 `pruned_bf16` | BF16 | 40.23 GB | 30.3 → ✗ | ❌ 显存不可能（且无 bf16 算力） | — |
| H3 GGUF Q2_K/Q3_K/Q4_K | GGUF w2a16 | 6.7–11.4 GB | 15.3–17.9 | ✅ 低显存档（现路线） | DiT 加载器 + ConvRot 判定（R1） |
| Wan2.2 / FLUX.1 / Qwen-Image / HunyuanVideo `fp8_scaled` | 与 H3 **完全同构**（comfy-quants `fp8_e4m3` 一族） | 12–26 GB | — | ⚪ **远期权** | 卡在"本仓库无 diffusion 运行时"；H3 运行时建成后接下一个是"换层名"级成本 |

> **注意反直觉点**：Community 的 ComfyUI/comfy-kitchen 在无 FP8 硬件的卡上跑 `fp8_scaled` 会 `except → 反量化成 bf16` 再算（显存翻倍到 ~40 GB，`comfy_kitchen/tensor/fp8.py`）——**H3 fp8 权重在 Z100 上"能用"恰恰依赖你这套零拷贝内核**，这是它在扩散域不可替代的理由。

---

## 4. H3 用 FP8 的深入结论（承接上一轮 H3 调研）

### 4.1 契约对齐（逐条）

| 内核要求 | H3 `fp8_scaled` 事实 | 判定 |
|---|---|---|
| weight scale ∈ {per_tensor, per_channel} | per-tensor（字节级：int8 比 fp8 多 12,174,008 B = 4 B×3,043,502 行 ⇒ int8 才是 per-row） | ✅ |
| act scale ∈ {per_tensor, per_token} | per-tensor（带 `input_scale` 则 static，否则 dynamic `amax/448`） | ✅ |
| `B.t()` 后 `[N,K]` 必须 contiguous（拒绝拷贝） | `[out,in]` row-major ⇒ `B.t()` 连续 | ✅ |
| scale `numel==1` ⇒ `stride=0` | 标量 fp32 | ✅ |
| fp8 dtype = 平台 `fp8_dtype()` = e4m3fn（非 fnuz） | `F8_E4M3` | ✅ |
| epilogue `(acc·a_s)·w_s → cast → +bias` | 无 bias（H3 的 4 个 GEMM 都没 bias），同式 | ✅ |

### 4.2 四方对比（BF16 / INT8-ConvRot / FP8 / GGUF Q2_K）

| 形态 | DiT 体积 | 权重/卡 TP2（含 TE+VAE） | 算力实测 | 画质证据 | 落地难度 |
|---|---:|---:|---|---|---|
| BF16 | 40.23 GB | 30.3 ✗ | 7.3 TFLOP/s（且需 96.9 GB fp16 副本，不可能） | 基线 | ✗ |
| **FP8 per-tensor** | 20.96 GB | **21.3** | **10.2 TFLOP/s** | **官方门禁：LPIPS 0.1156 / PSNR 23.63 dB / 音频 cos 0.9589 / 峰值 −22%**（vLLM-Omni 已合并 PR，外部） | 中（量化侧零工作，大头是 attention） |
| INT8-ConvRot | 20.97 GB | 21.3 | **22.2 TFLOP/s**（上限最高） | 无官方实测（社区口述 quality 优于 fp8） | 中-高（要 ConvRot 内核 + 坐标系自证） |
| GGUF Q2_K_XL | 8.06 GB | **15.3** | 6.6–8.1 | 无官方 benchmark（2.1–2.5 bit） | 低-中（内核已通，但无元数据 + ConvRot 未知） |

### 4.3 结论

- **H3 的 FP8 路线成立，且比 GGUF 更适合做"主画质档"**：契约干净（去掉无元数据/ConvRot 两个风险）、画质有官方数字可挂、算力 1.4×。
- **但不是替换**：DiT 显存 7.51 → 19.52 GiB（2.6×），(c) 档 243 帧会临界，必须**分阶段装载**（TE→释放→DiT→释放→VAE，峰值 ~18 GiB）；且**不能开 layerwise offload**（会破坏 `[N,K]` 连续性，内核是 raise 而非拷贝）。
- **TP 语义红线**：per-tensor 的 `w_s` 必须各 rank 拿到**同一个 checkpoint 标量**，绝不能在分片后重算 amax；row-parallel 下 `s_a·s_w` 可提到 all_reduce 之前（数学严格等价，这是 per-tensor 相对 block-scaled 的 TP 优势）。
- 只有 201 个 GEMM 被量化（50 block × 4 + condition_proj）；`token_refiner` 与每层 `adaln_proj` **仍 bf16**，patch/out 投影与 VAE 保持高精度 —— 正好落在安全一侧（官方 online-FP8 会量化 AdaLN，我们的文件不会）。

---

## 5. 增量价值判断（避免自我高估）

| 判断 | 结论 |
|---|---|
| **因为 FP8 才第一次可行** | **27B–32B 稠密**（fp16 需 27–32 GiB/卡，加 KV 就超 30 GiB）；24B 级为"半增量"（fp16 单卡能跑，FP8 才"好用"） |
| **只是省显存** | ≤14B 稠密；这部分 FP8 的价值是更长 KV / 更高并发 |
| **完全无增量** | 所有 MoE、MLA、FP4/MXFP8、W8A16、FP8 attention（SAGE/MXFP8 在 gfx906 不可行） |
| **速度别高估** | 同参数量 decode ≈1.3× fp16（405 GB/s vs 590–625 GB/s 等效）；prefill 大 M ≈1.5× bf16 GEMM。**16.1× 是对 Triton 标量回退的倍数** |
| **FP8 vs 同机 INT8 sdot4** | 同为 1 B/参；INT8 算力上限更高（22.2 vs 10.2）、但需本地再量化且**改 greedy token**。FP8 的独占价值 = **直接吃生态现成 checkpoint** |

---

## 6. 建议路线（按性价比排序）

| # | 事项 | 成本 | 解锁什么 | 判据 |
|---|---|---|---|---|
| 1 | **Qwen3-32B-FP8 TP=2 端到端**（`VLLM_DCU_USE_FP8_EMU=1`、`--max-model-len 32768`、util 0.88） | 0.5–1 天（下载 ~32 GB） | 32B 级稠密（唯一零内核工作的高价值空白） | ready + 输出连贯 + decode t/s ≈ 27B×(27/32) |
| 2 | **KV cache fp8 真伪探针**（27B/32B，`--kv-cache-dtype fp8 --max-model-len 4096` 起，再 8k/32k 压测） | 0.5 天 | 131k 上下文 + KV 减半（也让 40–50B 稠密进入边界可行） | token 流落在噪声底 + 无 dmesg 故障；否则写死"不可用" |
| 3 | **`PREFILL_CFG=auto` / `DECODE_CFG=16` 默认化** | 0.5 天（纯配置 + A/B） | DiT/长 prefill 提速 1.40× | 生产 A/B 不回退 |
| 4 | **e5m2 接线**（平台 fp8_dtype 改为按 checkpoint dtype 透传，或加 e5m2 分支） | 1–2 天 | e5m2 权重 checkpoint（少数 FP8 checkpoint 用它） | 一个 e5m2 权重的 token parity |
| 5 | **FP8 embedding/lm_head**（`Fp8Config` 分发 + gather/dequant） | 2–3 天 | vocab 权重 1.2–2.5 GB（27B/32B 的 8%）⇒ 更长的 KV | 显存账 + logits 一致性 |
| 6 | **H3 `fp8_scaled` 头解析（纯 CPU，0.5 天）→ 参数名映射 → 单层数值门禁** | 3–4 天 | 扩散域的 FP8 主画质档 | 201 个 fp8 weight + 201 个标量 scale 全映射；单层 rel_err ≤1e-3 |
| 7 | **fp8 grouped-GEMM（MoE）可行性评审**，再决定"做 / 写死不可用" | 评审 1 天，实现另算 | 全部 MoE 模型（Qwen3-30B-A3B / GLM-Air / MiniMax-M2 …） | 设计评审给出 tile 复用方案与预期 TFLOP/s |
| 8 | Wan2.2 / FLUX / Qwen-Image 等 `fp8_scaled` | 远期权 | 多模态生成全家 | 依赖 H3 运行时骨架先落地 |

---

## 7. 风险清单

| ID | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| R1 | **MoE 的 36× 退化是外推不是实测**（本机无 MoE 权重、4 卡被占满） | 中高 | 若要上 MoE 先跑一次小 MoE（E=8 级）实测，别用外推数字做决策 |
| R2 | **`supports_fp8()` 在 gfx906 返回 True**（`on_cdna()` 匹配 `"gfx9"` 子串）⇒ 把无硬件支持的 fp8 组合放行给 Triton | 中高 | 在 `supports_fp8()` 加 `not on_gfx906()`，让 MoE fp8 明确拒绝而非静默慢跑 |
| R3 | **KV fp8 门禁通过但零实测**，且 `fp8_per_token_head` 运行期抛 `NotImplementedError` | 中高 | 先做 §6-2 探针；否则在 README 钉死 per-tensor only |
| R4 | **e5m2 权重被 TORCH_CHECK 拒**（A 恒 e4m3fn） | 中 | §6-4 接线 |
| R5 | **FP8 embedding/lm_head 未量化**，大 vocab 模型显存被低估 8% | 中 | 预算时乘 1.05 系数（已有口径）；§6-5 |
| R6 | **C++ fp8 内核不在 DP4X guard 内**（guard 只包 Triton） | 中高 | 继续只走卡 0/1；若必须用卡 2/3，先用 hip shim 审计 VGPR（判据 ≤203） |
| R7 | **H3 的 per-tensor 精度**：整层一个 amax，离群值压有效位；官方 LPIPS 0.1156 说明"可接受但有可见差异" | 中 | AdaLN/patch/out/VAE 保高精度（本文件即如此）；必要时 `ignored_layers` |
| R8 | **H3 显存临界**：(c) 档 243 帧 ≈29 GiB / 31.2 预算 | 中高 | 分阶段装载；禁用 layerwise offload |
| R9 | **attention 仍是最大成本中心**（(b) 30.5% / (c) 72.4% FLOPs），FP8 不解决 | 高 | 独立攻坚（可行性文档 R3/R4） |

---

## 8. 证据路径

| 内容 | 路径 |
|---|---|
| 内核能力矩阵 / 分派 / 缺口 | `.s25/fp8-kernel.md`（§1、§4）、`csrc/rocm/fp8_emu_gemm.cu`、`csrc/rocm/gfx906_emu_common.cuh` |
| 主机侧选核与开关 | `vllm/model_executor/kernels/linear/scaled_mm/{gfx906_fp8.py,gfx906_common.py}`、`vllm/model_executor/kernels/linear/__init__.py:465-537` |
| FP8 量化方法全谱映射 | `.s25/fp8-llm-models.md`（§2）、`vllm/model_executor/layers/quantization/{fp8.py,compressed_tensors/**,modelopt.py,quark/**}` |
| MoE FP8 后端选择 | `vllm/model_executor/layers/fused_moe/oracle/fp8.py:48-145`、`.../experts/triton_moe.py:153-172` |
| FP8 实测（kernel / fixed time / 阶段归因） | `docs/archive/result/z100-11f-fp8-bf16-emu-kernels-result.md`、`docs/result/z100-fp8-emu-fixed-time-investigation.md`、`.s23/fp8_emu_kernel_fixed_times.md`、`.s20/S20_KERNEL_RESULT.md` |
| H3 fp8 契约与对比 | `.s25/fp8-diffusion.md`（§1、§2、§3） |
| Comfy fp8 格式与运行时语义 | <https://raw.githubusercontent.com/Comfy-Org/comfy-quants/main/docs/formats/fp8.md>、<https://raw.githubusercontent.com/Comfy-Org/comfy-kitchen/main/comfy_kitchen/tensor/fp8.py> |
| H3 官方 FP8 质量/显存门禁（外部，已合并 PR） | <https://api.github.com/repos/vllm-project/vllm-omni/pulls/5737>（LPIPS 0.1156 / PSNR 23.6316 / audio cos 0.9589 / 68.52→53.51 GiB） |
| H3 预算与本仓库缺口 | `docs/result/z100-minimax-h3-quant-feasibility.md`、`.s24/budget.md` |
