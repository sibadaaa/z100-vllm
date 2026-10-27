# MiniMax-H3 的 FP8 版本：选用清单 + 内核路由 + 落地计划

> 日期：2026-10-19 · 面向：自研 gfx906 FP8 内核（`Gfx906Fp8ScaledMMLinearKernel` / `csrc/rocm/fp8_emu_gemm.cu`）
> 前提口径：TP=2 on 卡 0/1（仅这两张 Z100L 无 DP4X 缺陷），每卡预算 ≤30 GiB，主机内存 31 GiB（可用 6–8 GiB）
> 关联：`docs/result/z100-fp8-kernel-model-extension.md`（内核能力面）、`docs/result/z100-minimax-h3-quant-feasibility.md`（H3 可行性）、`.s25/fp8-diffusion.md`（FP8 契约取证）、`.s24/budget.md`（H3 预算）

## 0. 结论：你要的"H3 版本"是这个

**H3 的 FP8 版本 = Comfy-Org 的 `fp8_scaled` 档（per-tensor W8A8，e4m3）**，不是 unsloth 的 GGUF 档，也不是官方 `int8_convrot` 档。

| 你要下载的 | 大小（字节） | 说明 |
|---|---:|---|
| `Comfy-Org/MiniMax-H3` → `diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | **20,958,205,608** | **主 DiT（t2va / fl2va / i2v）**；201 个 FP8 GEMM |
| `Comfy-Org/MiniMax-H3` → `diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors` | 20,958,205,608 | 只有要做「参考图锁角色 ref2va」时才需要（与 fl2va **不能同卡共存**） |
| 文本编码器：你已有的 `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` | 13,102,161,024 | **继续用 GGUF 版**（Comfy 的 bf16 51.51 GB / int8_convrot 27.14 GB 都不适合本机） |
| 视频 VAE：`vae/minimax_h3_video_vae_fp16.safetensors` | ≈5,207,742,160（你已有；镜像为 5,207,808,496，差 66 KB，下载后核对） | 必需 |
| 音频 VAE：`vae/minimax_h3_audio_vae_fp32.safetensors` | ≈605,149,280（你已有） | **必须 fp32**，换 fp16 会没声音 |
| 可选：`loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 1,956,192,992 | 4 步蒸馏 LoRA（第二阶段再上） |

- HF 仓库：<https://huggingface.co/Comfy-Org/MiniMax-H3>；国内镜像同路径 <https://modelscope.cn/models/Comfy-Org/MiniMax-H3>（本报告的字节数来自镜像 API 逐文件核对）。
- 磁盘：新增 20.96 GB（+ 可选 20.96 / 2.0），`/mnt/game` 余 328 GB 够；**不要放根分区（只剩 18 GB）**。
- ⚠️ **它今天还不能"直接跑"**：本仓库没有 `vllm/video`、没有 H3 模型实现、没有非因果高性能 attention。本文件给的是"选哪个版本 + 怎么接到你的 FP8 内核 + 分几步验"，其中**第一步（读 safetensors 头）纯 CPU、零风险、0.5 天**。

---

## 1. 为什么是这一档（四种 H3 权重对你的内核而言）

| 形态 | DiT 体积 | 契约 | 你的 per-tensor 内核能否直吃 | 判定 |
|---|---:|---|---|---|
| **`pruned_fp8_scaled`** | 20.96 GB / 19.52 GiB | `F8_E4M3` + **per-tensor** weight scale + per-tensor 激活 | **可以，9 项契约逐条对齐** | ✅ **选它** |
| `pruned_int8_convrot` | 20.97 GB | INT8 + per-row scale + **Hadamard-256 旋转坐标系** | ❌（需在线 ConvRot 内核；且坐标系需自证） | 🔶 备选，仅当你后续要极限算力（22.2 TF/s） |
| `pruned_bf16` | 40.23 GB | BF16 | ❌ 单卡 18.7 GiB 权重 + gfx906 无 bf16 算力 | ❌ |
| `pruned_w6a8` | 15.98 GB | W6A8 | ❌ 无此反量化格式 | ⚪ 以后若要更省显存 |
| GGUF Q2_K / Q3_K / Q4_K（你现在的） | 6.72–11.42 GB | GGML，**无元数据**，ConvRot 未知 | ❌ 不走 FP8 内核（走 `GGUFLinearMethod`） | ✅ 保留为**低显存档** |

**为什么 fp8 值得做主档（不是速度，是画质 + 契约）**：
- 官方画质门禁可挂：vLLM-Omni 已合并的 H3 online-FP8 PR 报 **LPIPS 0.1156（≤0.20 PASS）/ PSNR 23.63 dB / 音频谱 cosine 0.9589 / 峰值 −22%**（外部证据）。
- **消掉 GGUF 路线的两个高风险**：GGUF `kv_count=0`（config 只能从形状反推）、ConvRot 坐标系未知（错则静默出噪点）。`fp8_scaled` 有 `comfy_quant` 标记、权重未旋转、张量名与原生 checkpoint 1:1。
- 算力顺带小幅改善：DiT 大 M 实测 **10.23 TFLOP/s/卡**（cfg10，M=7920/N=3584/K=5120），比 GGUF 的 6.6–8.1 高 ≈1.4×。

---

## 2. 契约（下载后**第一件事**就是核这三条）

```
W  = W_q(e4m3, [N,K] row-major) * s_w        s_w: fp32 标量（per-tensor），**每层 1 个**
x  = round_fp8(x / s_a)                      s_a: fp32 标量（per-tensor；有 input_scale=static，否则 dynamic amax/448）
y  = (x_q @ W_q^T) * s_a * s_w  [+ bias]     H3 这 4 个 GEMM **没有 bias**
```

**字节级已证的两条**（不需要下载也能算）：
1. `int8_convrot` 比 `fp8_scaled` 大 **12,174,008 B = 4 B × 3,043,502 行** ≈ ΣN(3,046,400) ⇒ **int8 带 per-row f32 scale，fp8 是 per-tensor**；
2. 该文件只量化 **201 个 GEMM**（50 block × 4 + `condition_proj` = 19.295 G 元素），`token_refiner` 与每层 `adaln_proj` **仍 bf16** ⇒ 与官方"AdaLN/patch/out 保高精度"的意图一致（对我们是好消息）。

**待核（下载后读 8 B 长度头 + JSON 头，纯 CPU）**
- [ ] `*.weight` 为 FP8 的张量数 = **201**；
- [ ] 每层 `*.weight_scale` **`numel()==1`**（若是 `[N]` 就是 per-row，语义会变，必须按 per-channel 走内核并改 loader）；
- [ ] 是否存在 `*.input_scale`（决定 `activation_scheme="static"` 还是 `"dynamic"`）；
- [ ] `*.comfy_quant` 的 JSON 内容（`format`/`full_precision_matrix_mult`）。

> **加载器硬断言**：`assert weight_scale.numel() == 1`。把 per-tensor 当 per-row（或反之）不会崩，会**静默数值错**。

---

## 3. 张量 → vLLM 模块 → 内核路由（201 个 FP8 GEMM 全表）

| H3 张量（每层） | 形状 `[out,in]` | 参数量/层 | vLLM 模块 | TP=2 切分 | 走内核 | 说明 |
|---|---|---:|---|---|---|---|
| `blocks.N.attn.qkv_proj.weight` ×50 | `[21504, 5376]` | 115.6 M | `QKVParallelLinear`（56 头 → **28 头/rank**） | 切 N → `[10752,5376]` | per-tensor FP8 GEMM | 3×7168 = q/k/v 合并；**必须按 head 边界重排**（在 `load_weights()` 里、调用 vLLM loader 之前） |
| `blocks.N.attn.out_proj.weight` ×50 | `[7168, 5376]` | 38.5 M | `RowParallelLinear` | 切 K → `[7168,2688]` | 同上 | row-parallel：`s_a·s_w` 可提到 all_reduce 前（标量，数学等价） |
| `blocks.N.mlp.fc1.weight` ×50 | `[28672, 5376]` | 154.1 M | `MergedColumnParallelLinear`（gate+up） | 切 N → 2×`[14336,2688]` | 同上 | gate/up 合并，需拆成两段给 vLLM |
| `blocks.N.mlp.fc2.weight` ×50 | `[14336, 5376]` | 77.1 M | `RowParallelLinear` | 切 K → `[14336,2688]` | 同上 | — |
| `condition_proj.weight` ×1 | `[5120, 5376]` | 27.5 M | Column | 切 N | 同上 | 文本条件 5120 → 5376 |
| **小计** | | **19.30 G（96%）** | | | | |
| `blocks.N.adaln_proj.linear` ×50 | `[8, 96768]` | 0.77 M | 裸 torch | 复制 | **不进 FP8 内核**（文件里本就是 bf16） | 输入维只有 8（低秩曲线基），per-tensor fp8 会被放大误差；additionally 它承担调制 |
| `token_refiner.blocks.{0,1}.*` | 同 block 形状 | 0.771 G | 同上四类 | 同 | 主 GEMM 若也是 fp8 走内核；本文件为 bf16 → 走 `FP16LinearMethod` | 文本条件 refiner（2 层） |
| `video_patch_proj` / `audio_patch_proj` | `[96,5376]` / `[32,5376]` | 0.70 M | Column（F32） | 切 N | **留 fp32** | 输入是最原始的 patch/潜变量，官方明确排除 |
| `final_layer.{video,audio}_out` | `[5376,96]` / `[5376,32]` | 0.52 M | 裸 torch（F32） | 复制 | **留 fp32** | 输出头直接决定像素/波形 |
| `adaln_t_table` / `rope.inv_freq` / 各 norm | 小张量 | ~8 K | — | 复制 | 原精度 | — |

**内核路由（不需要改内核，只需选对 cfg）**：DiT 每步是"整条 packed 序列 × 全部层" ⇒ M = N（1,770 / 12,323 / 73,483），全部 > `_GEMV_MAX_M`=8 ⇒ 走 **tiled GEMM** 分支。

| 负载 | M | 建议 cfg | 依据 |
|---|---:|---|---|
| (a) 640×384×22 | 1,770 | auto（0）→ cfg5/cfg10 | `VLLM_DCU_FP8_PREFILL_CFG=auto` |
| (b) 1344×768×39 | 12,323 | **cfg10** | N 宽、M 深；实测 N=3584:5120 时 cfg10 比 cfg5 快 **1.40×** |
| (c) 1344×768×243 | 73,483 | cfg10 | 同上 |

---

## 4. 接线清单（量化侧几乎零内核工作，工作量在加载器与模型骨架）

1. **权重加载器**（唯一实打实的工作）
   - 读 Comfy safetensors：`weight`（FP8）+ `weight_scale`（1 元素 fp32）+ 可选 `input_scale`；`comfy_quant` (uint8 JSON) **直接跳过**。
   - qkv 按 head 重排、fc1 拆 gate/up —— **必须在 `load_weights()` 内、交给 vLLM loader 之前**做（官方 PR 同做法）。
   - **不得拷贝/转置成新布局**：`apply_scaled_mm` 检查 `w.is_contiguous()`，不连续直接 raise（好事：不会静默错）。
2. **量化配置构造**：Comfy safetensors 没有 `config.json`/`quantization_config`，不能走 `Fp8Config.from_config` ⇒ 在模型代码里手工构造
   `Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme=("static" if input_scale else "dynamic"))` 交给 `Fp8LinearMethod`。
3. **env**：
   ```bash
   VLLM_DCU_USE_FP8_EMU=1          # 必需；默认 OFF，关掉则 per-tensor FP8 直接失败
   VLLM_DCU_FP8_PREFILL_CFG=auto   # DiT 全走 GEMM 路径，cfg10
   # VLLM_DCU_FP8_DECODE_CFG / GEMV_MAX_M 与 DiT 无关（那是 LLM decode 的 256 次/步场景）
   ```
4. **TP 语义红线（per-tensor 的优势也是陷阱）**：`w_s` 是整张 `[N,K]` 的 amax 导出 ⇒ **各 rank 必须拿到同一个 checkpoint 标量，绝不能在分片后重算 amax**；row-parallel 下 `s_a·s_w` 可在 all_reduce 前乘掉。
5. **禁止 layerwise offload**：offload 会改变 weight stride，内核会 raise（vLLM-Omni 也记录过 CUTLASS FP8 内核拒绝 offload 后的 stride）。省显存只能靠**分阶段装载**：`TE → 释放 → DiT → 释放 → VAE`。
6. **attention 不在 FP8 范围**：QK-norm、RoPE、非因果全序列 attention 仍需 fp16/fp32（Triton unified attention 或 SDPA 分段），这是 H3 落地最大的独立工作量（见可行性报告 R3/R4）。

---

## 5. 预算（TP=2 on 卡 0/1）

| 项 | /卡 |
|---|---:|
| DiT（19.52 GiB / 2） | 9.76 GiB |
| 文本编码器 GGUF Q2_K_M（12.20 / 2） | 6.10 GiB |
| 视频 VAE + 音频 VAE（tile 并行 ⇒ **每卡复制**） | 5.41 GiB |
| **权重合计** | **21.27 GiB** |
| +(b) 档激活 1.08 GB | 22.35 GiB ✅ |
| +(c) 档激活 6.45 GB | **27.72 GiB**（≤30 极限区，必须分阶段装载才能压到 ~18） |

| 负载 | 总 FLOPs | @10.23 TF/s/卡（FP8） | @7.3（GGUF 基线） |
|---|---:|---:|---:|
| (a) 4 步 | 0.30 PF | **15 s** | 21 s |
| (b) 20 次 | 14.27 PF | **697 s ≈ 11.6 min** | 977 s ≈ 16.3 min |
| (c) 49 次 | 524 PF | **25,618 s ≈ 7.1 h** | 35,900 s ≈ 10 h |
| (b) 若加 4 步 turbo LoRA | 2.85 PF | **139 s ≈ 2.3 min** | — |
| (c) 若加 4 步 turbo LoRA | 42.8 PF | **≈35 min** | — |

> 时间账只算 DiT 前向；端到端另加：冷加载 ~17 s、文本编码 ~12 s（几乎全是 13.1 GB 权重搬运）、VAE 解码 (b) ≈35 s / (c) ≈8 min。

---

## 6. 最小验证矩阵（成本从低到高，每步都有判据）

| # | 步骤 | 成本 | 判据 |
|---|---|---|---|
| 1 | **读 safetensors 头**（纯 CPU，不必上卡）：统计 fp8 权重数、`weight_scale` shape、`input_scale` 有无、`comfy_quant` 内容 | 0.5 天 + 21 GB 下载 | 201 个 fp8 weight、201 个 `numel==1` scale；据此定 static/dynamic |
| 2 | **参数名映射表**：201 个 fp8 张量 + bf16/f32 残留全部映到 vLLM 名字 | 0.5 天（纯 CPU） | 532 张量 100% 映射、无残留、无重复 |
| 3 | **单层数值门禁**：取 `blocks.0.attn.qkv_proj [21504,5376]`，用真权重 + per-tensor 激活跑 `Gfx906Fp8ScaledMMLinearKernel`，与 CPU 参考 `(x_q·s_a)@(w_q·s_w)^T` 比 | 1 天 / 卡 0 | `rel_err ≤ 1e-3`；且与 fp16 rocBLAS 的偏差 ≤ fp8 量化误差本身 |
| 4 | **TP=2 正确性**：qkv 切 N、fc2 切 K，验证 `s_a·s_w` 提到 all_reduce 前与原式一致 | 0.5 天 | 逐位一致（标量重排，应完全相等） |
| 5 | **端到端最小片**（D1 档）：640×384×22 / 4 步，fp8 DiT + GGUF TE + VAE | 2–3 天 | 能出片；与 GGUF DiT 同种子相对 L2 ≤0.05（预期 fp8 更接近 bf16）；显存 ≤19 GiB/卡 |
| 6 | **画质档切默认**：(b) 1344×768×39 / 21σ，开 `PREFILL_CFG=auto` | 1 天 | 端到端 ≤25 min（预算 11.6 min + VAE/编码）；峰值 ≤28 GiB/卡且无 dmesg 故障 |

> 第 1–2 步**不需要 GPU**（4 卡被生产 27B-FP8 占满也能做），建议今天就开始；第 3 步起需要协商维护窗口。

---

## 7. 风险（相对 GGUF 路线新增的）

| ID | 风险 | 缓解 |
|---|---|---|
| F1 | **per-tensor 精度**：整层一个 amax，离群值压有效位（官方 LPIPS 0.1156 = 可接受但有可见差异） | AdaLN/patch/out/VAE 保高精度（本文件天然如此）；必要时对噪声敏感层用 `ignored_layers` 留 bf16 |
| F2 | **scale 形状误判**（per-tensor ↔ per-row）静默错 | 加载器硬断言 `numel==1`；§2 的 12.17 MB 字节差已排除 per-row |
| F3 | **DiT 显存 2.6×**（7.51 → 19.52 GiB），(c) 档 243 帧 = 27.7 GiB 临界 | 分阶段装载（峰值 ~18 GiB）；禁止 layerwise offload |
| F4 | **attention 仍是最大成本中心**（(b) 30.5% / (c) 72.4% FLOPs），FP8 完全不解决 | 独立攻坚；VSA 不建议首期做（参考实现自己没过质量门） |
| F5 | **TP 分片后重算 amax** | §4.4 红线：scale 必须 replicated |
| F6 | **4 步 turbo LoRA 与 FP8 权重叠加**（LoRA 是 bf16 delta） | 第二阶段再做；先确认 vLLM 侧 DiT LoRA 支持与 fp8 层兼容 |
| F7 | **本仓库无 H3 运行时**（模型、加载器、视频管线、非因果 attention 全缺） | 按可行性报告 D1→D2 分档推进；FP8 只解决"权重与 GEMM"，不解决运行时 |

---

## 8. 证据路径

| 内容 | 路径 |
|---|---|
| fp8 契约逐条取证（comfy-quants / comfy-kitchen / sd.cpp 三方交叉） | `.s25/fp8-diffusion.md:§1` |
| 与内核 9 项对齐表、`_block` 变体不需要 | `.s25/fp8-diffusion.md:§2` |
| 四方对比与官方画质门禁 | `.s25/fp8-diffusion.md:§3` |
| 内核契约（零拷贝 / scale stride / epilogue 顺序） | `csrc/rocm/fp8_emu_gemm.cu:35-44,1217-1245`、`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py:267-336` |
| prefill cfg 与 10.23 TFLOP/s 实测 | `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py:56-96` |
| H3 结构 / token 模型 / 逐张量预算 | `.s24/budget.md:§2-§4`、`docs/result/z100-minimax-h3-quant-feasibility.md:§2` |
| H3 各档权重字节（Comfy-Org，镜像 API 核对） | `docs/result/z100-minimax-h3-hf-model-candidates.md:§2` |
| 官方 FP8 质量/显存（外部，已合并 PR） | <https://api.github.com/repos/vllm-project/vllm-omni/pulls/5737> |
