# Z100 适配「量化版 MiniMax-H3」可行性调研

> 日期：2026-10-19 · 范围：`/home/lyh/z100-vllm`（vLLM Hygon Z100 / gfx906 分支）
> 方法：参考实现源码通读 + 目标机现状实测 + 量化体全量解析 + 预算测算（**全程只读，未改任何代码，未上卡运行负载**）
> 素材：`/mnt/game/1cat-vllm`（1Cat-vLLM / V100-SM70 分支，内含 native MiniMax-H3）、`/mnt/game/h3-test/models/`（用户已下载的量化体）、`/mnt/game/h3-test/*.log`（同一份 GGUF 在本机跑过的 ggml 实测）
> 关联文档：`docs/result/z100-moe-feasibility-investigation.md`、`docs/result/F24-investigation-20261018.md`、`docs/result/z100-rocm101-acceptance.md`
> 取证附件（本报告的三份原始取证，行号级）：`.s24/h3-port.md`、`.s24/z100-constraints.md`、`.s24/budget.md`

## 0. 结论摘要（先看这 12 条）

1. **可行，但可行域不是由算力决定，而是由三个耦合量决定：量化形态、卡选型、主机内存。** 三者中任何一条走错都会直接让整条路径不可行（§5、§6、§9）。
2. **「量化版」指的是 GGUF w2a16**：`minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf`（DiT，20.11 G 参数 / 8.063 GB / 532 张量 / **`kv_count=0` 无任何元数据**）+ `qwen3vl_32b_minimax_h3-Q2_K_M.gguf`（文本编码器，25.75 G / 13.102 GB / 902 张量）+ 两个 VAE safetensors（§1.1）。
3. **参考实现不吃 GGUF**。1cat 的 H3 路径只支持 **BF16** 与 **ComfyUI INT8-ConvRot（W8A16 + Hadamard256）** 两种权重；`minimax_h3/`、`vllm/video/`、`tools/minimax_h3/` 里 `gguf` 命中 0（`.s24/h3-port.md:21`）。⇒ **GGUF 加载层必须自建在本仓库已有的 `GGUFLinearMethod` 上，不能从参考实现照抄**。
4. **fp16 反量化常驻出局**：DiT 40.2 GB + 文本编码器 51.5 GB = **96.9 GB**，既超过 4×32 GiB 卡总显存，也是主机 31 GiB 的 3 倍。重量化/反量化路线（路径 C）在硬件上不成立（§5.4）。
5. **显存上 TP=2（卡 0/1）是唯一「全档可跑且不碰 DP4X 卡」的配置**：权重 15.26 GiB/卡（VAE 不按头分片需复制）+ (c) 档激活 6.45 GB ≈ **24 GiB**；TP=1 除 (a) 档外必 OOM（`t2.log` 实测在 864×480 就 `need 12895.68 MB > available 12749.42`）。TP=4 余量最足（≈13–16 GiB）但一半 GEMM 落在卡 2/3（§6.3）。
6. **算力是本报告最硬的量化结论**：同机、同一份 GGUF、ggml 四组配置反推出 **gfx906 有效算力 6.6–8.1 TFLOP/s/卡**（≈打包 fp16 峰值 26.1 TFLOP/s 的 25–31%）；工程化上限按手写 int8 sdot4 内核 42% 效率外推为 **10–13 TFLOP/s/卡**。V100 参考实现是 **51.9–59.5 TFLOP/s/卡**（§6.4）。
7. **端到端预期**（含加载/文本编码/VAE）：(a) 640×384×22/4 步 ≈ **40–70 s**；(b) 1344×768×39/21σ ≈ **11–17 min**；(c) 1344×768×243/50σ ≈ **3.3–10 h**（§6.5）。即 **H3 的 720p 长片在 Z100 上属于过夜任务**，(b) 是唯一现实可验收的完整质检档。
8. **移植面是「数值逻辑可搬、内核全部要换」**：模型/采样/打包/媒体编排共 ~16.3k 行 Python 可基本原样复用；CUDA/SM70 专属面 **20 项**（§3.3），其中**主干注意力（CUTLASS SM70 FMHA + WMMA + VSA 稀疏）是极高难度黑洞**，主 GEMM（cuBLAS/cuBLASLt Volta 计划）可换 rocBLAS，小内核（ConvRot/dequant/scaled-add/4 组 Triton kernel）难度低-中。
9. **DP4X（F24）是「栈绑定」问题，不是硬件问题**：故障只在 **DP4X-unaware 用户态**（自编 ROCm 10.1 / HIP 7.16）出现，DTK 26.04 用户态的 HSA loader 带 DP4X 支持（`vllm/platforms/hygon_dp4x.py:11-18`、`docs/result/F24-investigation-20261018.md:197-225`）。**建议 H3 移植直接锁 DTK 26.04 栈**，让 F24 整类出局；但 C++ 侧 GGUF 内核的 VGPR 仍需用 hip shim 审计一次（R2）。
10. **栈现状需要校订文档口径**（本次实测）：仓库文档记录的是 `torch 2.16 + ROCm 10.1`，但**本机 `/opt/rocm` 不存在，`.venv` 实际是 `torch 2.11.0+das.opt1.dtk2604` / `vllm 0.18.1+das.3266200.dtk2604` / `triton 3.8.0`（DTK 树另用 `/mnt/game/triton-gfx906/install` 的 `3.6.0+git7c56a5e4`）**；`tools/dcu/use_stack.sh status` 显示当前生效的是 DTK 那套 `.so`；生产服务用 `DTK_ROOT=/mnt/game/host-runtime/dtk2604`（`tools/dcu/serve_host_dflash2.sh:1-30`）。§1.3 有完整证据。
11. **DiT 的全序列非因果 attention 没有现成高性能后端**：本仓库 `flash-attn` 在 gfx906 不可用（`vllm/v1/attention/backends/fa_utils.py:38-58`），Triton unified attention 虽有 bidirectional 分支（`vllm/v1/attention/ops/triton_unified_attention.py:1238-1251`）但**强制 paged KV**（`:312-320`），每步都要伪造一份全序列 KV，成本是 `2·S·H·D` 的显存+带宽。这是除算力之外的第二个成本中心（R5）。
12. **两条「不要做」**：不要首期做 VSA（参考实现自己在 V100 上就没过质量门：L2 0.390670/0.088636、PSNR 24.28 dB）；不要把 (c) 档（243 帧）当验收目标（PFS 524 PF，注意力占 72.4%）。

---

## 1. 调研对象与素材

### 1.1 用户已下载的量化体（全量解析，`[实测]`）

| 文件 | 字节 | 参数量 | 张量 | dtype 构成 | 结构 |
|---|---:|---:|---:|---|---|
| `minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf` | 8,063,029,344 | **20.11 G** | 532 | Q2_K×169 / Q5_K×40 / F16×102 / F32×10 / BF16×211 | 50 层 DiT + 2 层 token_refiner；**block 0–39 全 Q2_K，block 40–49 全 Q5_K**（unsloth dynamic） |
| `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` | 13,102,161,024 | **25.75 G** | 902 | Q2_K×300 / Q5_K×50 / BF16×552 | Qwen3-VL **50 层**（d=5120、64 头×128、8 KV 头）+ `embed_tokens` BF16 1.556 GB + **视觉塔 27 block/351 张量全 BF16**；无 `lm_head` |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5,207,742,160 | 2.604 G | F16×562 | F16 | `vae_ratio=16`、`vae_ratio_t=4`、`z_channels=24`；**36 层 2048 维 ViT 解码器占 2.42 G（4.83 GB）** |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 605,149,280 | 0.151 G | F32×917 | F32 | BigVGAN、40 Hz、32 潜通道、32 kHz |

- **GGUF 里没有任何元数据**（`kv_count=0`，连 `general.architecture` 都没有）⇒ 所有结构常量只能由张量形状反推；反推结果与参考实现的 `MiniMaxH3DiTArchConfig` 默认值**逐项吻合**（`.s24/budget.md:74-88`）。
- 命名与参考实现的原生参数名 **1:1 对应**（`blocks.N.attn.qkv_proj.weight`、`blocks.N.mlp.fc1/fc2`、`blocks.N.adaln_proj.linear.weight[8,96768]`、`adaln_t_table[8,1025]`），说明它是从同一份原生 checkpoint 转出来的；`adaln_t_table` + `[8,96768]` 同时确认它是 **AdaLN-pruned** 变体（对应 `comfy_checkpoint.py:20-34` 的分支）。
- **文件里没有任何 ConvRot / INT8 / per-row scale 信息**。若这批权重源自 Comfy INT8-ConvRot 且转换时未反旋转，运行期必须补 Hadamard-256，否则**静默错误**——这是最大单点不确定性（R1）。

### 1.2 参考实现：1Cat-vLLM 的 native MiniMax-H3

- 代码在 `/mnt/game/1cat-vllm/vllm/model_executor/models/minimax_h3/`（36 文件 / 13,914 行）与 `vllm/video/`（11 文件 / 2,416 行），改编自 Apache-2.0 的 vLLM-Omni `7be014bc…`（`minimax_h3/UPSTREAM.md:1-10`）。
- 部署契约**写死 SM70**：Python 3.12 / Torch 2.10.0+cu128 / CUDA 12.8 / `TORCH_CUDA_ARCH_LIST=7.0` / CUTLASS v4.4.2（`docs/design/minimax_h3/README.md:20-26`）。
- 三个自制扩展（`tools/minimax_h3/build_extensions.py:25-50`）：`_h3_w8a16_C`（`csrc/sm70_turbomind/ops/h3_w8a16.cu`，cuBLAS/cuBLASLt）、`_h3_flashattn_C`（`flash-attention-v100/kernel/h3/forward.cu`，CUTLASS `arch::Sm70` FMHA）、`_h3_flashinfer_C`（`flashinfer-sm70`，`nvcuda::wmma`）；另有 VSA 稀疏与 peer 行规约扩展。
- **好消息**：H3 路径没有任何 CUDA graph / `torch.compile` 捕获（只有 1 处 `@torch.compiler.disable`，`transformer.py:451`）；主干是 **vLLM TP 线性层 + 3 个自制算子 + 6 个 Triton kernel**，不是满屏手写 CUDA。

### 1.3 目标机现状（本次实测，`[实测]`）

| 项 | 实测值 | 证据 |
|---|---|---|
| GPU | 4× Hygon Z100 / gfx906，VRAM 100 % 占用（生产服务 `Qwen3.8-27B-FP8` TP=4 在跑），功耗墙 280 W×2（卡 0/1，Z100L）+ 250 W×2（卡 2/3，Z100） | `hy-smi` |
| 主机内存 | **total 31 GiB、available ≈ 6–8 GiB、swap 3 GiB 已满** | `free -g` |
| 磁盘 | `/mnt/game` 余 328 GB；根分区余 18 GB | `df -h` |
| 生效栈 | `torch 2.11.0+das.opt1.dtk2604` / `vllm 0.18.1+das.3266200.dtk2604` / `triton 3.8.0`（venv），DTK 树另用 `triton 3.6.0+git7c56a5e4`（`PYTHONPATH=/mnt/game/triton-gfx906/install`） | `.venv/.../site-packages`、`serve_host_dflash2.sh:42` |
| 栈切换 | `tools/dcu/use_stack.sh status` 输出：DTK 树（HEAD `ab2a89b1a`）在用；ROCm 10.1 树在 `/mnt/game/port/gfx906-vllm-rocm101` | 命令输出 |
| 用户态树 | `/opt/rocm` **不存在**；DTK 用户态在 `/opt/dtk-26.10`（`dcc 26.04.0`）与 `/mnt/game/host-runtime/dtk2604`（生产用，因为 DTK 26.10 只有 `libhipblas.so.3` 而 DTK-torch 轮子要 `.so.2`） | `ls /opt`、`serve_host_dflash2.sh:1-8` |

> **为什么这条要单列**：本报告后续所有"DP4X 风险"结论都是**栈相关**的。文档里的 `torch 2.16 + ROCm 10.1` 是容器化路径（`z100-vllm-rocm101-rt`），当前裸机跑的是 DTK 26.04 那套。两份栈的 F24 行为**不同**（§4.2）。

---

## 2. 模型与计算量画像

### 2.1 结构常量（形状反推，与参考实现默认值一致，`[实测]`）

```
num_layers=50, token_refiner_num_layers=2
hidden=5376, heads=56×128 (=7168 inner), ffn=14336 (fc1 → 28672 gate+up)
video latent 24ch, patch(1,2,2) → 行宽 96；audio latent 32
text_dim=5120, adaln_out=18×5376, final_adaln=2×5376
AdaLN-pruned: adaln_t_table[8,1025] + adaln_curve_dim=8（无 dense time_embedder）
每 block: qkv[5376,21504] + out[7168,5376] + fc1[5376,28672] + fc2[14336,5376] ≈ 385.35 M
```

### 2.2 Token 模型（被参考实现两次撞中，`[实测]`）

```
帧对齐 frames ∈ 17n+5            潜帧 T_lat = ((frames-5)//17)*5 + 2
视觉 token V = T_lat × (H/32) × (W/32)      音频 token A = 2 × round(秒×40)
每步序列 N = 文本 + 关键帧条件 + A + V (+pad)
```

| 画布/帧 | V | A | N | 对账 |
|---|---:|---:|---:|---|
| 640×384×22 | 1,680 | 74 | **1,770** | ggml 日志 `sample 40x24x7` ✓ |
| 864×480×56 | 6,885 | 186 | **7,171** | ggml `sample 54x30x17` ✓ |
| 1344×768×39 | 12,096 | 130 | **12,323** | 1cat `CONTROL.md` 原文 "actual H3 N12323" ✓ |
| 1344×768×243 | 72,576 | 810 | **73,483** | 1cat `CONTROL.md` 原文 "N73483" ✓ |

### 2.3 每步 FLOPs

`每步 FLOPs = 2·P·N + 50·4·N²·7168`（P=20.11 G；非因果全序列 attention），DiT 调用次数 = σ 位置数 − 1。

| 负载 | N | 线性项 | 注意力 | 合计/步 | 调用 | 总 FLOPs | 注意力占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) 640×384×22 / 4 步 | 1,770 | 0.071 PF | 0.005 | **0.076 PF** | 4 | 0.30 PF | 5.9 % |
| (b) 1344×768×39 / 21σ | 12,323 | 0.496 PF | 0.218 | **0.713 PF** | 20 | 14.3 PF | 30.5 % |
| (c) 1344×768×243 / 50σ | 73,483 | 2.956 PF | 7.741 | **10.70 PF** | 49 | **524 PF** | **72.4 %** |

> 公式与参考实现自报口径一致：把 V100 的 1280×736×124/4 步代入，反算 51.98 TFLOP/s/卡 vs 文档自报 51.939（偏差 0.08 %）；把 39 帧/20 更新代入，反算与文档 `VALIDATION.md:20-24` 的 rank FLOPs 差 3.0 %。⇒ 后续所有预算可引用。

### 2.4 「量化版」的四类形态（决策必须区分）

| 形态 | 谁支持 | 硬件门槛 | 结论 |
|---|---|---|---|
| BF16（官方原分片，**每分区约 134 GiB**） | 1cat 原生 / vLLM-Omni | 需 ≥200 GiB 主机 RAM | ❌ 本机不可能 |
| 在线 FP8（BF16 checkpoint 上运行时量化） | vLLM-Omni（CUDA/gfx942/gfx950） | 需 FP8 通路；gfx906 只有仿真 | ❌ 出局 |
| **ComfyUI INT8-ConvRot（W8A16 + Hadamard256）** | **1cat 原生**（`quantization.py`、`comfy_checkpoint.py`） | 需 `.safetensors` + ConvRot 内核 | 🔶 需离线转换（§5.3） |
| **GGUF Q2_K/Q5_K（用户手上的）** | **本仓库**（`GGUFLinearMethod`：dequant / MMVQ / MMQ 三条 HIP 实现） | 无 FP4/FP8 需求，sdot4 + fdot2 即可 | ✅ **唯一可落地形态** |

---

## 3. 参考实现（1cat native H3）的移植面

### 3.1 计算图 → 内核映射（关键 8 段）

| 图段 | 参考实现内核 | 语言/依赖 | 出处 |
|---|---|---|---|
| 文本/视觉编码（Qwen3-VL 前 50 层） | vLLM 风格 TP 线性 + M-RoPE | torch + vLLM linear | `encoder.py:1-27,1059` |
| 编码器注意力（causal GQA） | `flash_attn_v100.flash_attn_func`，SM70 强制 cuDNN SDP | CUDA flash-attn | `encoder.py:445-466,18` |
| **主 GEMM（所有线性层）** | `h3_fp16_gemm`：`cublasGemmEx(TN, COMPUTE_32F, TENSOR_OP)`；块内 cuBLASLt **硬编码 Volta 计划**（algo id=21/tile=24/splitK=1） | **cuBLAS + cuBLASLt** | `h3_w8a16.cu:206-250`、`h3_column_major_gemm.h:14-124` |
| ConvRot / dequant / fp16-prepare | `convrot256` / `dequantize_rows|columns` / `prepare_fp16_rows` | CUDA C++ | `h3_w8a16.cu:14-120,131-204` |
| **主干注意力（非因果 varlen D128）** | CUTLASS FMHA `arch::Sm70` + 自制 softmax/PV 流水；写死 246 regs / 34 KB smem | **CUTLASS v4.4.2 SM70** | `flash-attention-v100/kernel/h3/forward.cu:17-70` |
| 备选注意力 | `nvcuda::wmma` m16n16k16 / 纯 torch SDPA 分段 | CUDA WMMA / torch | `flashinfer-sm70/csrc/h3_noncausal_sm70.cu`、`attention.py:51-60,143-146` |
| 归一化/调制/激活 | 6 个 **Triton** kernel（qk_norm_rope、rms_norm+scale_shift、indexed gate、silu_prepare…） | Triton | `modulation.py`、`qk_norm_rope.py`、`activation.py` |
| VAE 编解码 | torch conv3d + **cuDNN**，装载走 checkpoint 自带 `auto_map` 远程 python | torch/cuDNN | `vae.py:35-127` |
| 残差 TP 规约 | vLLM NCCL all_reduce 或 raw CUDA IPC 行规约 | NCCL / CUDA IPC | `collectives.py:56-91`、`exact_row_reduce.cu` |

> profile 归因（4 步 W8A16，V100 TP4）：**GEMM 41.10 % / attention 39.81 % / communication 13.52 % / ConvRot 1.02 %**（`CONTROL.md:40-41`）。这三项的 gfx906 替代方案分别决定移植的难度分布。

### 3.2 必须为 gfx906 重写/替换的清单（20 项浓缩）

难度口径：低=换调用；中=写 Triton/HIP 并对齐舍入；高=新写矩阵/注意力内核 + 性能工程；极高=算法级重设计或算力不可达。

| 难度 | 项目 | 替代方案 |
|---|---|---|
| **极高** | `_h3_flashattn_C` 主干注意力（CUTLASS SM70 FMHA） | 无直译可能；只能 (a) 复刻本仓库 Triton unified attention（自带非因果分支）(b) 手写 `v_dot2_f32_f16` D128 非因果 varlen 内核 (c) 先 `TORCH_SDPA` 保正确性 |
| **极高** | VSA 块稀疏注意力 + 布局融合（CUTLASS SM70 sparse） | 建议首期**不做**（参考实现自己没过质量门） |
| **极高** | peer 行规约（raw CUDA IPC，要求 ≥80 SM，`--fmad=false`） | **直接禁用**，退回 RCCL all_reduce |
| **高** | `_h3_flashinfer_C`（WMMA m16n16k16） | 无 WMMA ⇒ 放弃该后端（参考实现里它本就是回滚选项） |
| **中** | 主 GEMM（cuBLAS TENSOR_OP + FP32 累加语义） | rocBLAS/Tensile fp16 GEMM（gfx906 唯一大 M 现实路径，VGPR=32 且天然低寄存器） |
| **中** | cuBLASLt 零 workspace Volta 计划（硬编码 algo id） | hipBLASLt 无 gfx906 kernel（仅 gfx928）⇒ **删掉该快路径，只留 rocBLAS** |
| **中** | ConvRot256（32-lane radix-4 butterfly + FP32 中间量 + /16） | HIP 直译，但 gfx906 是 wave64，需把"32 lane 持一组 256 通道"改成"64 lane 持两组" |
| **中** | `qk_norm_rope`（`tl.gather` + `enable_fp_fusion=False`） | 改 `tl.where`/显式索引；必须保住"拒绝 FMA 收缩" |
| **中** | capability==70 硬校验（6 处：`quantization.py:203`、`h3_w8a16.cu:122`、`forward.cu:120`、`h3_column_major_gemm.h:24`、`sm70_diffusion.py:117`、`exact_row_reduce.cu:145`） | 统一改"平台判定 + 真实 ISA 判定" |
| **中** | 编码器 causal GQA（flash-attn v100） | 走 `F.scaled_dot_product_attention`（ROCm 落 math/efficient），需数值 A/B |
| **中** | VAE（conv3d + cuDNN 开关 + 远程 python 装载） | MIOpen 或 ATen 原生 conv（照抄 `causal_conv1d.py:190-198`）；**两个 safetensors 缺 config/auto_map，得补** |
| **中** | 主机内存驻留策略（阈值按 128 GiB 主机写的 pinned 路径） | 必须改"GGUF 直 mmap"，禁用 pinned host master（§6.6） |
| **低-中** | `modulation.py` 4 个 Triton kernel / `activation.py` silu_prepare | 直接跑；注意 `hidden=5376→next_power_of_2=8192` 的 VGPR 压力 |
| **低** | `prepare_fp16_rows` / `dequantize_rows|columns` / `scaled_add_rows` epilogue | HIP 直译（保留 frexp 与 `__fmaf_rn` 舍入顺序） |
| **低** | `pynvml` GPU 租约 / `h264_nvenc` / NVML 监控 | 换 `amdsmi`/`torch.cuda.get_device_properties` + libx264（默认本就是软编） |
| **极高（认知）** | 把 V100 的秒数/TFLOPS 当验收目标 | gfx906 无 tensor core，必须重定验收基线（§6.4） |

### 3.3 可基本原样复用

模型与采样数值逻辑：`transformer.py`、`modulation.py`（Triton 以外的部分）、`denoise_loop.py`、`sigma_schedule.py`、`packed_sequence.py`、`packed_tokens.py`、`conditioning.py`、`preprocess`/`validation`/`time_request.py`；量化契约与检查点解析：`quantization.py`（除 capability 门）、`comfy_checkpoint.py`、`prepared_weights.py`、`weight_cache.py`；编排：`pipeline.py`、`vllm/video/{engine,server,protocol,media}.py`；以及 **vLLM 的 TP 线性层本身**（本仓库已有 ROCm 实现）。

---

## 4. Z100 / gfx906 侧约束

### 4.1 ISA（机器码级审计结论）

| 能力 | 状态 | 对 H3 的含义 |
|---|---|---|
| `v_mfma*` / `v_wmma*` | **不存在** | DiT 大 M GEMM 只能 rocBLAS/Tensile 或手写 VALU/fdot2；attention 只能 Triton 或自研 |
| `v_dot4_i32_i8`（sdot4） | 存在（`_C.abi3.so` 中 71,442 条） | GGUF/INT8 量化内核的算力来源；手写内核实测 22.2 TFLOP/s |
| `v_dot2_f32_f16`（fdot2） | 存在（`__builtin_amdgcn_fdot2`） | fp16 GEMM/attention 的唯一"加速"指令；打包峰值 ≈26.1 TFLOP/s |
| 原生 bf16 / fp8 / fp4 | **全部不存在** | 文本编码器的 BF16 权重必须转 fp16 或仿真；FP8/FP4 路线出局 |
| 本仓库自研 GEMM | `bf16_emu_gemv` 仅 **M≤8**；`fp8_emu_gemm` 是 W8A8；skinny GEMM 是 MFMA-only 桩（已关） | **DiT 没有可复用的自研大 M GEMM**，必须走 rocBLAS |

### 4.2 DP4X / F24：栈绑定的故障面

| 项 | 事实 | 出处 |
|---|---|---|
| 故障 SKU | 卡 2/3 = Z100（`0x54B7`，DP4X=1）；卡 0/1 = Z100L（`0x55B7`，DP2X=0） | `hygon_dp4x.py:5-9,53-55` |
| 触发判据 | kernel descriptor VGPR 取整到 **256**（253/254/255 同罪）⇒ 首条 global store 后 wave 被中止、只写 72 元素、**dmesg 零故障**；`≤203` 全对；204–253 灰区未测 | `F24:121-144` |
| 栈依赖 | **DTK 26.04 用户态（HIP 6.3）有 DP4X loader 路径**；自编 ROCm 10.1（HIP 7.16）**没有任何 DP4X 代码** ⇒ F24 只在新栈发作 | `hygon_dp4x.py:11-18`、`F24:197-225` |
| guard 覆盖 | 只包 `triton.runtime.jit.JITFunction.run`（强制 `num_warps≥8`）；**C++ HIP / MIOpen / Tensile / RCCL 全裸奔** | `hygon_dp4x.py:342-415`、`F24:431-434` |
| RCCL 特例 | `ncclDevKernel_Generic_4<5120>` **VGPR=248**，纯 4-rank all_reduce 循环即可 XID:81→76 kill；三条修复路线（custom AR / NCCL_PROTO / 重编 RCCL）全失败 | `F24:483-547` |
| 第二故障模式 | DP4X 的 CWSR 抢占本身会 page fault：**对 Z100 worker 挂 profiler 即崩** | `F24:501-503` |
| 可行审计 | 静态 `vgpr_audit.py` 只看 Triton `.hsaco`；C++ 内核必须用 hip shim（`LD_PRELOAD=…/libhiplog.so`，`HIPLOG_VGPR_WARN=248`） | `F24:536-541` |

> **本报告的选型推论**：H3 移植应**锁定 DTK 26.04 栈**（与当前生产一致），从而绕开 F24 与 RCCL 248 两个已知无法修的问题；代价是不能用 `tools/dcu/env-rocm101.sh` 那套 triton 修复（DTK 树用 `triton-gfx906/install` 的 3.6.0 补丁版）。若必须走 ROCm 10.1 栈，则 H3 的 GGUF C++ 内核要逐一双重验证（DTK 上对拍 + shim 审计），成本显著更高。

### 4.3 其余可用性（逐项，均有代码门禁）

| 组件 | 状态 | 对 H3 |
|---|---|---|
| aiter | **硬性不可用**（`get_cdna_version()` 对 gfx906 返回 0） | ROCm 原生的 packed varlen attention（vLLM-Omni 的 AMD 路径）出局 |
| flash-attn | gfx906 不可用（AMD shim 对 gfx906 抛 KeyError） | encoder/DiT 的 FA 路径全部要走 Triton 或 SDPA |
| hipBLASLt | 只有 gfx928 kernel | 参考实现的 cuBLASLt 快路径整条删掉 |
| MIOpen / conv3d | 缺 gfx906 CK grouped-conv solver，运行时 JIT 偶发 `hipErrorInvalidKernelFile` | VAE 的 3D conv 是**未经验证的新面**（R8） |
| 主机内存 | 31 GiB（可用 6–8 GiB） | 见 §6.6 |
| 首跑 JIT | cache 冷时 4B 模型首启约 6 min；H3 内核数更多 | 冷启可能 20–60 min 级（R11） |

---

## 5. 量化路径三分叉（本报告的核心判断）

| 路径 | 做法 | 优点 | 主要风险 | 判定 |
|---|---|---|---|---|
| **A. GGUF w2a16 直接推理**（用户已下载） | 复用本仓库 `GGUFLinearMethod`：权重保持 Q2_K/Q5_K，运行时按需 MMQ（大 M）/MMVQ（小 M）/dequant+matmul | 权重仅 25.13 GiB；主机压力最小（mmap）；**本仓库内核已覆盖 Q2_K/Q5_K**（`gguf.py:758-781`、`dequantize.cuh:102`、`mmq.cuh:359-372`） | 需新写 DiT 权重加载器（GGUF 名 → vLLM 参数名）+ 新模型实现；C++ 内核 VGPR 未审计；MMQ 在 gfx906 的实测效率未知 | ✅ **唯一可行主路径** |
| **B. 离线转 ComfyUI INT8-ConvRot** | 反量化 GGUF → 重排 → 按 ConvRot 量化 → 走参考实现的 W8A16 契约 | 可直接复用 1cat 的量化契约与（移植后的）ConvRot 内核 | 需要原始 BF16/HF 权重做量化校准（本机没有，且官方需授权）；多一道离线工序；ConvRot 坐标系仍需自证 | 🔶 可选优化臂 |
| **C. 反量化成 fp16/BF16 常驻** | 一次性转 fp16 分片 | 内核最简单（rocBLAS 全包） | **96.9 GB 权重**：DiT 40.2 + TE 51.5 ⇒ 单卡必爆、主机必爆 | ❌ 出局 |
| D. 在线 FP8 | 运行时量化 | 参考 vLLM-Omni 能省 22 % | gfx906 无 FP8，仿真 38× 慢（`z100-rocm101-acceptance.md:81`） | ❌ 出局 |

**结论**：走 A；B 作为"MMQ 效率不达标时的兜底"；C/D 不进入路线图。

> 路径 A 的三条硬约束（写进设计红线）：
> ① **绝不物化 fp16 权重副本**（与 `docs/archive/海光Z100-性能优化方案.md` §4B 的既有约束一致）；
> ② DP4X 卡上必须先把 GGUF C++ 内核的 VGPR 审计掉（或只在卡 0/1 上跑）；
> ③ GGUF 无元数据 ⇒ 加载器必须从张量形状重建 config，不能用 `config.json` 的常规路径。

---

## 6. 显存 / 算力预算

### 6.1 权重（GiB，原始量化字节）

| 组件 | 原始 | 全驻 | TP=1 | TP=2 | TP=4 |
|---|---:|---:|---:|---:|---:|
| DiT（可按头分片） | 8.063 GB | 7.51 | 7.51 | 3.75 | 1.88 |
| 文本编码器（可分片） | 13.102 GB | 12.20 | 12.20 | 6.10 | 3.05 |
| 视频 VAE（tile 并行 ⇒ **需复制**） | 5.208 GB | 4.85 | 4.85 | 4.85 | 4.85 |
| 音频 VAE（复制） | 0.605 GB | 0.56 | 0.56 | 0.56 | 0.56 |
| **合计** | 26.98 GB | **25.13** | 25.13 | **15.26** | **10.34** |

若允许"分阶段装载并释放"（TE → 释放 → DiT → 释放 → VAE），TP=2 峰值可压到 `max(3.75, 6.10, 5.41) + 激活`。

### 6.2 激活峰值（`h = N·5376·2/TP`，qkv/fc1/qkv 张量/residual FP32 逐项累加）

| N | TP=1 | TP=2 | TP=4 | 朴素 N² score 矩阵 |
|---|---:|---:|---:|---:|
| 1,770 | 0.31 GB | 0.16 | 0.08 | 0.01 GB |
| 12,323 | 2.16 GB | 1.08 | 0.54 | 0.30 GB |
| 73,483 | **12.90 GB** | 6.45 | 3.23 | **10.80 GB/头** ⇒ 必须 flash 分块 |

### 6.3 每卡总占用（权重 + 激活 + 固定缓冲 ~1.5–2.5 GB）

| 配置 | 安全卡 | (a) | (b) | (c) | 结论 |
|---|---|---:|---:|---:|---|
| TP=1 | 卡 0 或 1 | ~27 GiB ✓ | ~29 GiB ⚠ | **39.5 GiB ✗** | 只有 (a) 能跑；`t2.log` 实测 864×480 即 OOM 印证 |
| **TP=2（权重分片 + VAE 复制）** | **卡 0+1 ✅** | ~19 GiB ✓ | ~19 GiB ✓ | **~24 GiB ✓** | **全档可行且只用安全卡**（推荐） |
| TP=2（分阶段装载） | 卡 0+1 ✅ | ~12 GiB ✓ | ~13 GiB ✓ | ~15 GiB ✓ | 更宽松，需处理换入换出 |
| TP=4 | 卡 2/3 含 DP4X | ~13 GiB ✓ | ~13.4 GiB ✓ | ~16 GiB ✓ | 余量最足，但一半 GEMM 落在 DP4X 卡 |

（单卡可用显存实测 31.98 GiB；sd.cpp 自报 budget 31.2 GiB。）

### 6.4 有效算力取值（gfx906）

| 项 | 值 | 来源 |
|---|---|---|
| CU 数 | 60（KFD `simd_count=240`） | `[实测]` |
| 打包 fp16 峰值 | ≈26.1 TFLOP/s（60 CU × 64 lane × 4 FLOP × 1.7 GHz，时钟为估算） | `[估算]` |
| int8 手写 sdot4 实测 | **22.2 TFLOP/s**（理论 42 %） | `[实测]` |
| **本负载 ggml 反推** | **6.6–8.1 TFLOP/s/卡** | `[实测反推]`（4 组配置散布 <20 %） |
| V100 对照 | 51.9–59.5 TFLOP/s/卡 | `[实测]`（1cat 文档） |
| **E1 保守**（现状内核效率） | **7.3 TFLOP/s/卡** | 本报告统一口径 |
| **E2 工程化** | **10–13 TFLOP/s/卡** | 26.1 × 40–50 % |

> 与 `sd.cpp` 日志的交叉校验通过：N 从 1,770→7,171（×4.05）时模型预测单步 FLOPs ×4.78、实测单步时间 ×5.05（偏差 5.7 %）。

### 6.5 端到端耗时

| 负载 | 总 FLOPs | TP=2 @7.3T | TP=2 @11T | TP=4 @11T | **端到端（TP=2 @7.3T）** |
|---|---:|---:|---:|---:|---|
| (a) 4 步 | 0.30 PF | 21 s | 14 s | 7 s | **≈ 68 s**（加载 17 + TE 12 + 去噪 21 + VAE 18） |
| (b) 20 次 | 14.3 PF | 977 s | 649 s | 324 s | **≈ 17 min** |
| (c) 49 次 | 524 PF | 35,900 s | 23,823 s | 11,911 s | **≈ 10.1 h**（TP=4 @11T ≈ 3.4 h） |

> (a) 的估算 68 s 与 sd.cpp 单卡实测 wall 74.77 s 同量级（sd.cpp 的 DiT 是单卡，见下条）。

### 6.6 与 `sd.cpp` 实测的关系（避免误判"已经能跑就行"）

| 项 | sd.cpp/ggml 实测 | 说明 |
|---|---|---|
| 640×384×22 / 4 步 | 去噪 46.08 s（10.0 s/步）；开 `--diffusion-fa` 后 42.65 s（9.37 s/步，计算缓冲 −52 %） | 单卡 DiT |
| 864×480×56 / 8 步 | 去噪 414.86 s（**50.5 s/步**），wall 474 s，`MAXRSS 7.2 GB` | **DiT 全在单张 ROCm0 上跑**（TE 在 ROCm1、VAE 在 ROCm2）：ggml **没有 DiT 张量并行** |
| 主机侧 | `RAM 0.00MB`、MAXRSS 3.5–6.9 GiB、权重读带宽 1.7–2.1 GB/s（冷加载 26.98 GB ≈ 15 s） | **证明"GGUF 直 mmap + 权重驻 VRAM"在本机可行**，这是路径 A 的实测前提 |
| vLLM 化后的收益 | — | 纯并行度：同效率下 864×480×56 可从 415 s → 207 s（TP=2）/ 68 s（TP=4）；再加 FA 省缓冲 |

> **这个对照很重要**：用户手上已有"能出片"的基线（ggml/sd.cpp，1 分 15 秒出 22 帧）。适配 vLLM 的价值不是"从不能跑变成能跑"，而是 **TP 并行 + 服务化（吞吐/并发/媒体管线）+ 与生产栈同栈**。若只要"能出片"，sd.cpp 已经是现状答案（§7 D0 档）。

---

## 7. 可行性判定（分档）

| 档 | 组合 | 判定 | 依据 |
|---|---|---|---|
| **D0｜已经可行（现状）** | `stable-diffusion.cpp` + 这份 GGUF，单卡 DiT | ✅ **今天就能出片**（`/mnt/game/h3-test/sd-cli`，22 帧 75 s） | §6.6 实测；不是本次适配的增量 |
| **D1｜最小增量（建议作为第一步）** | vLLM 侧新模型骨架 + `GGUFLinearMethod` + `TORCH_SDPA`/分段 attention，**TP=1，(a) 档 only** | ✅ **可行，1–2 周量级** | TP=1 显存 ~27 GiB ✓；SDPA 慢但 (a) 档 N=1,770 可接受 |
| **D2｜工程可行（目标档）** | 接 Triton unified attention（非因果）+ MMQ 大 M 内核，**TP=2 on 卡 0/1，(b) 档** | 🔶 **可行但需要注意力工程**（最大工作量） | 显存 24 GiB ✓；算力 11–17 min/条；风险 R5/R6 |
| **D3｜不建议** | VSA 稀疏、243 帧 (c) 档当验收、fp16 反量化、TP=4 用故障卡、在线 FP8 | ❌ | 质量未过 / 过夜任务 / 显存不可能 / DP4X / 无硬件 |

**一句话回答"Z100 能不能跑量化版 MiniMax-H3"**：**能，而且今天就有 ggml 的实测基线；vLLM 化的难点不在权重格式（Q2_K 本仓库已通）也不在显存（TP=2 就够），而在"非因果全序列 attention 的高性能 gfx906 实现"与"DP4X/主机内存这两个平台约束"。**

---

## 8. 风险清单

| ID | 风险 | 严重度 | 触发条件 | 后果 / 检测 / 缓解 |
|---|---|---|---|---|
| **R1** | **GGUF 无元数据 + ConvRot 坐标系未知**：无法判断权重是否已做 Hadamard-256 旋转 | **高** | 直接按"未旋转"加载 | 输出全错但不报错（画面噪点/音频失真）。检测：取一个小 linear 层与官方 Comfy INT8 检查点做旋转前后能量/相关性比对（**纯 CPU 可做**）。缓解：先在 sd.cpp 的 QC 日志上验证该 GGUF 的中性输出，再决定是否补 ConvRot |
| **R2** | **GGUF C++ 内核（MMQ/MMVQ/dequant）VGPR 从未审计**，不在 DP4X guard 内 | **高**（仅 ROCm 10.1 栈；DTK 栈降为中） | 在卡 2/3 上跑 Q2_K 权重 | 静默错算。检测：hip shim（`HIPLOG_VGPR_WARN=248`，判据 `numRegs≤203`；204–253 视为不可用）。缓解：**只在卡 0/1 上跑**；或 `ENABLE_GGUF_FORCE_DEQUANT_MATMUL=1` 把计算挪到 rocBLAS（Tensile VGPR=32） |
| **R3** | **DiT 非因果全序列 attention 没有现成高性能后端** | **高** | 采用本仓库 unified attention | 需伪 per-step paged KV（`block_table`/`seqused_k`），额外显存 `2·S·H·D`、每步全量写读；S 大时 32 GiB 卡紧张且带宽主导 |
| **R4** | **非因果 attention 无数值门禁**：bidirectional 分支只在 DFlash drafter 场景验证过 | **高** | 直接假设等价于 SDPA | "不报错但数值不对"（gfx906 有前科：triton overlay 缺失导致 attention 全错/NaN） |
| **R5** | **主机内存 31 GiB / 可用 6–8 GiB / swap 已满** | **高** | 用 pinned host master 或 fp16 副本 | 加载期 OOM / swap 抖动。缓解：GGUF 直 mmap（sd.cpp 实测 MAXRSS 3.5–6.9 GiB）；`prepared_weight_cache_gib` 从 128 下调到 ≤20 并指 `/mnt/game` |
| **R6** | **MMQ 在 gfx906 上的实际效率未知**（本仓库 ROCm tile 64×128×8 warps） | 中高 | 依赖 MMQ 跑大 M | 若远低于 7.3 TFLOP/s，则 (b) 档从 17 min 退化。缓解：A/B `ENABLE_GGUF_FORCE_DEQUANT_MATMUL`（dequant+rocBLAS） |
| **R7** | **文本编码器 25.75 G / 13.1 GB 是权重预算最大块**，且其计算量仅 0.8 TFLOP（几乎全是权重搬运，实测 12 s） | 中高 | 每请求加载 TE | 必须常驻。缓解：纯 t2va 跳过视觉塔 351 张量（sd.cpp 已 `551/902` 这么干）；TE 用 TP 分片 |
| **R8** | **VAE 3D conv / ViT 解码器走 MIOpen，gfx906 有 JIT 失败前科** | 中 | VAE 解码 | 崩溃或 NaN（延迟到下一个 op 才抛）。缓解：强制 ATen 原生 conv；VAE 保持 fp32；VIP 解码器 2.42 G 参数要预留 4.85 GiB |
| **R9** | **两个 VAE safetensors 缺 `config.json` + `auto_map`** | 中 | 沿用参考实现的"动态加载 checkpoint 自带 python" | 装不上。缓解：补 config 或本地重写 VAE adapter |
| **R10** | **device id 硬校验**：`major==7&&minor==0`、`get_min_capability()==70` 共 6 处 | 中 | 直接搬代码 | 初始化即拒绝。缓解：统一改平台/ISA 判定 |
| **R11** | **首跑 JIT 成本**：H3 内核数远多于 LM，冷启可能 20–60 min | 中 | 首次部署/换 dtype/换 shape | 误判为挂死。缓解：预热 `TRITON_CACHE_DIR` + 固定 shape 集合 warmup 落盘 |
| **R12** | **无空闲 GPU**：4 卡被生产 27B-FP8 TP=4 占满（VRAM 100 %） | 中高 | 任何 H3 验证 | 无法测或受干扰。缓解：协商维护窗口；或固定卡 0/1 并下调 `--gpu-memory-utilization` |
| **R13** | **权重/缓存落盘位置**：根分区仅 18 GB | 低中 | prepared cache 默认路径在 `/` | 写失败。缓解：全部指 `/mnt/game`（328 GB） |
| **R14** | **验收口径不可复用 V100 的秒数/TFLOPS**；且本平台"重启后 bit 一致"不成立 | 中 | 用 V100 的 58.29 s / 59.5 TFLOP/s 当门禁 | 永远过不了。缓解：改用"同输入 A/B + 固定栈 + eager + 与 sd.cpp 中性输出的相对 L2"作为门禁 |

---

## 9. 下一步最小验证矩阵（按成本从低到高，每步都有判据）

> 前置：4 卡当前被生产服务占满（`hy-smi` VRAM 100 %），任何上卡验证都要先协商窗口；主机可用内存 6–8 GiB，验证前需卸载/降载生产 worker。

1. **【0.5 天 / 不需要 GPU】ConvRot 坐标系判定（关闭 R1）**
   从 GGUF 取 `blocks.0.attn.qkv_proj` 的一行权重，与官方 Comfy INT8 检查点同位置行做 Hadamard-256 前后的相关/能量比对。**判据**：未旋转相关性 ≈1 → 不需要 ConvRot；否则必须补。
2. **【0.5 天 / 不需要 GPU】GGUF→vLLM 参数名映射表 + 形状反推 config（关闭"无元数据"）**
   按 `.s24/budget.md:74-88` 的映射生成 `MiniMaxH3Config` 纯 Python 桩，加载 532 张量的形状校验。**判据**：532/532 张量全部映射成功，无残留。
3. **【1 天 / 需要 1 张卡（卡 0）】Q2_K 线性层数值门禁**
   单卡上对 `qkv[5376,21504]` 等 4 个形状跑 `GGUFLinearMethod`（MMQ / dequant+matmul 两臂），与 CPU dequant 参考实现比对。**判据**：`rel_err ≤ 1e-2`；两臂耗时比记录为 R6 的基线。
4. **【1 天 / 卡 0】非因果 attention 数值门禁（关闭 R4）**
   在 `triton_unified_attention` 上用小形状（D=128、S≤2048、fp16、非因果 + cu_seqlens）对拍 `F.scaled_dot_product_attention(math)`。**判据**：`max abs err` 在 fp16 舍入量级（≤2e-2）。
5. **【2–3 天 / 卡 0】D1 档端到端最小片**
   vLLM 侧最小模型骨架（只 fl2va、TP=1、SDPA 分段 attention）+ GGUF DiT + GGUF TE + VAE（ATen conv），生成 640×384×22/4 步。**判据**：能出片、与 sd.cpp 中性输出相对 L2 ≤0.05、显存与耗时落在 §6.3/§6.5 的 (a) 档 ±30 %。
6. **【3–5 天 / 卡 0+1】D2 档（TP=2 + Triton attention + MMQ）**
   **判据**：1344×768×39/21σ 在 25 min 内完成（@7.3T 预算 17 min，留 50 % 余量）；峰值显存 ≤26 GiB/卡；无 dmesg 故障。
7. **【文档】把结论回写 README**
   新增"MiniMax-H3（GGUF w2a16）"条目：可用档 D1/D2、明确不可用（VSA、(c) 档、fp16 副本、在线 FP8）、以及"卡 0/1 与卡 2/3 的 SKU 差异在 DTK 栈下的状态"。

---

## 10. 证据路径

| 内容 | 路径 |
|---|---|
| 参考实现整体（SM70 契约 / 计算图 / 性能数字 / 重写清单） | `/mnt/game/1cat-vllm/vllm/model_executor/models/minimax_h3/`、`docs/design/minimax_h3/{README,CURRENT_STATUS,VALIDATION,CAMPAIGN_RESULTS,CONTROL,VSA_QUALITY_SPEED,LAYER_WEIGHT_OFFLOAD,SHARED_HOST_WEIGHTS}.md` |
| 三段自制扩展 | `tools/minimax_h3/build_extensions.py:25-50`、`csrc/sm70_turbomind/ops/h3_w8a16.cu`、`flash-attention-v100/kernel/h3/forward.cu`、`flashinfer-sm70/csrc/h3_noncausal_sm70.cu` |
| 量化契约（INT8-ConvRot） | `minimax_h3/{quantization.py,comfy_checkpoint.py}`、`vllm/model_executor/layers/sm70_diffusion.py` |
| 本仓库 GGUF 支持 | `vllm/model_executor/layers/quantization/gguf.py:749-781`、`:799-807`、`:2122-2450`；`csrc/quantization/gguf/{dequantize.cuh:102,mmvq.cuh:290,mmq.cuh:359-372}` |
| gfx906 attention 落点 | `vllm/v1/attention/ops/triton_unified_attention.py:312-320,1238-1251`；`vllm/v1/attention/backends/fa_utils.py:38-58`；`vllm/model_executor/models/diffusion_gemma.py:6-9` |
| DP4X / F24 | `vllm/platforms/hygon_dp4x.py`、`docs/result/F24-investigation-20261018.md` §1、§2.2、§12、§13 |
| ISA 审计 | `docs/archive/result/z100-11d-z100-isa-capability-audit-result.md:84-222` |
| 栈身份与栈切换 | `tools/dcu/use_stack.sh`、`tools/dcu/serve_host_dflash2.sh:1-30`、`docs/result/z100-rocm101-acceptance.md:39-50` |
| ggml 实测（同机同 GGUF） | `/mnt/game/h3-test/{smoke,t2,t3,t2c}.log`、`run_*.sh` |
| 量化体解析与预算原始数据 | `.s24/budget.md`（含逐张量表、token 公式、FLOPs 与交叉校验） |
