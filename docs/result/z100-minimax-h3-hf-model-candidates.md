# MiniMax-H3 本地可部署模型清单（面向 Z100 / gfx906 4×32 GiB）

> 日期：2026-10-19 · 用途：给本机（`/home/lyh/z100-vllm`，4×Hygon Z100，主机内存 31 GiB，`/mnt/game` 余 328 GB）挑 H3 权重
> 判定口径：见 §0；数据来源：ModelScope 镜像 API 逐个文件核到的**字节数**（HF 仓库直连在本机不可用，镜像与 HF 同步），HF 仓库 ID 同时给出
> 关联：`docs/result/z100-minimax-h3-quant-feasibility.md`（可行性）、`.s24/budget.md`（显存/算力预算）

## 0. 判定口径（先看这个，否则表里的 ✅ 会看不懂）

| 约束 | 取值 | 影响 |
|---|---|---|
| 卡 | 4×32 GiB，**只有卡 0/1 是 Z100L（无 DP4X 缺陷）** | **有效配置 = TP=2 on 卡 0/1**，每卡可用 ≈30 GiB |
| 主机内存 | 31 GiB（当前可用 6–8 GiB） | **不能**持有 fp16/bf16 主机 master；权重只能 GGUF 直 mmap |
| 计算精度 | 无 MFMA/WMMA、无原生 bf16/fp8/fp4 | **FP8 / NVFP4 / BF16 权重一律出局** |
| 唯一可行格式 | **GGUF w2a16**（本仓库 `GGUFLinearMethod` 已支持 Q2_K…Q6_K/Q8_0） | 路径 A；ConvRot/INT8 需额外内核移植 |
| 显存预算公式 | `DiT/2 + TE/2 + VAE 5.41 GiB（需复制）+ 激活`；激活 (b)N=12323 为 1.08 GiB、(c)N=73483 为 6.45 GiB | 见各表末两列 |

判定图例：✅ 路径 A 下可直接用（只需加载器/模型移植，无新内核）· 🔶 可用但要先做一件事（自写 ConvRot/W6A8 内核、换调度、额外 VRAM 余量）· ❌ 本机不建议/不可行
（GiB = GB × 0.9313；"TP=2 后/卡"指权重分片后单卡占用）

---

## 1. 主 DiT：GGUF 阶梯（**首选**，路径 A 唯一无阻塞的格式）

### 1.1 unsloth/MiniMax-H3-GGUF（推荐，覆盖 FL2VA 与 Ref2VA 两套）

| 档位 | FL2VA 字节 (GB) | Ref2VA 字节 (GB) | TP=2 后/卡 (GiB) | +(b) 峰值 GiB | 判定 | 说明 |
|---|---:|---:|---:|---:|---|---|
| Q2_K | 6.72 | 6.68 | 3.13 | **15.7** | ✅ 最省显存 | 你现在这份 `UD-Q2_K_XL` 的低一档；画质最低 |
| **UD-Q2_K_XL** | **8.06（你已有）** | — | 3.76 | 16.3 | ✅ 已在用 | 与本地文件字节完全一致（8,063,029,344） |
| Q3_K | 8.76 | 8.72 | 4.08 | 16.7 | ✅ | — |
| UD-Q3_K_XL | 9.56 | — | 4.45 | 17.0 | ✅ **建议起点** | dynamic 混合精度，同体积画质优于 Q3_K |
| Q4_K | 11.42 | 11.38 | 5.32 | 17.9 | ✅ | 质量/体积甜点 |
| Q5_0 | 13.92 | 13.89 | 6.48 | 19.1 | ✅ | — |
| Q6_K | 16.59 | 16.55 | 7.73 | 20.3 | ✅ | — |
| Q8_0 | 21.44 | 21.41 | 9.99 | 22.6 | 🔶 (c) 档 27.9 GiB，余量小 | 接近 bf16 画质，显存最贵 |
| TE：`qwen3vl_32b_minimax_h3-Q2_K_M.gguf` | 13.10（你已有） | — | 6.10 | ↑ 已计入 | ✅ | 含视觉塔 351 张量，vLLM 可直接用 |
| TE：`qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 18.22 | — | 8.48 | +2.4 | ✅ 质量更好 | 配合 Q4_K 以下 DiT 时总峰值 ≈20.3 GiB |

### 1.2 Abiray/MiniMax-H3-Pruned-GGUF（备选量化器，K-quant 模板更细）

| 档位 | FL2VA (GB) | Ref2VA (GB) | TP=2 后/卡 (GiB) | +(b) 峰值 GiB | 判定 |
|---|---:|---:|---:|---:|---|
| Q3_K_M | 8.90 | 8.90 | 4.15 | 16.8 | ✅ |
| Q4_K_S / Q4_K_M | 11.56 / 11.56 | 11.56 | 5.38 | 18.0 | ✅ |
| Q5_K_S / Q5_K_M | 14.07 / 14.07 | 14.07 | 6.55 | 19.2 | ✅ |
| Q6_K | 16.73 | 16.73 | 7.79 | 20.4 | ✅ |
| Q8_0 | 21.58 | 21.58 | 10.05 | 22.7 | 🔶 |

> 与 unsloth 的关系：unsloth 用 `Q*_K`（老式 K-quant）+ `UD-*`（dynamic）；Abiray 用 `Q*_K_S/M`。**同一档体积差异 <3%**，选一个即可，不要混用（张量布局相同但量化模板不同，画质 A/B 要各自成对比较）。

---

## 2. 官方原版（Comfy-Org/MiniMax-H3）——本机大多不适合，列出来是为了"别下错"

| 文件 | 字节 (GB) | 格式 | TP=2 后/卡 (GiB) | 判定 | 原因 |
|---|---:|---|---:|---|---|
| `minimax_h3_fl2va_bf16` | 66.28 | BF16 | 30.87 | ❌ | 单卡就超 30 GiB，且 gfx906 无原生 bf16 |
| `minimax_h3_fl2va_int8_convrot`（未剪枝） | 34.04 | INT8+ConvRot | 15.85 | ❌ | 加 TE/VAE 必爆；且有 ConvRot 依赖 |
| `minimax_h3_fl2va_pruned_bf16` | 40.23 | BF16 | 18.73 | ❌ | 权重+TE+VAE ≈30.2 GiB 已贴顶，且 bf16 无算力 |
| `minimax_h3_fl2va_pruned_fp8_scaled` | 20.96 | FP8 scaled | — | ❌ | **gfx906 无 FP8 硬件**，只能用仿真（38× 慢） |
| `minimax_h3_fl2va_pruned_int8_convrot` | 20.97 | INT8+ConvRot | 9.77 | 🔶 | 需自写 ConvRot256 内核；配 GGUF TE 时峰值 22.4 GiB ✅ |
| `minimax_h3_fl2va_pruned_w6a8` | 15.98 | W6A8 | 7.44 | 🔶 | 体积最省，但需自写 W6A8 反量化内核（本仓库无此格式） |
| `minimax_h3_ref2va_*`（同六档） | 同上 | 同上 | 同上 | 同上 | Ref2VA 分区与 FL2VA **不能同卡共存**，要换模型 |
| `text_encoders/qwen3vl_32b_bf16` | 51.51 | BF16 | 23.99 | ❌ | 单卡 24 GiB + VAE 就爆 |
| `text_encoders/qwen3vl_32b_int8_convrot` | 27.14 | INT8+ConvRot | 12.64 | 🔶 | 与 pruned_int8 DiT 同用时 28.9 GiB，余量危险 |
| `text_encoders/qwen3vl_32b_nvfp4_awq` | 15.69 | NVFP4 | — | ❌ | 无 FP4 硬件 |
| `vae/minimax_h3_video_vae_fp16` | 5.21 | F16 | 4.85（复制） | ✅ 必需 | 你已有 |
| `vae/minimax_h3_audio_vae_fp32` | 0.61 | F32 | 0.56（复制） | ✅ 必需 | **必须 fp32，换 fp16 会没声音** |

---

## 3. 融合 / 蒸馏加速模型（Z100 上收益最大的一类）

| 模型（HF ID） | 字节 (GB) | 内容 | 步数 | TP=2 后/卡 (GiB) | 判定 | 备注 |
|---|---:|---|---:|---:|---|---|
| `MATLOWAI/minimax-h3-fused-turbo-int8-convrot` | 20.98 | FL2VA+Ref2VA **二合一** + 8 步 Turbo 蒸馏 + Mystic LoRA（4 步出片） | 4 | 9.77 | 🔶 **强烈建议但要 ConvRot 内核** | 省掉"两套 DiT 各 20 GB"；4 步把 (b) 从 20 次调用砍到 4 次 |
| `OpenVDN/vdn-minimax-h3`（stage-dmd-step-250） | ~5.0（LoRA+linear 分支） | 8 步 DMD 学生 + 窗口/线性混合注意力 | 8 | 基线 DiT + ~5 GB | ❌ 不建议 | 需要 `VDNH3_ATTN` 专用后端，等于再写一套 attention |
| `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` | 1.49 | 4 步 LoRA（adapter_model.safetensors） | 4 | ~1.4 GB 额外 | 🔶 | 需 DiT LoRA 支持才能用 |
| `FlashGen/Minimax-H3-4step-lora-flashgen`（`minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16`） | 1.26 | 4 步 LoRA | 4 | ~1.2 GB 额外 | 🔶 | 同上 |

> **4 步蒸馏对 Z100 的意义（关键）**：1344×768×39 帧的算力从 20 次调用 14.3 PF 降到 **4 次调用 2.85 PF** ⇒ TP=2 @7.3 TFLOP/s/卡 从 ~16 分钟降到 **~3.3 分钟**；243 帧从 ~10 小时降到 **~49 分钟**。这是本机唯一能让 H3 变"可交互"的杠杆。

---

## 4. 官方加速 LoRA（Comfy-Org，小体积、与 DiT 正交）

| 文件 | 字节 (GB) | 用途 | 判定 |
|---|---:|---|---|
| `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16` | 1.96 | 官方 4 步蒸馏（768p） | 🔶 需 DiT LoRA 支持 |
| `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16` | 1.96 | 官方 8 步蒸馏 | 🔶 同上 |
| `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16` | 1.96 | Ref2VA 的 4 步 | 🔶 同上 |

## 5. 后处理/放大（省算力的正确姿势）

| 模型（HF ID） | 字节 | 作用 | 判定 | 备注 |
|---|---:|---|---|---|
| `LBH-123-AI/Minimax_h3_latent_Upscaler`（`minimax_h3_latent_upscaler_3d_bf16.safetensors`） | 0.69 GB | 在 **24 通道 latent 上** 1×–4× 放大，避免"解码→像素放大→重编码"再付一次 5B VAE 的代价 | ✅ **可行** | 纯 3D conv（torch/conv3d），无 CUDA 专属内核；低分辨率出片 + latent 放大是本机性价比最高路线 |
| TAE 微型预览解码器（Kijai KJNodes） | <0.1 | 快速预览 | ⚪ 仅预览 | 不用于出片 |

---

## 6. 建议组合（照抄即可）

| 配方 | 组成 | 每卡峰值 (b) | 预期耗时 (b: 1344×768×39) | 适合场景 |
|---|---|---:|---|---|
| **S1 最稳（今天就能试）** | `UD-Q2_K_XL`（你已有）+ TE `Q2_K_M`（已有）+ 两个 VAE（已有） | 16.3 GiB | 20 次 ≈ **16 分钟**（TP=2 @7.3T） | 跑通链路、对拍 sd.cpp 输出 |
| **S2 质量档** | unsloth `Q4_K`（11.42 GB，**需下载**）+ TE `Q4_K_M`（18.22 GB，需下载）+ VAE | 20.3 GiB | 同上 ≈16 分钟 | 画质优先的验收 |
| **S3 速度档（推荐目标）** | `UD-Q3_K_XL` 或 `Q4_K` + `minimax_h3_fl2v_turbo_4step` LoRA + TE `Q2_K_M` | 17–18 GiB | 4 次 ≈ **3.3 分钟** | 交互式试用 |
| **S4 一步到位（需内核工作）** | `MATLOWAI` fused turbo int8（二合一，4 步）+ TE `Q2_K_M` | 22.4 GiB | 4 次 ≈ **3.3 分钟** | 同时覆盖 t2va/i2v/ref2va，省一套 DiT |
| S5 高清 | S3 + latent upscaler（0.69 GB）→ 放大后再轻步数精修 | ≈同上 | +1–2 分钟 | 出 1080p |

> 磁盘：S1 已占 26.98 GB；S2 再 +29.6 GB；S4 再 +21 GB。`/mnt/game` 余 328 GB，全部放 `/mnt/game`（**不要放根分区，只剩 18 GB**）。

---

## 7. 必须知道的三个坑

1. **文本编码器的 GGUF 只有 vLLM 能用，ComfyUI 不能用**：社区 ComfyUI 教程普遍写"H3 的 GGUF 编码器缺 mmproj 视觉塔、`ClipLoaderGGUF` 必崩"。**对我们不成立**——你已有的 `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` 里实测含 351 张量视觉塔（`visual.patch_embed/pos_embed/blocks*/merger/deepstack_merger_list`，全 BF16），vLLM 侧可直接加载。反之若要转 ComfyUI 路线，就得改用 Comfy-Org 的 safetensors 编码器（int8_convrot 27.14 GB / bf16 51.51 GB，后者本机不可行）。
2. **FP8/NVFP4 一律别下**：本机无 FP8/FP4 硬件，这两个格式只能仿真，慢一个量级。
3. **许可**：MiniMax H3 Community License 带 "Applicable Territory"（地区排除条款）与商业营收门槛，商用前读仓库内 `LICENSE` 全文（`unsloth`、`Comfy-Org`、`MATLOWAI` 都沿用该许可）。第三方再量化/融合模型（Abiray、MATLOWAI）另需遵守各自 NOTICE。

---

## 8. 数据来源与复核方法

| 内容 | 方法 |
|---|---|
| 所有字节数 | ModelScope 镜像 API `repo/files` 逐文件返回的 `Size`（unsloth/MiniMax-H3-GGUF、Comfy-Org/MiniMax-H3、MATLOWAI/minimax-h3-fused-turbo-int8-convrot、Abiray/MiniMax-H3-Pruned-GGUF），HF 同名仓库同步 |
| 你已有文件 | 本地 `ls -l /mnt/game/h3-test/models/`：`UD-Q2_K_XL` = 8,063,029,344；TE `Q2_K_M` = 13,102,161,024；video VAE 5,207,742,160；audio VAE 605,149,280 —— 与镜像字节一致 |
| 每卡峰值 | `DiT/2 + TE/2 + VAE 5.41 GiB + 激活`（GiB = GB×0.9313；激活按 N=12,323 计 1.08 GiB，见 `.s24/budget.md` §7） |
| 耗时 | 每步 `2·P·N + 50·4·N²·7168`，P=20.11 G，N=12,323；有效算力取同机 ggml 实测 7.3 TFLOP/s/卡（`.s24/budget.md` §5/§8） |
| 蒸馏件体积 | `FastVideo` / `FlashGen` 取自 `/mnt/game/1cat-vllm/vllm/model_executor/models/minimax_h3/UPSTREAM.md` 的冻结 revision（1,485,626,152 / 1,258,539,599 B）；latent upscaler 691 MB 取自 vLLM-Omni 官方 recipe |
| 未核实项 | `OpenVDN/vdn-minimax-h3` stage 目录约 5 GB（标注为"约"，未逐个文件核字节） |
