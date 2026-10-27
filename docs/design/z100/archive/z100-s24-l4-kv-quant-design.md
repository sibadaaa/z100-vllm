# S24 L4 设计：KV cache 量化（gfx906 fp8 存储读路径 dequant）

> 状态：设计稿（W3，另立质量门禁，本轮不实现）。
> 关联：计划 §L4；S24 实测的容量/带宽账（本结果文档 §1/§4/§5）。

> **R5 前置核查（2026-10-20，降低工作量估计）**：本 fork 的量化 KV 基建比本稿
> 预估的完整——
> * `triton_unified_attention.py` 的 2D/3D 内核已有 fp8 KV 读路径：
>   `K_load.dtype.is_fp8()` → `K = K_load.to(f32) * tl.load(k_scale)`（per-tensor
>   descale，经 `k_descale/v_descale` 标量传入），`unified_attention` 签名已接受
>   `kv_quant_mode`（NONE / FP8_PER_TENSOR）；
> * `TritonAttentionBackend.supported_kv_cache_dtypes` 已含 `fp8`、`fp8_e4m3`、
>   `fp8_e5m2`、`fp8_per_token_head`、`int8_per_token_head`、`int4_per_token_head`；
>   SM<89 的 fp8 拒绝分支只在 `current_platform.is_cuda()` 下生效，**gfx906
>   (ROCm, cap 9.0) 不会被拒**；
> * `triton_attn_diffkv.py`（DiffKV 变体后端）与 `is_quantized_kv_cache` /
>   `kv_cache_uses_per_token_head_scales` 工具已在树上；
> * `k_scale_cache/v_scale_cache`（[num_blocks, block_size, num_kv_heads] fp32）
>   参数已在 `unified_attention` 签名中，但设置时 raise NotImplementedError
>   ⇒ per-token(-head) 粒度需要实现这两个 cache 的内核分支；
> * 剩余工作集中为：① KV 写路径量化（含 per-tensor scale 的来源：静态校准或
>   `--calculate-kv-scales` 动态）；② per-token-head 内核分支（若 per-tensor
>   过不了质量门禁）；③ `--kv-cache-dtype fp8_e4m3` 与 `--mamba-cache-mode align`
>   的页对齐交互复查。


## 1. 依据（S24 实测口径）

* 每 token 每卡 KV = 22.34 KB（full 16 层×1 KB + draft 5 层×1 KB bf16 + GDN 1.34 KB）。
  池 15.54 GiB/卡 = 748,982 token；MBT=4096 后 16.21 GiB = 783,993 token。
* 长上下文 decode/verify 是重读 KV 的带宽问题：32k 单步 KV 读 ≈0.5 GB/卡
  （full 16 层 × 32k × 1 KB），128k ≈1.9 GB/卡。
* KV 以 e4m3 存储（per-token-per-head scale）：**容量 ×2**（full+draft 21 KB →
  10.7 KB/token，池 token ≈ 1.36M）且 **decode KV 读带宽 ÷2** ⇒ 长上下文
  ITL −20~35%（与 L3.2 叠加）。

## 2. 方案

* `--kv-cache-dtype fp8_e4m3`（gfx906 后端支持，默认关闭）：
  K/V 在写回时量化（per-token-per-head fp32 scale，amax/448），读入 Triton
  unified attention 后用已有的精确 fp8→fp16 LUT + 一次乘法 dequant。
* 三条路径：decode（3D split-KV）、verify（M≤8 3D）、prefill（2D/3D）——
  dequant 只加在读 tile 之后、QK^T 之前，K 与 V 各一次乘法。
* GDN（linear attention）状态不量化（1.34 KB/token 占比小，状态是累加器，
  量化会引入漂移累积）。
* 写路径：独立的 quant kernel（K/V 投影输出 → fp8+scale），第一步不融合进
  投影 epilogue；每层每步多一次 kernel，decode 时 M 小、代价 ~µs 级。

## 3. 质量门禁（硬，全部通过才允许默认开启）

1. **8 条固定 prompt 的输出 perplexity / 与 fp16-KV 的 logit KL 偏差 ≤1%**
   （贪心 parity 必然破坏——本项是计划里唯一允许改数值的工作项）。
2. 长稳 1000 步 decode 无 NaN/Inf（含 64k 上下文驻留）。
3. decode ITL −20% **或** 池容量 ×1.8 至少一项在 e2e 成立。
4. 回归面：prefill 精度（fp8 KV 也会进 prefill 的 KV 读）纳入门禁 1 的口径。

## 4. 风险

* gfx906 无 fp8 ALU：dequant 是 fp16 乘法（LUT 后），attention 内核的
  FLOPs 增加可忽略；真正的风险是 scale 的 per-token 粒度带来的离群 token
  精度损失——备选 per-token-per-head 已是较细粒度，仍不达标则退回本设计不启用。
* DFlash2 draft 的 KV（5 层 bf16）是否同步量化：分开开关，先 target 后 draft。
* 与 L5b 的组对齐（880 页）交互：页字节数减半会把 attention 组页大小降到
  880 对齐以下 → 需要复查 `--mamba-cache-mode align` 的页对齐逻辑
  （mamba 页 880 token 不变，attention 页变小则回退到非对齐模式）。

## 5. 工作量估计

kernel（Triton 三路径 dequant 分支 + quant kernel）~200 行；config/validation
~100 行；门禁脚本复用 s23_parity 口径改造。建议排期在 L3.2 落地之后。
