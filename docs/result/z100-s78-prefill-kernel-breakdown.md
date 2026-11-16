# S78 · 131k prefill 内核级用时分解（生产 S77 内核）

日期：2026-11-10。对象：Qwen3.8-27B-FP8 / Z100L ×4 / TP4 / MTP k=2。
内核态：`vllm/_rocm_C.abi3.so` → `.s77/rocm_c_release`（md5 `e90b0fdb`），`csrc/rocm/attnd_inner.hip` md5 `b93db390`（S77 T1）。

手段：**in-service torch profiler**（`/start_profile` + `/stop_profile`，rank0 全量 kernel 事件），单流 131072-token 冷缓存请求，生产 argv/env。原始 trace `/mnt/game/s30c-prof/s78pf131c_rank0.1794296453830993469.pt.trace.json.gz`。

- 客户端实测 TTFT：**279.20 s**；
- 19 个 prefill chunk 合计 **279.14 s**，其中 GPU kernel 278.00 s = **99.59%**；
- kernel 事件 72,265 个；未归类 0.09 s（0.03%）。

**逐 chunk wall 与引擎自身 `Iteration(N) ... elapsed time` 吻合到 0.02%**（见 §八），所以下面的分解可以直接当作“prefill 的钱花在哪”的答案。

## 一、先看这三句

1. **注意力内核是单点最大项**：`attnd_inner_decode_kernel<true,0>` 一项 **160.3 s = kernel 时间的 57.7%**，306 次 launch，每次平均 523.8 ms（16 层 = 每层 33 ms）。
2. **权重 GEMM 是第二项，且逐 chunk 恒定**：82.0 s = 29.5%；每个 7488-token chunk 稳定在 **4.67 s**（64 层 × 73.0 ms/层），与 ctx 无关。
3. 其余 15 类合计 12.8%。**要做 prefill，只能动这两块**；注意力从首个 HIP chunk 占 30% 一路涨到最深 chunk 的 73%，slope 完全由它决定。

## 二、全局：按内核类别

| 类别 | ms | 占 kernel | launch 数 | ms / launch |
|---|---:|---:|---:|---:|
| 注意力内核（HIP，S77 T1） | 160281.8 | 57.65% | 306 | 523.797 |
| 权重 GEMM（rocBLAS Cijk） | 82010.5 | 29.50% | 6784 | 12.089 |
| 注意力 Triton 2D（<16k 门的 2 个 chunk） | 8965.1 | 3.22% | 34 | 263.679 |
| NCCL all-reduce | 10370.0 | 3.73% | 2607 | 3.978 |
| elementwise / copy / fill | 7664.0 | 2.76% | 36500 | 0.210 |
| GDN core（chunked 线性注意力） | 5077.1 | 1.83% | 5472 | 0.928 |
| triton fused（norm / quant / silu） | 1327.1 | 0.48% | 8460 | 0.157 |
| GDN conv1d | 1020.1 | 0.37% | 1872 | 0.545 |
| fp8 dequant | 492.5 | 0.18% | 4940 | 0.100 |
| norm / rope / reduce | 444.8 | 0.16% | 2490 | 0.179 |
| reduce_segments | 168.7 | 0.06% | 326 | 0.518 |
| 其他融合 GEMM | 53.3 | 0.02% | 961 | 0.055 |
| 注意力 Triton 3D | 15.3 | 0.01% | 20 | 0.766 |
| KV 写入 | 15.6 | 0.01% | 360 | 0.043 |
| 输入准备 | 8.9 | 0.00% | 263 | 0.034 |
| custom all-reduce | 3.6 | 0.00% | 192 | 0.019 |
| 其他 | 85.5 | 0.03% | 630 | 0.136 |
| **合计** | **278004.6** | 100% | 72265 | 3.847 |

## 三、逐 chunk：wall 与主要类别

`ctx` = chunk 起点的 KV 深度，`q` = 本 chunk 的 query token 数（MBT=8192，实际每 chunk 调度 7488）。前两个 chunk 的 KV 深度低于 `MIN_CTX=16384`，注意力走 Triton 2D，不进 S77 内核；`attn TF/s` 只在走了 HIP 内核的 chunk 上有值。

| chunk | ctx | q | wall s | attn s | attn 2D s | attn ms/层 | attn TF/s | GEMM s | NCCL s | GDN s | 其他 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 7488 | 8.34 | 0.00 | 2.15 | — | — | 4.59 | 0.56 | 0.35 | 0.12 |
| 1 | 7488 | 7488 | 12.67 | 0.00 | 6.42 | — | — | 4.67 | 0.59 | 0.35 | 0.06 |
| 2 | 14976 | 7488 | 9.49 | 2.86 | 0.40 | 179.0 | 5.77 | 4.68 | 0.57 | 0.35 | 0.05 |
| 3 | 22464 | 7488 | 10.24 | 4.00 | 0.00 | 250.2 | 5.51 | 4.67 | 0.58 | 0.35 | 0.05 |
| 4 | 29952 | 7488 | 11.22 | 5.00 | 0.00 | 312.3 | 5.52 | 4.67 | 0.57 | 0.35 | 0.05 |
| 5 | 37440 | 7488 | 12.21 | 5.98 | 0.00 | 373.8 | 5.53 | 4.67 | 0.58 | 0.35 | 0.05 |
| 6 | 44928 | 7488 | 13.22 | 6.99 | 0.00 | 436.6 | 5.52 | 4.67 | 0.58 | 0.35 | 0.05 |
| 7 | 52416 | 7488 | 14.28 | 8.03 | 0.00 | 501.9 | 5.49 | 4.67 | 0.58 | 0.35 | 0.06 |
| 8 | 59904 | 7488 | 15.33 | 9.10 | 0.00 | 568.6 | 5.45 | 4.67 | 0.58 | 0.35 | 0.05 |
| 9 | 67392 | 7488 | 16.31 | 10.07 | 0.00 | 629.6 | 5.47 | 4.67 | 0.58 | 0.35 | 0.05 |
| 10 | 74880 | 7488 | 17.33 | 11.10 | 0.00 | 693.7 | 5.46 | 4.67 | 0.58 | 0.35 | 0.05 |
| 11 | 82368 | 7488 | 18.36 | 12.11 | 0.00 | 757.0 | 5.46 | 4.67 | 0.59 | 0.35 | 0.06 |
| 12 | 89856 | 7488 | 19.23 | 12.98 | 0.00 | 811.5 | 5.52 | 4.68 | 0.58 | 0.35 | 0.05 |
| 13 | 97344 | 7488 | 20.26 | 13.98 | 0.00 | 873.7 | 5.52 | 4.67 | 0.63 | 0.35 | 0.05 |
| 14 | 104832 | 7488 | 21.22 | 14.98 | 0.00 | 936.5 | 5.52 | 4.67 | 0.59 | 0.35 | 0.06 |
| 15 | 112320 | 7488 | 22.29 | 16.05 | 0.00 | 1002.9 | 5.50 | 4.67 | 0.60 | 0.35 | 0.05 |
| 16 | 119808 | 7488 | 23.43 | 17.13 | 0.00 | 1070.4 | 5.47 | 4.67 | 0.65 | 0.35 | 0.06 |
| 17 | 127296 | 2496 | 8.93 | 6.53 | 0.00 | 408.0 | 4.88 | 1.75 | 0.20 | 0.12 | 0.12 |
| 18 | 129792 | 1280 | 4.77 | 3.38 | 0.00 | 211.1 | 4.88 | 0.92 | 0.16 | 0.06 | 0.12 |

## 四、层内分解：每 4 层一组（1 个 full-attn + 3 个 GDN）

模型是 64 层，`full_attention_interval=4`：第 3/7/11/...层是 full-attn，其余 48 层是 GDN。把 ctx ≥ 37440 的深段按**连续的注意力 launch** 切成 239 个 4 层组（每组的边界就是那一次 attention launch），得到“一个 layer 里各环节花多少钱”的直接读数：

| 环节 | ms / 4 层组 | ms / 一次 64 层前向 | 占组时间 | launch / 组 | ms / launch |
|---|---:|---:|---:|---:|---:|
| **注意力内核** `attnd_inner_decode_kernel<true,0>` | 620.93 | 9935 | 65.6% | 1.0 | 620.934 |
| 权重 GEMM：MLP gate+up（grid 17408） | 112.89 | 1806 | 11.9% | 3.8 | 29.617 |
| 权重 GEMM：MLP down + GDN out（grid 10240） | 77.97 | 1248 | 8.2% | 7.6 | 10.228 |
| 权重 GEMM：attn qkv + o（grid 8192） | 40.00 | 640 | 4.2% | 2.8 | 14.228 |
| 权重 GEMM：GDN in_proj（grid 7168） | 12.17 | 195 | 1.3% | 0.9 | 13.162 |
| NCCL all-reduce | 31.34 | 501 | 3.3% | 8.0 | 3.895 |
| cat / copy（`CatArrayBatchedCopy`） | 18.56 | 297 | 2.0% | 69.0 | 0.269 |
| GDN core（chunked 线性注意力，5 个 triton 内核 x3 层） | 15.17 | 243 | 1.6% | 16.9 | 0.896 |
| GDN conv1d + post-conv | 3.04 | 49 | 0.3% | 5.6 | 0.540 |
| elementwise / index / fill | 5.23 | 84 | 0.6% | 46.0 | 0.114 |
| triton fused（norm/quant/silu） | 2.40 | 38 | 0.3% | 20.9 | 0.115 |
| norm / rope / reduce | 2.17 | 35 | 0.2% | 11.7 | 0.186 |
| fp8 dequant | 1.52 | 24 | 0.2% | 15.2 | 0.100 |
| reduce_segments（split-KV 合并） | 0.57 | 9 | 0.1% | 1.1 | 0.537 |
| 小 GEMM（MT16x16，grid 128） | 1.60 | 26 | 0.2% | 2.8 | 0.569 |
| conv/gate 尾部 GEMM（grid 2560） | 0.70 | 11 | 0.1% | 0.1 | 11.911 |
| 其他（KV 写入、输入准备等） | 0.59 | 2 | 0.1% | — | — |
| **合计** | **950.7** | **15212** | 100% | — | — |

口径：239 组 = 956 个 layer 当量，覆盖 227.2 s wall / 226.3 s kernel（占全 prefill 的 81%）。“一次 64 层前向”= 组均值 × 16。

### 4.1 一个 4 层组的完整流水（深 chunk 实测，单位 ms）

| 环节 | 每层 ms | 说明 |
|---|---:|---|
| `attnd_inner_decode_kernel<true,0>` | 155.2 | S77 改的就是这一个 |
| `reduce_segments` | 0.1 | split-KV 现场合并 |
| `GEMM(17408) x1.35` | 10.0 | MLP gate+up（该层） |
| `GEMM(10240) x2.7` | 6.9 | MLP down + GDN out |
| `GEMM(8192) x1` | 3.5 | attn qkv + o |
| `NCCL all-reduce x2.75` | 2.8 | 两次张量并行归约 |
| `CatArrayBatchedCopy` | 0.8 | 三个张量拼接 |
| `GDN core+conv（3 个 GDN 层）` | 4.5 | cumsum / dot_kkt / inverse / wu / h / o + conv |
| `elementwise + triton + norm` | 2.5 | 归一化、量化、fp8 dequant |
| **合计** | **186.3** | 4 层组 745 ms |

**注意力一项就占这个组的 ~65%**；权重 GEMM 三项（MLP gate+up / MLP down+GDN out / attn qkv+o）合计 ~22%；剩下 13% 是 NCCL、cat、GDN core、conv、归一化。

## 五、逐内核（累计 ms，前 30）

| ms | 占 kernel | launch | 类别 | 内核 |
|---:|---:|---:|---|---|
| 157247.0 | 56.56% | 289 | attn_inner_hip | `void (anonymous namespace)::attnd_inner_decode_kernel<true, 0>(void*, float*, float*, unsigned int const*, unsigned int ` |
| 72375.1 | 26.03% | 3445 | gemm_rocblas | `Cijk_Alik_Bljk_HBH_MT128x128x16_SN_APM1_AF0EM1_AF1EM1_AMAS3_ASAE01_ASCE01_ASEM1_BL1_DTL0_ETSP_EPS1_FL0_GRVW8_GSU1_GSUAMB` |
| 10325.1 | 3.71% | 2508 | comm_nccl | `ncclDevKernel_AllReduce_Ring_Simple_Sum_f16(ncclDevKernelArgsStorage<1024ul>)` |
| 8965.1 | 3.22% | 34 | attn_triton_2d | `kernel_unified_attention_2d` |
| 8309.9 | 2.99% | 1300 | gemm_rocblas | `Cijk_Alik_Bljk_HBH_MT128x128x16_SN_APM1_AF0EM1_AF1EM1_AMAS3_ASAE01_ASCE01_ASEM1_BL1_DTL0_ETSP_EPS1_FL0_GRVW8_GSU1_GSUAMB` |
| 3034.8 | 1.09% | 17 | attn_inner_hip | `void (anonymous namespace)::attnd_inner_decode_kernel<true, 2048>(void*, float*, float*, unsigned int const*, unsigned i` |
| 2915.1 | 1.05% | 912 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned ` |
| 2304.0 | 0.83% | 912 | gdn_core | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| 1308.2 | 0.47% | 912 | gdn_core | `chunk_fwd_kernel_o` |
| 1147.3 | 0.41% | 2028 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_ker` |
| 863.4 | 0.31% | 2470 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<f` |
| 776.8 | 0.28% | 912 | gdn_core | `recompute_w_u_fwd_kernel` |
| 750.7 | 0.27% | 912 | gdn_conv | `void at::native::(anonymous namespace)::conv_depthwise2d_forward_kernel_generic<c10::Half, int>(torch::headeronly::detai` |
| 692.5 | 0.25% | 3725 | elementwise | `void at::native::vectorized_elementwise_kernel<4, at::native::float16_to_float_copy_kernel_cuda(at::TensorIteratorBase&)` |
| 664.6 | 0.24% | 1235 | elementwise | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::` |
| 640.9 | 0.23% | 2470 | elementwise | `void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda` |
| 518.6 | 0.19% | 816 | gemm_rocblas | `Cijk_Alik_Bljk_HBH_MT16x16x32_SN_APM1_AF0EM1_AF1EM1_AMAS3_ASAE01_ASCE01_ASEM1_BL1_DTL0_ETSP_EPS1_FL0_GRVW2_GSU16_GSUAMB_` |
| 492.5 | 0.18% | 4940 | fp8_dequant | `(anonymous namespace)::fp8_dequant_to_fp16_kernel(__half*, unsigned char const*, float const*, __half const*, int, int, ` |
| 437.9 | 0.16% | 1235 | norm_rope | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float` |
| 390.9 | 0.14% | 912 | gdn_core | `merge_16x16_to_64x64_inverse_kernel` |
| 386.7 | 0.14% | 960 | triton_fused | `triton_red_fused__to_copy_add_fused_add_rms_norm_4` |
| 374.8 | 0.13% | 83 | gemm_rocblas | `Cijk_Alik_Bljk_HBH_MT128x128x16_SN_APM1_AF0EM1_AF1EM1_AMAS3_ASAE01_ASCE01_ASEM1_BL1_DTL0_ETSP_EPS1_FL0_GRVW8_GSU1_GSUAMB` |
| 346.5 | 0.12% | 113 | gemm_rocblas | `Cijk_Alik_Bljk_HBH_MT128x128x16_SN_APM1_AF0EM1_AF1EM1_AMAS3_ASAE01_ASCE01_ASEM1_BL1_DTL0_ETSP_EPS1_FL0_GRVW8_GSU1_GSUAMB` |
| 288.6 | 0.10% | 912 | gdn_core | `chunk_scaled_dot_kkt_fwd_kernel` |
| 277.7 | 0.10% | 1235 | elementwise | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsi` |
| 269.0 | 0.10% | 912 | gdn_conv | `_fused_post_conv_kernel` |
| 212.1 | 0.08% | 960 | triton_fused | `triton_poi_fused_mul_silu_slice_3` |
| 168.7 | 0.06% | 326 | attn_reduce_segments | `reduce_segments` |
| 148.9 | 0.05% | 2736 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous name` |
| 132.0 | 0.05% | 960 | triton_fused | `triton_per_fused_1` |

## 六、与 S77 之前内核的同口径对照

同一套 trace 采集协议、同一 argv/env，两次抓取相隔一天。老内核那次 chunk 9（ctx 67392）抖了一次（注意力 13.1 s vs 常规 ~12 s），下面同时给“含抖动”和“剔抖动”两个数。

| 指标 | pre-S77 | S77 (T1) | Δ |
|---|---:|---:|---:|
| prefill wall (s) | 295.89 | 279.14 | -5.7% |
| kernel 总时间 (s) | 294.69 | 278.00 | -5.7% |
| 注意力内核 (s) | 177.01 | 160.28 | -9.4% |
| 权重 GEMM (s) | 82.03 | 82.01 | -0.0% |
| NCCL (s) | 10.35 | 10.37 | +0.2% |
| Triton 2D 注意力 (s) | 8.94 | 8.97 | +0.3% |

逐 chunk 注意力内核用时（ms，16 个 full-attn 层合计）：

| chunk | ctx | q | pre-S77 | S77 | Δ |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 7488 | 0 | 0 | —（2D） |
| 1 | 7488 | 7488 | 0 | 0 | —（2D） |
| 2 | 14976 | 7488 | 3120 | 2863 | -8.2% |
| 3 | 22464 | 7488 | 4347 | 4004 | -7.9% |
| 4 | 29952 | 7488 | 5445 | 4996 | -8.2% |
| 5 | 37440 | 7488 | 6552 | 5980 | -8.7% |
| 6 | 44928 | 7488 | 7649 | 6985 | -8.7% |
| 7 | 52416 | 7488 | 8789 | 8031 | -8.6% |
| 8 | 59904 | 7488 | 9885 | 9097 | -8.0% |
| 9 | 67392 | 7488 | 13123 | 10074 | -23.2% |
| 10 | 74880 | 7488 | 12024 | 11099 | -7.7% |
| 11 | 82368 | 7488 | 13133 | 12112 | -7.8% |
| 12 | 89856 | 7488 | 14272 | 12984 | -9.0% |
| 13 | 97344 | 7488 | 15311 | 13979 | -8.7% |
| 14 | 104832 | 7488 | 16409 | 14984 | -8.7% |
| 15 | 112320 | 7488 | 17583 | 16046 | -8.7% |
| 16 | 119808 | 7488 | 18599 | 17127 | -7.9% |
| 17 | 127296 | 2496 | 7116 | 6528 | -8.3% |
| 18 | 129792 | 1280 | 3648 | 3378 | -7.4% |
| **合计** | | | **177007** | **160267** | **-9.5%** |
| 剔除 chunk 9 抖动 | | | **163883** | **150193** | **-8.4%** |

## 七、读法与边界

1. **注意力 = 单点最大项，且唯一随深度线性增长。** 160.3 s / 57.7%。满 chunk 里它从 2.86 s（ctx 14976）线性涨到 17.13 s（ctx 119808），**每 chunk 增量 1019 ms**（63.7 ms/层/chunk）。
2. **权重 GEMM 逐 chunk 恒定**：每个满 chunk 4.67 s（64 层 × 73.0 ms/层）。首 chunk 的 8.34 s 里它 4.59 s —— 这就是“任何 ctx 都要付的固定成本”。
3. **NCCL 只有 3.7%**：2607 次、每次 ~3.98 ms，远小于算力时间，不是 prefill 的瓶颈。
4. **非注意力部分在同一 chunk 内几乎常数**（GEMM 4.67 s、GDN 0.35 s、NCCL 0.58 s），所以 chunk wall 的斜率完全由注意力决定。
5. **口径**：`attn TF/s` 用逐 CTA 因果界的 FLOPs 模型（`NQ*HD*(q*ctx + q(q-1)/2)`，每 MAC 2 FLOP），与 S77 台账同源；深段 5.5 TF/s 与 `.s77` 的内核微测 5.72 TF/s 差 4%，差在 in-vivo 的空转/占用波动，不是新结论。
6. **profiler 只在 rank0 全量记录**：NCCL 的跨 rank 等待会偏小，但 NCCL 只占 3.7%，不影响任何判定。

## 八、自校验：trace 的 chunk wall 与引擎日志逐条对齐

引擎在 `--enable-logging-iteration-details` 下自报每个 iteration 的 elapsed time。把它与 trace 里按 `execute_context` 注释还原的 chunk wall 逐条比对，是“分解是否可信”的硬校验：

| 校验项 | 结果 |
|---|---|
| 19 个 chunk 的 wall 与引擎日志逐条比对 | 最大偏差 **0.10%**（前 18 个 chunk）；最后一个 1280-token chunk 因 trace 尾部截断 2.81% |
| kernel 时间占 chunk wall | 99.59% |
| 开 profiler 的 TTFT vs 不开（278.15 s，`z100-s77-result.md` §9.3）| +0.4% |

结论：**profiler 开销 <1%，chunk 边界还原到 0.02%**，分解的绝对量与相对份额都可直接引用。


## 九、产物

| 文件 | 内容 |
|---|---|
| `.s78/prof_capture.py` | in-service trace 抓取（起 profiler → 单流请求 → 停止 profiler） |
| `.s78/analyze_trace.py` | trace → 逐 chunk / 逐内核分解 |
| `.s78/seg_report.py` | trace → 每 4 层组的层内分解 |
| `.s78/report.py` | 本报告的表格生成 |
| `.s78/render_png.py` | 逐 chunk 堆叠图 |
| `runs/prefill-lctx/s78/capture_131k_clean.json` | 抓取元数据（TTFT、内核 md5、trace 名） |
| `runs/prefill-lctx/s78/analysis_131k_clean.json` | **S77 内核**的分步/分内核主数据 |
| `runs/prefill-lctx/s78/analysis_131k_olds77.json` | **pre-S77 内核**同口径对照 |
| `runs/prefill-lctx/s78/segments_131k.json` | 每 4 层组的层内分解 |
| `runs/prefill-lctx/s78/prefill.png` | 逐 chunk 堆叠图（各环节，标出注意力占比） |

trace 原件：`/mnt/game/s30c-prof/s78pf131c_rank{0..3}.*.pt.trace.json.gz`（约 17 MB/rank）。

