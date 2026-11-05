# S40-B0:lm_head 数值口径裁决 + 形状账本修正 — 2026-10-31

## 0. 形状/速率账本修正(设计图基线勘误,trace 实证)

S39 账本把 lm_head 记为 "[38484,5120] fp16 分片,947 µs/发,417 GB/s"。
**trace 定案**(`s34nospec_rank3.1793405011427528091.pt.trace.json.gz`,
38 步窗):

- `bf16_emu_gemv_dot2_kernel<1,…>` 大发 **40 次 ≈ 1 发/step**,
  **grid [62080,1,1] × block [64,1,1]** → 真实形状 **[62080, 5120] fp16
  = 635.7 MB/发**(词表 62080,lm_head 未按 TP 分片,与
  utils.py A1 注释 "unquantized lm_head (N=62080, K=5120)" 一致)。
- 中位 **899.3 µs/发 → 706.9 GB/s**(设计图 417 GB/s 是把 636 MB 流量
  套进 394 MB 形状的账)。
- **窗口口径复核**(本机 harness,.s40/gemv_bw_bench.py,copies=3 轮转):
  **896.49 µs/rep → 709.3 GB/s**,与 trace 差 0.3% —— 该内核长
  (~0.9 ms),无 launch-gap 税,窗口=trace 自洽。

**影响**:桶大小 0.95 ms/step 不变;但现役速率已是 ~707 GB/s(非 417),
设计图 B2 门 "≥550 GB/s" 失效。**重述 B2 门(窗口口径,同形状同轮)**:
**stream ≥ prod × 1.06 ≈ 752 GB/s**(下限),stretch 780(= 裸形 deep4
K-E 783 的窗口证据);对应收益下限 **−0.05 ms/step**,上限 −0.11 ms/step
(780 GB/s → 815 µs)。收益预期较设计图(−0.25~0.35)下修,但仍是
decode 第二大单核,方向与几何证据(663→783)不变。

## 1. 累加序裁决(bf16_emu_gemv.cu:79-127 读账)

现役 `bf16_emu_gemv_dot2_kernel`:

- 线程内:`for i = t; i < K2; i += nt` 跨步串行 dot2 累加(K2 = K/2 个
  packed dword);
- 跨线程:shared-mem 树归约(`red[m*nt+t]`,off = nt>>1 … 1)。

S40 stream 新内核(fp16_gemv_stream_m1):

- lane 内:slice 外层、chunk 内层,每 slice 16 个 fp16 元素 8 次 dot2
  串行(2×uint4),slice 间串行;
- 跨线程:wave64 `__shfl_down` 树(off 32…1)。

**两序不同 → 非位同,构造上即不可能 torch.equal**(现役 nt=64 跨步序 vs
stream lane-chunk 序,浮点加法不可交换)。数值门定案:

1. `max|Δ| ≤ 3e-4`(对现役 dot2 输出,fp16 out,同输入同 bias);
2. S29 logprob 板 ≤ 0.2255 nat(`.s38/parity_probe.py --capture/--compare`);
3. 服务贪心流差异按 S34 先例只记录不阻塞(重排级)。

bit-identical 备选(复刻 i=t+j·nt 序)被设计图明确排除(放弃 slice 几何
= 放弃收益)。

## 2. 几何定案(新内核 fp16_gemv_stream_m1)

- rows=4 / nt=256 / RW=1(与 m1 生产几何同),grid = ceil(62080/4) =
  **15520 blocks**(设计图 "9621" 沿用了错误的 N)。
- slice = kStreamSlice = 1024 fp16 元素 = 2 KB A / 2 KB W(行);lane 每
  slice 2×uint4 W(1 KiB/请求粒度,与 fp8 m1 的 16B×64=1 KiB 同);
  pf[4] 预取环直接按 A 轨成品形。
- 无 LUT、无 sl、无 w_s/a_s 折算(LDS = 2 KB A-slice);epilogue =
  shfl 树 + bias。
- 路由门:`VLLM_DCU_FP16_GEMV_STREAM=1`(默认 0)且
  `M==1 && N>=2048 && K%128==0 && dtype==fp16`;其余一切形状走现役。

## 3. 微测基线(D0 窗口口径,本裁决用对照面)

| 项 | 值 |
|---|---|
| lm_head prod 窗口(copies=3) | **896.49 µs/rep → 709.3 GB/s** |
| lm_head prod trace | 899.3 µs → 706.9 GB/s |
| 锚点 sum(1 GiB) 窗口 | 792.8 GB/s(两轮 793.1/792.8,±0.04%) |
| gate_up prod 窗口(copies=18) | 510.1 GB/s(7 形状全表见 s40-d0-baseline.txt) |
