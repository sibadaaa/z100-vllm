# S30c 施工图：纯 decode 原生路线（csrc）+ 显存最小化 —— 实施任务书（2026-10-26）

> **交给实施 agent 的完整任务书**。设计依据：`docs/design/z100-s30c-pure-native-design.md`
> （决策 D1–D6 与拒绝项，先读）；上游基线：`docs/design/z100-s30b-pure-decode-plan.md`。
> 本文件取代 `docs/design/z100-s30b-construction-gdn-m1-triton-gemv-hostgap.md` 的
> WP1/WP2 执行路线（该文 = **Plan-B**，仅当 WP1-N parity 硬门失败时按其路线重试，
> 平时不得执行）。按 WP 顺序执行，每个 WP 独立可验收、独立回滚。
>
> **基线**（勿重推导）：nospec 臂（`.s29/graph_profile_launch_nospec.sh`，piecewise
> cudagraph [1,2,4,8,16]、workspace pin、BM16），7469 doc / 96 tok / 3 reps：
> **step p50 = 39.24 ms，TTFT = 739 ms，21.33 t/s**（`.s29/s20_s29-nospec_*.json`）。
> 目标：**27–29 ms/tok（34–37 t/s）+ TTFT 保持 + 显存快照差入账**。

---

## 0. 环境须知（先读，全是踩过的坑）

1. **GPU 实测必须在 DSH 沙箱外**（沙箱挡 HSA，`hsa_init=4104` 是假象）。正路：
   宿主 shell，或 `docker exec z100-regress bash -c '...'`（构建用）。
2. **服务生命周期**（重启 ~4.5 min；任何时刻一个服务）：
   ```bash
   cd /home/lyh/z100-vllm
   pkill -f "entrypoints[.]cli"; sleep 12; ps aux | grep "[e]ntrypoints" | wc -l  # 停，应为 0
   MAX_SEQS=4 bash .s30c/launch_native.sh > .s30c/serve-<arm>.log 2>&1 &           # 起
   for i in $(seq 1 24); do sleep 15; code=$(curl -s -o /dev/null -w "%{http_code}" \
     --max-time 3 http://127.0.0.1:8000/health); [ "$code" = 200 ] && break; done  # 等 ready
   ```
3. **csrc 构建 + 换装（C7 纪律，z100-opt-round3-construction.md）**：
   ```bash
   # 改前：备份旧 .so + 记 md5
   md5sum /mnt/game/port/build-sync/temp/_rocm_C.abi3.so   # 记入 WP 日志
   cp /mnt/game/port/build-sync/temp/_rocm_C.abi3.so \
      /mnt/game/port/build-sync/temp/_rocm_C.abi3.so.bak-s30c
   # 构建（增量 ~100 s；禁止 build-sync2.sh —— 强删 stamp 会全量重 hipify）
   docker exec -e MAX_JOBS=5 z100-regress bash -lc 'bash /mnt/game/port/build-sync.sh _rocm_C'
   # 换装后记新 md5；禁止手改 build-sync/temp 里的产物
   md5sum /mnt/game/port/build-sync/temp/_rocm_C.abi3.so
   ```
   生产符号链：`vllm/_rocm_C.abi3.so -> /mnt/game/port/build-sync/temp/_rocm_C.abi3.so`，
   换装后重启服务生效。二进制回滚 = `cp` 回 `.bak-s30c`；**env kill-switch 永远优先**。
4. **探针鉴权**：API key 在 `/mnt/game/host-serve/api-key`，套壳
   `tools/dcu/s23_common.py install_auth()`（S29 施工图 §0.3 模板）。
5. **环境核验**：每次起服务核对 serve log `[host-serve] policy: FP_FULL ...` 指纹 +
   `/proc/<pid>/environ` 里本 WP 的开关值；新 csrc 分支另用 DEBUG 打印确认 HIT。
6. **度量纪律**：每 WP 真实 shape 微测 → 数值对照 → smoke → 同口径 e2e → 文档；
   JSON 落 `.s30c/`；结果文档 `docs/result/z100-s30c-*.md`；微测命令写进文档。
7. **共用门禁**（每个数值类 WP 必跑）：
   ```bash
   # E2E：7469 真实文档 / 96 tok / 3 reps（双口径：srv_step_ms + ttft_ms）
   .venv/bin/python tools/dcu/s20_single_probe.py \
     --prompt-file docs/design/z100-dual-prefill-reconciliation-projection.md \
     --truncate-tokens 7469 --max-tokens 96 --repeat 3 --tag s30c-<arm> --out .s30c
   # 贪心数值板：temperature=0 流式抓 token_ids + top5 logprobs（复用 .s29/capture.py）
   ```
   通行标准：96-token 贪心流**零翻转**；`srv_tok_per_step==1.0`；TTFT +3% 内；
   e2e 报 N≥2 重启分布。

---

## WP1-N：GDN M=1 原生并行化（−8~9 ms，主杠杆，最先做）

### 1.1 前置核实（半天内完成，对不上就停下报告）

```bash
grep -n "dec_v2_shape_ok\|state_stride_v == 1" csrc/mamba/gfx906_decode_kernels.cu  # :2866-2872
grep -n "use_transposed_state_for_packed_decode = False" \
  vllm/model_executor/layers/mamba/gdn_linear_attn.py                               # :446
```
确认：生产 decode 池原生 k 连续 ⇒ DEC_V2 门（要求 stride_v==1）MISS ⇒ 串行 kernel。
模型 shape：48 GDN 层、秩上 HV=12 / H=4 / K=V=128、mixed_qkv 2560、state
`[slots,12,128,128]` fp32（stride_v=128、stride_k=1）、`state_indices [T] int32`、
out `[T,1,12,128]` fp16。记录基线 `.so` md5。

### 1.2 kernel 实现（`csrc/mamba/gfx906_decode_kernels.cu`，约 120 行）

在 `_mtp_v2_kernel`（:731）之后新增。**骨架 = v2 的转置**：lane 持 2 个 k
（float2，64 lane × 8B = 512B 整行合并），warp 持 VW 个 v 行；v2 的 cross-warp
smem dot 交换换成 wave64 `__shfl_xor` butterfly（6 步，全波广播）。

```cpp
template <typename scalar_t, typename state_t, int VW>   // VW: v-rows per warp
__global__ void fused_recurrent_gated_delta_rule_gfx906_m1_native_kernel(
    const scalar_t* __restrict__ mixed_qkv, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ A_log,
    const scalar_t* __restrict__ dt_bias, state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices, scalar_t* __restrict__ out,
    int64_t tokens, int64_t heads, int64_t kv_heads,
    int64_t mixed_stride_t, int64_t a_stride_t, int64_t b_stride_t,
    int64_t state_stride_slot, int64_t state_stride_h,
    int64_t out_stride_t, int64_t out_stride_h,
    double scale, bool use_qk_l2norm_in_kernel, bool use_tiled_qk_head_mapping) {
  // grid (v_tiles, kv_heads, tokens)，blockDim.x = 64（wave64，单 warp）
  const int v_tile = blockIdx.x, hv = blockIdx.y, t = blockIdx.z;
  const int lane = threadIdx.x;                 // 0..63
  constexpr int KD = 128, VD = 128;             // host 门已保证
  const int v0 = v_tile * VW;
  const int q_head = use_tiled_qk_head_mapping ? (hv % (int)heads)
                                               : (hv / (int)(kv_heads / heads));
  const int64_t tok = (int64_t)t * mixed_stride_t;      // stride_d==1 由门保证
  // ---- 每块一次：q/k l2norm（对齐 :1326-1336 数学）----
  float q0 = (float)mixed_qkv[tok + q_head*KD + 2*lane];
  float q1 = (float)mixed_qkv[tok + q_head*KD + 2*lane + 1];
  float k0 = (float)mixed_qkv[tok + heads*KD + q_head*KD + 2*lane];
  float k1 = (float)mixed_qkv[tok + heads*KD + q_head*KD + 2*lane + 1];
  float qs = q0*q0 + q1*q1, ks = k0*k0 + k1*k1;
  for (int o = 32; o; o >>= 1) { qs += __shfl_xor(qs, o, 64); ks += __shfl_xor(ks, o, 64); }
  float q_s = use_qk_l2norm_in_kernel ? rsqrtf(qs + 1e-6f) * (float)scale : (float)scale;
  float k_s = use_qk_l2norm_in_kernel ? rsqrtf(ks + 1e-6f) : 1.0f;
  // ---- 每块一次：门控（对齐 :1531-1544）----
  const float x  = (float)a[t*a_stride_t + hv] + (float)dt_bias[hv];
  const float sp = x <= 20.0f ? log1pf(expf(x)) : x;
  const float decay = expf(-expf((float)A_log[hv]) * sp);
  const float beta_t = 1.0f / (1.0f + expf(-(float)b[t*b_stride_t + hv]));
  const int64_t v_off = tok + 2*heads*KD + (int64_t)hv * VD;
  // ---- 无效 slot：out 写 0、不碰 state（对齐 :1357-1361）----
  const int32_t slot = state_indices[t];
  if (slot < 0) {
    #pragma unroll
    for (int i = 0; i < VW; ++i)
      out[(int64_t)t*out_stride_t + (int64_t)hv*out_stride_h + v0 + i] = (scalar_t)0.0f;
    return;
  }
  state_t* base = state + (int64_t)slot*state_stride_slot + (int64_t)hv*state_stride_h;
  const float kq0 = k0*k_s, kq1 = k1*k_s, qq0 = q0*q_s, qq1 = q1*q_s;
  #pragma unroll
  for (int i = 0; i < VW; ++i) {
    const int64_t v = v0 + i;
    // float2 读：state[s][h][v][2k,2k+1]，512B 整行合并（指针 8B 对齐由 host TORCH_CHECK 保证）
    float h0 = (float)base[v*KD + 2*lane] * decay;
    float h1 = (float)base[v*KD + 2*lane + 1] * decay;
    float dot = h0*kq0 + h1*kq1;
    for (int o = 32; o; o >>= 1) dot += __shfl_xor(dot, o, 64);   // dot[v]，全波广播
    const float vp = ((float)mixed_qkv[v_off + v] - dot) * beta_t;
    h0 += vp * kq0; h1 += vp * kq1;
    float o_out = h0*qq0 + h1*qq1;
    for (int o = 32; o; o >>= 1) o_out += __shfl_xor(o_out, o, 64);
    base[v*KD + 2*lane] = (state_t)h0;
    base[v*KD + 2*lane + 1] = (state_t)h1;
    if (lane == 0)
      out[(int64_t)t*out_stride_t + (int64_t)hv*out_stride_h + v] = (scalar_t)o_out;
  }
}
```

实现注意：
- **state 仅原生布局**（stride_v==KD、stride_k==1），索引直接 `v*KD + k`；
  transposed 池不被本 kernel 支持（host 门挡掉，落回原路径）。
- state_t=float 时可用 `reinterpret_cast<const float2*>` 成对读写；state_t=Half
  用两次标量读（v2 的 `gdn_mtp_v2_load_kchunk` 同款纪律）。host 侧
  `TORCH_CHECK(8B 对齐)`（torch 分配 256B 对齐，防御性检查）。
- VLLM_DISPATCH_FLOATING_TYPES 双层结构照抄 :2999-3050（state fp32 / 同 dtype 两路）。
- grid.x = VD / (blockDim.x/64 × VW)；cfg 表（host 选档，env
  `VLLM_GFX906_GDN_DEC_M1_CFG`，默认暂定 1，微调后固化）：
  | CFG | block | VW | 覆盖 v/block | v_tiles | blocks(T=1) |
  |---|---|---|---|---|---|
  | 1 | 64 | 4 | 4 | 32 | 384 |
  | 2 | 64 | 8 | 8 | 16 | 192 |
  | 3 | 128 | 4 | 8 | 16 | 192 |
  | 4 | 256 | 4 | 16 | 8 | 96 |
- 仿 :2873-2891 / :2900-2989 加 `VLLM_GFX906_GDN_DEC_M1_DEBUG`（HIT/MISS 一次性
  fprintf）与 `VLLM_GFX906_GDN_DEC_M1_TIME`（128 事件环、64 次聚合）。

### 1.3 host 门（:2862 `dec_v2_enabled` 区域旁，先于 dec_v2 判断）

```cpp
static const bool m1_enabled = [] {   // VLLM_GFX906_GDN_DEC_M1，默认 1（parity 通过后）
  const char* e = std::getenv("VLLM_GFX906_GDN_DEC_M1");
  return e == nullptr || std::atoi(e) != 0; }();
const bool m1_shape_ok =
    state_stride_v == key_dim && state_stride_k == 1        // 原生池，与 dec_v2 门互补
 && key_dim == 128 && value_dim == 128 && tokens >= 1       // 不限 tokens==1，c>1 直接受益
 && mixed_qkv.stride(1) == 1 && a.stride(1) == 1 && b.stride(1) == 1
 && out.stride(3) == 1 && out.stride(1) == out.stride(2) * kv_heads
 && state_indices.stride(0) == 1;
if (m1_enabled && m1_shape_ok) { /* 按 CFG 选档 launch，return out; */ }
```

- 位置在 dec_v2 判断**之前**（m1 命中即 return）；miss 落回 dec_v2/串行，原逻辑
  一字不改。**不加 static tensor**（decode 的 state_indices 已是 per-token 平铺，
  无需 cu_seqlens）。
- op 签名/bindings/Python 全部不动；plain 与 combined 入口自动同时生效。

### 1.4 构建换装 + 微测

1. §0.3 流程构建换装，md5 对入 WP 日志。
2. **新工具 `tools/dcu/s30c_gdn_m1_bench.py`**（仿 `perf1_gdn_bench.py` 结构）：
   - case：T ∈ {1,4}（T=4 验 batched）、H=4、HV=12、K=V=128、fp32 state 原生布局、
     含 slot=-1 用例；种子固定。
   - `--impl {serial,m1}` 在 import torch/vllm **之前** set env（host 静态量进程级
     读一次）；`--cfg 1..4` 选档。每 run 把 out/state 存 `.s30c/gdn_m1_<impl><cfg>.npz`。
   - `--compare` 读两份 npz 出表：max|Δout|（fp16 域）、max|Δstate|（fp32 域）、
     fp16 不同元素数；另含 eager torch 参照第三列。
   - 计时：event loop + torch profiler（match `m1_native|packed_decode`），
     出 per-档 µs 表。
3. **判据 A**：最优档 per-op **≤40 µs**（现 216 µs；T=1 与 T=4 都报）。
4. **判据 B**：`max|Δstate| ≤ 1e-5`、`max|Δout| ≤ 1e-5`（树形归约合理域；
   超限 = 实现 bug，先查 butterfly 边界与 decay 乘法顺序）。
5. **判据 C（硬门）**：服务级 96-token 贪心流零翻转。

### 1.5 服务验证与收尾

1. smoke：§0.2 起 `.s30c/launch_native.sh`（= nospec 臂 + `export
   VLLM_GFX906_GDN_DEC_M1=1` + `VLLM_GFX906_GDN_DEC_M1_DEBUG=1`）；environ 核验 +
   serve log 出现 `GFX906_GDN_DEC_M1 HIT`。
2. `VLLM_GFX906_GDN_DEC_M1_TIME=1` 单独一跑：avg µs 与微测对齐（±20%）。
3. kdiff 复核（`.s29/kdiff.py` 同口径）：`m1_native` per-op 达标且 **conv kernel
   时间与基线持平**（本方案不动 conv，偏差 >2 µs/层 = 有未预期交互，停下查）。
4. e2e ×3 reps ×N≥2 重启。**判据 D**：step p50 **−5 ms 以上**（≤34.2 ms；预期
   −8~9 ms ⇒ ~30-31 ms），TTFT ±3%。
5. 文档 `docs/result/z100-s30c-gdn-m1-native.md`：cfg 扫描表、parity 表、kdiff
   前后块表、e2e 双口径、`.so` md5 新旧、指纹、回滚方式。

**回滚**：`VLLM_GFX906_GDN_DEC_M1=0`（或臂脚本不设）即回串行路径；代码默认值
在判据 D 达成后才固化为 1。

---

## WP2-N：fp8 GEMV M=1 → 400 GB/s（−1.5~2 ms，两级）

### 第一级：env 扫描（零代码，先做）

1. 形状清单：`/mnt/game/s29-prof/s29ns_rank0.*.pt.trace.json.gz` 经
   `.s29/kdiff.py` 逻辑导出 M=1 fp8 gemv 的 distinct (N,K) × calls/round ×
   µs/call → `.s30c/s30c_gemv_shapes.json`；只调覆盖 ≥90% 时间的形状。
2. `tools/dcu/p0_gemv_bench.py`（W8A16 route）对每形状 × {mode 1 × ROWS
   4/8/16/32/64} ∪ {mode 2}，每组合独立进程（env 进程级生效）；GB/s 入表。
   `VLLM_DCU_FP8_STREAM_MAX_M` 保持 unset。
3. 数值：`tools/dcu/p0_gemv_probe.py`（STREAM 值入输出）对照 fp32 反量化参照；
   随后服务级贪心流零翻转（**硬门**——rows/变体会改累加序，O4 先例）。
4. 判据：热形状 per-op **−15%** 以上；服务 A/B（nospec 臂 + 选定 env）e2e
   **−1.5 ms** 以上（与第二级合计口径）。达标即停，不进第二级。

### 第二级：MS=1 专用 kernel（仅当第一级 <400 GB/s）

1. `csrc/rocm/fp8_emu_gemm.cu` 新增 `fp8_emu_gemv_stream_m1_kernel`：M=1 时删
   A-restage 与 M 维索引、加深 K unroll（P0 证据 451–621 GB/s，:591-599）；
   env `VLLM_DCU_FP8_GEMV_M1=1` 门（默认 0，parity 后固化），`run_block_w8a16`
   的 stream 分派处优先命中 M==1。
2. 同 §0.3 构建流；微测 = 第一级的形状表复跑 + per-op −15% 判据 + 贪心流零翻转
   + e2e。
3. 文档 `docs/result/z100-s30c-fp8-gemv-m1.md`（两级全表、采纳项、回滚）。

---

## WP3：host gap ~3 → ~1.5 ms（Python 侧，无数值影响的削减）

1. `ITER_DETAILS=1` 起服务跑共用探针 1 rep，出 host 段分布表（字段语义以
   `vllm/v1/utils.py::compute_iteration_details` 与
   `vllm/v1/core/sched/scheduler.py:1631` 附近实际输出为准）。
2. 只做无数值影响的削减（重复张量构造、可缓存查询、多余 device 同步、纯 Python
   分支）；禁止改 kernel 参数/顺序/dtype/调度策略。
3. 每削减点单独 e2e ×3 reps；**判据**：合计 step p50 **−1 ms 以上** + 贪心流
   零翻转；单项 <0.3 ms 不采纳。落空则如实出表停手，不得为凑数引入数值风险。
4. 文档 `docs/result/z100-s30c-host-gap.md`。

## WP4：AR 7.44 ms —— 记录不动

在 final 文档引用 `tools/dcu/perf1_comm_report.md` 结论一段话（每层 2 次 × TP4
结构性，64.7 µs/op 低于 90 µs 地板），不做实验。

---

## V-WP：显存最小化（设计图 D5，速度门禁不回退为前提）

### V1：纯 decode cutover（主红利，与 WP1-N 判据 D 达成合并执行）

1. **切换前快照**（dflash 仍在 :8000 时）：serve log 启动统计（weight GiB /
   KV GiB + #tokens / 非 torch 内存）+ `hy-smi` 四卡 used → `.s30c/vram_dflash.json`。
2. cutover：`.s30c/launch_native.sh` 上位成现役臂（`UTIL=0.85` 不变），重复同一
   快照 → `.s30c/vram_native.json`。
3. 差值入账 = draft 塔权重 + draft KV + spec 缓冲释放量；固定 UTIL 下差值自动
   转 KV tokens 容量（速度中性）。**只记实测，不预估**。
4. dflash 臂退役留档（30.9 ms/tok / 32.4 t/s 已在案），不再维护。

### V2：cudagraph capture 裁剪

1. 臂脚本 `.s30c/launch_native_ctrim.sh`：唯一差异 =
   `--compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4]}'`。
   依据：MAX_SEQS=4 且无 spec ⇒ decode batch ≤4，capture 8/16 永不命中。
2. 验证：启动日志 capture 清单只含 [1,2,4]；e2e step ±2%；贪心流零翻转；显存差
   （启动统计 + hy-smi）入账。
3. ⚠️ 注记进文档：未来调高 MAX_SEQS 必须 re-add。

### V3：可选臂（按场景选用，默认不做，逐臂 e2e 门禁）

- `UTIL=0.75`：总 HBM 占用 ↓；前提单流 262k 容量仍满足（KV tokens 预期 ~640k），
  e2e 与 TTFT 不变才收。
- `MBT=2048`：activation workspace 再减半（S24 L5 先例）；TTFT ±3% 门，prefill
  敏感则弃。

### 拒绝项（不得重开，设计图 D5.4）

KV fp8_e4m3（前测容量 ×1.63 但 **ITL +41% 不晋级**，z100-longctx-vram-optimization-
result.md §15）；SSM state fp16（递推精度锚）；decode 池布局迁移（D1 使其永久
无必要）。

---

## 终局（全部 WP 后）

1. `s30c-full` 臂（采纳项叠加）×N≥3 重启 e2e；双口径报告 step 与 TTFT。
2. final 文档 `docs/result/z100-s30c-final.md`：WP 状态表 + 显存快照差表 +
   全部指纹/md5 + 对照留档（dflash 30.9 ms/tok；纯 decode 终值）。
3. 现场恢复：**纯 decode 原生臂留在 :8000**；`.s30c/` 全部 JSON 归档。
4. 纪律重申：贪心流零翻转硬门、双口径结论、N≥2 重启分布、每 WP 独立回滚。

## WP 状态表（实施时逐项更新）

| WP | 判据 | 状态 | 证据 |
|---|---|---|---|
| WP1-N m1_native | per-op ≤40 µs；step p50 −5 ms；流零翻转 | 未开始 | |
| WP2-N 一级 env | per-op −15%；累计 e2e −1.5 ms | 未开始 | |
| WP2-N 二级 kernel | （仅一级 <400 GB/s 时） | 未开始 | |
| WP3 host gap | step p50 −1 ms；流零翻转 | 未开始 | |
| WP4 AR | 记录即可 | 未开始 | |
| V1 cutover | 快照差入账；e2e 不回退 | 未开始 | |
| V2 capture 裁剪 | capture 清单 [1,2,4]；step ±2% | 未开始 | |
| V3 可选臂 | 按场景 | 未开始 | |
| 终局 | 双口径 + 显存表 | 未开始 | |
