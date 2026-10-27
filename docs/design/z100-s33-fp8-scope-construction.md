# S33 施工图：fp8 约束内对齐天花板 —— 实施任务书（2026-10-26）

> **交给实施 agent 的完整任务书**。设计依据：`docs/design/z100-s33-fp8-scope-design.md`
> （先读；含勘误①–④，勿再引用 S31/S32 的旧数字）。基线（勿重推导）：
> **s30c-full 现役臂 step p50 = 29.48 ms**（N=3：29.24/29.48/30.05），热 TTFT
> 701–746 ms。目标：确定性项 → ~21–23.5；裁决项后再到 ~20–21。
> 产物目录 `.s33/`；结果文档 `docs/result/z100-s33-*.md`。

---

## 0. 环境须知（先读，全是实测踩过的坑）

1. **GPU 可用性自检先行**：本会话 DSH 沙箱内 GPU 可用，但每次开工先跑：
   ```bash
   cd /home/lyh/z100-vllm
   export DTK_ROOT=/mnt/game/host-runtime/dtk2604
   set +u; source tools/dcu/env.sh; set -u
   .venv/bin/python -c "import torch; x=torch.randn(8,8,device='cuda'); print(torch.cuda.is_available())"
   ```
   若 torch 报 `libgalaxyhip.so.5` 缺失 = env 没加载，不是机器坏了。
   `hy-smi` 可直接看卡（4 卡 HBM 占用，服务在役时 ~88%）。
2. **服务生命周期**（重启 ~4.5 min；任何时刻一个服务；现役 = s30c ctrim 臂）：
   ```bash
   bash .s30c/restart.sh <tag> [EXTRA_ENV ...]        # 起 .s30c/launch_native.sh
   LAUNCH_VARIANT=ctrim bash .s30c/restart.sh <tag>   # 起 capture[1,2,4] 变体
   # restart.sh 自带: pkill → 起服 → 等 /health → 打印 policy 行 + environ 核验
   ```
   **pkill 禁忌**：不要在内联命令里手写 `pkill -f "VLLM::Worker"` ——模式会匹配
   自己的命令行把自己杀掉（2026-10-26 实测）。清理僵尸用 `pgrep -f
   "entrypoints[.]cli"` 拿 pid 后 `kill -9 <pid>`。
3. **csrc 构建换装（C7 纪律）**：GDN 内核在 **`_C`**，fp8 内核在 **`_rocm_C`**——
   `build-sync.sh` 的目标名必须选对（改 mamba → `_C`；改 fp8_emu_gemm →
   `_rocm_C`；两者都改则各跑一次）：
   ```bash
   # 改前备份 + 记 md5（_C 与 _rocm_C 都在 /mnt/game/port/build-sync/temp/）
   docker exec z100-regress bash -c 'cp /mnt/game/port/build-sync/temp/_rocm_C.abi3.so \
     /mnt/game/port/build-sync/temp/_rocm_C.abi3.so.bak-s33 && md5sum /mnt/game/port/build-sync/temp/_rocm_C.abi3.so*'
   docker exec -e MAX_JOBS=5 z100-regress bash -lc 'bash /mnt/game/port/build-sync.sh _rocm_C'
   ```
   **竞态坑**：若 build log 只有 `[1/6] Running hipify` 且 `.so` md5 未变 =
   hipify 拷贝与 ninja 竞态，**原样重跑 build-sync 即恢复**（已实测两次），
   禁止手改 temp 产物或删 stamp。改后必须 `nm -DC` 或微测确认新符号生效。
4. **profiler collect 会 wedge 服务**：`s27_five_phase_profile --phase collect`
   在部分臂上会 `sample_tokens` RPC 超时并使引擎退出（已知危害）。规则：
   collect 前先跑完该臂的 e2e/capture；collect 后立刻重启服务再继续。
5. **贪心流判定口径**：基线有 **两个重启世界**（`.s29/capture-s29-bm16-r1.json`
   与 `-r2.json` 在 pos 2 的 0-margin 平局互相翻转）。判定"零翻转" = 新臂流
   **等于 r1 或 r2 之一**（`cmp_capture.py` STREAMS IDENTICAL）且会话内 3 reps
   位同。只对 r1 比对会得到假阴性。
6. **指纹工具**：新增任何 `VLLM_*` knob 必须同步加进
   `tools/dcu/dcu_policy_fingerprint.py` 的 KNOBS（否则每次启动刷 WARN 且
   A/B 指纹不可比）。
7. **度量工具全套现成**（勿重写）：
   - e2e：`bash .s30c/probe.sh <tag>`（7469 doc / 96 tok / 3 reps，写 `.s30c/`）
   - 贪心流：`PYTHONPATH=tools/dcu .venv/bin/python .s29/capture.py --tag X --out .s33`
     / 比对 `.s29/cmp_capture.py a.json b.json`
   - trace/kdiff：`s27_five_phase_profile --phase collect` + `.s29/kdiff.py`
   - GEMV 微测：`tools/dcu/p0_gemv_bench.py --mode ref|bench`（进程级 env）
   - GDN 微测：`tools/dcu/s30c_gdn_m1_bench.py`
8. **每 WP 通用验收流**：真实 shape 微测 → 数值对照 → 服务 smoke（environ +
   policy 指纹核验）→ e2e（3 reps × N≥2 重启）→ 文档。改模型输出的项加
   eval + 用户裁决。
9. MBT/UTIL：restart.sh 固定 `MAX_SEQS=4 MBT=8192`，与基线一致，**不要动**。

---

## WP0：核间空隙分账（只测量，~半天）

1. 现役臂起服 → `bash .s30c/probe.sh s33-gap`（热身）→
   `s27_five_phase_profile --phase collect --profile-prefix s33gap --trace-dir
   /mnt/game/s30c-prof --profile-steps 20 --out .s33 --tag s33gap`
   （collect 后按须知 4 重启）。
2. 分账脚本：kdiff 全表 `us_per_round` 求和（GPU busy）→ 与同窗 e2e step p50
   相减 → 差值再按"窗口 skew 工件（nccl 行跨 trace 方差）/ 核间空隙（kernel
   count × 平均 gap）/ 其他"三分。参考：`.s30c/kdiff-s29ns-s30cm1b.csv` 口径。
3. ITER_DETAILS 佐证：`ITER_DETAILS=1` 起服跑 1 rep，
   `grep "iteration elapsed" .s33/serve-*` 取 decode step 分布。
4. **判据**：分账覆盖差值 ≥90%；产出 `.s33/gap_accounts.json` +
   `docs/result/z100-s33-gap-accounts.md`。**不改任何代码**。
   本 WP 结果决定 WP3 融合的优先级（空隙大 → 融合升值）。

## WP1：fp8 GEMV whole-A 内核（数值位同，e2e −1.5~−2.5 ms）

### 1.1 前置核实（对不上就停下报告）
```bash
grep -n "kStreamSlice = 1024\|kStreamChunk = 16\|kStreamSmemCap" csrc/rocm/fp8_emu_gemm.cu
grep -n "fp8_emu_gemv_stream_m1_kernel\|stream_m1_enabled\|VLLM_DCU_FP8_GEMV_M1" csrc/rocm/fp8_emu_gemm.cu | head
grep -n "VLLM_FP8_EMU_BLOCK_W8A16(T, AT)" csrc/rocm/fp8_emu_gemm.cu   # 派发宏（M==1 m1 分支在其内）
```
确认：m1 kernel（WP2-N 遗产，默认关）已是 M=1 专用骨架（整核无 M 维、
rows 走 `stream_block_select`、2-slice 预取、每 slice 2 barrier）。
形状/字节账：qkv 5120×2048、o 5120×1536、GDN-in 5120×2560、GDN-out 5120×1536、
gateup 5120×8704（A 行均 10 KB）；down（17408 或 4352 视 K 切法）**排除**。

### 1.2 实现（`csrc/rocm/fp8_emu_gemm.cu`，m1 kernel 之后）
新核 `fp8_emu_gemv_wholea_m1_kernel<RW, T>`，规格：
- **一次 stage**：整行 A（K×2B，10 KB）+ 折算 scale（nkb floats）进动态 LDS；
  stage 后**一次** `__syncthreads()`；K 循环内零 barrier。
- K 循环：W 走 uint4 预取环 `pf[3]`（或 [4]，以寄存器不溢为准——编译后查
  scratch spill），每 lane 每 step 1×uint4（16 fp8）→ `fp8x4_to_f16x2x2` ×4
  → 8×`dot2` → `acc += loc * scale[kb]`；**slice/chunk 遍历顺序与 m1 kernel
  完全一致**（位同的前提，逐行对照旧循环写）。
- 行归属/n0/行越界写零/bias/lane0 归约：照抄 m1 kernel 现行实现。
- **down 不适用**（34 KB → 每 CU 1 block）：留在现路径。
- 派发：`run_block_w8a16` 宏 m1 分支扩为 `M==1 && m1_enabled` 时按新 env
  `VLLM_DCU_FP8_GEMV_WA`（默认 0）选 whole-A 或 slice-m1；`W8A8` 宏不动。
- bank 冲突（WP-f3 并入）：hip 汇编/isa 流程审计 `ds_read` 4-way 嫌疑
  （lane 读 32 B 跨步）；若确认，A slice 行距 pad 1×uint4（只改地址不改序）。
- 指纹 KNOBS 加 `VLLM_DCU_FP8_GEMV_WA`。

### 1.3 构建与微测
1. §0.3 构建 `_rocm_C`（备份 md5 记 `.s33/wp1_so_md5.log`）。
2. 微测（每形状独立进程，env 先置）：
   ```bash
   VLLM_DCU_FP8_GEMV_STREAM=1 VLLM_DCU_FP8_GEMV_STREAM_ROWS={0,4} \
   VLLM_DCU_FP8_GEMV_WA=1 python tools/dcu/p0_gemv_bench.py --mode bench --out /tmp/p0ref
   # 位同对照：--mode bench 输出含 max_abs_delta（对 /tmp/p0ref 的 ref 值）
   ```
3. **判据 A（带宽）**：gateup ≥600 GB/s、qkv ≥450、o ≥400、GDN-in/out ≥450
   （任一不达 → 停，出表报告，不得硬上服务）。
   **判据 B（位同）**：全形状 `max_abs_delta` 与现路径逐值相同（0 差异级）。
4. 服务 A/B：`VLLM_DCU_FP8_GEMV_WA=1` 重启（N≥2）→ e2e 判据 **−1.5 ms 以上**
   + TTFT ±3% + 贪心流落基线世界（位同应免检，仍跑一遍确认流程完好）。
5. 文档 `docs/result/z100-s33-wholea-gemv.md`（微测表/位同表/e2e/md5/回滚）。

**回滚**：`VLLM_DCU_FP8_GEMV_WA=0`（默认即关）或臂脚本不设；二进制 cp 回 bak。

## WP2：gfx906 one-shot AR 复活+验证（e2e −2.5~−4，验收门 −2）

### 2.1 前置核实
```bash
sed -n '260,300p' csrc/custom_all_reduce.cuh        # VLLM_CUSTOM_ALLREDUCE_ALGO: 1stage/oneshot/2stage/twoshot
sed -n '270,300p' vllm/distributed/device_communicators/custom_all_reduce.py   # gfx906 opt-in
sed -n '525,560p' vllm/distributed/device_communicators/custom_all_reduce.py   # dtype/size 门（bf16→RCCL）
grep -rn "uncached\|finegrain\|FineGrain" csrc/custom_all_reduce.cuh | head   # 历史阻塞点
```
历史结论（comm report）：S13 e2e ±0.4%（中性）+ 细粒度分配失败 → 本 WP =
根治分配 + oneshot 强制 + 按本图口径重测，**不是新写内核**。
载荷口径：decode AR = hidden 5120 fp16 = 10 KB/次，128 次/round。

### 2.2 任务
1. 根治 uncached alloc：细粒度缓冲分配走已验证的 fine-grain 路径
   （`HSA_FORCE_FINE_GRAIN_PCIE=1` 已在 env.sh；S17 先例 8.9×@1MiB）。
2. 强制 oneshot：`VLLM_DCU_CUSTOM_AR=1` + `VLLM_CUSTOM_ALLREDUCE_ALGO=oneshot`；
   bf16 一律保持 RCCL（现 gate 行为不动）；size 门 ≤64 KiB → oneshot，其余
   RCCL（**prefill 大载荷不受影响，TTFT 不回退**）。
3. piecewise graph capture 兼容验证：capture 期（服务启动日志 Capturing CUDA
   graphs 段）无 hang、replay 正确（registered buffer 路径）。
4. 微测：10/30/64 KiB 三档 × {无扰动, 模拟 skew（一 rank 人为延迟 10–30 µs）}，
   出 oneshot vs nccl 表。**skew 档必测**（勘误①：等人的时间省不掉）。
5. 服务 A/B（N≥2）+ 长跑 30 min（decode 连发，0 hang/0 错）+ TTFT 复测。

### 2.3 判据
- 微测：无扰动 10 KiB ≤ 25 µs/call；skew 档 ≤ nccl −40%。
- e2e：**−2 ms 验收门（目标 −2.5~−4）**；TTFT ±3%。
- 数值：固定 rank 序 → 逐 run 确定；贪心 diff 板交付，流仍须落基线两世界
  之一，**落世界外 → 停，按 WP-C 流程交用户裁决**。
- 文档 `docs/result/z100-s33-oneshot-ar.md`（含分配修复说明 + 微测/e2e/长跑）。

**回滚**：去掉 `VLLM_DCU_CUSTOM_AR`（默认 0 = RCCL），env 一行。

## WP3：融合层（依赖 WP2；e2e −0.8~−1.4 ms）

### 3.1 前置核实（先映射，后动手）
- norm 实例归属：trace 中 `triton_red_fused_*fused_add_rms_norm*` 48/16 c/r
  逐个对应到层位置（post-attn / post-MLP），确认哪些紧跟 AR 之后
  （grep `vllm/model_executor/models` 相应 Qwen3.5 前向 + 现有
  `_C`/triton 调用点）。
- silu-mul 配对：`triton_poi_fused_mul_silu_slice_*` 64 c/r，行配对
  (i, i+4352) 的 gate/up 语义核实。
- 勘误③红线：**norm 不可融进行并行 GEMV epilogue**（那里只有 AR 前的部分和）；
  唯一正确融合点是 AR 核（归约完成即得完整行）。

### 3.2 任务
1. **AR 核内 norm+residual+quant**：one-shot AR 归约完成后，同核同 warps 做
   residual-add + rms_norm + fp8-quant（输出直接写下一 GEMV 的 fp8 输入）。
   数学逐条对齐现 triton 核（rsqrtf+1e-6f 同款、累加顺序对齐），fp32 域
   max|Δ| 报表 ≤1e-6（同序应位同；若实现需重排，走零翻转硬门）。
   核签名变化：多传 residual 指针 + weight + scale 输出；仅 decode 载荷启用。
2. **gateup silu-mul epilogue**：whole-A/m1 gateup 核按行配对 (i, i+4352)
   输出 silu(a)*b（消 `mul_silu_slice` 64 c/r）。
3. **GDN CFG 扩展（可选）**：m1_native CFG 表加档（每 lane 2 v-行布局），
   微测 ≤11 µs 才收。
4. 核数验收：decode 每 round 独立 kernel 数下降 ≥40（kdiff c/r 表前后对照）。

### 3.3 判据
微测 per-op −20% 以上；e2e −0.8 ms 以上；贪心流零翻转；单项 <0.3 ms 不采纳。
文档 `docs/result/z100-s33-fusion.md`。

**回滚**：融合核走独立 env（`VLLM_DCU_AR_FUSED_NORM=1`、
`VLLM_DCU_GEMV_FUSED_SILU=1`，默认 0）。

## WP4：attention decode 分段（+按需 split-K；e2e −0.3~−0.9 ms）

1. 前置核实：`vllm/v1/attention/ops/triton_unified_attention.py:129`
   `GFX906_DECODE_SEGMENTS`（已存在的钩子，默认 0）；`reduce_segments` 核
   （prefill 3D 已用）的复用面。
2. 实现：decode seq 分 S 段（S 按 seq 自适应：≤8k→4；32k→8；更大→16），
   段内部分 softmax/sum 写 `[S]` 工作区，`reduce_segments` 归约。
   **改累加序 → 贪心流零翻转硬门 + max|Δ top-1 logprob| 报表（BM16 流程）**。
3. 微测：7.5k ctx per-op −30% 以上；32k/128k 档各一跑（收益放大验证）。
4. 服务：e2e −0.3 ms 以上（7.5k ctx 收益小，属长文投资）；TTFT ±3%。
5. 文档 `docs/result/z100-s33-attn-segments.md`。
   回滚：`VLLM_GFX906_ATTN_DECODE_SEGMENTS=0`（现默认）。
   **split-K GEMV 小 N 仅当 WP1 后 qkv/o 仍 <450 GB/s 才启动**，规格同
   S31-WP1（零翻转硬门）。

## WP5：lm_head fp8 + KV fp8 LUT（eval + 用户裁决；不默认执行）

1. **lm_head fp8**：checkpoint 的 lm_head 转 W8A16（block 128×128 scale，
   服务路径现成 `run_block_w8a16`），907→~460 µs。交付：贪心 diff 板
   （96-token 流 + 8-prompt top5 logprob 偏移报表）+ eval（对齐
   `docs/contributing/model/tests.md` 精神）→ **用户裁决后才可进现役**。
2. **KV fp8 LUT**：先微测证伪"ITL +41% = 反量化未融合"假设（LUT 反量化进
   attention kernel，参照 fp8 GEMV 的 LUT 方案）；证实后才进服务级；
   262k 档补测。
3. 文档 `docs/result/z100-s33-fp8-extensions.md`。默认两 env 全关。

## 终局（全部 WP 后）

1. `s33-full` 臂（采纳项叠加）×N≥3 重启 e2e，双口径报告；每步叠加前单独
   A/B（本图各 WP 已含）。
2. final 文档 `docs/result/z100-s33-final.md`：WP 状态表 + 分账表 + 全部
   md5/指纹 + 与 s30c 的对照（29.48 → 实测值）。
3. 现役恢复：按用户裁决保留 s33 臂或回 s30c 臂；`.s33/` 全部 JSON 归档。
4. 纪律重申：零翻转硬门、双口径、N≥2、每 WP 独立回滚、改输出必 eval+裁决。

## WP 状态表（实施时逐项更新）

| WP | 判据 | 状态 | 证据 |
|---|---|---|---|
| WP0 空隙分账 | 覆盖 ≥90%；只测量 | **完成**（trace-window 覆盖 100%；真实核间空隙 ≈3.8 ms/round，ITER_DETAILS 佐证） | `.s33/gap_accounts.json`、`docs/result/z100-s33-gap-accounts.md` |
| WP1 whole-A | gateup≥600/qkv≥450/o≥400；位同；e2e −1.5 ms | **位同 PASS；判据 A 未达 → 按图停**（rows=4 最优：gateup 586.9/qkv 402.5/o 361.3/gdn_in 440.8/gdn_out 363.6；折算 e2e −0.9~−1.1 ms < −1.5 门）；默认关，待用户裁决 | `docs/result/z100-s33-wholea-gemv.md`、`.s33/wp1_bench_*.json`、md5 `.s33/wp1_so_md5.log` |
| WP2 one-shot AR | 微测 10 KiB ≤25 µs；e2e 门 −2 ms；长跑 0 hang | **分配修复 ✓、e2e −3.40 ms 超门，但贪心流两次落世界外 → 按图停**（graph 臂确定性劣化 / eager 臂非确定；现役已回滚）；待用户裁决是否立项修复 | `docs/result/z100-s33-oneshot-ar.md`、`.s33/wp2_car_*.json`、`.s33/capture-s33wp2-*.json` |
| WP3 融合 | per-op −20%；e2e −0.8 ms；核数 −40 | **不执行**（AR-norm 随 WP2 停；silu epilogue 载体 whole-A/m1 未采纳；GDN CFG 扩展按"单项 <0.3 ms 不采纳"规则出局：12.9→11 µs 仅 −0.09 ms e2e） | 本表 + WP2/WP1 停点报告 |
| WP4 split-KV(+split-K) | per-op −30%@7.5k；e2e −0.3 ms；零翻转 | **维持现状**（分段已产线激活：auto=128 段，kdiff grid [1,1,128]；设计图"默认 0=关"前提过时；扫档 seg32=30.17 / seg64=29.16 vs auto=29.33，无过门增益） | `.s30c/s20_s33wp4-seg32_*.json`、`.s30c/s20_s33wp4-seg64_*.json` |
| WP5 lm_head/KV fp8 | eval + 用户裁决 | **按图不默认执行**；备料（量化预期）入 final 文档，随 WP1/WP2 裁决一并交用户 | `docs/result/z100-s33-final.md` |
| 终局 | s33-full N≥3；final 文档 | **完成**（无采纳项 → s33-full = 现役 s30c ctrim 复测 N=3） | `docs/result/z100-s33-final.md` |
