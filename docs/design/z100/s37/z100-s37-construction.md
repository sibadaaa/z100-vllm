# S37 施工图:W3 收口 + W4 AR 融合 + split-K 评估 + B2 MTP 裁决

> 前置阅读:同目录 `z100-s37-design.md`。S36 终局
> (`docs/result/z100-s36-final.md`)为事实基线;其数据文件
> `docs/perf/data/s36-*.txt` 与工作目录 `.s36/`(restart 脚本、位同语料、
> H1b patch)直接复用。
> **部署基线**:无 MTP 服务在役(base2 实例,启动器 `/tmp/launch_nospec.sh`,
> 基线 `.so` md5 `86470c5d`,备份 `.r3/so-backup/_rocm_C.s36-before.so`)。

---

## 0. 通用纪律(在 S36 §0 之上增补)

1. csrc 改动:备份在役 `.so` + md5 → 容器重编译(`docker start z100-regress;
   docker exec z100-regress bash -c 'cd /mnt/game/port/build-sync/temp && ninja
   _rocm_C.abi3.so'`)→ 微测 → 数值对照 → 服务 smoke → e2e;失败恢复备份并回
   md5。注意容器构建后 `.so` 属主变 root,回滚用 `docker exec cp`。
2. py 改动:纯 env/参数类先做;逻辑类走 feature 分支;每处改动 `git diff` 入台账。
3. 每步留档 `docs/perf/data/s37-<wp>-<date>.txt`(原始输出,不誊写)。
4. **长 ctx 探针一律带预热线**(health 200 后先 128k warm 2-token,见 A1)与
   `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200`(臂级 env)。
5. 服务重启统一用 `.s36/restart_seg_arm.sh <tag> <SEG|none> <SEGMAX|none>`
   (含 in-flight 检查、孤儿处置、健康等待、env 指纹核验)。

---

## 1. A1/W3 收口:ABAB 交错确认 + executor 治理(0.5d)

### 1.1 第零步:测量工具参数化(0.5h)

- 复制 `/tmp/ctx_sweep.py` → `.s37/ctx_sweep.py`,改两处:
  `ask(p, 2)`(warm 不变)与 `ask(p, 16)` → `ask(p, NTOK)`,
  `NTOK = int(sys.argv[1])`,ctx 列表改为 `sys.argv[2:]`;
  每 ctx 测 2 rep(循环两次,输出两行,统计取两行 p50 的较小值——保守侧)。
- 预热线:`.s37/prewarm128k.sh` = 单次 `ctx_sweep 131072` 的 warm 段
  (或直接跑一次 `ctx_sweep.py 48 131072` 弃数)。

### 1.2 第一步:ABAB 四实例(约 3h,含重启与 128k warm)

```bash
# A = 256, B = 128(默认);每实例:重启 → prewarm → 4 ctx × 2 rep
bash .s36/restart_seg_arm.sh s37-A1 none 256
bash .s37/ctx_sweep.py 48 16384 32768 65536 131072   # rep1+rep2
bash .s36/restart_seg_arm.sh s37-B1 none none
bash .s37/ctx_sweep.py 48 16384 32768 65536 131072
bash .s36/restart_seg_arm.sh s37-A2 none 256         # A 重复
bash .s37/ctx_sweep.py 48 16384 32768 65536 131072
bash .s36/restart_seg_arm.sh s37-B2 none none        # B 重复
bash .s37/ctx_sweep.py 48 16384 32768 65536 131072
```

- 产出:`docs/perf/data/s37-w3-abab-{A1,B1,A2,B2}.txt`;
  汇总表(A1/A2 中位 vs B1/B2 中位,逐 ctx,附各实例 A/A drift)。
- **判据**:128k A 中位 − B 中位 ≥3% 且 <0 → 默认改 256;16k 同向 ≥3% 佐证;
  32k 只记录不裁决(S36 已证其基线离散 ~5%)。
- **1.3 步默认值改动**(过门才做):`vllm/v1/attention/ops/
  triton_unified_attention.py` 常量缺省 `"128"` → `"256"`(一行),重启
  生产臂,跑 `ctx_sweep 48 16384 131072` 冒烟确认 ITL 与 A 臂一致,留档。

### 1.3 第二步:executor 自杀治理(与 1.2 并行)

- 臂级 env(加入重启命令,不改 vllm_env.sh):
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200`。
- 任何实例若再现 60s×3 shm 警告序列:立刻
  `py-spy dump --pid <worker TP0>` ×5(30s 间隔)→ `.s37/exec-stall-*.txt`;
  服务自杀则重启后在该臂补 `prewarm` 再续测量。
- py-spy 不可用时降级:worker 侧 `VLLM_LOGGING_LEVEL=DEBUG` + 
  `timeout 5 strace -p <pid> -f -c` 留系统调用分布。

---

## 2. A2/W4:push AR 出口融合(1-1.5d,引用 wp4 §2-§3)

**照抄** `docs/design/z100/s34/z100-s34-wp4-ar-fusion.md` §2-§3 执行,此处只列
S37 增补:

### 2.1 第零步:当日口径重测(1h)

- 无 MTP 臂 32k profile(`/start_profile` → 1 请求 → `/stop_profile`),
  取 rank0 表:push 核 µs×次数/步、rmsnorm/quant 小核 µs×次数/步,
  重写 W4 收益账(`docs/perf/data/s37-w4-baseline-profile.txt`)。
  S36 参考值:push 28.7µs×125/步;S36 设计图曾引 69.4µs×128(口径不同,
  以当日为准)。

### 2.2 实施顺序(按 wp4 §3,增补两条)

1. 触点:`csrc/custom_all_reduce.cuh`(push 核归并出口)、
   `vllm/distributed/device_communicators/custom_all_reduce.py`
   (签名 + `VLLM_DCU_CUSTOM_AR_FUSION` env,默认 0);
   影响目标 `.so`:`_C_stable_libtorch.abi3.so` —— **先备份+md5**
   (对齐 `.r3/so-backup/` 惯例,命名 `_C_stable_libtorch.s37-before.so`)。
2. 重编译走 build-sync(0. 条),构建后 spill/VGPR 复核(§4 工具);
   融合核若改归约序,位同门直接走 logprob 板,不硬凑 torch.equal。
3. 微测:`tools/dcu/car_bench.py` 加融合档(T=1/T=4,fp64 误差表);
   门:per-op −20%(wp4 原判据)。
4. e2e:无 MTP 臂 7469-doc 探针 3 reps,门 −0.8 ms/round;
   再 60k/128k 各 1 rep 看长 ctx 方向(只记录,不设门)。
5. 回滚:env 默认 0 + revert py;`.so` 恢复备份。
6. **S37 增补**:若 A1 已把 SEG_MAX 默认改 256,W4 的 e2e A/B 两臂都必须在
   该默认上跑(口径统一),并在台账注明。

---

## 3. A3:split-K 评估(0.5d 可行性 + 1d 实现,kill-fast)

### 3.1 第零步:形状分布核实(0.5h)

- 从 S36 32k profile(rank2 表)核对:fp8 m1 248 launch/步 = 62 层 ×
  {qkv,o,gateup,down};把每形状 per-step launch 数写进
  `docs/perf/data/s37-splitk-shapemix.txt`(收益账的输入)。

### 3.2 第一步:可行性微测(零生产风险,0.5d)

写 `.s37/splitk_feasibility.py`(packed 口径,复用 s36_packed 方法):

- **实验 1(并行度响应)**:现役内核,N ∈ {1280, 2560, 3840, 5120} @
  K=6144,量 GB/s-vs-blocks;再测 K=6144×2 拼接块(模拟 S=2 总负载),
  外推 o@640/960 块可达带宽。
- **实验 2(分片数值代价预演)**:对同一输入,全 K 单次调用 vs 两个
  半 K 调用 + fp32 求和(torch.sum(partial, dim=0)),统计
  max|Δ|/|ref| 分布(100 组随机 seed),给 logprob 板定标尺。
- **过门条件**:外推 o@640 块 ≥450 GB/s 且 max|Δ| 相对 ≤3e-4(留 3 倍
  余量给核内实现);否则 WP5 存档关闭,记录负结果,WP 终止。

### 3.3 第二步:最小实现(仅当 3.2 过门,~80 行)

| 位置 | 改动 |
|---|---|
| `csrc/rocm/fp8_emu_gemm.cu` m1 区(~884 起) | 新增 `fp8_emu_gemv_stream_m1_splitk_kernel<S, RW, T>`:slice 1024 几何不变(S36 已证局部最优),block 只扫 `[seg·K/S, (seg+1)·K/S)`,partial 写 fp32 workspace |
| 同文件 reduce 核 | `reduce_splitk_m1_kernel`:**固定 s=0..S-1 段序**逐段加 + bias/out 写出(复用现有 cross-lane 语义) |
| 同文件 dispatch(`dispatch_gemv_stream_m1` ~1045) | env `VLLM_DCU_FP8_GEMV_SPLITK`(默认 0,读法对齐 `stream_m1_enabled()` ~1067);仅 M==1 && N≤2048 && K≥4096 生效 |
| 同文件 op 入口 | workspace:op 内 static 缓存 (S,N) fp32(decode 单流;多流风险记录在案) |
| `tools/dcu/...` 微测 | packed 五形状 × {S=1(现役), 2, 3} |

- VGPR/spill 复核(§4 工具):m1 splitk 版 spill 必须为 0。
- **数值门**:max|Δ|≤1e-3 相对 + top5 logprob 板(**非** torch.equal,
  S29 披露口径);微测门:o ≤18µs、qkv ≤20µs(packed),gateup/down 零回归
  (env 隔离应不触)。
- e2e 门:step p50 −0.3 ms 以上采纳;否则 env 保持默认 0 存档。
- 回滚:env 归零 = 行为回到现役路径;`.so` 恢复备份 + md5。

---

## 4. 通用工具(沿用 + 增补)

- 内核元数据提取(VGPR/spill):S36 施工图 §4 原样可用(rocprofv2
  `--kernel-trace`,注意本机版本**不接受** `--` 分隔符;SCR 列即 spill)。
- ABAB/重启:`.s36/restart_seg_arm.sh`;探针:`.s37/ctx_sweep.py`(1.1 步)。
- 位同语料:`.s36/parity/`(120 张量)+ `/tmp/s36_parity.py check`
  ——W4/split-K 改动后凡涉及现役路径的行为差异,先跑此门。

---

## 5. B 轨:B2 —— MTP dynk 修复与重启裁决(0.5d,可与 A 并行)

### 5.1 第一步:durable 修法(~10 行 + 单测)

- `vllm/v1/core/sched/scheduler.py:302-305`:
  ```python
  self.dynk_ema_good = min(5.0, 0.75 * self.num_spec_tokens)
  self.dynk_mid_k = min(3, self.num_spec_tokens)
  ```
  (k=7 → ema_good 5.25→min 5.0、mid_k 3:行为逐值不变;k=2 → 1.5/2)。
- 单测 `tests/v1/core/test_dynk_spec_len.py`(新增,最小化):k' 矩阵
  k∈{2,7} × ctx∈{1k,16k,65k,128k} × EMA∈{None,1.0,1.6,5.1},
  断言 k=7 行为与改前一致(先在改前跑一遍存基线)。
- 门:`pytest tests/v1/core/test_dynk_spec_len.py -v` 全绿;diff 入台账。

### 5.2 第二步:MTP 臂 A/B(裁决)

- 臂 M(参考):现役 MTP 部署口径(dynk on;`.s34/launch_mtp2.sh` 原样,
  vllm_env.sh 注入 VLLM_S24_DYNK_LO=16384)。
- 臂 D(修复):同上 + `VLLM_S24_DYNK_LO=0`(dynk 整体关;若 5.1 已落地,
  也可不设 env 直接由 durable 修法生效——**两条路径不要同时改**,先 env 后
  代码,读数干净)。
- 探针(每臂):7469-doc 7.5k/96tok/3rep + 60k/256tok/1rep + 128k/256tok/1rep
  (全带预热线);metrics:`spec_decode_num_drafts_total`、
  `num_draft_tokens_total`、`accepted_tokens_per_pos`。
- **判据**:D 臂 draft_tokens/drafts ∈ [1.9, 2.1] 且 pos1 ≥0.6,且 60k/128k
  step p50 比无 MTP 基线快 ≥10% → **MTP 重启获推荐**;任一不满足 →
  MTP 继续冻结,B 轨关档并把读数写进终局报告。
- 隔离:B 轨不改任何 A 轨触点;A 轨 e2e 永远在无 MTP 臂。

---

## 6. 顺序与工作量

```
A1 W3 收口(0.5d)→ A2 W4(1-1.5d)→ A3 split-K 可行性(0.5d)→ [实现(1d)]
B1 durable 修法(2h)+ B2 MTP A/B(0.5h 探针 ×2 臂)   ← 与 A 轨并行,裁决独立
```

- A3 可行性未过门则该轨当轮关闭(省 1d);
- B2 裁决若"MTP 重启获推荐",MTP 默认部署的切换(生产口径)是独立决策,
  不在本轮施工图内自动执行——产出裁决材料,交人工拍板。
