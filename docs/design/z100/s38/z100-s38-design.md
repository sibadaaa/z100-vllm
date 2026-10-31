# S38 设计图:AR 复合路径收口 + FP8 GEMV 小 K 特化(无 MTP 臂 decode 二期)

> 日期:2026-10-30。输入:S38 前置实测 `runs/s34nospec/REPORT.md`(无 MTP 臂
> 39-step trace 分环节 + FP8 GEMV 分级带宽专题)、S37 终局
> (`docs/result/z100-s37-final.md`,W4 融合核已建未部署、A3 split-K 判死)、
> s34-wp4 融合图(`docs/design/z100/s34/z100-s34-wp4-ar-fusion.md`)、
> KERNEL_REPORT 天花板(只读 784.8 / D2D 777.5–809.6 GB/s)。
> **与 S37 的关系**:S37 留下三个未收口(W4 模型层穿透+e2e、W4 融合 `.so`
> 未进在役库、A3 split-K 关档);本设计把实测新证据(AR 占 38%、o/out 仅
> 309 GB/s)折成两条施工轨。

---

## 0. 基线快照(2026-10-30,全部判据的对照面)

| 项 | 值 |
|---|---|
| 在役臂 | 无 MTP(`SPEC_OFF=1`,systemd `z100-vllm-mtp2`),FP `2f3793deb97e138f` |
| 服务级 | W1 贪心 37.45/37.84 t/s,W2 采样 37.07,TTFT 0.72 s,step ≈ 26.7 ms |
| step 构成(GPU busy 24.2/26.7 ms) | fp8 GEMV 10.84(44.8%)、AR 复合 ≈10.1 sum 口径(38%:push 63.3 µs×129 = 8.16,nccl 回退 481 µs×3.3 = 1.59,AllGather 365 µs×1 = 0.37;push union 7.45)、norm/elem 2.86、attn 1.23+0.32、GDN 0.89、bf16 0.52 |
| `.so` | `_rocm_C.abi3.so` md5 `ca05f746`(splitk 代码在库,env 默认 0);`_C_stable_libtorch.abi3.so` md5 `7dcce4f2`(**W4 融合核不在内**) |
| GEMV 分类速率 | gate_up 615 / down 575 / qkv 524 / packed-in 477 / **o,out(K=1536)309 GB/s**;rows=4 已证最优;split-K 判死 |
| 测量工具(已就绪) | `runs/s34nospec/{profile_window,parse_trace,gemv_bw_bench}.py`、`runs/specbench/e2e_ts.py`、`.s37/car_fused_bench.py` |

**排序与结构**

```
Track A(AR 复合路径,服务/宿主轨;预期 −1.9~−3.1 ms/step)
  A1  W4 收尾    融合核部署 + 模型层 ar_epilogue 穿透 + e2e      −0.7~−1.2 ms(门 −0.8)
  A2  回退归位   3.3 次/step nccl ring AR → push-AR              −1.0~−1.6 ms(门 −0.5)
  A3  logits     lm_head/AllGather 路径定位 + 优化               −0.2~−0.3 ms(兼闭字节账)
Track B(FP8 GEMV 小 K 特化,内核轨;与 A 独立可并行)
  B1  三案竞速   RW=2 双行(0.5h)→ slice512(1d)→ 尾 slice lane 拆分(视前两者)
  B2  部署       形状门自动生效 + trace/e2e 复测                 −0.6 ms(门 −0.3)
不施工(记录):norm/elementwise 深融合(二阶,−0.5~1 ms 待 AR/GEMV 落地后重测)、
attention decode BW(绝对值小)、rows=8(已证伪)、split-K(S37 判死)。
```

排序理由:**确定性优先**——A1 的内核/host/绑定/python 四层 S37 已交付并微测
(0.81–0.88×),只剩重建 + 穿透 + e2e,是全案性价比最高的一步;A2 收益最大但
病灶未定位,设计为"定位(0.5d, kill-fast)→按位点修";B1 的最小方案(RW=2)
只是派发层改动,与 A 轨零耦合,可并行;A3 排最后(单项小,但有闭合字节账的
附属价值)。A1 落地后 AR 占比下降,A2/A3 的 e2e 门在**新基线上重算**
(门值不变,对照步长更新)。

**与 MTP 的关系(重要披露)**:S37 B2 已裁决 MTP 重启 GO(+29~42%),由操作者
拍板;本设计全部 e2e 在**无 MTP 臂**同口径(测量隔离原则不变)。依赖性披露:
A1/A2/A3 在 MTP 臂同样成立(AR 路径与 verify 行数无关,融合核 S37 已测 rows=12);
**B1 的 M=1 特化在 MTP 下基本失效**(verify M=3 走 stream_block 而非 m1 内核),
若切回 MTP,B 降级为"为无 MTP 场景预留",收益不入 MTP 口径。

---

## 1. A1/W4 收尾:融合 AR 出口部署(0.5–1d)

### 1.1 现状(S37 §4 遗留)

- 已交付:`cross_device_reduce_push_fused`(push 五步与归约序不变,出口
  add-rmsnorm,residual 位同)、host `allreduce_fused`、绑定
  `_C_custom_ar.all_reduce_fused`、python 链
  `custom_all_reduce_fused`(env `VLLM_DCU_CUSTOM_AR_FUSION` 默认关 + 静态
  形状门 + 回退语义)、微测 `s37-w4-fused-microbench.json`(rows 1/4/8/12:
  0.81–0.88×,per-op 省 5.5–10.1 µs)。
- 未做:(1) 融合核未进在役 `_C_stable_libtorch`(基线 md5 `7dcce4f2`);
  (2) 模型层 `ar_epilogue` 穿透;(3) e2e 验收(门 −0.8 ms/round)。

### 1.2 设计:穿透点与图边界

- **穿透点(Qwen3NextDecoderLayer,`vllm/model_executor/models/qwen3_5.py`)**:
  每层两处 RowParallelLinear 出口(attn 侧 → 本层 post_attention_layernorm;
  mlp 侧 → 下层 input_layernorm)。当融合可用(形状门过 + env 开)时,
  把"AR → 写存 → fused_add_rms_norm 读回"替换为单次
  `vllm.all_reduce_fused_rmsnorm(hidden, residual)`(mutates residual)。
- **图边界**:该 custom op 是 piecewise 分裂点 → AR+norm 都移出图外。
  代价: norm 的 ~10 µs GPU + 一次发射脱离图捕获;收益:AR 出口少一轮
  写存/读回(S37 微测已覆盖净效应,0.81–0.88×)。回退语义:融合不可用时
  = 现役 AR + `fused_add_rms_norm`,逐位一致。
- **部署面**:`.s34/serve_deploy.sh` 增加 `export VLLM_DCU_CUSTOM_AR_FUSION=1`
  (采纳态;回滚 = 注释该行)。指纹漂移预期(新增 env 不在指纹前缀内,记录)。

### 1.3 门与风险

- **微测门(复测)**:`car_fused_bench.py` rows∈{1,4} 融合/未融合 ratio ≤0.85
  且 max|Δout| ≤5e-04、Δresidual = 0(位同);不过 → 停,不重建部署。
- **e2e 门**:同实例 AB(先 B 后 A 各 2 实例)step p50 **−0.8 ms/round**
  (S37 原门);W1 三 rep 贪心流 in-arm 一致;parity 板 8 prompt
  max|d top-1 logprob| ≤ S29 先例界(0.2255 nat)。
- **风险**:穿透改到 decoder 层 → 影响面大;以 env 默认关 + 逐位点 enable
  控制,可单点回退。norm 移出图后 host 发射 +~128 次/step,须在 trace 里
  确认 host 空隙不涨(§3 的 host gap 指标)。

---

## 2. A2:nccl ring 回退 AR 归位 push-AR(1d,定位先行)

### 2.1 现状(trace 实测)

- decode 每步 129 次 push-AR(63 µs)之外,还有 **3.3 次/step nccl ring AR
  (481 µs,共 1.59 ms/步)**。位置分布(窗口总量):GDN core 内 65、
  3 个 CompiledFxGraph 内 48(27+12+9)、`fp8_emu_mm_block_w8a16_auto` 内 8、
  python 层 7。
- 两个候选病因(定位步骤裁决):
  H1 **图捕获回退**:DCU push-AR 在 graph capture 下不可用(finegrained IPC
  flag?)→ 编译区内的 AR 被 nccl 替代并被捕获;H2 **尺寸门**:该 AR 消息
  > `VLLM_DCU_CUSTOM_AR_MAX_KIB=128`(M=1 hidden 5120 fp16 仅 10 KB,H2 概率低,
  但 GDN packed 出口若有更大缓冲则成立)。

### 2.2 设计:先定位,按位点修

- **定位(0.5d, kill-fast)**:一次性 env `VLLM_DCU_CUSTOM_AR_FALLBACK_LOG=1`
  (rank0,每 (op,shape) 只打一行):捕获 nccl 路径的调用点/消息字节数/
  是否 capture 中。1 个 profile 窗口即可裁决 H1/H2。
- **修法(按位点)**:
  - H1 → 两条:(a) 把这些 AR 调用移出编译区(piecewise 分裂点已含
    `vllm.all_reduce`,核对为何该点未分裂——多半是 fused/concat 后的隐式
    AR);(b) 或让 DCU push-AR capture-safe(核对 finegrained flag 在
    capture 下的合法性;若 hipMemcpy/主机同步被卷入则放弃 (b))。
  - H2 → 单独评估该消息是否走 push(two-shot)或拆分。
- **验收门**:trace 中 `ncclDevKernel_AllReduce_Ring` decode 步内计数 = 0;
  e2e step p50 **−0.5 ms**(保守,3.3×(481−63)µs 上限 1.4 ms 的 1/3 起门);
  贪心流 parity 同 A1 门。

### 2.3 风险与回滚

- capture 边界改动会影响编译缓存与捕获形状集(现有 [1,2,4]);改后必须
  全形状 smoke + 冷启动一轮(首编译时间回归记录)。
- 回滚:修复以 env/常量默认现役路径为基线,patch 走 `.s38/*.diff` 台账,
  可单文件还原。

---

## 3. A3:logits / AllGather 路径定位与优化(0.5d)

### 3.1 现状(两个未解)

- decode 每步 1 次 **365 µs AllGather**(python 层,疑似 logits 交换);
  但**看不到独立的 logits GEMV 大核**(无 >150 µs 调用、无 Cijk/aten::mm)
  ——lm_head 的 M=1 计算路径不明,字节账有 ~0.6 GB/步 变数。
- 若 AllGather 传的是 [1, vocab/4] fp16(124 MB),365 µs = 340 GB/s,对
  124 KB 量级的 M=1 logits 明显偏大;若真是 124 MB,则 lm_head 计算本体
  0.6 GB/步 也应可见——两者矛盾,必须定位。

### 3.2 设计

- **定位(0.5h)**:读 `compute_logits`(logits_processor + ParallelLMHead
  的 M=1 分支)+ shard 头确认 lm_head dtype/形状/切法;一次
  `record_shapes=True` 的离线单层复现(或 trace 里对 AllGather 的 args.size)。
- **优化(视发现,kill-fast)**:候选:小消息 custom one-shot allgather
  (对齐 push-AR 量级 ≤50 µs,−0.3 ms);或 logits 尾块与 sampler 重叠;
  或确认 logits 走了某条已有快路径后仅闭账不改码。
- **门**:AllGather ≤100 µs(trace);e2e −0.2 ms;不改码则仅更新字节账。

---

## 4. Track B:FP8 GEMV 小 K 特化(1–2d)

### 4.1 现状

- `fp8_emu_gemv_stream_m1`:2-slice 双缓冲预取 + 寄存器累加 + V6 shuffle
  归约(结构已对,LDG 先行已达成);加权 561 GB/s(读上限 71%)。
- **短板 = K=1536 的 attn_o/gdn_out(309 GB/s,64 次/步共 1.64 ms)**:
  K=1536 = 1.5 个 kStreamSlice(1024),双缓冲刚起飞就收尾,跑道不足 +
  半 slice 尾巴。gate_up/down(73–78%)为二阶。
- 约束(S37 A3 教训):**不得增加核数**;split-K 两核结构已判死。

### 4.2 三案竞速(全部单核内,kill-fast 顺序)

| 案 | 改动 | 机理 | 成本 | 风险 |
|---|---|---|---|---|
| **B1-a RW=2 双行** | 派发层:K≤2048 且 N≥4096 的形状用 rows=8/nt=256(RW=2);kernel 零改(模板已有) | 每 warp 2 条独立累加链,同 slice 的 LDG 在飞数翻倍,块数减半(1280→640,仍 10 波) | ~0.5h(含微测) | 行映射变化需核 `row < N` 边界;数值期望位同(每行累加序不变) |
| **B1-b slice512** | `kStreamSlice` 模板化,小 K 形状 512 | 3 个整 slice,预取充分展开;半 slice 消失 | ~1d;累加分组变 → 非位同 | syncthreads ×3/slice-set;kb 折叠分组变化 → max\|Δ\| 门 3e-4 |
| **B1-c 尾 slice lane 拆分** | 主 peel 循环不变,残段(K%1024)在 block 内跨 lane 对半拆 | 消灭半空转 lane | ~1d,改动最深 | peel 宏重写,位同性需逐案证明 |

- **微测门(同 harness `gemv_bw_bench.py`)**:o/out 中位 **−20%** vs 基线
  harness,且 gate_up/down/qkv 回归 ≤2%;数值:o/out `torch.equal`(a 案)
  或 max|Δ| ≤3e-4(b/c 案);门不过 → 下一案,全不过 → 关档留档。
- **trace 复测门**:o/out ≥ **450 GB/s**(现 309);e2e 门 **−0.3 ms**。
- **部署**:形状门自动生效(K<2048),env `VLLM_DCU_FP8_GEMV_SMALLK=0` 一键
  回现役路径;`.so` md5 台账照旧。

### 4.3 与 MTP 的口径披露(重申)

B1 收益只在 M=1 路径(no-MTP 臂 decode 与 MTP 的 draft 单步)。MTP verify
M=3 不经过该内核;若操作者切回 MTP,B2 的 e2e 门改在 draft/无 MTP 场景评估,
不入 MTP 主口径。

---

## 5. 预期终态与验收总表

| WP | e2e 门(无 MTP 臂,step p50) | trace 门 | 数值门 |
|---|---|---|---|
| A1 W4 | −0.8 ms | 融合核出现、norm-in-graph 数下降、host gap 不涨 | Δresidual=0,Δout ≤5e-04 |
| A2 回退归位 | −0.5 ms | `AllReduce_Ring` decode 计数 = 0 | 贪心 parity(S29 界) |
| A3 logits | −0.2 ms(或不改码闭账) | AllGather ≤100 µs | — |
| B2 小 K | −0.3 ms | o/out ≥450 GB/s | 位同(a) / ≤3e-4(b,c) |
| **合计** | **−1.8 ms(门)/ −2.6~3.7 ms(上界)** | | step 26.7 → ~23–24 ms,贪心 ~40–43 t/s(+8~16%) |

全部 WP 各自独立回滚(env 或单文件 diff);任何一门不过 → 该 WP 关档留证,
不影响其余 WP。完成判据 = 本表逐行落档 `docs/result/z100-s38-final.md`。
