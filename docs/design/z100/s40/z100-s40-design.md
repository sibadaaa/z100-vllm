# S40 设计图:GEMV 读上限逼近(pf[4] 预取环 + fp16 lm_head stream 化 + A-slice 双缓冲)

> 日期:2026-10-31。输入:S39 终局(`docs/result/z100-s39-final.md`,三条主打
> 路径证伪、账本修正)、**ceiling-probe**(`.s39/ceiling-probe.md`,
> 读带宽阶梯 905/828/783/723/615 实测)、S39 收档台账(`.s39/*-close.md`)。
> **与 S39 的关系**:S39 证伪了"改通信、消 LUT、位技巧"三条路,并给出新账本
> ——decode 步 24.15 ms(GPU busy)中 GEMV 12.30、**fp16 lm_head 0.95
> (417 GB/s,第二大单核)**;ceiling-probe 证明 GEMV 形状纯读可达
> 783(轮转)/828(单块顺序),生产 615 的缺口主要在**预取深度、lm_head
> 旧内核、A-slice 同步结构**,三者在内核结构层、数学不变。

---

## 0. 基线快照(2026-10-31,全部判据的对照面)

| 项 | 值 |
|---|---|
| 在役臂 | 无 MTP(`SPEC_OFF=1`,systemd `z100-vllm-mtp2`),FP `2f3793deb97e138f` |
| `.so` | `_rocm_C.abi3.so` md5 `ca05f746`;`_C_stable_libtorch.abi3.so` md5 `54ccf2e2`;备份 `.r3/so-backup/` |
| 服务级(当日带,漂移大) | W1 贪心 35.4–39.1 t/s(step ≈ 25.6–27.3 ms);TTFT 0.71–0.78 s |
| decode 步构成(新窗 1793405011,38 步,GPU busy 24.15 ms) | **GEMV 12.30**、**fp16 lm_head 0.95(947 µs/发,417 GB/s)**、push-AR 3.76(自旋)、attn 1.42、norm/碎核 ~2.6、GDN 1.00、bf16 小核 0.55、AG 0.19、**nccl AR 0** |
| 读带宽阶梯(ceiling-probe,窗口口径) | 纯读墙 905 = 88% pin;单块 1GiB 顺序+deep4 **828**;轮转+deep4 **783**;+LUT **723**;生产 trace **615** |
| 预取环现状 | m1/stream 内核 `pf[2][RW]`(fp8_emu_gemm.cu:721/945),生产几何恒 RW=1 → 每 lane 2 个 uint4 在飞;probe 实测 1→4 深 = +20%(裸形窗口口径) |
| lm_head 现役路径 | `bf16_emu_gemv_dot2_kernel`(bf16_emu_gemv.cu:79):**每 block 一行、256B 请求粒度、无预取环、A 行每 block 从全局重读** → 38484 block × 1 发/step @417 GB/s |
| barrier 现状 | m1 每 slice 2 次 `__syncthreads()`(A 暂存写→读、读→下次写);slice = 2 KB fp16/行 |
| 测量纪律(新增,S39 教训) | 微测一律**窗口口径**(30 rep 单事件对;每 rep 事件对短核压 34%)+ 服务 trace 双口径对账 |
| 测量工具 | `.s39/widen_tax.cu`(9 模式,含 deep4/window/1GiB-seq)、`runs/s34nospec/{profile_window,parse_trace,gemv_bw_bench}.py`、`.s39/gemv_bw_bench.py`、`runs/specbench/e2e_ts.py`、`.s38/parity_probe.py`(S29 板) |

**排序与结构**

```
Track A(pf[4] 预取环,_rocm_C;probe 已证方向,最大桶)
  A0  寄存器/vgpr 预检(RW=1 实例 pf[2]→pf[4] +8 regs)      0.5h kill-fast
  A1  m1 内核 peel-by-4 改造 + env 门                        0.5d
  A2  微测(窗口)+ 位同(torch.equal,消费序不变)            0.5h
  A3  MS>1 内核跟随(仅 RW≤2 实例;RW≥4 保持 pf[2])          视 A2
Track B(fp16 lm_head stream GEMV,_rocm_C;第二大单核,最孤立)
  B0  bf16_emu_gemv.cu 读账 + 数值口径裁决(序不同→非位同)    0.5h
  B1  新内核 fp16_gemv_stream_m1(~150 行,免 LUT/免 scale)   1d
  B2  微测窗口 ≥550 GB/s + python 路由(env 门)+ logprob 板   0.5h
Track C(A-slice 双缓冲,barrier 2→1/slice;time-box 0.5d)
  C1  微设计 + kill-fast 微测(先探针后施工)                 0.5d
Track D(测量基建,0.5h,先行)
  D0  gemv_bw_bench.py 窗口口径模式 + lm_head 形状 harness
不施工(记录,S39 已证伪):LDS 消除/位技巧/nccl 归位/AG one-shot/
eager 采纳/slice512/合并权重分配(轮转 TLB −6.9%,工程风险大)。
```

排序理由:**确定性优先**——D0 是 A/B 的一切判据(先做,0.5h);A 方向已被
probe 裸形实验证实(783 vs 652),改动局部(peel-by-4)、消费序不变可位同;
B 是第二大单核且完全孤立(新内核、新 op、env 门,零现役路径风险),收益
下限最硬(417→550 即 −0.22 ms);C 是 probe 数据指向的下一个结构税,但
S38/S39 教训 = 探针先行、time-box、不过即关。

---

## 1. Track A:pf[4] 预取环(目标:GEMV 桶 −0.7~1.0 ms)

### 1.1 现状与机理(全部代码事实)

- m1 内核(fp8_emu_gemm.cu:885)`pf[2][RW]` 双缓冲:第 s+2 slice 的 W 块
  在第 s slice 计算时发射,每 lane 每 slice 一个 uint4(16B),64 lane =
  1 KiB/请求。**2 深意味着 DRAM 延迟(~600 ns)只能被 2 KiB/行在飞量吸收**;
  probe 的 deep4(4 条独立流,步长 64 chunk)把裸形窗口从 652 → 783 GB/s。
- 生产几何恒 RW=1(rows=4/nt=256):`pf[2][1]` = 8 寄存器 → `pf[4][1]` =
  16 寄存器,**+8 regs,无 spill 风险**(A0 预检确认)。
- MS>1 内核(:721 同构):仅 RW≤2 实例跟随;RW≥4(M≥5 或 SMALLK/rows_ov
  特例)寄存器翻倍,保持 pf[2]。

### 1.2 A0:预检(0.5h,kill-fast)

`hipcc -W` + 编译后 vgpr 统计(m1 内核 RW=1 实例):`pf[4]` 后 VGPR 不超
现役 occupancy 档(40 warp/CU 需 ≤64 VGPR/线程上限内)。超档 → A 轨降级为
"m1 专用 pf[4]",MS 内核不动(A3 关档)。

### 1.3 A1:peel-by-4 改造(0.5d)

- 位置:fp8_emu_gemm.cu m1 内核(:945 `pf[2][RW]` 声明、:948-958 预填、
  :960 `VLLM_FP8_STREAM_M1_SLICE` 宏、:1014-1016 peel 循环)。
- 结构:4 buffer 旋转,peel-by-4(4 个字面量 buffer 下标),尾部
  (<4 slice)回退现序;**slice 消费序不变** → per-lane k 序、dot2 链、
  折算乘加全部不动。
- env 门:`VLLM_DCU_FP8_GEMV_PF4=1`(默认 0 = 现役 pf[2],一次编译 A/B)。
- MS 内核(:721)同构改造,**仅 RW≤2 实例启用**,A3 跟随。

### 1.4 A2:门(0.5h)

- 微测(窗口口径,D0 harness):gate_up、o_out **≥ +6%**(对照 pf2 臂),
  其余形状 ≤2% 回归。
- 位同:七形状同输入 `VLLM_DCU_FP8_GEMV_PF4=0/1` **`torch.equal` 全绿**
  (消费序不变,按构造成立;失败 = 实现 bug)。
- 不过门 → 关档留证(回滚 = env 归零)。

---

## 2. Track B:fp16 lm_head stream GEMV(目标:−0.25~0.35 ms/step,下限最硬)

### 2.1 现状(bf16_emu_gemv.cu:79,代码事实)

- 每 block 负责一行(launch 38484 blocks),线程 t 以 nt=256 步长串行吃
  dwords:**256 B/请求**(fp8 stream 的 1/4)、无预取环、A 行每 block 从
  全局重读(38484 × 5 KB,靠 L2 吸收)、无 LDS 复用。
- 实测 947 µs/发 = 417 GB/s(394 MB = [38484,5120] fp16 分片),decode
  第二大单核。fp8 系内核同形工作点 600–780 GB/s → 差距纯结构。

### 2.2 B0:数值口径裁决(0.5h)

- 读 bf16_emu_gemv.cu 确认累加序:现役 = 线程内 i=t+j·nt 串行 dot2 +
  shared-mem 树归约;stream 新序 = lane 内 chunk 串行 + shfl 树。**两序
  不同 → 非位同**,数值门走 `max|Δ| ≤ 3e-4`(对现役输出)+ **S29 logprob
  板 ≤ 0.2255 nat** + 贪心流差异按 S34 先例只记录不阻塞。
- 若想保 torch.equal:必须复刻 t+j·nt 序(放弃 slice 几何)= 放弃收益,
  **不做**(S29 板就是为这类重排设立的)。

### 2.3 B1:新内核 fp16_gemv_stream_m1(1d,~150 行)

- 位置:`csrc/rocm/fp8_emu_gemm.cu`(与 m1 内核同文件,复用
  kStreamSlice/kStreamChunk 常量与 launch 骨架)。
- 结构 = m1 内核删减:W 以 uint4 读 **8 个 fp16 元素**(slice = 2048 B =
  128 uint4,lane 每 slice 2 个 uint4,保持 1 KiB/请求粒度);**无 LUT、
  无 sl 暂存、无 a_s/w_s 折算**;A 行 fp16 直入 LDS(2048 B/slice);
  pf[4] 预取(直接按 A 轨成品形);epilogue = shfl 树 + bias。
- 绑定/路由:`torch.ops._rocm_C.fp16_gemv_stream_m1` +
  `vllm/model_executor/layers/utils.py` 的 GEMV 选择器加
  `VLLM_DCU_FP16_GEMV_STREAM=1`(默认 0)且 `M==1 && N>=2048 && K%128==0`
  才路由;其余一切形状走现役 bf16 路径,零风险。
- shape 特化:lm_head 分片 [38484,5120] 编译期常量化 N 不必要——内核按
  运行时 N/K 走,probe 显示几何收益与 N 无关(行内 K 走法决定)。

### 2.4 B2:门(0.5h)

- 微测(窗口):lm_head 形状 **≥550 GB/s**(+32%,即 −0.22 ms/step 下限);
  stretch 650。
- 数值:`max|Δ| ≤ 3e-4` + S29 板 ≤0.2255 nat;W1/W2 服务 smoke 正常出字。
- 不过门 → 关档留证(env 归零 = 现役)。

---

## 3. Track C:A-slice 双缓冲(time-box 0.5d,barrier 2→1)

- 现状:m1 每 slice 两次 barrier(A 写→MAC 读、MAC 读→下一轮 A 写)。
- 改法:LDS 内 A-slice ×2 旋转(M=1 +2 KB,MS=8 +16 KB,44 KB 门内),
  暂存 s+1 与计算 s 重叠 → 每 slice 一次 barrier。
- **探针先行**:widen_tax 加"barrier 减半"模式(纯读形,排除其他变量),
  预期 +3~5%;<+3% 直接关档(不进内核)。
- 门:窗口微测 ≥+3%;位同(序不变,`torch.equal`);并入部署合并臂。

---

## 4. Track D:测量基建(0.5h,先行)

- `gemv_bw_bench.py` 增 `--window` 模式(30 rep 单事件对;消除 launch gap
  偏置——S39 实测短核被压 34%)+ lm_head 形状(38484×5120,copies=4 ×
  394 MB = 1.6 GB,轮转同样 defeat L2)。
- trace 分类脚本补两桶:lm_head kernel 名、(若 pf4 采纳)GEMV 桶速率。
- 一切 A/B/C 判据以窗口口径为准,服务 trace 为第二口径。

---

## 5. 预期终态与验收总表

| WP | 微测门(窗口) | trace 门 | e2e 门 | 数值门 |
|---|---|---|---|---|
| A pf[4] | gate_up/o_out ≥+6%,余 ≤2% | GEMV 分类 ≥650 | 合并臂 **−0.8 ms** | 七形状 `torch.equal` |
| B lm_head stream | ≥550 GB/s | lm_head 桶 ≥600 GB/s | 并入合并门 | `max|Δ|≤3e-4` + S29 板 ≤0.2255 |
| C 双缓冲 | ≥+3%(探针) | — | 并入合并门 | `torch.equal` |
| D harness | (工具,无门) | — | — | — |

合计:GEMV 12.30 → ~10.9–11.4、lm_head 0.95 → ~0.65 → **step
−1.0~1.5 ms**;W1(以 ABAB 配对口径)+4~6%;4 并发口径同步受益(M 路径
同内核)。全部 WP env 门独立回滚;任一门不过 → 该 WP 关档留证。
完成判据 = 逐行落档 `docs/result/z100-s40-final.md`。

## 6. 风险

- **pf[4] 寄存器**:A0 预检不过 → 降级 m1 专用;MS=8 实例永不启用 pf4
  (A3 明确排除 RW≥4)。
- **lm_head 非位同**:S29 板是判据,贪心流允许重排级差异(S34 先例);
  若 logprob 漂移 >0.2255 nat = 实现 bug,修到位,不开 3e-4 之外的退路。
- **窗口口径的新偏差**:免 gap 计时放大时钟漂移敏感度 → 每次微测前后
  各跑一次 K-A 参照锚点(±3% 内才算数)。
- **合并臂归因**:沿用 S39/S38 惯例——合并批先测总量,过门后逐项单开
  归因;当日带漂移 ±2 tps,ABAB 配对内才有裁决力。
- C 为 time-box:探针 <+3% 即关,不为 0.2 ms 级收益冒险动 slice 结构。
- profile 规程(S38 事故教训,强制):一发请求 + 立即 `/stop_profile`;
  异常楔死 → `sudo systemctl restart z100-vllm-mtp2`(~60 s,编译缓存热)。
