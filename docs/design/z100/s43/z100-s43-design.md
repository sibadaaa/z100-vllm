# S43 设计图:INT4 w4a16 GEMV(质量先行 + 几何探针 + wholea 骨架换载荷)

> 日期:2026-10-31 深夜。输入:S41 终局(`docs/result/z100-s41-final.md`,
> wholea 采纳)、S42 双关档(`.s42/b0-verdict.md`:a-reg −21%/ldcs 0%,
> fp8 内核结构空间穷尽)、INT4 预研(`docs/design/z100/s42/
> int4-w4a16-prework.md`)。**本轮问题:权重位宽从 fp8(1.03 B/元素)
> 换 INT4(0.53 B/元素)能否在质量门内把 GEMV 桶 11.15 → ~6.5 ms/step,
> 单流 39 → 46-52 t/s。**前置:GEMV 是 2 FLOP/B 的带宽主导算子,字节
> 减半是唯一剩下的一阶杠杆(S33-S42 已把内核结构空间全部证伪)。

---

## 0. 基线快照(2026-10-31,S41 采纳态,一切判据的对照面)

| 项 | 值 |
|---|---|
| 在役臂 | 无 MTP(`SPEC_OFF=1`)+ S40 段(`VLLM_DCU_FP16_GEMV_STREAM=1`)+ S41 段(`VLLM_DCU_FP8_GEMV_WA=1`,wholea);`.so` md5 `f72bb39d`(S41/S42 零代码改动) |
| 服务级(S41 当日 ABAB) | W1 贪心 38.7-40.0 t/s(B 臂带);TTFT 0.706-0.75 s |
| decode 步构成(S41 trace 重切口径) | **fp8 GEMV 11.15 ms/step**(256 发,单发中位 40.8 µs)、push-AR ~3-4(账本 3.76)、attn 1.37、gdn 0.97、bf16 小核 0.58、lm_head 0.81、norm/碎核 ~2.6;GPU busy ~20.7 ms/step |
| GEMV 形状份额 | K≥4096 类 **10.2 ms/step**(qkv 3584×5120、gdn_in 2560×5120、gate_up 8704×5120、z 1536×5120、down 5120×4352);**o/out(5120×1536)0.95 ms/step** |
| fp8 物理口径(探针,rows4/nt256) | wholea 骨架 **643 GB/s**;裸形 K-E 782;纯读锚 792 |
| S42 已证伪(不再重试) | a-reg(K=5120 −21%,64 VGPR→占用塌;K=1536 +5.7% 不够门)、ldcs(0%)、kPf=2 短 K(+2.5%,不够门)、叠加无效 |
| o/out 账 | 75% lane 利用率结构上限(96 chunk/64 lane),全部几何/占用路已证伪 → **本轮 o/out 不接管,留 fp8**(预研 §1 的份额账已按此计) |
| INT4 数值事实 | 内核数值管线(fp16 宽展→fp32 累加→per-128 scale 折算)与 fp8 同构不劣化;新增误差只有权重表示一项:GPTQ g128 模型级 PPL 典型 +1~5%,27B 预期 +2~4% |
| 测量纪律 | 窗口口径 + K-A 锚点 ±3% + 物理 GB/s(0.53 B/元素)口径;**门全部换轨:torch.equal/3e-4/S29 板不适用(换权重),改 RTN 基线对照 + PPL + 任务小集**;profile 楔死预案照抄 S41(判读用单发中位/内核总账) |

**排序与结构**

```
Track Q(质量先行,零内核成本,kill-fast 最优先)
  Q0  GPTQ g128 对称量化 checkpoint(校准集=生产 prompt 采样)      0.5d
  Q1  PPL 评估(HF 直跑,不动 vLLM)+ RTN 基线对照                 0.5d
  Q2  任务小集 + 贪心流语义对照                                   0.5h
  门:PPL ≤ fp8 基线 ×1.03(用户裁决点);GPTQ 显著优于 RTN
Track P(几何探针,0.5d)
  P0  widen_tax mode 20/21:wholea 骨架 + INT4 载荷(两种 pack 变体)0.5d
  门:gate_up 形物理 GB/s ≥ 550(= 现役 643 × 0.855)
Track K(内核化,1.5-2d,视 P0)
  K0  w4a16 内核(wholea 骨架换载荷,slice-2048 tile)            1-1.5d
  K1  loader/packer(HF GPTQ pack → 布局预 pack)+ 形状门         1d
  K2  数值门:内核 vs fp64 参考 ≤1e-3(同量化权重,验算术)        0.5h
  K3  窗口门:≥550;fp8 形状(o/out)零回归                        0.5h
Track DEP(部署 ABAB,0.5d)
  checkpoint 级 A/B(非 env 翻转:模型文件不同)→ trace 桶门
  → W1 门 → 质量门复验
不施工(记录):o/out INT4(0.75 tile 几何,利用率 37.5%,留 fp8)、
a-reg 复活(寄存器预算已证死)、actorder(与折算结构冲突)。
```

排序理由:**质量门是唯一不可工程化的风险**,且能零内核成本先测——Q 轨
kill-fast 排最前;P 轨验证几何利用率(K=5120 在 INT4 tile-2048 下只有
83% lane 利用率 vs fp8 的 100%,是探针要量化的主要未知数);K/DEP 沿用
S41 全流程纪律。

---

## 1. Track Q:质量先行(0.5-1d,零内核成本,kill-fast)

### 1.1 机理与预期

- 换权重的误差全部来自量化表示:fp8 e4m3 权重 RMS 相对误差 ~2-3%,
  INT4 g128 对称 RTN ~4-8%(离群值主导),GPTQ 用 Hessian 逆把误差推到
  不敏感方向——量级不变、方向变好,模型级 PPL 增幅典型 +1~5%,27B 预期
  +2~4%,卡在 ±3% 门附近,**必须实测**。
- 点积输出相对误差 ≈ 权重 RMS 相对误差(与 K 无关);fp16 激活、fp16
  KV、fp16 lm_head、fp32 累加全部不动——**误差是一次性的、静态的**。
- 贪心流与 fp8 基线必然分叉(S41 观察到的 0.0107 nat 近平手扰动会放大
  为可见级)——S29 板(0.2255 nat/0 错位)对此类变更**设计上就不适用**,
  换 Q2 的语义对照。

### 1.2 Q0:量化 checkpoint(0.5d)

- 方案:GPTQ,group=128,**对称 zp=0**,actorder 不用;int32 沿 K 打包
  (8 nibble/dword),`q_s` fp32 [N, K/128](与现役 w_s 同构)。
- 范围:只量化 K≥4096 类大权重(qkv/gdn_in/gate_up/z/down);
  **o/out 留 fp8**(几何见 §2);GDN 小参数(A_log/dt_bias/conv)、
  embed/norm、lm_head 全部不动。
- 校准集:生产 prompt 分布采样 256 条 × 2048 token(runs/specbench 的
  prompt_ids 池),不用公开数据集(域要对齐)。
- 工具:auto-gptq 或 llm-compressor;输出独立目录
  `/mnt/game/models/qwen3.8-27b-gptq-int4/`(fp8 基线目录只读不动)。
- 同时产出 **RTN g128 checkpoint**(同校准集,仅作 Q1 对照与 K2 的
  内核数值参考,不上服务)。

### 1.3 Q1:质量评估(0.5d)

- PPL:wikitext-2(en)+ 自定义域集(生产 prompt 拼接),HF
  transformers 直跑(量化模型走 HF int4 kernel 路径即可,与我们的内核
  无关——**测的是权重的质量,不是内核**);stride 512 完整评估。
- 对照三臂:fp8 基线 / GPTQ g128 / RTN g128。
- **门:PPL(GPTQ) ≤ PPL(fp8) × 1.03**;GPTQ 显著优于 RTN(否则是
  量化配置错,先修配置再谈)。

### 1.4 Q2:任务小集 + 语义对照(0.5h)

- runs/specbench probe 集 + 5 个固定任务,贪心 96-512 token,人工 +
  规则双查(无乱码/无复读/长度分布正常);与 fp8 基线输出并排留档。
- Q1 或 Q2 不过 → **整轨关档**(总投入 1d,零内核成本);缓解阶梯
  (非对称 zp / 敏感层混合 fp8 / group 64)只有在"差一点"时经用户
  裁决才追加,默认不追。

---

## 2. Track P:几何探针(0.5d,kill-fast)

### 2.1 机理与两个 pack 变体

- lane 的 16 B 载荷 = 4 dword = 32 nibble = **32 个 K 元素**(fp8 是 16);
  tile(每行每 slice)= 64 lane × 32 = 2048 元素 = 1 KB/行。
- **利用率问题(本轮最大未知数)**:K=5120 = 2.5 tile → 尾半 tile 只有
  lanes 0-31 有活,**83% 利用率**(fp8 同形状是 5.0 chunk/lane 整 → 100%);
  K=4352 = 2.125 → 85%。两个 pack 变体正面对比:
  - **V1 tile-2048**(mode 20):直 pack [N, K/8] int32,lane 每 tile 1×
    uint4(32 元素),K=5120 利用率 83%;
  - **V2 tile-1024 + dword 载荷**(mode 21):pack 重排为 1 KB tile
    (1024 元素 = 64 lane × 16 元素),lane 每 tile 1×dword(8 B),相邻
    lane 的 dword 读由 wavefront 合并成 64 B 段(访存效率实测验证);
    K=5120 = 5 整 tile → **100% 利用率**,代价是每 lane 每元素组减半的
    向量宽。
- LUT:16 entry × fp16 = 32 B = 8 寄存器,**寄存器驻留**(S33 的 LDS
  4-way 冲突一章以"消灭查表"收尾);备选无查表位技巧(nibble 符号扩展
  + cvt,整数 ≤7 在 fp16 精确)。
- A 路径:**维持 wholea LDS 整行 staging**(a-reg 已证死,寄存器预算
  不碰);A 读 bank 冲突(64 B lane 步距,4-way)与现役同,接受。
- 折算:lane 的 32 元素块不跨 128 组边界(l*32 % 128 ∈ {0,32,64,96}),
  每 lane 每 tile 一次 scale 乘,与 fp8 折算同构。

### 2.2 P0:探针(0.5d)

- `cp .s42/widen_tax.cu .s43/`,加 mode 20(V1)/mode 21(V2);
  权重数据:离线脚本对现 fp8 权重做 group-128 对称 RTN 生成两套 pack
  bin(V1/V2 布局各一),LUT 沿用 16-entry int4 表。
- 口径:**物理 GB/s(0.53 B/元素)**;等效元素吞吐 = 物理 ×1.94 才与
  fp8 账可比。
- **门:gate_up 形物理 GB/s ≥ 550**(= 现役 fp8 骨架 643 × 0.855,
  即元素吞吐 ≥ 现役 ×1.65)。两变体取优;全不过 → 整轨关档。
- 对照锚:同轮跑 mode 15(wholea fp8)防设备漂移(±3%)。

---

## 3. Track K:内核化(1.5-2d,视 P0)

### 3.1 K0:内核(1-1.5d)

- `csrc/rocm/fp8_emu_gemm.cu`:新增 `fp8_emu_gemv_wholea_m1_kernel` 的
  INT4 载荷兄弟模板(P4 宽展 + slice-2048 walk + LUT-reg + per-32 折算),
  lane→tile 映射与 per-lane k 序按 P0 赢家形态;**A/scale/LUT staging、
  折算乘加序、归约结构逐行对齐 wholea**(内核算术确定性按构造)。
- 路由:`M == 1 && K >= 4096 && int4_enabled()` 且该层权重为 INT4 打包
  时接管;o/out 与 lm_head 不接管。
- env:`VLLM_DCU_INT4_GEMV=1`(默认 0)+ 权重层 dtype 由 loader 标记。
- A0 预检:编译无 spill + VGPR ≤ wholea+8(S42 教训:寄存器预算即
  占用,点亮前先看 ISA)+ 微测不降速。

### 3.2 K1:loader/packer(1d)

- 离线 packer:HF GPTQ checkpoint → 我们布局(V1 或 V2)+ `q_s` fp32;
  产物随 checkpoint 目录走,loader 按 dtype 标记分发。
- 加载路径:launch 指向 INT4 checkpoint 目录;fp8 层(o/out/lm_head/
  GDN 小参数)从基线目录原样加载——**checkpoint 目录 = 量化层 INT4 +
  保留层 fp8 的混合目录**,由 packer 一次性生成,避免运行时双源加载。

### 3.3 K2/K3:门(0.5h + 0.5h)

- **K2 数值门(分层,与质量门正交)**:内核(量化权重)vs fp64 逐元素
  参考(同一量化权重)max 相对误差 ≤ 1e-3——验的是**内核算术**,权重
  误差已在 Q1 验过。两套权重(GPTQ/RTN)各跑一次,7 形状(5 量化形状
  实测 + 2 fp8 形状证不接管)。
- **K3 窗口门**:gate_up/qkv 物理 GB/s ≥ 550(探针同口径);fp8 形状
  (o/out)与 lm_head 窗口零回归(同 .so 同 env 复测)。

---

## 4. Track DEP:部署 ABAB(0.5d,checkpoint 级)

- **与 S41 的关键差异**:AB 两臂是不同模型文件(env 翻转不够)——
  A 臂 = fp8 checkpoint(现役),B 臂 = INT4 混合目录;每次切换 = 改
  launch 的模型路径 + restart(~270 s,与现役相同)。B/A/B/A 四臂。
- 每臂采:smoke、e2e(W1 96×3 + W2 512×2)、trace 桶(预案:楔死则
  restart,判读用单发中位/内核总账)、Q2 任务小集复跑。
- **门**:trace GEMV 桶 ≤ **6.5 ms/step**(理论 5.8-6.2 + 折损余量);
  W1 ≥ **+15%**(桶 −46% 的服务折算下限,配对口径);Q2 语义门维持;
  TTFT 无回归(量化层加载更快的 offset 可接受)。
- 回滚:launch 模型路径改回 fp8 目录 + restart(现役逐位恢复);`.so`
  内 INT4 路径 env 默认 0,不触 fp8 路径。

## 5. 预期终态与验收总表

| WP | 门 | 判据 |
|---|---|---|
| Q 质量 | PPL ≤ ×1.03;GPTQ<RTN;任务小集不塌 | HF 直跑,零内核成本 |
| P 探针 | 物理 GB/s ≥ 550(gate_up 形) | 两 pack 变体取优 |
| K 内核 | 算术 ≤1e-3(vs fp64);窗口 ≥550;fp8 形状零回归 | 分层数值观 |
| DEP | trace 桶 ≤6.5 ms/step;W1 ≥+15%;语义门维持 | checkpoint 级 ABAB |

**合计现实预期**:GEMV 桶 11.15 → **6.0-6.8 ms/step**(o/out 0.95 留
fp8);W1 39 → **46-50 t/s**;TTFT 持平或略优(权重读取减半惠及
prefill)。完成判据 = 逐行落档 `docs/result/z100-s43-final.md`。

## 6. 风险

- **质量门不过(最高,且不可工程化)**:Q 轨 1 天 kill-fast 先行;
  缓解阶梯(非对称/混合层/group 64)默认不追,差一点时经用户裁决。
- **INT4 几何利用率**:K=5120 在 tile-2048 下 83%(vs fp8 100%)——
  P0 双变体正面量;若 V1<550 而 V2≥550,packer 按 V2 布局(每层
  重排一次,离线成本)。
- **GPTQ 对 GDN 混合架构的适配**:conv/decay/A_log 不量化已定;gdn_in
  的 gate/z 形状若质量敏感,Q0 时逐层留 fp8 开关(packer 参数)。
- **DTK codegen**:nibble 宽展的 cvt/LUT-reg 路径需 A0 预检(S42 的
  NT-builtin 类型限制经验:builtin 只收原生向量)。
- **profile 楔死**:S41 两臂 100% 复现——trace 判读一律用单发中位 +
  内核总账口径,桶中位数仅在步界完整时可用。
