# F24 独立复核报告：Z100 上的 vmfault / TP=4 worker 死亡与静默错算

> 日期：2026-10-18 · 复核对象：`docs/z100-torch216-rocm101-adapt-plan.md` §9.2 F24、
> `docs/result/z100-rocm101-acceptance.md` §5/§6、`/mnt/game/z100-rocm101-validation/f23-evidence/F24-vendor-report.md`
> · 环境：自编 ROCm 10.1 + torch 2.16 + triton 3.6.0（容器 `z100-vllm-rocm101-rt`），
> DTK 26.04 基线（容器 `z100-regress`）。
> 证据总目录：`/mnt/game/f24-investigation/`（`00-LEAD-SUMMARY.md` 为实验索引，
> `01-hw-forensics.md` / `02-stack-diff.md` / `03-kernel-analysis.md` / `04-external-research.md`
> 为四条并行调查线，`lead-logs/` 为全部原始日志）。

## 0. 结论摘要（先看这个）

1. **根因框架已定位：Z100 卡以 DP4X 模式启动，而当前 ROCm 10.1 用户态栈完全不认识 DP4X。**
   - KFD topology（只读 sysfs）显示**唯一与"好卡/坏卡"完全对应的硬件差异**：
     Z100（`1d94:54b7`，BDF 38/3b，HIP 2/3）`dp4x_status=1`；
     Z100L（`1d94:55b7`，BDF 2e/33，HIP 0/1）`dp4x_status=0`。
     KMD 通过固件 MP0 `C2PMSG_33`（`0xDB48DB48`=DP4X / `0xDB28DB28`=DP2X）确认该模式；
     连带 `ctl_stack_size`（24576 vs 20480）与 `cwsr_size`（22372352 vs 33550336）。
   - 与本 KMD（`hycu` 6.4.2-V2.1.0，KFD uapi minor 16，配 DTK 26.10 的 HIP 7.2/ROCR 1.11）
     **配套的是 DP4X-aware 的 DTK 用户态**：DTK `libhsa-runtime64` 有 DP4X loader 路径
     （`EF_GPU_DP4X`、"The input codeObject need to be compiled without dp4x"、
     "Get vgpr granularity"、HSA attr 101），DTK HIP 暴露 `hasDp4x`，DTK clang 有 dp4x 编译选项。
   - 而我们加载的 ROCm 10.1 用户态（ROCR 1.21.0、HIP 7.16，靠
     `HSA_OVERRIDE_GFX_VERSION=9.0.6` 才认卡）**没有任何 DP4X 代码**（0 命中），
     也不修正 DP4X 下的 VGPR 粒度/CWSR 几何。
2. **可复现、可判定的最小故障判据：在 Z100 卡上，任何 VGPR 分配达到 gfx906 上限
   （256 VGPR/thread，kernel descriptor `VGRPBlocks 63`）的 kernel 会"静默错执行"；
   VGPR ≤ 203 的 kernel 全部正确。**
   - 同一份 attention HSACO（`.text/.rodata/.note` 字节相同）在 Z100 上只写出
     `72/10240` 个输出（`out[0,0]` 的 dim 0..71，确定性重复 3 次），**dmesg 零故障**；
     在 Z100L 上 10240/10240 正确；同一份数据在 DTK 栈上 4 卡全对。
   - head 16/32/64（VGPR≤204）通过；head 96/128/192/256（VGPR=256）失败；
     改变 `num_warps`/`BLOCK_M` 使 VGPR 降到 128/64 后**全部通过**（含生产 256-head
     GQA6 的 `BLOCK_M=32` 形状）。
   - 这不是 attention 特有问题：4B/27B 使用的 GDN（FLA）`chunk_fwd_kernel_o` 实测
     **VGPR=256、scratch 2840 B/thread**，同样在故障包络内。
3. **两个故障类同源**：
   - **A 类（静默错算）**：VGPR=256 kernel 的 wave 在第一条 global store 处被中止，
     只提交了前 9 个 lane 的 16B 存储（72 元素）、其余未写、值为垃圾；无 dmesg。
   - **B 类（worker 死亡/vmfault）**：同一批被中止/挂住的 wave 在其它地址形态下产生
     UTCL2 无效 PTE（XID:81）→ 队列无法 preempt（XID:76，9 s 超时）→ KMD kill 进程
     → 进程回收阶段 vmfault 风暴。多 rank 时一个 rank 死亡即表现为 NCCL split 失败
     / worker died —— 与 F24 原始 TP=4 现象一致。今天在 Z100 卡上重新复现了该签名。
4. **已拿到最小独立复现器（无需 vLLM/模型，厂商可直接用）**：
   `/mnt/game/f24-investigation/minimal_vgpr_repro.py` —— 单 CTA、2 warps 的普通
   elementwise Triton kernel（编译后 VGPR=256、15 spills）：Z100 卡写出
   `72/16384`，Z100L 卡写出 `16384/16384`；同一 binary 4 卡矩阵（card0/1 PASS、
   card2/3 FAIL）、0 dmesg。把 `num_warps` 提到 4/8（VGPR 144/80）后 Z100 全部通过。
   这消除了“需要 vLLM/attention 才能复现”的疑问，也直接证明问题是 VGPR 分配几何
   而非任何具体 kernel 语义。
5. **已拿到并通过端到端验证的 workaround（仅绕过，不修根因）**：
   把 attention 的 Triton kernel 保持在 VGPR 上限以下——2D prefill 与 3D decode
   都强制 `num_warps=8`：
   - Qwen3-0.6B 单卡 Z100：由垃圾输出 → `' Paris. The capital of Italy is Rome'`（与 Z100L 一致）；
   - **Qwen3-0.6B TP=4（4 卡，含 2 张 Z100）：READY 36 s，输出完全正确，dmesg 零故障**
     （未打补丁时同一 TP=4 在请求阶段出现 sq_intr + vmfault、请求失败）。
   - 局限：GDN/27B 的其它 Triton kernel 仍是 VGPR=256，尚未处理（4B GDN 模型在 Z100
     上仍输出垃圾；把 attention 单独修好后反而在 init 阶段触发了 B 类 hang，说明
     GDN kernel 必须先做同类处理），因此**27B-FP8 TP=4 生产形态仍不可用**。
6. **对既有 F24 结论的修订**（详见 §7）：
   - 保留：Z100 卡在新栈上异常、TP=4 被阻塞、厂商报告方向正确。
   - 修正：①"vmfault 导致 NaN/部分写出"不成立（静默错算 0 条 dmesg）；②故障 VA 不是
     host 指针（是进程 GPU-VA，large-BAR 映射，PTE 无效）；③"attention 配置矩阵全失败"
     的证据无效——`f24-cfg-w4/w8/s2` 臂实测编译出的 kernel 仍是 `num_warps=2`
     （overrides 未生效），矩阵缺少唯一起作用的维度（把 VGPR 降到 ≤203）；④"simple
     kernel 全对"不能证明 SKU 健康（全部是 1 CTA、低 VGPR kernel）；⑤不是"Z100 硬件坏"
     ——同一份代码字节在 DTK 栈上 4 卡全对，且 workaround 下同一张 Z100 卡完全正确。
7. **建议的下一步**（详见 §8）：给厂商的最小复现 + 6 条问题；维护窗口做
   `hycu.dp4x_enable=0`（或 DP2X VBIOS）冷启动实验（决定性）；把 workaround 扩展到
   FLA/GDN kernel 并评估性能；完成 27B TP=4 复测。
8. **当日后续已执行（见 §11，2026-10-18 下午）**：
   - **`hycu.dp4x_enable` 被证明是死参数**：live `.ko` 中它只有 `.rela__param`
     注册重定位、无任何代码引用；DP4X 是 MP0/VBIOS 决定的硬件模式，KMD 只读。
     **不需要、也不应该为它重启机器。**
   - **运行时清 `SPI_CONFIG_CNTL_2.DP4X_EN` 实验**（card2，可逆）已做：清零后 repro
     反而出现更不一致的写坏（连 torch reduction 都被污染），恢复后回到 baseline
     签名；说明"仅翻 bit、不重跑 GPU init"拿不到干净 DP2X 状态，DP4X 关闭需要
     KMD/BIOS 级路径（厂商项）。
   - **guard 已工程化落地并验证**：`vllm/platforms/hygon_dp4x.py`（DP4X 检测 +
     全局 Triton 发射守卫 `num_warps>=8` + `VLLM_GFX906_DP4X_WATCH` 自检）+
     `tools/dcu/dp4x_guard_selftest.py`。端到端结果：
     - **27B-FP8 TP=2（两张 Z100）**：ready **96 s**、输出正确、0 dmesg；
     - **27B-FP8 TP=4（全部 4 卡，含 2×Z100）**：ready 110–162 s、输出正确、
       两次 3×64-token bench 均 **0 dmesg 故障**，TTFT p50 **1.74–1.77 s**、
       decode **13.5–14.3 t/s**（DTK TP4 B0 23.92 t/s 的 ~58%，FP8 emu + guard 成本）；
     - **4B GDN TP=4 / 0.6B TP=4**：init+首请求通过；
     - **残留风险**：4B TP=4 连续请求下仍偶发 DP4X page fault（guard 只覆盖 Triton
       kernel，C++/库 kernel 不在覆盖范围，见 §11.3）。

## 1. 今天的复现与最小事实

### 1.1 基线复现矩阵（`replay_attn.py attn-calls/call-0.pt`）

| card | 型号 | 栈 | 写出 | finite | maxdiff | dmesg |
|---|---|---|---|---|---|---|
| 0 | Z100L | ROCm10.1 | 10240/10240 | 是 | 1.7e-3 | 0 |
| 1 | Z100L | ROCm10.1 | 10240/10240 | 是 | — | 0 |
| **2** | **Z100** | ROCm10.1 | **72/10240** | **否** | inf | **0** |
| **3** | **Z100** | ROCm10.1 | **72/10240** | **否** | inf | **0** |
| 2 | Z100 | DTK | 10240/10240 | 是 | 1.7e-3 | 0 |

- card-2 失败是**确定性**的：连续 3 次运行写出完全相同的 72 个元素、相同的值
  （`lead-logs/e6-detail-card2.log`）。
- 同一调用在 card-0 与 DTK/card-2 上正确 → **不是数据、不是 code object 字节差异**
  （kernel-analyst 复核：`.text` md5 `700dab8d…`、`.rodata` `9efa8578…`、kdesc note
  `06c0f6fe…` 在 DTK 与 ROCm10.1 两栈完全相同）。
- `synthetic_attn_probe.py --kv 5`（全新 tensor）在 card-2 同样失败、dmesg 仍为 0。

### 1.2 失败签名（kernel-analyst 精确解码）

- 编译产物：`kernel_unified_attention_2d`，VGPR=256（`VGRPBlocks 63`，Occupancy 1
  wave/SIMD），SGPR 60，LDS 8192，`private_segment_fixed_size=32`，7 次 spill，
  `num_warps=2/num_stages=1`。
- 输出 store 布局：`sizePerThread [1,8] / threadsPerWarp [4,16] / warpsPerCTA [2,1]`，
  每线程 2 条 `buffer_store_dwordx4`。**72 = CTA(0,0) 第 0 条 store 的 lane 0–8**
  （每 lane 8 个 fp16）→ 即"第一条 global store 只提交了前 9 个 lane 就被中止"，
  其余未写、前 8 个值是 fp16 位模式 `00 01 02 … 0F`（字节计数，不是 attention 数据），
  72 个值里 3 个 NaN。Triton 的 mask 是 row/16-lane-group 一致（不可能产生 9-lane 前缀），
  因此这不是 mask/逻辑问题，而是**执行提交被截断**。

### 1.3 head / launch-config 矩阵（card-2，`probe_synth.py` / `probe_warps.py`）

| kernel 配置 | n_regs | n_spills | 写出 | 判定 |
|---|---|---|---|---|
| head64 w2（默认） | 203 | 0 | full | **PASS** |
| head64 w2 BLOCK_M=32 | 256 | 17 | 72 | FAIL |
| head64 w2 BLOCK_M=64 | 256 | 122 | 72 | FAIL |
| head64 w4 BLOCK_M=32 | 203 | 0 | full | **PASS** |
| head96 w2（默认） | 256 | 8 | 72 | FAIL |
| head128 w2（默认 serving） | 256 | 8 | 72 | FAIL |
| head128 w2 BLOCK_M=8 | 253→256 | 0 | 72 | FAIL |
| head128 w4 BLOCK_M=16 | 254→256 | 0 | 136 | FAIL |
| head128 w8 BLOCK_M=16 | **128** | 10 | 10240 | **PASS** |
| head128 w16 BLOCK_M=16 | **64** | 72 | 10240 | **PASS** |
| head128 w8 BLOCK_M=32 | 128 | 115 | 10240 | **PASS** |
| head192 w2（默认） | 256 | 268 | 72 | FAIL |
| head192 w8 | 128 | 114 | full | **PASS** |
| head256 w2（默认） | 256 | 268 | 72 | FAIL |
| head256 w4 BLOCK_M=32（=生产 tuned GQA6 prefill 形状） | 256 | 268 | 136 | FAIL |
| head256 w8 BLOCK_M=32 | 128 | 296 | 30720 | **PASS** |

**判据（12 个失败 / 9 个通过配置）：失败 ⇔ VGPR 分配向上取整到 256。**
VGPR=203/128/64 的配置全部通过。spills（0～296）、LDS、grid、head size、`num_warps`
本身都不能单独区分两类；它们只是影响 VGPR 的旋钮。

对照：1D/2D/3D 简单 kernel（store/load/LDS，grid 1…1024）在 card-2 全对、0 故障
（`grid_probe.py`/`grid2d_probe.py`）→ 不是 dispatch/网格问题。这同时解释了为什么
先前"simple kernel 矩阵全对"无法证明 SKU 健康：那些 kernel 都是 1 CTA、低 VGPR。

### 1.4 GDN/FLA kernel 也落在故障包络内

triton cache 中 4B（Qwen3.5-4B，24×GDN + 8×full-attn，head 256/GQA4）使用的
`chunk_fwd_kernel_o` kernel descriptor：**VGPR=256、scratch=2840 B/thread**。
→ GDN 模型在 Z100 上同样会静默错算，且 attention-only 的 workaround 不够。

### 1.5 最小独立复现器（新增，vendor 交付件）

`/mnt/game/f24-investigation/minimal_vgpr_repro.py`：单 CTA、2 warps、`BLOCK=16384`
的普通 elementwise Triton kernel（不含 attention、不含相对 vLLM 的逻辑依赖）。
编译后 `n_regs=256`、`n_spills=15`（gfx906 上限）。同一 binary、同一 triton cache、
同一进程环境，4 卡实测：

| card | SKU | dp4x_status | written | 结果 | dmesg |
|---|---|---|---|---|---|
| 0 (BDF 2e) | Z100L | 0 | 16384/16384 | PASS (maxdiff 3.8e-06) | 0 |
| 1 (BDF 33) | Z100L | 0 | 16384/16384 | PASS | 0 |
| 2 (BDF 38) | Z100 | 1 | **72/16384** | **FAIL** | **0** |
| 3 (BDF 3b) | Z100 | 1 | **72/16384** | **FAIL** | **0** |

同一 kernel 改变 launch config（card-2）：warps=4（VGPR=144）→ 16384/16384 PASS；
warps=8（VGPR=80）→ PASS；BLOCK=32768 warps=2（VGPR=256）→ 72/32768 FAIL；
BLOCK=8192 warps=2（VGPR=136）→ PASS。**这是可交给厂商的最小、确定性复现。**

## 2. 根因链：DP4X 模式 + 不认识 DP4X 的新用户态

### 2.1 唯一的 SKU 级硬件差异：DP4X

| node | HIP | BDF | DID | 名称 | dp4x_status | ctl_stack_size | cwsr_size | clk |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 2e | 0x55b7 | Z100L | 0 | 20480 | 33550336 | 1600 MHz |
| 2 | 1 | 33 | 0x55b7 | Z100L | 0 | 20480 | 33550336 | 1600 MHz |
| 3 | 2 | 38 | 0x54b7 | Z100 | **1** | 24576 | 22372352 | 1319 MHz |
| 4 | 3 | 3b | 0x54b7 | Z100 | **1** | 24576 | 22372352 | 1319 MHz |

其余全部相同：`gfx_target_version=90106`、`simd_count=240`、`fw_version=36`、
`capability=745775744`、`debug_prop=1494`；固件版本（MEC 36/RLC 7/SDMA 7/SMC
00.05.07.00）四卡一致；VBIOS 4.51 与 5.701 各在两张卡上出现一次（**与 SKU 无关**，
因此不是 VBIOS）；ECC/RAS/坏页四卡干净；同一 IOMMU group、`iommu=pt`、ATS/PASID
状态一致。故障与 DID、`dp4x_status` 完全对应（历史 dmesg 全部落在 38/3b，2e/33 零故障）。

KMD 侧证据（`01-hw-forensics.md`）：Z100 的 DP4X 模式来自固件
（MP0 `C2PMSG_33`：`0xDB48DB48`=DP4X，`0xDB28DB28`=DP2X）；KMD 有专门的
DP4X CWSR trap handler（"dp4x cp vgpr granularity error" workaround）、DP4X VGPR
init/RAS 路径，并在 `check_ctx_size_invalid()` 中按 DP4X 用 cu_num=64/vgpr=0x40000
校验 queue save area。

### 2.2 两栈的用户态能力差异

| 项 | DTK 26.04 用户态（配套） | ROCm 10.1 用户态（本次） |
|---|---|---|
| ROCR/HIP | ROCR 1.11（ROCR-Runtime_7.2 fork）/ HIP 7.2 | ROCR 1.21.0 / HIP 7.16.26385 |
| DP4X 支持 | 有：loader `EF_GPU_DP4X`、code-object 检查 "compiled without dp4x"、"Get vgpr granularity"、HSA attr 101；HIP `hasDp4x`；clang dp4x 选项 | **无**（`strings` 0 命中） |
| 设备识别 | 原生（Hygon DID 表） | 必须 `HSA_OVERRIDE_GFX_VERSION=9.0.6`，只改 ISA 名，不改任何设备属性/feature |
| KFD/driver 配套 | KFD uapi minor 16 配套 | HIP 7.16/ROCR 1.21 比 KMD 期望超前约 5 个 KFD minor / 10 个 ROCr major |

而代码对象字节在 DTK 与新栈完全相同（`.text` 一致）→ **差异完全在运行时行为**，
不是编译器/内核源码问题。

### 2.3 机制假设（按置信度）

- **高置信（实测规则）**：DP4X 卡 + 新用户态下，VGPR 分配到 256 的 kernel 的 wave
  在第一条 global store 处被中止（部分 lane 提交），dmesg 无记录。VGPR ≤203 正常。
- **中高置信（机制链条）**：DP4X 改变 VGPR 粒度/CWSR 保存几何；本 KMD 知道 DP4X，
  但新用户态不声明/不配置 DP4X（无 `EF_GPU_DP4X` code-object 标记、无 "Get vgpr
  granularity" 修正、queue context save 尺寸按 stock 公式算），使得满 VGPR kernel
  落在 KMD 的 DP4X 特例路径上；KMD 的 dp4x trap handler 把该异常"吞掉"（所以静默），
  而 wave 已提交的部分 store/后续地址访问在其它形态下变成 UTCL2 无效 PTE（XID:81）；
  同一 kernel 若卡住不可 preempt，则在 9 s 后 XID:76 kill 进程，回收阶段引发 vmfault
  风暴（历史 F24 的 B 类签名）。
- **待厂商确认**：DP4X 下是否允许 VGPR=256（是否有"≤203"之类的文档上限）；code object
  是否必须带 DP4X 标记；CWSR 尺寸的精确契约。

## 3. 两个故障类与 TP=4 死亡的关系

### 3.1 A 类：静默错算（本次新发现，确定性）

- 单卡 Z100：attention（0.6B）或 GDN（4B）输出垃圾，**无任何 dmesg**。
- 0.6B card-2 实测：`'lock顾_guardIGHT\\版äge ultimate'` vs card-0
  `' Paris. The capital of Italy is Rome'`；4B card-2：`'ductduct…'` vs card-0
  `' Paris.\nA. True\nB'`。
- 这解释了历史上把 bf16 NaN 误判为 F21、以及"DTK 能跑/新栈不行"的大量现象，
  但**不是** F24 报告里"vmfault 导致 NaN"的因果。

### 3.2 B 类：worker 死亡/vmfault

- 历史 6 个 episode（10-17，全部在 Z100 卡）：UTCL2 no-retry page fault（PTE 无效，
  XID:81）→ `Queue preemption time out`（XID:76）→ kill 进程 → 进程回收 vmfault。
  `sq_intr err_type 2` = MEMVIOL；`priv 1` 是 PRIV 上下文位（非诊断信息，历史报告读法已修正）；
  `[80xx]` 是 PASID 而不是错误码；fault VA 是进程 GPU-VA（large-BAR 直接映射），
  不是 host 指针。
- 今天复现：
  - **TP=4（0.6B，stock 栈，4 卡）**：READY 后首个请求即出现 3 条 sq_intr + 4 条 vmfault
    （38 卡），服务端 `RuntimeError: cancelled`（engine 无响应）——即用户描述的
    "Z100 上 worker 直接 vmfault 死亡"类；打 workaround 后同一 TP=4 完全正常。
  - **4B GDN（attention-only workaround 配置，card-2）**：init/warmup 阶段
    XID:81 → XID:76（4 s 内）→ killed process → vmfault 风暴，与历史 F24/TP=4 签名一致。
    （unpatched 4B 只是静默垃圾输出，说明 B 类是否发生取决于当时执行的 kernel/地址形态；
    触发条件尚未按 kernel 完全分离。）
- **与 TP=4 的关系**：多 rank 时只要 Z100 rank 触发 B 类，其余 rank 在 NCCL/RCCL
  split/collective 上就会看到 "remote process exited / connection closed" —— 这正是
  F24 原始 TP=4 27B 的 worker-2 死亡日志。因此 TP=4 死亡 = B 类（同一 DP4X 根因家族），
  而不是某个特定 attention kernel 的独立 bug。

## 4. 已验证的 workaround（绕过，不是修复）

仅在影子副本 `/mnt/game/f24-investigation/wa-repo`（`cp -a` 自
`/mnt/game/port/gfx906-vllm-rocm101`，未改动受验证树）中打两处补丁：

1. `vllm/v1/attention/ops/triton_unified_attention.py`（2D 非 tuned prefill 分支）：
   `num_warps` 2 → 8；
2. 同文件 `_decode_num_warps()`：统一 `return 8`（3D split-KV decode）。

效果（全部为今天实测）：

| 场景 | stock | warps=8 workaround |
|---|---|---|
| 0.6B TP1 card-2 | 垃圾输出（0 故障） | 与 Z100L 完全一致的正确答案 |
| **0.6B TP4 全 4 卡** | 请求失败 + sq_intr/vmfault | **READY 36 s、答案正确、0 dmesg 故障** |
| 4B GDN TP1 card-2 | 垃圾输出 | 未完成（GDN kernel 仍 VGPR=256，init 触发 B 类） |

要点与限制：

- 生产 tuned head-256/GQA6 prefill 路径走 `GFX906_PREFILL_WARPS`，已有 env 旋钮
  `VLLM_GFX906_ATTN_PREFILL_WARPS=8` 可用（本次未做 27B 端到端验证）。
- GDN/FLA（`chunk_fwd_kernel_o` 等）必须做同类"降 VGPR"处理，否则 4B/27B 仍不可用
  —— 这是 27B-FP8 TP=4 生产形态的剩余阻塞点。
- 性能影响未测量：`num_warps=8` 改变 tiling/occupancy，prefill 可能变慢，需要
  benchmark 后才能作为生产 workaround。
- 该 workaround 只是让 kernel 避开故障包络，不能替代厂商修复。

## 5. 为什么之前的"排除矩阵"没有得到正确结论

| 先前排除项 | 本次复核 |
|---|---|
| **env 矩阵**（fine-grain PCIe / SDMA / 中断 / `HIP_LAUNCH_BLOCKING` / override 9.0.0/9.0.8/9.0.9） | 这些开关只影响 RCCL 链路/ISA 名，不改变 kernel 的 VGPR 分配；`HSA_OVERRIDE_GFX_VERSION` 只覆盖 ISA 名，不补 DP4X/设备属性。未测到真正的维度（launch config→VGPR）。 |
| **simple kernel 矩阵**（add/smem/barrier/exp/dot/trans/atomic 全对） | 全部是 1 CTA、低 VGPR kernel，从不触及故障包络；不能证明 Z100 SKU 健康。 |
| **attention 配置矩阵**（warps/stages/tile/BLOCK_M/tuning/强制 3D 全失败） | 复核发现 `f24-cfg-w4/w8/s2` 臂编译出的 kernel **仍然是 `num_warps=2`**（overrides 在该形状下不生效，见 §1.3 的生效条件）→ "warps/stages 失败"证据无效；tile64/tune0 等确实改了配置但 VGPR 仍=256。矩阵缺少 `num_warps=8/16` 这一唯一能把 VGPR 降到 ≤203 的维度。 |
| **TP=4 在 skip warmup + eager fallback 下仍死** | eager fallback 只替换 attention，GDN/FP8 等其它 Triton kernel 仍是 VGPR=256；且 B 类可以在 model init/teardown 的普通 GPU 工作中触发。故不能推出"与 kernel 无关"。 |

## 6. 与既有 F24 结论的对照修订

| F24 原文 | 修订 |
|---|---|
| "同一份 HSACO 在两栈字节相同，DTK 4 卡全对、新栈 Z100 NaN/部分写出" | 保留（本次独立复核：`.text/.rodata/.note` 相同；DTK card-2 正确）。 |
| "vmfault 导致 NaN/只写出 72/10240" | **修正**：静默错算 dmesg 为 0；vmfault 与静默错算是同一根因下的两种表现，不是因果。 |
| "fault VA 是 host 指针 / 主机地址" | **修正**：是进程 GPU-VA（large-BAR 映射，PTE invalid）；`[80xx]` 是 PASID。 |
| "Z100 SKU 在 ROCm 10.1 上执行异常（需厂商 runtime 修复）" | 方向保留，但**收窄为 DP4X 模式 + DP4X-unaware 用户态 + VGPR=256 kernel**；不是"所有 kernel"、不是"硬件缺陷"；且已找到可验证的 workaround。 |
| "env/simple kernel/attention config 矩阵已排除" | **修正**：见 §5；其中 attention config 矩阵的 warps/stages 臂实际是 no-op。 |
| 影响："TP=4 阻塞，验证固定 0/1" | 保留（27B TP=4 仍阻塞于 GDN kernel）；但 0.6B TP=4 在 workaround 下已可用，说明阻塞是"per-kernel VGPR 包络"问题，可工程化绕过。 |

## 7. 建议的下一步（按优先级）

1. **厂商报告 v2 + 最小复现（已交付，含 §11 的 dp4x_enable/寄存器实验结果）**：
   给厂商一个不依赖 vLLM 的最小 kernel（高 VGPR + spill，确定性写出截断）和
   `replay_attn.py` 现成复现；问题清单见 v2 §Ask。
2. **维护窗口决定性实验**：用 `hycu.dp4x_enable=0`（当前 0444 RO，需要 boot 参数/重编
   KMD）或能切 DP2X 的 VBIOS/固件冷启动后，在 card-2 跑
   `replay_attn.py`：若 `finite/untouched=0` 即闭环证明 DP4X 是唯一变量。
3. **冷启动复测**（低成本，先做）：冷启动后立即跑 card-2 最小复现，排除"故障后残留状态"。
4. **workaround 工程化**：
   - 对 `VLLM_DCU_GCN_ARCH=gfx906` 且 `dp4x_status=1` 的卡，自动把 attention
     prefill/decode 的 `num_warps` 提升到 8（或引入 per-kernel VGPR 上限检查，
     把 n_regs=256 的 kernel 自动降 VGPR：调 warps/BLOCK_M/maxnreg）；
   - 对 FLA/GDN kernel（`chunk_fwd_kernel_o` 等）做同样处理并逐一验证正确性；
   - 用 `VLLM_GFX906_ATTN_PREFILL_WARPS=8` 覆盖生产 tuned head-256/GQA6 prefill；
   - 为 workaround 补性能基准（prefill/decode 与 DTK TP4 B0 对比）。
5. **27B-FP8 TP=4 复测计划**（GDN 修复后）：`serve_rocm101.sh --model 27b --tp 4`，
   记录 ready 时间/KV cache/token 与 DTK 对照。
6. **文档同步**：更新计划书 §9.2 F24、验收报告 §5/§6、env 契约的 F24 注释
   （把"卡 2/3 不能用于验证"改成"DP4X 卡需避让/降 VGPR"），并撤回
   vendor report 中两条被修正的表述。

## 8. 复现命令（最小集）

```bash
# 1) 静默错算（Z100 卡，约 10 s；dmesg 无新行）
docker exec -e HIP_VISIBLE_DEVICES=2 z100-vllm-rocm101-rt bash -c '
  . /src/tools/dcu/env-rocm101.sh >/dev/null 2>&1
  python3 /mnt/game/z100-rocm101-validation/replay_attn.py \
    /mnt/game/z100-rocm101-validation/attn-calls/call-0.pt --device 0'
#   期望 card-2: untouched=10168/10240, finite=False
#   对照 card-0: untouched=0/10240, finite=True

# 2) 最小独立复现（无 vLLM/模型；4 卡对照，vendor 交付件）
for c in 0 1 2 3; do
  docker exec -e HIP_VISIBLE_DEVICES=$c z100-vllm-rocm101-rt bash -c '
    . /src/tools/dcu/env-rocm101.sh >/dev/null 2>&1
    python3 /mnt/game/f24-investigation/minimal_vgpr_repro.py'
done
#   期望：card0/1 PASS(16384/16384)；card2/3 FAIL(72/16384)，且 dmesg 无新行

# 3) 判据确认（head/launch config 矩阵；written 完全写满=通过）
docker exec -e HIP_VISIBLE_DEVICES=2 z100-vllm-rocm101-rt bash -c '
  . /src/tools/dcu/env-rocm101.sh >/dev/null 2>&1
  python3 /mnt/game/f24-investigation/lead-logs/probe_synth.py 128 8 16 5 5 16 8'
#   h128 w8: n_regs=128 -> written=10240 (PASS)；w2: n_regs=256 -> 72 (FAIL)

# 4) 端到端（workaround 影子树）
TP=4 REPO=/mnt/game/f24-investigation/wa-repo \
  bash /mnt/game/f24-investigation/lead-logs/smoke_06b_tp.sh
#   期望：READY ~36 s，COMPLETION_TEXT=' Paris. The capital of Italy is Rome'，0 dmesg
```

## 9. 证据路径

| 内容 | 路径 |
|---|---|
| 实验索引/数据表（本报告 §1–§3 的来源） | `/mnt/game/f24-investigation/00-LEAD-SUMMARY.md` |
| 硬件/驱动/dmesg 取证（873 行） | `/mnt/game/f24-investigation/01-hw-forensics.md` |
| 栈差异 / DP4X / ABI 分析（539 行） | `/mnt/game/f24-investigation/02-stack-diff.md` |
| kernel 资源/签名分析（291 行） | `/mnt/game/f24-investigation/03-kernel-analysis.md` |
| 外部调研 / 上游支持状态 / 厂商问题 | `/mnt/game/f24-investigation/04-external-research.md` |
| 全部原始日志（52 文件） | `/mnt/game/f24-investigation/lead-logs/` |
| 最小独立复现器（vendor 交付件） | `/mnt/game/f24-investigation/minimal_vgpr_repro.py` |
| 最小复现 4 卡原始日志 | `/mnt/game/f24-investigation/lead-logs/minimal-vgpr-4card.log` |
| workaround patch（shadow diff） | `/mnt/game/f24-investigation/workaround-attn-warps8.patch` |
| 最终 dmesg 快照（含 B 类签名） | `/mnt/game/f24-investigation/lead-logs/dmesg-snapshot-20261018-final.txt` |
| DP4X 关闭实验计划/证据（task1） | `/mnt/game/f24-investigation/05-dp4x-disable-plan.md`、`/mnt/game/f24-investigation/dp4x-boot/`、`/mnt/game/f24-investigation/dp4x-off/experimentA.log` |
| VGPR=256 kernel 审计（task2） | `/mnt/game/f24-investigation/06-vgpr-256-kernel-audit.md` |
| DP4X guard 实现/自检 | `vllm/platforms/hygon_dp4x.py`、`tools/dcu/dp4x_guard_selftest.py` |
| 27B TP4 性能/正确性日志 | `/mnt/game/f24-investigation/lead-logs/bench-27b-tp4-auto*.log`、`27b-tp4-*.log` |
| 4B/0.6B TP4 验证与残留故障日志 | `/mnt/game/f24-investigation/lead-logs/bench-4b-tp4-auto.log`、`tp4-06b-*.log` |
| workaround 影子树（仅供验证，勿用于生产） | `/mnt/game/f24-investigation/wa-repo/` |
| 厂商报告 v2（含最小复现与问题清单） | `/mnt/game/z100-rocm101-validation/f23-evidence/F24-vendor-report-v2.md` |
| 历史证据（F23/F24） | `/mnt/game/z100-rocm101-validation/f23-evidence/` |

## 11. 当日后续执行结果（2026-10-18 下午）

### 11.1 Task 1：DP4X 关闭实验

1. **`hycu.dp4x_enable` 是死参数（决定性证据，`05-dp4x-disable-plan.md`）**：
   live `/usr/local/hyhal/dkms/hycu.ko`（build-id `2022130934…`，与
   `/sys/module/hycu/notes` 一致）中 `gpu_dp4x_enable` 只有 2 处重定位——一处
   `.rela__param`（注册）、一处 DWARF，**`.rela.text` 中 0 处**（对照：`cwsr_enable`
   有 PC32 代码引用）。因此 `modprobe hycu.dp4x_enable=N` / GRUB 参数都不可能改变
   任何行为，**没有为它重启机器的理由**。
2. **DP4X 的真实来源**：GC 寄存器 `SPI_CONFIG_CNTL_2` (0x31108) bit8；只读实测
   card0/1=0x11、card2/3=0x111；KMD 在 init 时通过 MP0 `C2PMSG_64←0xD0000`、
   `C2PMSG_33==0xDB48DB48` 读取并按键 DP4X 行为。无 sysfs/debugfs/参数可写。
3. **运行时位翻转实验（已执行、已恢复）**：card2 上 `mmio_rw -i 2 driver -w 0x31108 0x11`
   → 读回 0x11；最小复现反而出现 `written=7020917803559905225/16384, finite=False`
   （连 checksum reduction 都被污染）——因为驱动/引擎的 DP4X 簿记仍是 DP4X，而硬件
   已切 DP2X，状态不一致。恢复 0x111 后回到 `written=72/16384` baseline，GPU 健康
   （guard selftest card2 PASS、card0/2/3 复现签名一致、无 dmesg 故障）。
   **结论**：仅翻 bit 无法得到干净 DP2X；DP4X 关闭必须走 KMD 重写（Experiment B）
   或厂商 BIOS/VBIOS，属于厂商项。

### 11.2 Task 2：DP4X guard 工程化（已落地并验证）

**实现**（repo 内，非影子树）：
- `vllm/platforms/hygon_dp4x.py`（新）：KFD topology 读取 `dp4x_status`（按 PCI bus
  匹配 torch device，兼容 `HIP_VISIBLE_DEVICES`），`install_dp4x_triton_guard()`
  在进程内对 `triton.runtime.jit.JITFunction.run` 做一次包装：DP4X 设备上强制
  `num_warps>=VLLM_GFX906_DP4X_MIN_WARPS`（默认 8）；`VLLM_GFX906_DP4X_WATCH`
  对仍编译到 granulated VGPR=256（`n_regs>=253`，253/254/255/256 同属故障包络）
  的 kernel 发 warning；`VLLM_GFX906_DP4X_DEBUG=1` 输出每个 kernel 的
  device/overridden/n_regs（用于验证没有漏网 kernel）；`VLLM_GFX906_DP4X_GUARD`
  `auto|0|1`（auto = Hygon DCU + 存在 DP4X 卡 + HIP>=7 的上游 runtime；DTK 栈
  HIP 6.3 自动关闭，不影响已验证的 DTK 基线）。
- `vllm/platforms/rocm.py`：平台导入时安装 guard（覆盖所有会发射 Triton kernel 的
  进程；在 DTK 栈上 enable 检查立即返回 False）。
- `tools/dcu/dp4x_guard_selftest.py`（新）：单 CTA、VGPR-max 哨兵 kernel 部署自检；
  card2 期望 `written=16384/16384`（guard 生效，n_regs=80），card0 期望同样 PASS
  （guard 不触发，n_regs=256，DP2X 本就正常）。
- `tools/dcu/env-rocm101.sh`：F24 注释更新为新机制与 env 开关。

**验证结果（全部 0 dmesg 故障）**：

| 场景 | 结果 |
|---|---|
| guard 自检 card2 (Z100) | PASS，n_regs 256→80（w8 强制生效） |
| guard 自检 card0 (Z100L) | PASS，n_regs=256（不触发，DP2X 正常） |
| 0.6B TP1/TP2/TP4（含 mixed SKU TP2 0+2） | 正确输出，0 故障（TP4 warm cache 多次通过；冷 cache 下有历史偶发，见 11.3） |
| 4B GDN TP1 card2 | 正确输出，0 故障 |
| 4B GDN TP4 全部 4 卡 | init+首请求正确，0 故障；连续请求下偶发故障（11.3） |
| **27B-FP8 TP2（Z100 对）** | ready **96 s**，`' Paris.\nThe capital of Germany is'`，0 故障 |
| **27B-FP8 TP4（4 卡，auto 与 guard=all 两种）** | ready **110–162 s**，输出正确，两次 3-request bench 全通过：TTFT p50 **1744–1770 ms**、decode p50 **13.5–14.3 t/s**，0 故障；TP4 生产形态首次在含 Z100 的 4 卡上可用 |
| Z100L 回归 | guard 不触发（overridden=False），行为与之前一致 |

### 11.3 残留问题（未解决，需后续/厂商）

- guard 只覆盖 **Triton** kernel（通过 `JITFunction.run`）；**C++/库 kernel
  （vLLM `_C` HIP kernel、先编译的 MIOpen/rocBLAS/Tensile/RCCL kernel）不在覆盖范围**。
- 4B GDN TP4 在连续请求（第 2 个 64-token 请求的 decode 阶段）出现一次
  `XID:81 page fault (VLLM::Worker on 38) → XID:76 → kill` 的间歇故障（6 sq_intr /
  5 vmfault / 12 XID），随后服务不可用；0.6B TP4 在早期冷 cache 运行中也出现过同类
  间歇故障。27B-FP8 TP4 的 4 次运行（2 smoke + 2 bench）均干净，但样本仍有限。
- 说明存在少量**非 Triton 或尚未被 guard 覆盖的 DP4X 敏感 kernel**；下一步建议
  在 DP4X 卡上用 rocprof/CUPTI kernel trace 对比 4B TP4 失败前后，或由厂商给出
  DP4X-aware KMD/runtime 后彻底消除。当前生产建议：27B-FP8 TP4 可用（已实测），
  4B/0.6B TP4 在长时间/高并发前需复测，或固定 Z100L 双卡 TP2。

## 10. 交付物与回滚

- 本报告是**独立复核**，不改变已验收的 Z100L 结论（L0–L4、TP=1/2、27B TP=2 不受影响）。
- 未修改受验证的 `/mnt/game/port/gfx906-vllm-rocm101` 树，也未修改 `/src`（repo）；
  workaround 仅在影子副本中验证。
- 建议只在评审后把 §7-4 的工程化改动落到 repo（另开 commit/PR），并同步文档。
