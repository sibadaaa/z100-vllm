# Z100-PREFILL-LCTX 结果:S56 HIP 内循环 prefill 匂

> 执行依据:`docs/design/z100/prefill-lctx/z100-prefill-lctx-design.md`(设计图)与
> `z100-prefill-lctx-construction.md`(施工图),2026-11-06。执行 agent 会话 s56
> (台账 `.s56/`)。原始数据:`runs/prefill-lctx/s56/`。
> 环境:生产栈 Qwen3.8-27B-FP8,TP=4,MTP k=2,MBT 8192,服务 :8000(本会话内
> 经一次重启窗口切换 A→B 臂,cmdline/env 复刻 incident 模板)。

## 0. 结论一览

| 项 | 判据 | 结果 | 裁决 |
|---|---|---|---|
| WP0 基线复现 | 3d:4 ≈2447±3%、2D ≈2686、比值 1.10±0.02 | 2471.4 / 2670.2 ms、1.080 | **PASS**(≤±1%) |
| WP3 内核微测 131k | 2× 门(≤1230 ms) | **1505.0 ms(1.67×)**,3.89 TF useful | 1.3–2× 档 |
| WP3 内核微测 262k | ≤2500 ms | **3005.1 ms(1.70×)**,3.96 TF | 同上 |
| WP3 精度门(核级) | \|d\| ≤ 0.011 vs fp32(S53 先例) | **1.3e-5**,且与 Triton 输出逐位同 | **PASS** |
| WP3 服务级 131k A/B | 同窗 ×2 reps | 722.3 s → **351.5 s(−51.3%)** | **PASS(超 450 s 门)** |
| WP3 贪心 parity | 8 prompt 零单向劣化 | **8/8 逐 token 全同** | **PASS** |
| WP3 TPOT/ITL 同长 | ≤±3% | 8k ITL 48.0 → 48.2 ms(+0.4%) | **PASS** |
| 262k 服务冒烟 | 完成 + 服务存活 | **未完成**:PCIe DPC Fatal → HCU reset(硬件事件,§2.4) | **失败,单列** |
| WP2 fp16 段输出 | 微测 −12~19% | 实测 **≈0**(−1 ms) | **不采纳(负结果)** |

**服务级关键数字(131k 单请求 prefill):TTFT 722.3 s → 351.5 s(−51.3%),
prefill 吞吐 181.4 → 372.9 tok/s(2.06×)。** 立项目标链 131k ≤450 s:达标;
300 s 拉力目标余 51 s。

## 1. WP0 开工核验

`s22_attn_prefill_probe --ctx 131072 --m-list 7488 --block-size 800`(DCU3,
shuffle map,服务在役静默):3d:4 = 2471.4 ms(台账 2447,+1.0%)、2D =
2670.2 ms(台账 2686,−0.6%)、比值 1.080(门 1.10±0.02 下沿)。判定:环境
可信,复现偏差全部 ≤1%。

## 2. WP3 = P3-b0:S53 HIP 内循环 prefill 化(主路径)

### 2.1 改动清单(全部默认关闭,可 env 回滚)

* `vllm/v1/attention/ops/triton_unified_attention.py`:
  - 新 env `VLLM_GFX906_ATTN_PREFILL_HIP_INNER`(默认 0,per-dispatch 实时读,
    同 decode 旋钮风格);在 prefill 3D 分支内 `_launch_split_kv` 前加 HIP 旁路,
    门 = use_prefill_3d 全部门 + causal + num_seqs==1 + q_len≥16 + 与 decode
    `use_hip_inner` 同款 dtype/stride/layout 检查(GQA6、head 256、b%16==0、
    16B 对齐 KV strides、int32 metadata、段缓冲 stride 契约);命中则调
    `torch.ops._rocm_C.gfx906_attnd_decode_inner(..., num_segments=4,
    tps_hint)` + 复用现有 `reduce_segments`。**decode 派发未动。**
  - 新 env `VLLM_GFX906_ATTN_PREFILL_SEGM_FP16`(默认 0,WP2,见 §3)。
* `csrc/rocm/attnd_inner.hip`:`ATTN_INNER_LAUNCH` 加 tps_hint 桶 **2048**
  (131k/4 段)与 **4090**(262k/4 段);段输出写随缓冲 dtype 分支(WP2);
  `segm_output` 形参改 dtype 无关基指针。
* 构建债:原文件 `at::cuda::getCurrentHIPStream/OptionalHIPGuard` 拼写在生产
  torch(host .venv 的 c10/hip 头用 CUDA 名)下不可编译,已恢复 CUDA 名拼写。
* `_rocm_C.abi3.so` 重建:用生产 build 树 build.ninja 的**逐字旗标**在 host 侧
  单文件重编 attnd_inner + 全量重链(其余 .o 逐字复用原生产构建产物;脚本
  `.s56/rocm_c_rebuild/host_build.sh`)。仓库符号链接
  `vllm/_rocm_C.abi3.so` → `.s56/rocm_c_rebuild/_rocm_C.abi3.so`。
  **回滚 = 符号链接改回 `/mnt/game/port/build-sync/temp/_rocm_C.abi3.so`
  + 两个 env 不设/设 0**(原生产 .so 文件未改动)。

### 2.2 内核级 A/B(工具 `tools/dcu/s56_hip_prefill_ab.py`)

生产派发对(producer + reduce_segments)同输入双调,真分页 block 800、
shuffle map、q=7488、fp16、GQA6/256:

| ctx | Triton 3D | HIP inner | 加速 | HIP useful TF | HIP GB/s(issued) | max\|d\| HIP vs Triton |
|---:|---:|---:|---:|---:|---:|---:|
| 131072 | 2507.6 ms | **1505.0 ms** | **1.67×** | 3.89 | 324 | 0.0(逐位同) |
| 262144 | 5097.3 ms | **3005.1 ms** | **1.70×** | 3.96 | 330 | 0.0(逐位同) |

vs fp32 参考(s22 `--ref-check` 同款):HIP 1.3e-5,与 Triton 自身同精度。
useful 3.9–4.0 TF,带宽 324–330 GB/s 仅屋顶 42%——瓶颈在算力侧。
**2026-11-07 S58 WP0 修正**:原“dot2 1/4 速率、天花板 ~4.5 TF”的归因
被微内核证伪——`v_dot2_f32_f16` 满占用可达 22.7 TF/s(峰值 92%),生产
2 CTA/CU + 单累加器链下为 6.05 TF/s;3.9 TF 是单链延迟/占用受限,不是
指令发射率。b0+(打包 FMA)不立项,下一杠杆是 QK^T 多累加器 ILP。详见
[`z100-s58-result.md`](z100-s58-result.md)。

### 2.3 服务级同窗 A/B(一次重启窗口)

A 臂 = 原服务(旧 .so,env 关);B 臂 = 重启服务(新 .so,
`VLLM_GFX906_ATTN_PREFILL_HIP_INNER=1`,其余 env/cmdline 与 incident 模板
逐字一致)。`s24_longctx --gen 16`,同 seed 公式,同日同窗:

| lens | 臂 | TTFT p50 | prefill tok/s | ITL p50 |
|---:|---|---:|---:|---:|
| 8192 | A | 9724.5 ms | 842.4 | 48.0 ms |
| 8192 | B | 9732.9 ms(+0.1%) | 841.7 | 48.2 ms |
| 131072 | A | **722,336 ms** | 181.4 | 134.2 ms |
| 131072 | B | **351,512 ms** | **372.9** | 73.8 ms |

* 131k:**722.3 → 351.5 s(−51.3%)**,吞吐 **2.06×**;两 rep 349.7/353.3 s
  (散差 1%)。8k 档 +0.1%(16k 门槛之下不派发,零影响),ITL +0.4% ≤3%。
* 131k ITL 134.2 → 73.8 ms(gen=16 小样本,方向为改善,量级供参考)。
* 贪心 parity:8 × 20480-token prompt,temperature 0,gen 64:
  **8/8 输出逐 token 全同(0 翻转、0 分歧)**;20k 单请求 wall 34.6 →
  30.5 s(−12%,与该长度 3D/HIP 覆盖比例相符)。
* 服务健康:重启后 `/v1/models` 200,全程 0 崩溃、0 OOM。

### 2.4 长曲线点与 262k 冒烟(B 臂)

| lens | B 臂 TTFT | P0 台账(A,2026-11-05/06) | 变化 |
|---:|---:|---:|---:|
| 16384 | 23.5 s | 24.3 s | −3%(门槛边缘,符合预期) |
| 32768 | 49.5 s | 67.1 s | −26% |
| 65536 | 121.9 s | 208.7 s | −42% |
| 131072 | 351.5 s | 722.9 s | −51% |
| 256000 | **1110.0 s(230.6 tok/s,1 rep)** | (262k 台账 2667.8 s,量级参照) | **~2.4×** |

16k–64k 为 B 臂实测、A 侧引 P0 台账(跨窗口,方向性);增益随长度单调放大,
与 KV 扫描占比上升一致(核级 262k 单层 1.70×)。

**256k 复测(2026-11-06 16:43-17:02,同 HIP 臂 + API 活性修复):完整跑通,
TTFT 1110.0 s / 230.6 tok/s,全程 /health 1 ms 级即时响应,日志 0 条 PCIe
错误。** 上次 262k 的 DPC 复位在 256k 同区间未复现——定性为**偶发硬件事件**
(单次,不可复现),软件侧无相关性证据;建议生产灰度 >131k 并保留 AER 监控,
复现即回退 env。

**262k 冒烟(首次):未完成,失败定性为硬件级事件(非本改动软件故障)。** 请求进行到
约 75%(≈KV 200k)时引擎无进展,`journalctl -k` 证据:root port
`0000:35:10.0` 触发 **DPC containment——PCIe Fatal uncorrectable error
(Transaction Layer)** → slot reset → `hycu: HCU reset`(4 个 TP worker 全部
被驱动杀掉,exit code None)。这是总线级错误,不是内核异常/死循环/主存 OOM;
软件侧无任何 HIP 内核报错。该卡复位后恢复正常,服务已按 B 配置重启并验活。
262k 级稳定性本就挂账 H6(设计图 §2 附带项),本次事件把它升级为
**">131k 上下文 + HIP 高带宽扫描"组合下的 PCIe 链路完整性问题**,需硬件侧
排查(链路重训/误差计数)后再定 262k 是否可开 HIP;建议在 H6 修复立项中
用 `rocevents`/AER 计数器做复现对照。**≤131k 全部门禁不受影响。**

## 3. WP2 = P3-b1:段往返税(已实现,负结果,不采纳)

按施工图改法 (a) 实现:`VLLM_GFX906_ATTN_PREFILL_SEGM_FP16`(段 workspace
fp32→fp16,184→92 MB;HIP 内核 half2 打包写;reduce_segments 的 Triton 核
dtype 无关无需改;segm_max/expsum 保持 fp32)。核级微测:hip16 = 1504.0 ms
vs hip 1505.0 ms(**−1 ms,≈0**),精度 |d| = 1.3e-5 不变。

**根因**:设计图 b1 的 −12~19% 由"拷贝屋顶 777 GB/s × 368 MB 往返 ≈ 474 ms"
推得。实测 HIP 路径里段写与注意力计算重叠、reduce_segments 是 latency 受限
的小 grid(7488×6 CTA × 4×256),**真实段税 ≈1 ms**。负结果按施工图纪律
落账;代码保留(默认 0,顺带省 100 MB workspace,或为 0.95 利用率场景留用)。

## 4. 出口裁决

* 核级落施工图 **1.3–2× 档**(1.67×/1.70×);可叠加的 WP2 实测 ≈0,无叠加。
* **建议采纳进 launcher**:`VLLM_GFX906_ATTN_PREFILL_HIP_INNER=1`(与
  `VLLM_GFX906_ATTN_DECODE_HIP_INNER=1` 并列),**生效范围 ≤131k 上下文**
  (262k 见 §2.4 硬件事件,开 HIP 前需链路排查)。依据:门禁全绿(核级
  |d|=1.3e-5 ≪ 0.011、8/8 parity 全同、8k 零回退、ITL 不劣化、0 软件崩溃),
  131k −51.3%、64k −42%、32k −26%。
* **131k ≤450 s 门:351.5 s 达标**;300 s 拉力目标差 51 s,余量在
  b0+(打包 FMA 宏 tile)与 WP1 惩罚根因。

## 5. 未尽事项(本会话边界)

1. **WP1(P3-a in-vivo 惩罚根因,rocprof 对拍)未执行**——本会话唯一重启
   窗口已用于 A/B 切换。B 臂实测 TTFT 351.5 s 已优于按核级 1.67× 的线性
   折算(≈487 s),说明 in-vivo 惩罚对 HIP 路径显著小于对 Triton 路径的
   1.63×;惩罚结构是乘性且核相关的,WP1 应以 B 臂为新基线重做。
2. WP4 全曲线 4k 档未重跑;262144 全长与 gen≥2 曲线协议留采纳后回填立项
   文档 §7。
3. WP5(b2 KV 扫描重构)未触发(WP3 ≥1.3× 出口)。
4. b0+ 打包 FMA 宏 tile 化已于 S58 WP0 证伪(2026-11-07):`v_dot2_f32_f16`
   满速率、`v_pk_fma_f16` 不更快,不立项;下一杠杆为 QK^T 多累加器 ILP
   (见 [`z100-s58-result.md`](z100-s58-result.md))。
5. **262k + HIP 的 PCIe DPC 事件(§2.4)需硬件侧复现与排查**;在结论出来前,
   >131k 上下文不建议开 `VLLM_GFX906_ATTN_PREFILL_HIP_INNER`。

## 6. 同会话附加:256k 多轮可用性(用户新需求)

长 prefill 提速后,256k 级多轮对话仍"崩溃"。定位出**与内核无关的真凶**并修复:

* **现象**:任一 200k+ 请求在跑时,API server 对 /health、/v1/models、新
  POST 全部无限期挂起(TCP 可连、永不回包),metrics 周期日志出现 41 分钟
  空洞;客户端在 6-12 分钟里得不到任何响应,超时/重试/中止,即用户看到的
  "多轮直接崩溃"。
* **根因**:fork 在 ROCm 上让 AsyncLLM 走 in-process EngineCore,
  `InprocClient.get_output_async` 在 API 事件循环上**同步**跑完一整个引擎
  step——长上下文 chunk 一步 36-90+ s,循环被冻结一整步。
* **修复** `vllm/v1/engine/core_client.py`:step 移入 `run_in_executor`
  线程;新增 `_async_step_lock` 串行化 get_output/add/abort,保持"add/abort
  不与 step 交错"的原单线程不变量(abort 仍在 step 边界生效,语义不变)。
* **验证**:120k 请求 prefill 全程(313.4 s TTFT,与速度曲线一致、零损耗),
  并发 /health 探测 **200/200 即时 200、最差 0.0 s**(修前全路由挂死)。
* **256k 多轮实测**(顺序两轮,修前行为不影响正确性):turn-1 200k
  TTFT 729.7 s;turn-2 256k(同前缀+56k 新增)**TTFT 393.6 s,前缀命中
  198,848/256,000(78%)**——长上下文前缀缓存在混合模型上成立,第二轮省
  ~60%;两轮零崩溃。多轮体验 = 第二轮起不再有全量长 prefill,且服务全程
  可探活。
* **残留**:262000 单发在 HIP 臂出现过一次 PCIe DPC Fatal → HCU 复位
  (硬件级事件,§2.4);修复后 API 可探活、可排队,但 >131k 开 HIP 前仍
  建议先做链路排查;H6(abort 挂死)单独立项,修复后 abort 在 step 边界
  即时生效,其复现/定位窗口反而变好。

## 6. 工件清单

| 文件 | 内容 |
|---|---|
| `.s56/snapshot.md` | WP0 快照、服务/环境取证、构建债 |
| `runs/prefill-lctx/s56/wp0base.json` | WP0 基线探针原始数据 |
| `runs/prefill-lctx/s56/ab131k.json` / `ab262k.json` | WP3 内核 A/B |
| `runs/prefill-lctx/s56/ab131k_fp16.json` | WP2 fp16 臂(负结果) |
| `runs/prefill-lctx/s56/{base,hip}/longctx.jsonl` | 服务级 A/B 原始数据 |
| `runs/prefill-lctx/s56/{hip2,hip262}/longctx.jsonl` | B 臂长曲线 + 262k 冒烟 |
| `runs/prefill-lctx/s56/parity_{base,hip,verdict}.json` | 贪心 parity 与判决 |
| `.s56/rocm_c_rebuild/{host_build.sh,build_in_docker.sh}` | 内核重编脚本 |
| `.s56/relaunch_s56.py` | 重启驱动(env 注入 + 健康等待) |
| `tools/dcu/s56_hip_prefill_ab.py` / `s56_greedy_parity.py` | 本专项工具 |
| 改动源码 | `triton_unified_attention.py`(派发+env)、`attnd_inner.hip`(桶+fp16+dtype 修)、`vllm/logger.py`(循环导入 fail-open 修) |
