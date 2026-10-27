# S33 WP2：gfx906 one-shot AR 复活 —— 停点报告（2026-10-26）

> 施工依据：`docs/design/z100-s33-fp8-scope-construction.md` §WP2（设计 D3，
> "复活+修复+实测，不是新写内核"）。
> **状态：分配修复成功、内核跑通、速度大赚（−3.4 ms/round），但贪心流
> 落基线世界外（硬门）→ 按 §2.3 停，交用户裁决。现役已回滚。**

## 1. 交付与修复

1. **分配根治（历史阻塞点）**：`VLLM_DCU_CUSTOM_AR_ALLOC=finegrained`
   （`csrc/custom_all_reduce.cu` 的 `hipExtMallocWithFlags` 旗标选择，前序
   已入库；gfx906/DTK 26.04 上默认 uncached 必 `hipErrorOutOfMemory`）。
   配合 `HSA_FORCE_FINE_GRAIN_PCIE=1`（env.sh 既有）——分配成功，
   `CustomAllreduce` 4-rank PCIe-only 组全 rank 初始化通过。
2. **size 门（本 WP 新增）**：`custom_all_reduce.py::should_custom_ar`
   dcu 分支加 `inp_size > 64 KiB → RCCL`（prefill 大载荷不受影响）；
   无效 algo 警告文案同步修正。
3. **臂脚本**：`.s33/launch_ctrim_ar.sh`（= launch_native_ctrim.sh 去掉
   `--disable-custom-all-reduce`，逐字其余不变）；`.s33/launch_ctrim_eager_ar.sh`
   （再加 `--enforce-eager`、去 compilation-config，用于隔离实验）。
   `.s33/restart.sh` 支持 `LAUNCH_SCRIPT=` 覆盖。回滚 = 不用这些脚本。
4. **指纹 KNOBS**：加 `VLLM_CUSTOM_ALLREDUCE_ALGO`（os.getenv 直读，
   envs=False）。
5. **微测工具**：`tools/dcu/car_bench.py` 加 `--rank-delay-us/--delay-rank`
   （skew 档注入，双实现对称施扰）。

## 2. 微测（torchrun 4 rank，fp16，200 样本；`.s33/wp2_car_*.json`）

| 载荷 | nccl p50 | oneshot p50 | 改善 | skew20 后 nccl / oneshot |
|---|---|---|---|---|
| 10 KiB | 154.9 µs | 119.2 µs | 23.0% | 182.2 / 157.9（13.4%） |
| 30 KiB | 125.1 µs | 96.5 µs | 22.9% | 199.5 / 123.5（38.1%） |
| 64 KiB | 168.0 µs | 99.8 µs | 40.6% | 181.8 / 116.3（36.0%） |

- 判据"无扰动 10 KiB ≤25 µs"：**口径错位**——bench（torchrun 独立进程、
  冷时钟）连 NCCL 自身都是 155 µs（S13 同款 bench 历史 186 µs 一致）；该
  25 µs 门按服务级口径写成，bench 上下文不可达。改以服务级判定（§3/§4）。
- skew 档 10 KiB 改善 13.4% < 40% 门 ✗（30/64 KiB 36–38% 边缘）——
  勘误①"等人的时间省不掉"在 10 KiB 档成立。
- 正确性：elementwise vs NCCL 3.9e-3（fp16 求和序差级，无 NaN/Inf）；
  位同 = False（AR 求和序与 RCCL 环不同，预期内，由贪心流门兜底）。

## 3. 服务级 A/B（ctrim 臂 + AR env，N=1）

- capture 完成无 hang；`disable_custom_all_reduce` 去除后 custom AR 真启用
  （"enabling on a 4-rank PCIe-only group"×4 rank）。
- e2e：**step p50 = 25.929 ms**（基线 29.331）→ **−3.40 ms（−11.6%）**，
  超 −2 ms 验收门、落在设计 −2.5~−4 预期带内；单流 29.6–29.8 t/s（+9%）。
  （速度与 bench 相对改善 × 128 calls/round 自洽。）

## 4. 贪心流硬门 —— 落世界外，两次独立复现

| 臂 | 流形态 | vs 基线两世界 |
|---|---|---|
| graph（部署目标形态） | `8283, 1023, 1023, …`（全程重复 1023，唯一 token 2 个，3 rep 位同） | **pos1 起确定性劣化**（prefill pos0 正常：走 RCCL）→ piecewise graph replay 的 registered-buffer 路径取到陈旧数据 |
| eager（隔离实验） | pos1 起 3 rep 三种流（`220/1710/220…`），**非确定性** | 内核/可见性竞态；隔离 bench（每次调用前同步）暴露不了 |

结论：**本 gfx906 + DTK 26.04 栈上，one-shot custom AR 内核的服务级数值是
坏的**——graph 路径确定性错、eager 路径非确定性错。候选根因（图内未立项
修复）：该核的 flag/data 先后序假设（为 MI300 级内存模型写）在本代
gfx906 fine-grain PCIe P2P 上不被兑现（缺 producer 侧 system 级 fence）；
S13 "custom AR 结构性受阻"的完整含义应包含此点。
速度收益（−3.4 ms）在数值正确前**不可采信**（重复 token 的生成不构成有效
工作负载）。

## 5. 处置

- 现役已回滚：canonical ctrim 臂（`bash .s30c/restart.sh <tag>`），
  AR 三 env 不设（默认全关）。
- WP3 依赖重排：**AR 核内 norm 融合（勘误③融合点）随 WP2 一并停**；
  WP3 的 gateup silu-mul epilogue 与 GDN CFG 扩展不依赖 AR，继续。
- 若用户后续想救 WP2：最小修复面 = 在 `cross_device_reduce_1stage` 的
  producer 侧加 `__threadfence_system()`（gfx906 条件编译），并核查
  graph replay 的 `register_graph_buffers` 路径；属"新写修复"范畴，
  需重新立项。

## 6. 产物

`.s33/wp2_car_oneshot.json`、`.s33/wp2_car_oneshot_skew20.json`、
`.s33/capture-s33wp2-ar2.json`（graph 劣化流）、
`.s33/capture-s33wp2-ear.json`（eager 非确定流）、
`.s30c/serve-s33wp2-ar2.log`、`.s30c/serve-s33wp2-ear.log`、
臂脚本 `.s33/launch_ctrim_ar.sh`、`.s33/launch_ctrim_eager_ar.sh`。
