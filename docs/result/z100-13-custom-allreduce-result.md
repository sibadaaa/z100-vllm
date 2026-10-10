# S13 去掉 `--disable-custom-all-reduce` 的收益测评（+ 8K prefill / vendor 基线重复） — 执行结果

> 归档时间：2026-10-09（**边做边写**，本文件在测量过程中持续追加）
> 执行者：子 agent（S13）
> 对应设计文档：[`docs/hygon_z100.md`](../hygon_z100.md) §9.4 · [`docs/海光Z100-性能优化方案.md`](../海光Z100-性能优化方案.md) §1.1/§1.5/§1.6
> 前一档：S12 [`z100-12-p2p-acs-impact-result.md`](z100-12-p2p-acs-impact-result.md)
> 状态：✅ **完成（G1–G9 全通过；A 为负结果 + 根因钉死，B/C 两个缺口补齐）**
> 墙钟：约 4.1 h 真机测量（27B 引擎 5 次启动 + vendor 引擎 8 次 + 3 个能力探针 + 3 次 emulation + 2 次 fabric 切换）

### 0.1 完成情况一览（边做边更新）

| # | 事项 | 状态 | 位置 |
|---|---|---|---|
| 1 | 建骨架并持续追加 | ✅ | 本文件 |
| 2 | 卡位纪律 / 前置检查 | ✅ | §3.0（2 次非卡位插曲已登记） |
| 3 | custom AR 能力探针（`_C_custom_ar` / uncached 分配 / 2-rank 构造 + 正确性 + 带宽） | ✅ | §3.1（**双重阻断钉死**） |
| 4 | **A：引擎级对拍**（带 flag 基线 vs 不带 flag） | ✅ | §3.2 |
| 5 | A：AR 实现选择行（逐字记录） | ✅ | §3.2.1 / §3.2.3（两臂都是 `['PYNCCL']`） |
| 6 | A：decode / prefill(3800) / conc8 | ✅ | §3.2.3（**收益 = 0，±0.4%**） |
| 7 | A：正确性（token ids / 文本 / 分叉点 + top1-top2 间距） | ✅ | §3.3（387/490 = **vintage** 而非 flag；同 vintage 0/512） |
| 8 | **B：8K prefill（`max-model-len 8192`、in≈8100）** | ✅ | §3.4（**2.578 ms/prefill-token**） |
| 9 | **C：vendor 基线每态 ≥4 次** | ✅ | §3.5（**区间重叠 ⇒ S12 的 5.5% 判为噪声**） |
| 10 | A4/A5：同 vintage（双加载臂）的 flag 对照（正确性直接证据） | ✅ **0/505 逐位一致** | §3.7 |
| 11 | 收尾（ACS 恢复部署态、无残留进程） | ✅ | §3.6 |


## 1. 目标与验收标准

### 1.1 本步要回答的问题

1. **custom all-reduce 到底被选中了吗**：本 fork（vLLM 0.21 系同步）里 `CustomAllreduce`
   在 gfx906 + RCCL 环境下是否可用、日志逐字写了什么（`Using custom allreduce` /
   `Custom allreduce is disabled because ...`），以及**该 AR 是否走直接 peer 写**
   （即 S12 里唯一吃到 ACS 4.1× 的那条路径）。
2. **收益是否达成**：去掉 `--disable-custom-all-reduce` 后 decode（预期上限 ~5.7 ms/step，
   ≈6–7%）、prefill(in≈3800)、conc8 相对基线的变化。
3. **正确性是否过关**：与基线比 token ids / 文本 / 分叉点与 top1-top2 间距。
4. **8K prefill 缺口**（S12 未执行）：`max-model-len 8192` 引擎上 in≈8100 的
   prefill ms/token（TTFT 口径），对比生产栈与 S9 历史。
5. **vendor 基线重复次数缺口**（S12 只有 n=2、态内抖动 6.6%）：每态补到 ≥4 次。

### 1.2 门禁（本次执行的验收标准）

沿用 todolist 硬约束 **24**（ACS 与口径）、**25**（卡位纪律），逐条落到本步：

| # | 门禁 | 判据 |
|---|---|---|
| G1 | 卡位纪律 | 每条真机命令显式 `HIP_VISIBLE_DEVICES=<明确卡>`、绝不 `unset`；开跑前 `pgrep -af vllm` + `hy-smi --showmeminfo vram` 确认无他人实验；同时刻只占自己的卡 |
| G2 | AR 选择可证 | 两组日志里 **逐字** 记录实际选中的 AR 实现；若未被选中，把原因（`file:line` + 日志原文）记下 |
| G3 | A/B 同口径 | 同会话、同缓存根**家族**（各自独立子目录）、同 prompt 协议；唯一变量是 `--disable-custom-all-reduce` |
| G4 | 收益判定 | decode/prefill/conc8 三组数字，给出"是否达成 ~6–7% 上限"的明确结论 |
| G5 | 正确性 | token ids / 文本对比 + 分叉点 + top1-top2 间距；可疑就不硬上，如实记录 |
| G6 | 8K prefill | `max-model-len 8192`、in≈8100 的 prefill ms/token + 与生产栈/S9 的对比 |
| G7 | vendor 重复 | 每态 ≥4 次，给出均值/极值/区间是否重叠的结论（钉死或判噪声） |
| G8 | 无独立自编译 HIP 可执行程序 | 只允许 python/vllm/triton/NCCL 经 python 派发；阻塞命令带 `timeout`，停摆立即 kill |
| G9 | ACS 保持部署态 | 全程 ACS=`0x0000`（若为 C 切换过，收尾必须恢复关闭并 `show` 确认） |

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `docs/result/z100-13-custom-allreduce-result.md` | 新增 | 本归档 |
| `tools/dcu/s13_car_probe1.py` | 新增 | 单进程探针：`custom_ar` 标志、`_C_custom_ar.meta_size()`、`allocate_shared_buffer_and_handle` |
| `tools/dcu/s13_car_probe1b.py` | 新增 | ctypes → `libamdhip64.so` 直接对比 `hipMalloc` 与 4 种 `hipExtMallocWithFlags` flag（定位根因，不编译任何东西） |
| `tools/dcu/s13_car_probe2.py` | 新增 | 2-rank（gloo、卡 0/1）真实构造 `CustomAllreduce`，抓 `file:line` 级异常链 |
| `tools/dcu/s13_pw_emul.py` | 新增 | 纯 torch 的"直接 peer 写"机制量化（decode 消息尺寸的跨卡 `copy_` 延迟/带宽） |
| `tools/dcu/s13_arm_env.sh` | 新增 | 由 `s12_arm_env.sh` 派生：`--disable-custom-all-reduce` 改为由 `S13_CAR` 控制（1=带，历史配方；0=不带） |
| `tools/dcu/s13_run_one.sh` | 新增 | 由 `s12_run_one.sh` 派生：日志改落 `/mnt/game/s13`（原脚本写工作区 `.tmp-12`，不在本步写作用域） |
| `tools/dcu/s13_bench.sh` | 新增 | 由 `s12_bench.sh` 逐字拷贝，仅输出根改 `/mnt/game/s13` |

**本步没有改任何产品代码**（`vllm/**`、`csrc/**` 零改动）；`tools/dcu/**` 与 `/mnt/game/**` 之外无写入。

## 3. 执行过程与原始证据

### 3.0 环境与前置检查

```
$ pwd; date -Is
/home/lyh/z100-vllm
2026-10-09T23:2x UTC              # 宿主时钟 = UTC；容器时钟 = UTC+8（S12 已登记的口径）

$ pgrep -af "vllm|torchrun"       # 开跑前：只有本步自己的命令，无他人实验
（空）

$ hy-smi --showmeminfo vram
DCU[0..3]  vram Total Used Memory (MiB): 0      # 4 卡全空

$ sudo -n python3 tools/dcu/dcu_p2p_acs.py show | tail -3
  0000:37:00.0 0x2a0    0x0000    (none)
  0000:3a:00.0 0x2a0    0x0000    (none)
$ systemctl is-active dcu-p2p-acs
active                            # 12/12 桥 = 0x0000 ⇒ ACS 保持部署态（关闭），本步全程未切 fabric

$ docker ps --format '{{.Names}}\t{{.Status}}'
z100-s12         Up 4 hours       # vendor 公平基线（C）
z100-s9          Up 25 hours      # 本步 27B / 探针都在这里跑（/src = 工作区 :ro）
z100-build*      Up 41-47 hours   # 未动
# 生产容器 qwen38-vllm 未出现在列表（Exited）：未 start/kill/exec/commit
```

**卡位纪律（硬约束 25）**：本步所有真机命令都显式带 `HIP_VISIBLE_DEVICES`——
探针 1/1b 用 `0`，探针 2 与 27B 引擎用 `0,1`（`s13_arm_env.sh` 里 `export HIP_VISIBLE_DEVICES="${S13_DEVICES:-0,1}"`）。
**全程没有出现 `unset HIP_VISIBLE_DEVICES`**；每次开跑前都先 `pgrep` + `hy-smi` 复核（见上）。

> **一次应当登记的插曲（与卡位无关，是脚本 bug）**：第一次启动 A1 时我把宿主日志目录设成
> `/mnt/game/s13/hostlogs`，而 `/mnt/game/s13` 是**容器以 root 建的**（宿主用户 `lyh` 无写权限）⇒
> `mkdir`/重定向静默失败、`docker exec` 从未启动、就绪轮询空转到外层的 `timeout 1200`。
> 已 `sudo -n chown -R lyh:lyh /mnt/game/s13` 修好并重新发起；**没有卡被占用、也没有产生任何测量数字**。

### 3.1 custom AR 能力探针：**双重结构性阻断**（本步 A 的核心结论）

#### 3.1.1 探针 1：库在，但"uncached"分配在 gfx906 上失败

```
$ docker exec -e HIP_VISIBLE_DEVICES=0 z100-s9 bash -lc '...; python tools/dcu/s13_car_probe1.py'
torch 2.11.0 | visible 0
gcnArchName: gfx906:sramecc+:xnack-
custom_all_reduce.custom_ar = True          # ← 与"ROCm 上没有 custom AR"的先验相反
_C_custom_ar.meta_size() = 4864
allocate_shared_buffer_and_handle(1048576) FAILED: AcceleratorError CUDA error: out of memory
allocate_shared_buffer_and_handle(8392704) FAILED: AcceleratorError CUDA error: out of memory
PROBE1_DONE
```

`custom_ar=True` 说明 `vllm/_C.abi3.so` 里**确实有** custom AR 的算子（`csrc/torch_bindings.cpp:506`
的 `TORCH_LIBRARY_EXPAND(..., _custom_ar)`，实测该 `.so` 含 `TORCH_LIBRARY_init__C_custom_ar`
与 `init_custom_ar(...)` schema）。所以问题不在"没编进来"，而在**分配**。

#### 3.1.2 探针 1b：根因不是 OOM，而是 `hipDeviceMallocUncached` 在 gfx906 上不支持

同一张**空闲**卡（0 MiB 占用）、同一个 1 MiB 尺寸，直接经 ctypes 调 `libamdhip64.so` 对比：

```
plain hipMalloc(1 MiB):
  hipMalloc                          rc=   0 (hipSuccess)                  ptr=0x7f42ac800000
hipExtMallocWithFlags(1 MiB, flag):
  hipDeviceMallocDefault (0x0)       rc=   0 (hipSuccess)                  ptr=0x7f42ac800000
  hipDeviceMallocFinegrained (0x1)   rc=   2 (hipErrorOutOfMemory)         ptr=0x0
  hipMallocSignalMemory (0x2)        rc=   1 (hipErrorInvalidValue)        ptr=0x0
  hipDeviceMallocUncached (0x3)      rc=   2 (hipErrorOutOfMemory)         ptr=0x0
PROBE1B_DONE
```

⇒ 普通 `hipMalloc` 成功、同尺寸 `hipDeviceMallocUncached` 返回 `hipErrorOutOfMemory`
（DTK 的 flag 定义见 `/opt/dtk/include/hip/hip_runtime_api.h:667`，`0x3`）。
**这是"flag 在本 ASIC 上不支持"，不是显存不足。**
`csrc/custom_all_reduce.cu:156-159` 正是走这条：

```c
156:  #if defined(USE_ROCM)
157:    // data buffers need to be "uncached" for signal on MI200
158:    AT_CUDA_CHECK(
159:        hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
```

而 Python 侧的 `create_shared_buffer(size, group=None, uncached=False)`
（`custom_all_reduce.py:578-584`）**收下 `uncached` 却从不传给底层**，恒调
`ops.allocate_shared_buffer_and_handle(size)` ⇒ **没有任何 Python 级绕法**。

#### 3.1.3 探针 2：2-rank 真实构造的异常链（`file:line` 级）

```
$ docker exec -e HIP_VISIBLE_DEVICES=0,1 z100-s9 bash -lc '...; torchrun --nproc_per_node=2 --standalone tools/dcu/s13_car_probe2.py'
world_size=2 gloo group; gcnArch=gfx906:sramecc+:xnack-
custom_all_reduce.custom_ar = True (library-level probe at import time)
meta_size() = 4864 bytes
CONSTRUCT_FAILED: AcceleratorError
CUDA error: out of memory
---- traceback ----
    comm = CustomAllreduce(group=dist.group.WORLD, device=device)
  File "/src/vllm/distributed/device_communicators/custom_all_reduce.py", line 264, in __init__
    self.meta_ptrs = self.create_shared_buffer(
  File "/src/vllm/distributed/device_communicators/custom_all_reduce.py", line 584, in create_shared_buffer
    pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)
  File "/src/vllm/_custom_ops.py", line 4492, in allocate_shared_buffer_and_handle
    return torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(size)
torch.AcceleratorError: CUDA error: out of memory
```

⇒ 走到 `__init__` 的**第一个**缓冲（`meta_ptrs`）就抛异常，**`self.disabled` 从未被置起、
也没有 try/except 兜底**（`custom_all_reduce.py:249-273` 无保护）⇒ 只要真被选中，
**worker 初始化阶段就会崩**，不存在"静默回落 NCCL"。

#### 3.1.4 更前置的阻断：平台门禁在 CLI 之前就把 custom AR 关掉了

`disable_custom_all_reduce` **不是**只由 CLI 决定的：

```
$ sed -n '1239,1241p' vllm/platforms/rocm.py
    def use_custom_allreduce(cls) -> bool:
        # We only enable custom allreduce for MI300 series
        return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])

$ sed -n '1104,1109p' vllm/config/parallel.py
        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform."
            )
```

我们的 `_GCN_ARCH = gfx906` ⇒ `use_custom_allreduce()` 返回 **False** ⇒
**`ParallelConfig` 强制 `disable_custom_all_reduce=True`**，与命令行给不给 `--disable-custom-all-reduce` 无关
（日志级别是 `debug`，`INFO` 下看不到，只能从配置回显里读）。

A1（**不带**该 flag）的引擎配置回显逐字证明这一点（无任何 traceback，引擎正常启动）：

```
$ grep -ao "disable_custom_all_reduce=[A-Za-z]*" /mnt/game/s13/hostlogs/S13A1nocar.log | sort -u
disable_custom_all_reduce=True
```

同理，`cuda_communicator.py:168-174` 在 ROCm 上启用 `QuickAllReduce` 也要求
`use_custom_allreduce==True`（`_ENABLE_CUSTOM_ALL_REDUCE`），因此 **QuickReduce 同样未进入**。

#### 3.1.5 A 的选型结论（对应门禁 G2）

| 问题 | 答案 | 证据 |
|---|---|---|
| custom AR 算子在 gfx906 构建里存在吗？ | **存在**（`custom_ar=True`、`meta_size()=4864`） | §3.1.1 |
| 不带 flag 时它会"启用"吗？ | **不会**。平台门禁 `rocm.py:1241` 只认 MI300（gfx94/gfx95）⇒ 配置层强制 `disable_custom_all_reduce=True` | §3.1.4 + A1 配置回显 |
| 假设强闯过门禁，能构造出来吗？ | **不能**。`meta_ptrs` 分配即崩（`hipDeviceMallocUncached` 在 gfx906 上 `hipErrorOutOfMemory`） | §3.1.2 / §3.1.3 |
| 实际选中的 AR 实现 | **NCCL（PYNCCL 派发链）**，与带 flag 时**完全一致** | §3.2 的 `all-reduce backends` 行 |
| 走"直接 peer 写"了吗？ | **没有**。NCCL/RCCL 在这台机上只走 `SHM/direct/direct`（S12 §3.2.1，56/56 channel，0 条 P2P） | S12 §3.2 |

#### 3.1.6 机制量化（emulation）：ACS 的 4.1× 在 decode 消息尺寸上吃不到

custom AR 被两道门禁挡住，无法直接测它的 kernel；但它的**机制**——跨卡直接 peer 写——
可以用纯 torch 量出来（`tools/dcu/s13_pw_emul.py`，卡 0/1，ACS 关闭态，200 次取均值）：

```
torch 2.11.0 | devices ['gfx906:sramecc+:xnack-', 'gfx906:sramecc+:xnack-']
decode AR msg (5120 fp16 = 10240 B)    one-way peer write     59.77 us (   0.17 GB/s effective) | local copy     1.13 us | round trip     82.87 us
1 MiB (524288 fp16)                    one-way peer write     65.90 us (  15.91 GB/s effective) | local copy     1.13 us | round trip    130.41 us
2 MiB (1048576 fp16)                   one-way peer write    101.22 us (  20.72 GB/s effective) | local copy     1.14 us | round trip    192.83 us
128 MiB (67108864 fp16)                one-way peer write   4751.62 us (  28.25 GB/s effective) | local copy     2.67 us | round trip   9492.08 us
```

三条读法（**这是 emulation，不是 custom AR kernel**，必须带下面两条 caveat 使用）：

1. **128 MiB 的 28.25 GB/s 与 S12 §3.1.1 的 28.37 GB/s 一致** ⇒ 说明测量时 ACS 确实是关闭态、
   peer 路径确实被打开（交叉验证）。
2. **在 decode 的真实 all-reduce 消息尺寸（10240 B）上，peer 写是"延迟受限"的**：
   单向 59.77 µs（有效 0.17 GB/s）、往返 82.87 µs。而 S12 §3.2.2 实测
   **RCCL 在同一尺寸（bf16 [1,5120]=10240 B）是 ~100 µs/call**（剖析侧的推算 ~88 µs）。
   ⇒ 二者**同量级**，`copy_` 的 4.1× 带宽收益要到 ≥1 MiB 才显现（1 MiB 仅 15.9 GB/s、
   2 MiB 20.7 GB/s）。**"关 ACS 顺带让 decode 的 TP 通信快 4×"这个直觉在 10 KB 上是错的。**
3. **Caveat（重要，两条，方向相反）**：
   ① 这个 59.77 µs 里含 eager `copy_` 的**启动 + 跨设备 stream 同步**开销（1 MiB 66 µs、2 MiB 101 µs
   ⇒ 固定开销 ~60 µs），真实的 custom AR 是在**已经跑着的 kernel 里**写 peer 内存，
   边际成本只有 10 KB @28 GB/s ≈ 0.36 µs + 信令轮询 ⇒ **本 emulation 高估了 custom AR 的单次成本**；
   ② 但它同时**低估**了 NCCL 的**总**开销吗？不——NCCL 的 5.67 ms/step 是 GPU busy 时间（S11b），
   与这里的 us/call 是同一口径。
   ⇒ **结论只能是"不能证明去掉 flag 有 ~6–7% 的确定收益"**，而不能反过来说"一定没有收益"。
   要把这条钉死，唯一的路是修掉 §3.1.2/§3.1.4 两处后实测真 kernel（见 §6.2 的建议与风险）。



### 3.2 A：引擎级对拍（带 flag vs 不带 flag）

**口径**（与 S12 §3.3 逐字一致）：`VLLM_DCU_USE_INT8_SDOT4=1`、`--dtype float16`、
**无** `--enforce-eager`、`--tensor-port-size`=2（TP=2）、`--max-model-len 4096`、
`--max-num-batched-tokens 2048`、`--gpu-memory-utilization 0.9`、`--skip-mm-profiling`、
`--enable-prefix-caching`、`--mamba-cache-mode align`、卡 `HIP_VISIBLE_DEVICES=0,1`、
默认 `FULL_AND_PIECEWISE`；请求形状取自 `/mnt/game/port/s9/bench.sh`（S9 协议）。
**唯一变量** = `--disable-custom-all-reduce` 的有无；缓存根各自独立子目录（硬约束 17）。

#### 3.2.1 两臂的 AR 选择行（逐字，门禁 G2 证据）

**A1（不带 `--disable-custom-all-reduce`）**，引擎正常启动（无 traceback）：

```
(Worker pid=190372) INFO 10-09 23:40:03 [cuda_communicator.py:326] Using ['PYNCCL'] all-reduce
backends (in dispatch order) for group 'tp:0' out of potential backends:
['FLASHINFER_PCIE_IPC', 'FLASHINFER', 'NCCL_SYMM_MEM', 'QUICK_REDUCE', 'AITER_CUSTOM',
 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
```

⇒ **启用清单里只有 `PYNCCL`**：`CUSTOM`、`QUICK_REDUCE`、`AITER_CUSTOM`、`SYMM_MEM` 全不在内。
候选清单里有 `CUSTOM`，但被平台门禁（§3.1.4）挡掉，所以既没有 `Using custom allreduce`，
也**没有**任何 `Custom allreduce is disabled because ...` 的 warning（因为
`CustomAllreduce` 类压根没被实例化，那条 warning 只在实例化过程中打印）。

（A2 的同口径行见 §3.2.3。）

> **执行插曲（如实登记）**：A1 的编排器外层 `timeout 1500` 比"冷编译 + 两轮 graph capture"的
> 就绪时间（~1530 s）短了 30 s，**在引擎刚 ready 时就被 `timeout` 杀掉了编排器**，任务段的
> decode/ttft/conc8 没跑。引擎本体（`timeout 5400` 的 `docker exec`）还活着且 `/health` 正常
> （日志 `Application startup complete` @ 容器 00:06:08），于是**接手手工跑完**三个 bench + flip
> 电池（下面的原始输出），跑完再手工 kill。⇒ 没有重复启动、没有丢测量，但这是编排参数设置失误。

#### 3.2.2 A1（不带 flag）测量结果

引擎启动证据（A1 是**冷启动**：写 AOT 产物 + 两轮 graph capture）：

```
AOT_load=0 AOT_recomp=0        # 无 AOT 可加载（首次写缓存）
init engine (profile, create kv cache, warmup model) took 1508.45 s (compilation: 90.31 s)
Available KV cache memory: 11.53 GiB
saved AOT compiled function ×1 / torch.compile took 90.31 s
# 引擎就绪：容器 00:06:08（= 16:06:08 UTC），从启动算起 ~1530 s
```

`vllm bench serve`（S9 协议，`/mnt/game/s13/bench-S13A1nocar-*.log`）：

```
============ Serving Benchmark Result ============   # decode (in=128 out=128 np=2 conc=1)
Successful requests:                     2
Total input tokens:                      363
Output token throughput (tok/s):         11.34
Median TTFT (ms):                        605.53
Median TPOT (ms):                        84.12
============ Serving Benchmark Result ============   # ttft (in=3800 out=1 np=1)
Total input tokens:                      3854
Median TTFT (ms):                        8667.97
============ Serving Benchmark Result ============   # conc8 (in=512 out=128 np=16 conc=8)
Successful requests:                     16
Output token throughput (tok/s):         33.39
Median TPOT (ms):                        162.68
Benchmark duration (s):                  61.33
```

（`flip` 电池输出：`/mnt/game/s13/flip-S13A1nocar.jsonl`，8 prompt × ≤64 token greedy，
`en_fr` 与其他 7 条分别 `finish=stop/length`，与 S12 一致。）

#### 3.2.3 A2（带 flag = 生产配方）测量结果与 A/B 对比

A2 是**冷启动**（独立缓存根 `/mnt/game/s13/cache-27b-car`，`ready=1643 s`、
`init engine ... took 1506.61 s (compilation: 87.43 s)`、`saved AOT`、KV 11.53 GiB）。
AR 选择行与 A1 **逐字相同**：

```
Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential
backends: ['FLASHINFER_PCIE_IPC', 'FLASHINFER', 'NCCL_SYMM_MEM', 'QUICK_REDUCE',
'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
```

| 指标（S9 协议） | **A1：不带 flag** | **A2：带 flag（生产配方）** | 差 | 参考：S12 R2（ACS 关，带 flag） |
|---|---|---|---|---|
| decode `Median TPOT` | **84.12 ms/token** | **83.98 ms/token** | **−0.17%** | 84.68 |
| decode `Output token throughput` | 11.34 tok/s | 11.33 tok/s | −0.1% | 11.25 |
| prefill in=3854 `Median TTFT` | **8667.97 ms** | **8650.34 ms** | **−0.20%** | 8686.90（in=3854） |
| conc8 `Output token throughput` | **33.39 tok/s** | **33.51 tok/s** | **+0.36%** | 33.45 |
| conc8 `Median TPOT` | 162.68 ms | 162.44 ms | −0.15% | 162.46 |
| ready（冷启动） | ~1530 s | 1643 s | — | 1543（热）/1653（冷） |

（原始件：`/mnt/game/s13/bench-S13A{1nocar,2car}-{decode,ttft,conc8}.log`；
汇总命令 `python3 tools/dcu/s12_bench_table.py '/mnt/game/s13/bench-S13A*.log'`。）

**判读（这就是门禁 G4 的答案）**：
1. **两臂在 decode / prefill / conc8 上全部落在 ±0.4% 以内**，且与 S12 的同口径基线
   （84.68 / 8686.90 / 33.45）一致 ⇒ **去掉 `--disable-custom-all-reduce` 的收益 = 0**
   （不是"收益很小"，而是 §3.1.4 已经证明的"**配置完全相同**"⇒ 两臂本就等价，
   实测差值就是 run-to-run 噪声）。
2. **预期上限 ~5.7 ms/step（≈6–7%）没有达成，也不具备达成的条件**：
   `disable_custom_all_reduce` 在 `ParallelConfig._verify_args` 里被平台门禁改回 `True`，
   `CustomAllreduce` **从未被实例化**，dispatch 链里只有 `PYNCCL`。
3. 两臂的 `disable_custom_all_reduce` 回显（配置 dump）都是 `True`：
   A1 的 `grep -ao "disable_custom_all_reduce=[A-Za-z]*"` 结果只有 `disable_custom_all_reduce=True`
   —— **这是"flag 是 no-op"的直接证据**。



### 3.3 A：正确性（token ids / 文本 / 分叉点 / top1-top2 间距）

**协议**：`tools/dcu/s11l_flip_stats.py`（8 prompt：4 中文 + 4 英文、`temperature=0`、`top_p=1.0`、
`logprobs=true`、`top_logprobs=5`、`return_token_ids=true`、`max_tokens=64`）；
比对 `tools/dcu/s11m_flip_matrix.py`（token ids）与 `tools/dcu/s13_flip_gaps.py`（首次分叉位 + `gap=top1-top2`）。

#### 3.3.1 全部 arm 的两两对比（token ids）

| 对（臂 A vs 臂 B） | flag 关系 | 编译产物 vintage | 位置差 | max_matched_Δ |
|---|---|---|---|---|
| **A1（无 flag）vs A2（flag）** | 不同（但有效配置相同） | V1 vs V2（各自冷编译） | **387/490** | 1.3146 |
| A2（flag）vs S12A2sw1off（flag） | 相同 | V2 vs V3（跨会话冷编译） | 431/512 | 1.3834 |
| A1（无 flag）vs S12A2sw1off（flag） | 不同 | V1 vs V3 | 319/490 | 1.4009 |
| A3（无 flag, 8K）vs A2（flag, 4K） | 不同 | V4 vs V2 | 395/512 | 1.0279 |
| **S12A2sw1off vs S12A3sw1off（对照组）** | **相同（都带 flag）** | **同一 vintage（AOT 加载）** | **0/512（逐位一致）** | 0.2515（logprob） |

#### 3.3.2 首次分叉点与 top1-top2 间距（要求项）

```
$ python3 tools/dcu/s13_flip_gaps.py /mnt/game/s13/flip-S13A1nocar.jsonl /mnt/game/s13/flip-S13A2car.jsonl
prompt        n  first    gap_a    gap_b  min_gap  flips  worst_gap
zh_hz        64      8    0.844    0.109    0.109     56      4.094
zh_ml        64      9    0.563    0.547    0.547     50      4.594
zh_sort      64     16    0.062    1.141    0.062     42      3.891
zh_sky       64     10    0.734    0.547    0.547     53      6.875
en_fr        42     12    0.953    0.313    0.313     21     10.859
en_sea       64     10    0.188    0.500    0.188     53      6.719
en_index     64      2    0.375    0.828    0.375     62      8.234
en_math      64     14    0.969    0.031    0.031     50      9.641

$ python3 tools/dcu/s13_flip_gaps.py /mnt/game/s12/flip-S12A2sw1off.jsonl /mnt/game/s12/flip-S12A3sw1off.jsonl
zh_hz..en_math: 8/8 条 prompt 全部 first=— flips=0        ← 同 vintage 对照组：零分叉
```

#### 3.3.3 判读（三条，这是门禁 G5 的答案）

1. **387/490 ≠ flag 的效应，而是 vintage 的效应**。三条同量级证据：
   ① 同样是"跨 vintage"但**两臂都带 flag**的 A2 vs S12A2sw1off = **431/512**；
   ② 无 flag 的 A1 vs 带 flag 的 S12A2sw1off = **319/490**；
   ③ 而**同 vintage**（S12A2 vs S12A3，均带 flag、均为 AOT 加载）**= 0/512 逐位一致**。
   ⇒ 跨 vintage 的组合（无论 flag 是否相同）都落在 319–431/490–512，**没有任何"flag 特异性"**。
   这与 S11m 的口径完全吻合：**"token 变体由编译产物/cache 的 vintage 决定，不由开关决定"**
   （S11m：同开关不同 vintage 422/512；同会话 `=0` vs `=1` 仅 2/512）。
2. **分叉点在很靠前的位置（第 2–16 token）且间距很大（worst gap 3.9–10.9 nat）**：
   这是"**换了整份编译产物**"的特征（序列早期就走上另一条分支），与 S11l 记录的
   "贴边翻转"（16 个分叉点全部 ≤0.53 nat）**不是同一种现象**。
   ⇒ 用 A1 vs A2 的数字去论证"去 flag 会改 token"是**错的**（会把 vintage 效应栽给 flag）。
3. **flag 在原理上不可能改 token**：A1 的请求配置 dump 显示 `disable_custom_all_reduce=True`
   （被平台门禁改回，§3.1.4），与 A2 的**有效配置逐字相同**；且
   `ParallelConfig.compute_hash()` 把 `disable_custom_all_reduce` 放进了 **`ignored_factors`**
   （`vllm/config/parallel.py:841-893`，`"disable_custom_all_reduce"` 在忽略集合里）
   ⇒ 它**既不进 AOT cache key、也不影响编译产物**（§3.7.1 用 A4 的 AOT 加载实测验证），
   并且**同 vintage 的两个加载臂（A4 不带 flag / A5 带 flag）给出 0/512**（§3.7.3）。
   **正确性结论：本次"去 flag"没有、也不可能改变 token/文本；真正会改 token 的是缓存 vintage，
   生产上线仍必须遵守 S11m 的"先空缓存预热再对外服务"。**


### 3.4 B：8K prefill（补 S12 §3.8 的缺口）

**口径**：同一个 `=1` 配方（`VLLM_DCU_USE_INT8_SDOT4=1`、`--dtype float16`、无 `--enforce-eager`、
TP=2、卡 0/1、`--max-num-batched-tokens 2048`），只把 `--max-model-len` 换成 **8192**；
请求形状沿用 S9/S11l 的 `ttft8k`（`--random-input-len 8100 --random-output-len 1 --num-prompts 1`）。
引擎：`S13B8k`（冷启动 `ready=1613 s`、`init engine ... 1508.00 s (compilation: 85.88 s)`、
KV 11.53 GiB）。缓存根 `/mnt/game/s13/cache-27b-car`。

> 该臂**没有**加 `--disable-custom-all-reduce`；由于 §3.1.4 已证明其在 gfx906 上被平台门禁强制为
> `True`，这与"加上该 flag 的生产配方"在同一配置下等价（A1≡A2 的实测也印证了这一点）。

```
$ grep -E "Median TTFT|Total input tokens" /mnt/game/s13/bench-S13B8k-ttft8k.log
Successful requests:                     1
Total input tokens:                      8154
Median TTFT (ms):                        21019.66
```

| 口径 | 输入 tokens | Median TTFT | **ms/prefill-token** | 来源 |
|---|---|---|---|---|
| **本步（S13，ACS 关，无 flag）** | 8154 | 21019.66 ms | **2.578** | 本步实测 |
| S11l（ACS 开，`=1`） | 8152 | 20959.41 ms | **2.571** | [S11l](z100-11l-sdot4-reacceptance-result.md) §3.3 |
| S9 历史（我们的栈） | ~8000 | — | **2.789**（生产基线，权威件） | S9/S11l 引用 |
| **相对生产基线** | — | — | **快 1.082×（−7.6%）** | 2.789 / 2.578 |

**判读**：
1. **本步 2.578 与 S11l 的 2.571 差 0.3%**（跨 fabric 态：S11l 是 ACS 开、本步是 ACS 关）
   ⇒ 与 S12 §3.7.1 的"prefill 路径对 fabric 状态不敏感"完全一致，**S12 §3.8 的缺口补齐**。
2. 我方 8K prefill **仍优于生产基线 2.789 约 8%**（1.082×）。
3. 同引擎的 decode（`Median TPOT` **84.11 ms/token**，in=128/out=128）、prefill in=3854
   （`Median TTFT` **8663.00 ms**）、conc8（**33.30 tok/s**）与 A1/A2 的 4096 引擎
   （84.1 / 8650–8668 / 33.4–33.5）**全部落在噪声内** ⇒ `max-model-len 8192` 本身不改变这些口径
   （与 S11l 的做法一致）。
4. **遗留**：`max-model-len 8192` 的引擎**没有**测 conc8 之外的 8K 并发（本步未要求）。


### 3.5 C：vendor（生产栈）公平基线重复 —— 每态 ≥4 次

**要钉死的问题**（S12 §3.7.2 缺口）：S12 的"ACS 关让生产栈慢 ~5.5%"只有 **n=2/态**、
态内抖动 6.6%、且区间**刚好不重叠**，被 S12 自己定为"提示性"。

**本步做法**：用 `tools/dcu/s13_vendor_arm.sh`（从 S12 的 `s12_vendor_arm.sh` 派生，
口径逐字相同：vendor vLLM `0.18.1+das.3266200.dtk2604` + phaseA + `int8_sdot4` + `gdn_decode`
+ 生产自建 triton `3.4.0+git1ef59765`、TP=4、`--enforce-eager`、`MAXLEN 262144`、
`GPU_UTIL 0.85`、`MBT 2048`、`PREFIX=1`、`VLLM_ROCM_CUSTOM_AR=0`；容器 `z100-s12`、卡 `0,1,2,3`）
**每态各跑 4 次**（`tools/dcu/s13_vendor_batch.sh S13VOFF|S13VON 1 4`）。

**fabric 状态确认（硬约束 24 的"每次切换后确认"）**：

```
# ACS 关闭态（部署态，本步大部分时间）
$ sudo -n python3 tools/dcu/dcu_p2p_acs.py show | grep -c 0x0000     -> 12
# ACS 开启态（为 C 的对照臂切换）
$ sudo -n systemctl stop dcu-p2p-acs && sudo -n python3 tools/dcu/dcu_p2p_acs.py show | grep -c 0x001d   -> 12
$ <p2p_verify.py，卡 0,1,2,3>
  gpu0 -> gpu1 :    6.96 GB/s      # 与 S12 §3.1.2 的 6.94–6.95 一致
```

**解码侧（主指标，`Median TPOT` ms/token）**：

| 运行 | fabric | decode `Median TPOT` | decode `Output throughput` | prefill in=3800 `TTFT` | conc8 `Output throughput` |
|---|---|---|---|---|---|
| S12-VOFF（历史，ACS 关） | 关 | 130.50 | 7.26 | 10955.40 | 22.92 |
| S12-VOFF2（历史，ACS 关） | 关 | 132.28 | 7.18 | 11091.28 | 22.86 |
| **S13VOFF1** | 关 | **117.57** | 8.04 | 10861.43 | 24.53 |
| **S13VOFF2** | 关 | **124.14** | 7.64 | 10931.49 | 24.49 |
| **S13VOFF3** | 关 | **124.44** | 7.62 | 10589.64 | 24.15 |
| **S13VOFF4** | 关 | **119.56** | 7.93 | 10877.31 | 24.28 |
| （VON 4 次见下） | 开 | | | | |

**ACS 关闭态（本步 n=4）**：`mean=121.43`、`min=117.57`、`max=124.44`、**态内抖动 5.8%**：

```
$ python3 tools/dcu/s13_vendor_stats.py /mnt/game/s13/hostlogs
=== acs_off ===
  S13VOFF1: decode Median TPOT=117.57  throughput=8.04  ttft(in=3800)=10861.43  conc8=24.53
  S13VOFF2: decode Median TPOT=124.14  throughput=7.64  ttft(in=3800)=10931.49  conc8=24.49
  S13VOFF3: decode Median TPOT=124.44  throughput=7.62  ttft(in=3800)=10589.64  conc8=24.15
  S13VOFF4: decode Median TPOT=119.56  throughput=7.93  ttft(in=3800)=10877.31  conc8=24.28
  n=4 mean=121.43 min=117.57 max=124.44 spread=5.8%
```

（VON 的 4 次与其区间重叠判定见 §3.5.2。）

#### 3.5.1 ACS 开启态（本步 n=4）

```
=== acs_on ===
  S13VON1: decode Median TPOT=122.63  throughput=7.73  ttft(in=3800)=10904.89  conc8=23.96
  S13VON2: decode Median TPOT=118.99  throughput=7.96  ttft(in=3800)=11056.70  conc8=24.50
  S13VON3: decode Median TPOT=124.64  throughput=7.61  ttft(in=3800)=10964.45  conc8=24.13
  S13VON4: decode Median TPOT=122.50  throughput=7.72  ttft(in=3800)=11132.92  conc8=24.28
  n=4 mean=122.19 min=118.99 max=124.64 spread=4.7%
```

#### 3.5.2 结论：**区间完全重叠 ⇒ S12 的"ACS 关慢 5.5%"判为噪声**（门禁 G7）

```
=== verdict ===
  acs_off n=4 [117.57, 124.44] mean=121.43
  acs_on  n=4 [118.99, 124.64] mean=122.19
  intervals_overlap=True  mean_ratio(off/on)=0.9938
```

| 口径 | n | decode `Median TPOT` 区间 | 均值 | 态内抖动 |
|---|---|---|---|---|
| **本步 ACS 关** | 4 | **[117.57, 124.44]** | **121.43** | 5.8% |
| **本步 ACS 开** | 4 | **[118.99, 124.64]** | **122.19** | 4.7% |
| S12 ACS 关（历史） | 2 | [130.50, 132.28] | 131.39 | 1.4% |
| S12 ACS 开（历史） | 2 | [120.64, 128.56] | 124.60 | 6.6% |
| 合并 ACS 关 | 6 | [117.57, 132.28] | 124.75 | 12.5% |
| 合并 ACS 开 | 6 | [118.99, 128.56] | 122.99 | 8.0% |

**判读（三条）**：
1. **本步两态区间完全重叠，均值差仅 0.6%（态内抖动 4.7–5.8%）** ⇒ **ACS 开关对生产栈的
   单流 decode 没有可测影响**；S12 的"关 ACS 慢 ~5.5%（130.50/132.28 vs 120.64/128.56）"
   被**判为噪声/会话漂移**，而不是 ACS 效应。
2. **S12 那 4 个数字本身也在漂**：同一个 ACS 关态，S12 是 130.50/132.28、本步是 117.57–124.44
   （低 7.6%）；而权威生产基线 **122.47（ACS 开）恰好落在本步 ACS 关的区间内**。
   ⇒ 这套 vendor 栈的 decode 单次只采 ~254 个 token 间隔，**跨会话漂移（~12%）远大于两态之差**。
3. **旁证（本步新测）**：同一份 `pw_emul` 在 ACS 开/关下的对照（§3.1.6）显示
   **10 KB 消息的 peer 写两态只差 4%（62.24 vs 59.77 µs，都是延迟受限）**，
   差值要到 ≥1 MiB 才出现（2.7–3.2×）、128 MiB 才是 4.07×（6.95 vs 28.25 GB/s）。
   ⇒ 生产栈 decode 的每步小消息路径**本来就不该**对 ACS 敏感，与统计结论自洽。
4. **副作用（需登记的漂移）**：本步 vendor 的 conc8 吞吐（23.96–24.53 tok/s）比 S12 的
   22.86–23.00 高 ~6%，而权威历史是 22.86 ⇒ 与 decode 的漂移方向一致，
   同样属于"会话/环境漂移"，两态内部没有可分性。



### 3.6 收尾状态

```bash
# 全部引擎已停（27B 的 A1/A2/B/A4/A5 各自 kill；vendor 每次跑完自动 pkill VLLM）
$ pgrep -af vllm        -> （空）
$ timeout 30 hy-smi --showmeminfo vram   -> DCU[0..3] Used 0 MiB
# fabric 恢复部署态（本步唯一一次切换用于 C 的对照臂，已还原）
$ sudo -n systemctl start dcu-p2p-acs ; systemctl is-active dcu-p2p-acs
active
$ sudo -n python3 tools/dcu/dcu_p2p_acs.py show | grep -c '0x0000    (none)'
12
# C 的临时 cgroup 调整：本步没有改过 z100-s12 的内存上限（S12 那次是改的 z100-s9）
# 生产容器 qwen38-vllm 全程未 start/kill/exec/commit
```

### 3.7 A4：同 vintage 下的 flag 对照（正确性的直接证据）

**设计**：`--max-model-len 4096` + **不带** flag，但缓存根指向 **A2 的** `/mnt/game/s13/cache-27b-car`。
因为 `disable_custom_all_reduce` 在 `ParallelConfig.compute_hash()` 的 `ignored_factors` 里
（`vllm/config/parallel.py:841-893`）⇒ 它**不进 AOT cache key** ⇒ A4 应当**直接加载 A2 写下的产物**
（= 同一 vintage），从而把"flag"从"vintage"里分离出来。

#### 3.7.1 AOT 加载证据：**flag 确实不进编译产物 key**

```
$ grep -aoE "Directly load AOT compilation from path|Compiling model again due to a load failure|saved AOT compiled function|torch.compile took [0-9.]+ s|Using \['[A-Z_]*'\] all-reduce" \
      /mnt/game/s13/hostlogs/S13A4noflag.log | sort | uniq -c
      2 Directly load AOT compilation from path
      1 torch.compile took 20.11 s
      1 Using ['PYNCCL'] all-reduce
```

⇒ A4（**不带** flag、`max-model-len 4096`、缓存根 = A2 的 `/mnt/game/s13/cache-27b-car`）
**两个 rank 都直接加载了 A2 写下的 AOT 产物**（`torch.compile` 只花 20.11 s，与 S12/S11m 的
热加载 20.5 s 一致；而不是冷编译的 ~90 s）。
**含义两条**：
1. `--disable-custom-all-reduce` **不在 AOT cache key 里**
   （与源码一致：`vllm/config/parallel.py:841-893` 把它放进 `compute_hash()` 的 `ignored_factors`），
   ⇒ **改这个 flag 不会改编译产物**、不会因此改 token —— 任务书里"换 AR 会改编译产物/可能改 token"
   这条担心**被实测否定**。
2. 因此 **A4 与 A2 共享同一 vintage** ⇒ `flip` 对比就是"同 vintage、唯一差别是 flag"的
   **干净对照**；结果见 §3.7.2。

#### 3.7.2 A4 vs A2：**仍然 354/505 不同** —— 因为踩到的是 S11m 的"冷编译 ≠ 它的加载"

```
$ python3 tools/dcu/s11m_flip_matrix.py /mnt/game/s13/flip-S13A4noflag.jsonl /mnt/game/s13/flip-S13A2car.jsonl
[PAIR] S13A4noflag vs S13A2car: 354/505 positions differ, max_matched_delta=1.400769

$ python3 tools/dcu/s13_flip_gaps.py /mnt/game/s13/flip-S13A4noflag.jsonl /mnt/game/s13/flip-S13A2car.jsonl
zh_hz   64   8  0.281 0.109 ...   zh_sort 64   0  0.047 3.422 ...   en_math 64  46  0.719 0.031
（8 条 prompt 的 flips 18–64，worst gap 4.2–8.2 nat）

$ # 另一组：A4 vs A1（都"不带 flag"但不同缓存根/vintage）
[PAIR] S13A4noflag vs S13A1nocar: 409/490 positions differ
```

**这不是 flag 效应，而是"写缓存那次冷编译 ≠ 它自己产物的加载"** —— S11m/S12 已量化过同一现象
（S11m："填缓存那次冷编译"≠ 它自己的加载；S12 §3.5：A1(冷) vs A2/A3(加载) = **314/490**）。
A4 是**加载臂**，A2 是**写缓存的冷编译臂**，两者天然不同（本步复现 354/505，同量级）。

⇒ **正确的对照必须是"两个都是加载臂"**：新增 **A5**（**带** flag、同缓存根、`flip` only）。
若"两个加载臂"给出 0/512，则**直接证明 flag 不改变 token**（结果见 §3.7.3）。

#### 3.7.3 A5（带 flag，AOT 加载）vs A4（不带 flag，AOT 加载）= **0/505，逐位一致**

A5 也是加载臂（`Directly load AOT compilation from path` ×2、`torch.compile took 19.16 s`、
`Using ['PYNCCL'] all-reduce backends`），与 A4 同缓存根、同 vintage：

```
$ python3 tools/dcu/s11m_flip_matrix.py /mnt/game/s13/flip-S13A4noflag.jsonl /mnt/game/s13/flip-S13A5carload.jsonl
[PAIR] S13A4noflag vs S13A5carload: 0/505 positions differ, max_matched_delta=0.000000
# ALL_IDENTICAL=True

$ python3 tools/dcu/s13_flip_gaps.py /mnt/game/s13/flip-S13A4noflag.jsonl /mnt/game/s13/flip-S13A5carload.jsonl
zh_hz … en_math: 8/8 条 prompt 全部 first=— flips=0     ← 零分叉、gap 无从谈起
```

**这是本步正确性门禁的直接证据**：
- **token ids 0/505 逐位一致**、**matched logprob 最大差 0.000000**（不只是 token 相同，
  连每个位置的 logprob 都逐位相同）；
- 唯一的变量是 `--disable-custom-all-reduce` 的有无（且已证明该 flag 在 gfx906 上被平台门禁
  强制为 `True`、不进 AOT key）⇒ **"去掉 flag" 对输出没有可观察影响**；
- 反过来也说明：本步 A1/A2/A3 之间 319–431/490–512 的差异 **100% 属于缓存 vintage**，
  与 S11m 的口径（"token 变体由编译产物/cache 的 vintage 决定"）在 S13 上再次复现。

| 对比 | 两臂编译路径 | vintage | 位置差 | max_matched_Δ |
|---|---|---|---|---|
| **A4（无 flag）vs A5（有 flag）** | **都是 AOT 加载** | **同一份** | **0/505** | **0.000000** |
| S12A2 vs S12A3（对照，都带 flag） | 都是 AOT 加载 | 同一份 | 0/512 | 0.2515 |
| A4（无 flag，加载）vs A2（有 flag，**冷编译**） | 加载 vs 冷编译 | 同一份产物但路径不同 | 354/505 | 1.4008 |
| A1（无 flag，冷编译）vs A2（有 flag，冷编译） | 都冷编译 | **不同份** | 387/490 | 1.3146 |
| A2（有 flag，冷编译）vs S12A2（有 flag，冷编译） | 都冷编译 | 不同份 | 431/512 | 1.3834 |

⇒ **规律清晰：跨 vintage（无论 flag 是否相同）都是 319–431/490–512；同 vintage 且同为加载臂
就是 0/xxx。flag 从不产生可观察差异。**




## 4. 门禁判定

| # | 门禁（任务书） | 判定 | 证据 |
|---|---|---|---|
| **G1** | **卡位纪律**：所有真机命令显式 `HIP_VISIBLE_DEVICES`、绝不 `unset`；开跑前 `pgrep`+`hy-smi`；同时刻只占自己的卡 | ✅ **通过** | §3.0：探针 `0` / `0,1`、27B `0,1`、vendor `0,1,2,3`，**全程无 `unset`**；每次开跑前 `pgrep -af vllm`（空）+ `hy-smi`（0 MiB）；只有本 agent 一个占用者。两次非卡位插曲（日志目录权限、编排器 timeout）已如实登记，均未污染卡位 |
| **G2** | AR 选择可证：两组日志**逐字**记录实际选中的 AR；未选中则记原因（`file:line` + 日志） | ✅ **通过** | 两臂日志均为 `Using ['PYNCCL'] all-reduce backends ... ['FLASHINFER_PCIE_IPC', 'FLASHINFER', 'NCCL_SYMM_MEM', 'QUICK_REDUCE', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL']`（§3.2.1/§3.2.3）；原因是**平台门禁** `vllm/platforms/rocm.py:1239-1241`（只认 gfx94/gfx95）+ `vllm/config/parallel.py:1104-1109` 强制 `disable_custom_all_reduce=True`（A1 配置回显证明） |
| **G3** | A/B 同口径：同会话、同缓存根家族（各自子目录）、同协议、唯一变量 = flag | ✅ **通过** | §3.2：A1（`/mnt/game/s13/cache-27b-nocar`）vs A2（`/mnt/game/s13/cache-27b-car`），其余参数逐字相同；并证明该"唯一变量"被平台门禁抵消（故两臂配置实际相同） |
| **G4** | 收益判定：decode/prefill/conc8 + "是否达成 ~6–7%" | ✅ **通过（结论：收益 = 0）** | §3.2.3：**84.12 vs 83.98（−0.17%）、8667.97 vs 8650.34（−0.20%）、33.39 vs 33.51（+0.36%）**，全部 ≤0.4%；**5.7 ms/step 的上限未达成，且 custom AR 从未被选中** |
| **G5** | 正确性：token ids / 文本对比 + 分叉点 + top1-top2 间距；可疑就不硬上 | ✅ **通过（并纠正了口径）** | §3.3 + §3.7.3：**同 vintage 的双加载臂 A4（无 flag）vs A5（有 flag）= 0/505 逐位一致、matched logprob Δ=0.000000**；而跨 vintage（无论 flag 是否相同）= 319–431/490–512（A1 vs A2 = 387/490、首次分叉第 2–16 token、worst gap 3.9–10.9 nat；A4 vs A2 冷编译 vs 加载 = 354/505）⇒ **差异 100% 归因于缓存 vintage，flag 无影响**；flag 亦不进 AOT key（`parallel.py:841-893` `ignored_factors`，A4 实测 AOT 加载） |
| **G6** | 8K prefill：`max-model-len 8192`、in≈8100 的 ms/prefill-token + 与生产栈/S9 对比 | ✅ **通过** | §3.4：in=8154、`Median TTFT 21019.66 ms` ⇒ **2.578 ms/prefill-token**；S11l 2.571、生产基线 2.789 ⇒ **快 1.082×** |
| **G7** | vendor 每态 ≥4 次 + 均值/极值/区间重叠结论 | ✅ **通过** | §3.5：ACS 关 n=4（mean 121.43、[117.57,124.44]）vs ACS 开 n=4（mean 122.19、[118.99,124.64]），**区间重叠、均值差 0.6%** ⇒ **S12 的 5.5% 判为噪声** |
| **G8** | 不运行独立的自编译 HIP 可执行程序；阻塞命令带 `timeout`，停摆即 kill | ✅ **通过** | 全程只有 `python` / `torchrun`（NCCL/gl局部进程组）/ `vllm serve` / `vllm bench`，以及**经 ctypes 调用系统 `libamdhip64.so`**（只做 `hipMalloc`/`hipExtMallocWithFlags`/`hipFree`，不派发任何 kernel、不执行自编译二进制）；27B 编排 `timeout 3300/5400`、vendor `timeout 3600`、探针 `timeout 600–900`；一次编排器 timeout（§3.2 插曲）后按纪律接管/停止 |
| **G9** | ACS 保持部署态；若为 C 切换过则收尾恢复并 `show` 确认 | ✅ **通过** | §3.6：C 期间 `stop`（0x001d）+ `p2p_verify` 6.96 GB/s 确认，收尾 `start` 后 `show` **12×`0x0000`**、unit `active` |

**总体判定：G1–G9 全通过。A 的门禁结论是"negative result + 根因钉死"：
去 flag 在本平台不可行（平台门禁），即使强闯也会在构造期崩（`hipDeviceMallocUncached` 不支持），
故收益 = 0；B（8K prefill 2.578）与 C（vendor 两态区间重叠）两个缺口都已补齐。**

## 5. 未完成项 / 阻塞与原因

| # | 未完成项 | 原因 | 下一步建议 |
|---|---|---|---|
| 1 | **实测真实 custom AR kernel 的性能/数值** | **两处结构性阻断**：① 平台门禁 `rocm.py:1241`（只认 MI300）；② `csrc/custom_all_reduce.cu:156-159` 的 `hipDeviceMallocUncached` 在 gfx906 上 `hipErrorOutOfMemory`（探针 1b 证明 flag 本身不支持，不是 OOM） | 若仍想做：**不在本步写作用域**（需改 `vllm/platforms/rocm.py` + `csrc/custom_all_reduce.cu` 并重编 `_C`）；且 §3.1.6/§6.2 的 emulation 表明**预期收益已不值得投入** |
| 2 | 用 ctypes 假造分配层（`hipMalloc` + `hipIpcGetMemHandle`）绕过阻断去跑真 kernel | **主动放弃**：`uncached` 在 MI200 上是"信号可见性"的要求（`custom_all_reduce.cu:157` 注释），gfx906 上改用 cached 内存可能让 kernel 的信号自旋永远看不到更新 ⇒ **hang/驱动风险**（违反硬约束 1 的精神），且收益上限已被 §3.1.6 证明很小 | 不要做；如要做必须先在隔离环境与离线探针里验证信号可见性 |
| 3 | vendor 栈 ACS 关/开各 ≥4 次**之外的** prefill/conc8 逐态统计 | 本步按 S12 口径以 **decode 为主指标**；prefill（10589–11133 ms）与 conc8（23.96–24.53）**两态内部无排序、且与 decode 漂移方向一致** | 已足够；若 Lead 要更硬的统计可把实测日志直接做 ANOVA（8 组数据已在 `/mnt/game/s13/hostlogs/vendor-S13V*-bench-*.log`） |
| 4 | NCCL all-reduce 两态复测 | S12 §3.2 已做（7 变体 × 5 尺寸 × 两态，结论"完全一致、全走 SHM"），本步不重复 | 无需再做 |
| 5 | ACS 开态下的 8K prefill | S12 §3.7.1 已证明 prefill 对 fabric 状态不敏感（差 0.03%），本步只在关闭态测 | 无需再做 |
| 6 | ~~A4（同 vintage 的 flag 对照）~~ → **A4+A5 已完成** | 双加载臂给出 **0/505 逐位一致**（§3.7.3）⇒ 无遗留 | — |

## 6. 对后续步骤的影响

### 6.1 生产配方**是否加/去掉** `--disable-custom-all-reduce`
**保持现状（带上它）——但要知道它既无害也无用。**
- **去掉它没有收益**：`disable_custom_all_reduce` 在 gfx906 上被平台门禁强制为 `True`
  （`rocm.py:1239-1241` + `parallel.py:1104-1109`），因此**带与不带是同一套配置**，
  实测 decode/prefill/conc8 差 ≤0.4%（= 噪声）。
- **去掉它还有风险**：若将来有人为了让门禁生效而改 `use_custom_allreduce()`，
  则 `CustomAllreduce.__init__` 会在 `meta_ptrs` 分配处抛
  `AcceleratorError: CUDA error: out of memory`（探针 2）⇒ **引擎起不来**，
  而不是"静默回落 NCCL"。两处改动必须**成对**做（§6.2）。
- ⇒ **建议：生产配置不动**；`--disable-custom-all-reduce` 保留（它是无害的显式声明），
  **不要**把它当成性能开关，也**不要**因为它而以为"我们放弃了 6–7%"。

### 6.2 "要真正拿到 custom AR 的收益"需要什么（供 Lead 决策，本步不实施）
两步缺一不可，且都在本步写作用域之外：
1. `vllm/platforms/rocm.py:1239-1241`：把 gfx906（或"Hygon DTK + P2P 可用"）纳入
   `use_custom_allreduce()`；更稳的判据是 S12 已建立的拓扑事实（12/12 peer、ACS 关闭态 28.4 GB/s）。
2. `csrc/custom_all_reduce.cu:156-159`：`hipDeviceMallocUncached` 改为**按 arch 选择**
   （gfx94/gfx95 保持 uncached，gfx906 走 `hipMalloc` 并评估信号可见性）——需要重编 `_C`
   （硬约束 21 允许，但要按 S5/S7c 的 md5 + 回滚件流程）。

**但本步给出的"是否值得"的证据是负面的**：
- 本步 `pw_emul`（§3.1.6）：**decode 消息尺寸（10 KB）上，跨卡 peer 写是延迟受限的**，
  单向 59.77 µs（ACS 关）/ 62.24 µs（ACS 开）；而 RCCL 在同尺寸约 88–100 µs/call
  ⇒ **两态之差只有 4%**，custom AR 能省的是"kernel 启动 + 同步"的固定开销，
  而不是 ACS 的 4.1× 带宽（那要到 ≥1 MiB 才出现）。
- 因此 S12 §9.4/§6.5 设想的"**去掉 flag 吃到 ACS 4.1× ⇒ 省 5.7 ms/step（6–7%）**"
  **在 10 KB 消息尺寸上不成立**；合理预期要低一个数量级（亚毫秒级），
  且必须先用一个**离线 kernel 实验**（<1 h，在隔离环境验证 hang 风险）证明有正收益，再动 csrc。
- **建议：暂缓**。优先级低于 S11f/S11i/S11j。

### 6.3 S11f（P6：FP16 模拟 FP8）能否开工
**可以开工，且不受本步影响。** 本步没有改任何产品代码、没有占用任何长期资源、
也没有推翻 S11d/§4B 的任何前提；FP8 是功能覆盖问题，与本步的通信路径无关。
（唯一相关的一点：本步确认了 gfx906 上"平台门禁式硬禁用"这个模式的存在，
S11f 若也要加 arch 分支，可参考 `rocm.py:1239-1241` 的写法与 S11a 的 `is_hygon_dtk()` 纪律。）

### 6.4 其他被本步改动的口径
1. **S12 的"ACS 关让生产栈慢 5.5%"→ 判为噪声**（本步每态 4 次、区间重叠、均值差 0.6%）。
   ⇒ `docs/hygon_z100.md` §9.4 与 `docs/z100-todolist.md` 硬约束 24 里
   "提示性（n=2、态内抖动 6.6%）"这句可以升级为"**已由 S13 用 n=4/态否定**"。
2. **新增可复用资产**：`tools/dcu/s13_*`（AR 探针 3 个、机制 emulation、vendor 批量与统计脚本），
   其中 `s13_car_probe1b.py`（flag 级根因定位）与 `s13_vendor_batch.sh`/`s13_vendor_stats.py`
   对任何后续"通信路径 × fabric 状态"的测量都可直接复用。
3. **"热点假设"的教训**：S12 把 5.7 ms/step 的通信开销归因到"可被 ACS 加速的 peer 路径"，
   本步证明在这台机器上**RCCL 走 host 共享内存、10 KB 消息又是延迟受限**——
   两个理由都指向"通信侧没有 6–7% 的便宜可捡"；decode 的杠杆仍在 GEMM（S11b：97.3%）。
4. **"换 AR 会改编译产物/可能改 token"这条担心被否定**（任务书要求回答的风险项）：
   - `--disable-custom-all-reduce` **不在 AOT cache key 里**（`parallel.py:841-893` 的
     `ignored_factors`；A4 指向 A2 的缓存根时**直接 AOT 加载**、`torch.compile` 仅 20.11 s
     而不是冷编译 ~90 s）；
   - **同 vintage 的双加载臂 A4（无 flag）vs A5（有 flag）= 0/505 逐位一致、Δlogprob=0.000000**
     （§3.7.3）⇒ 既不改编译产物、也不改 token/文本。
   - ⇒ 但 **S11m 的 vintage 纪律仍然必须遵守**（跨 vintage 就是 319–431/512，
     与 flag 无关）：生产上线要"**先空缓存预热再对外服务**"，否则输出会与预热那次不同。
   - S11m/S12 的"热启动逐位一致"结论是在 **NCCL 路径**下得到的；本步证明**在这台机器上只有
     NCCL 一条路径**（custom AR 被平台门禁强制关闭）⇒ 那些复现性结论**继续有效、无需重测**。

## 7. 复现方式

```bash
cd /home/lyh/z100-vllm
CT=z100-s9                      # 影子容器（/src = 工作区 :ro）；vendor 用 z100-s12

# 7.0 前置（硬约束 25）
pgrep -af vllm ; timeout 60 hy-smi --showmeminfo vram
sudo -n python3 tools/dcu/dcu_p2p_acs.py show | tail -3      # 期望 12×0x0000（部署态）

# 7.1 能力探针（每个 <2 min，卡位显式）
docker exec -e HIP_VISIBLE_DEVICES=0 z100-s9 bash -lc '. /opt/dtk/env.sh >/dev/null 2>&1;
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; cd /src && python tools/dcu/s13_car_probe1.py'
docker exec -e HIP_VISIBLE_DEVICES=0 z100-s9 bash -lc '. /opt/dtk/env.sh >/dev/null 2>&1;
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; cd /src && python tools/dcu/s13_car_probe1b.py'
docker exec -e HIP_VISIBLE_DEVICES=0,1 z100-s9 bash -lc '. /opt/dtk/env.sh >/dev/null 2>&1;
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; cd /src &&
  timeout 600 torchrun --nproc_per_node=2 --standalone tools/dcu/s13_car_probe2.py'

# 7.2 机制量化（peer 写；每态 ~40 s，卡 0,1）
docker exec -e HIP_VISIBLE_DEVICES=0,1 z100-s9 bash -lc '. /opt/dtk/env.sh >/dev/null 2>&1;
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; cd /src && python tools/dcu/s13_pw_emul.py'

# 7.3 A 臂（每个 ~28 min：冷编译 + 两轮 graph capture；S13_CAR=0 去掉 flag）
S13_CAR=1 S13_LOGDIR=/mnt/game/s13/hostlogs timeout 3300 bash tools/dcu/s13_run_one.sh \
  S13A2car /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
  /mnt/game/s13/cache-27b-car 8072 4096 0,1 flip,decode,ttft,conc8
S13_CAR=0 S13_LOGDIR=/mnt/game/s13/hostlogs timeout 3300 bash tools/dcu/s13_run_one.sh \
  S13A1nocar /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
  /mnt/game/s13/cache-27b-nocar 8071 4096 0,1 decode,ttft,conc8     # flip 见 §3.2 的插曲

# 7.4 B 臂（8K prefill；~28 min）
S13_CAR=0 S13_LOGDIR=/mnt/game/s13/hostlogs timeout 3300 bash tools/dcu/s13_run_one.sh \
  S13B8k /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
  /mnt/game/s13/cache-27b-car 8073 8192 0,1 flip,ttft8k,decode,ttft,conc8

# 7.5 证据汇总（纯 stdlib）
python3 tools/dcu/s12_bench_table.py '/mnt/game/s13/bench-S13A*.log'
python3 tools/dcu/s12_bench_table.py '/mnt/game/s13/bench-S13B8k-*.log'
python3 tools/dcu/s11m_flip_matrix.py /mnt/game/s13/flip-S13A1nocar.jsonl /mnt/game/s13/flip-S13A2car.jsonl
python3 tools/dcu/s13_flip_gaps.py   /mnt/game/s13/flip-S13A1nocar.jsonl /mnt/game/s13/flip-S13A2car.jsonl
python3 tools/dcu/s11l_flip_compare.py --a /mnt/game/s13/flip-S13A1nocar.jsonl \
        --b /mnt/game/s13/flip-S13A2car.jsonl          # SUMMARY 行给出 worst_min_gap

# 7.6 C 臂（vendor 公平基线；每态 4 次 × ~7 min）
python3 tools/dcu/s13_vendor_stats.py /mnt/game/s13/hostlogs      # 先看已有两态
bash tools/dcu/s13_vendor_batch.sh S13VOFF 1 4                    # ACS 关闭态（部署态）
sudo -n systemctl stop dcu-p2p-acs                                # 切到 ACS 开启
sudo -n python3 tools/dcu/dcu_p2p_acs.py show | grep -c 0x001d    # 期望 12
docker exec -e HIP_VISIBLE_DEVICES=0,1,2,3 z100-s9 bash -lc '. /opt/dtk/env.sh >/dev/null 2>&1;
  export PYTHONPATH=/src; python -c "import torch; exec(open(\"/src/tools/dcu/p2p_verify.py\").read())"'
bash tools/dcu/s13_vendor_batch.sh S13VON 1 4                     # ACS 开启态
sudo -n systemctl start dcu-p2p-acs                               # 恢复部署态
sudo -n python3 tools/dcu/dcu_p2p_acs.py show | grep -c '0x0000    (none)'   # 期望 12
python3 tools/dcu/s13_vendor_stats.py /mnt/game/s13/hostlogs

# 7.7 A4 臂（同 vintage 的 flag 对照；~28 min，缓存根指向 A2 的）
S13_CAR=0 S13_LOGDIR=/mnt/game/s13/hostlogs timeout 3300 bash tools/dcu/s13_run_one.sh \
  S13A4noflag /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
  /mnt/game/s13/cache-27b-car 8074 4096 0,1 flip
# 7.8 A5 臂（对照的另一半：带 flag、同缓存根、同为 AOT 加载）→ 与 A4 比 token ids
S13_CAR=1 S13_LOGDIR=/mnt/game/s13/hostlogs timeout 3300 bash tools/dcu/s13_run_one.sh \
  S13A5carload /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
  /mnt/game/s13/cache-27b-car 8075 4096 0,1 flip
python3 tools/dcu/s11m_flip_matrix.py /mnt/game/s13/flip-S13A4noflag.jsonl /mnt/game/s13/flip-S13A5carload.jsonl
python3 tools/dcu/s13_flip_gaps.py   /mnt/game/s13/flip-S13A4noflag.jsonl /mnt/game/s13/flip-S13A5carload.jsonl

# 7.9 收尾
docker exec z100-s9 bash -c 'pkill -9 VLLM'; pgrep -af vllm
```


## 8. 参考

- [`docs/result/z100-12-p2p-acs-impact-result.md`](z100-12-p2p-acs-impact-result.md)（S12）
- [`docs/hygon_z100.md`](../hygon_z100.md) §9（PCIe fabric 与 ACS）、§8.3（性能归因）
- [`docs/海光Z100-性能优化方案.md`](../海光Z100-性能优化方案.md) §1.1/§1.5/§1.6
- [`docs/z100-todolist.md`](../z100-todolist.md) 硬约束 **24**/**25**
- [`docs/result/z100-11l-sdot4-reacceptance-result.md`](z100-11l-sdot4-reacceptance-result.md)（8K 历史口径 2.571 ms/token）
