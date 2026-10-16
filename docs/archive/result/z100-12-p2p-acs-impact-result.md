# S12 ACS 关闭后的通信/并行度复测 + 公平基线 — 执行结果

> 归档时间：2026-10-09（**边做边写**，本文件在测量过程中持续追加）
> 执行者：子 agent（S12）
> 对应设计文档：[`docs/海光Z100-性能优化方案.md`](../海光Z100-性能优化方案.md) §3 · [`docs/hygon_z100.md`](../hygon_z100.md) §9
> 前一档：S11m [`z100-11m-aot-restart-reproducibility-result.md`](z100-11m-aot-restart-reproducibility-result.md)
> 状态：🔶 **完成（G1/G2/G3/G6/G7/G8 通过；G4/G5 部分通过，缺口均写明"未执行 + 原因"）**
> 墙钟：约 4.5 h 真机测量（27B 启动 4 次 + 4B 启动 3 次 + vendor 启动 4 次 + TP=4 尝试 2 次）

### 0.1 完成情况一览

| # | 事项 | 状态 | 位置 |
|---|---|---|---|
| 1 | 建骨架并持续追加 | ✅ | 本文件 |
| 2 | 卡位纪律 / 前置检查 | ✅ | §3.0（含 1 次污染如实登记 §3.3.3） |
| 3 | 两态状态确认（`show` + `p2p_verify`） | ✅ | §3.1.1 / §3.1.2 |
| 4 | **NCCL 4-rank all-reduce 两态 A/B**（7 变体 × 5 尺寸） | ✅ | §3.2（**结论：两态一致，RCCL 根本不走 P2P**） |
| 5 | 工作负载两态（decode / prefill 3800 / conc8） | ✅ 同会话 A/B | §3.3.2 / §3.7.1（**≤0.7%，无可测差异**） |
| 6 | 冷/热启动电池 3 次 | ✅（logprob 有 1 处差异） | §3.5 |
| 7 | 公平基线（生产栈复现，两态各 2 次） | ✅ | §3.6 / §3.7.2 |
| 8 | 27B TP=4 | ❌ 两次 `oom_kill` | §3.4.1 |
| 9 | 4B TP=1/2/4 缩放曲线（G5 替代证据） | ✅ | §3.4.2（**TP=2→4 仅 1.07×**） |
| 10 | 8K prefill（in≈8000） | ⛔ **未执行**（时间盒） | §3.8 |
| 11 | **修 `dcu_p2p_acs.py` 的 `revert` bug** | ✅（本步唯一的代码修复） | §2.1 |

## 1. 目标与验收标准

### 1.1 本步要回答的问题

1. **ACS 关闭（`0x0000`）到底让哪些东西变快**：NCCL 层（all-reduce us/call）、
   真实工作负载层（27B W8A8 INT8 TP=2 的 decode / prefill / conc8）；
2. **TP=4 是否值得切**：ACS 关闭后 TP=4 相对 TP=2 的收益（历史阻塞点是宿主 ~31 GB RAM）；
3. **公平基线**：历史"生产 122.47 ms/token"是 **ACS 开启 + TP=4 + vendor 栈**，
   跨 fabric 状态不可比 —— 必须在**同一 fabric 状态**下重测 vendor 栈；
4. **S11m 冷/热电池是否复现**：S11m 的 3 次启动电池横跨了 ACS 改动，
   需在 ACS 关闭的稳定状态下独立重跑（同缓存根、同开关 `=1`、3 次启动 × 8×64 greedy）。

### 1.2 门禁（本次执行的验收标准）

沿用 todolist 硬约束 **24**（ACS 与口径）与 **25**（卡位纪律），逐条落到本步：

| # | 门禁 | 判据 |
|---|---|---|
| G1 | 卡位纪律 | 每条真机命令都显式 `HIP_VISIBLE_DEVICES=<明确卡>`；开跑前 `pgrep -af vllm` + `hy-smi --showmeminfo vram` 确认无他人实验 |
| G2 | 状态确认 | 每次 fabric 切换后，`dcu_p2p_acs.py show` **与** `p2p_verify.py` 各确认一次 |
| G3 | NCCL 前后可比 | 同一会话、同一容器、同一脚本，ACS 关/开各跑一遍，尺寸 ≥3 档 |
| G4 | 工作负载两态 | 27B W8A8 INT8、`VLLM_DCU_USE_INT8_SDOT4=1`、`--dtype float16`、无 `--enforce-eager`、`max-model-len 4096`、TP=2、卡 0/1；两态各一遍 decode + prefill + conc8 |
| G5 | TP=4 | 给出"ACS 关闭后 TP=4 相对 TP=2 的收益"与"是否值得切 4 卡"的结论（实测或明确外推） |
| G6 | 公平基线 | vendor 栈同 fabric 状态跑一次，或明确写"未执行 + 原因" |
| G7 | 冷/热电池 | ACS 关闭稳定态下 3 次启动：热启动之间逐位一致；冷编译 ≠ 其加载 |
| G8 | 无独立自编译 HIP 可执行程序 | 只允许 python/vllm/triton/NCCL 基准经 python 派发；阻塞命令带 `timeout` |

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/ar_bench.py` | 新增 | 从 `/home/lyh/server/diag/ar_bench.py` **逐字节拷贝**（用户既有基准，出处：`docs/hygon_z100.md` §9 提到的 `/home/lyh/server/diag/`） |
| `tools/dcu/ar_sweep.sh` | 新增 | 从 `/home/lyh/server/diag/ar_sweep.sh` **逐字节拷贝**（同上） |
| `tools/dcu/s12_ar_bench.py` | 新增 | 由 `ar_bench.py` 扩写：尺寸补 `20480`、带 `S12_TAG` 便于两态汇总 |
| `tools/dcu/s12_nccl_sweep.sh` | 新增 | 单 fabric 状态的 6 变体扫描 + `NCCL_DEBUG=INFO` 拓扑证据 |
| `tools/dcu/s12_arm_env.sh` | 新增 | 容器侧引擎启动器（27B / 4B，TP 与卡位参数化），派生自 `s11m_arm_env.sh` |
| `tools/dcu/s12_bench.sh` | 新增 | 容器侧 `vllm bench serve` 客户端（协议逐字拷贝自 `/mnt/game/port/s9/bench.sh`，新增 `ttft8k`） |
| `tools/dcu/s12_run_one.sh` | 新增 | 宿主侧编排：起引擎 → 等 ready → 跑任务 → 关引擎 → 写 summary |
| `tools/dcu/s12_vendor_arm.sh` | 新增 | 宿主侧：在 `z100-s12` 里复现**生产栈**（vendor vLLM 0.18.1+das + phaseA + int8_sdot4 + gdn_decode）跑 TP=4 公平基线 |
| `tools/dcu/s12_nccl_table.py` / `s12_bench_table.py` | 新增 | 纯 stdlib 的证据汇总脚本 |
| `tools/dcu/dcu_p2p_acs.py` | **`cmd_revert` 修 bug** | 见 §2.1（这是本步唯一的"改产品代码"，且是必需项） |

### 2.1 `dcu_p2p_acs.py` 的 `revert` 真 bug（`tools/dcu/**` 作用域内，已修）

**症状**：`systemctl stop dcu-p2p-acs`（`ExecStop=revert`）**静默失败**，ACS 仍是 `0x0000`；
手动 `revert` 抛：

```
$ sudo -n python3 tools/dcu/dcu_p2p_acs.py revert
Traceback (most recent call last):
  File ".../dcu_p2p_acs.py", line 244, in main
    return cmd_revert()
  File ".../dcu_p2p_acs.py", line 224, in cmd_revert
    bdf, off, old = line.split()
ValueError: too many values to unpack (expected 3)
```

**根因**：`cmd_apply` 在修"重复 apply 覆盖旧快照"那个坑时，把快照格式从
`bdf acs_off old_ctl`（3 列）改成了 **`bdf acs_off old_ctl new_ctl`（4 列）**
（`dcu_p2p_acs.py:188/200`），但 `cmd_revert` 的解析没跟着改（`:224` 仍按 3 列解包）。

⇒ **硬约束 24 里写的"`revert` 可还原"目前是不成立的**：任何由当前 `apply` 写出的快照都无法还原，
而 `ExecStop=revert` 正是文档给出的回滚路径（`docs/hygon_z100.md` §9.3）。

**修法**（最小改动，向后兼容两种格式，`old_ctl` 在两种格式里都是第 3 列）：

```python
            parts = line.split()
            if len(parts) < 3:
                print("跳过无法解析的行情: %r" % line)
                continue
            bdf, off, old = parts[0], parts[1], parts[2]
```

**验证**（同一个快照、无需重写）：

```
$ sudo -n python3 tools/dcu/dcu_p2p_acs.py revert
已还原 12 个桥的 ACS Control。            # 12/12
$ sudo -n python3 tools/dcu/dcu_p2p_acs.py show | grep -c 0x001d
12
```

**遗留**：`/home/lyh/server/diag/dcu_p2p_acs.py`（用户自己的那份）有同样的 bug，
但不在本步写作用域内 → **请 Lead/用户同步这一处修复**。
另外 `systemctl reset-failed dcu-p2p-acs` 需要跑一次，否则 unit 停在 `failed` 状态
（`ExecStart` 仍能正常 `start`）。

> 硬约束遵守：**未**修改 `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`；
> **未**改 `VLLM_DCU_USE_INT8_SDOT4` 默认值；产物/缓存只落 `/mnt/game`；
> `VLLM_CACHE_ROOT` 用独立子目录。

## 3. 执行过程与原始证据

### 3.0 环境与前置检查

```
$ pwd; date
/home/lyh/z100-vllm
2026-10-09（**口径注意：宿主时钟 = UTC，容器时钟 = UTC+8，差 8 h**；
本文所有 `date -Is` 的容器/脚本时间戳是 UTC+8，`ls`/宿主命令的 mtime 是 UTC）

$ free -g
               total        used        free      shared  buff/cache   available
Mem:              31           3          16           0          11          27
Swap:              3           0           3

$ docker ps  --format ...        # 影子容器，生产容器 qwen38-vllm 仍 Exited (137)
z100-s9          Up 21 hours     # 本步使用的影子容器（--memory=32g、--ipc=host、
                                 #   /dev/kfd,mkfd,dri、/mnt/game、/src(ro)、/opt/hyhal(ro)）
z100-build        Up 43 hours     # 2.10 基线，本步未动
z100-build-211    Up 36 hours     # 未动
qwen38-vllm       Exited (137) 35 hours ago   # 生产容器：未 start/kill/exec/commit

$ pgrep -af "vllm|torchrun|python.*serve"     # 开跑前确认无他人实验
1360 python3 /usr/local/bin/api-gateway serve   # 与本项目无关的常驻进程

$ hy-smi --showmeminfo vram                   # 4 卡全空
DCU[0..3]  vram Total Used Memory (MiB): 0
```

**卡位纪律（硬约束 25）**：本步所有真机命令都带 `HIP_VISIBLE_DEVICES`——
NCCL 扫描与 p2p_verify 用 `0,1,2,3`（基准需要 4 rank），27B 工作负载用 `0,1`，
4B 缩放曲线按点位 `0` / `0,1` / `0,1,2,3`。**全程没有出现 `unset HIP_VISIBLE_DEVICES`。**

### 3.0.1 工具出处

用户既有基准在 `/home/lyh/server/diag/{ar_bench.py,ar_sweep.sh}`（`docs/hygon_z100.md` §9 的
P2P 验证同目录）。本步**逐字节拷贝**它们到 `tools/dcu/`（`cmp` 为空），再用
`s12_ar_bench.py` 扩了一个 `20480` 尺寸、`s12_nccl_sweep.sh` 扩了 `RING_1ch` 变体与
`NCCL_DEBUG=INFO` 拓扑证据。**协议与用户脚本一致**（bf16 `all_reduce`、20 次预热 + 300 次计时、us/call），
以便与用户过往的测量直接比较。

### 3.1 ACS 状态确认（每次切换后各一次）

#### 3.1.1 ACS 关闭态（`0x0000`）——切换前已在位，无需 `systemctl start`

```
$ sudo -n python3 tools/dcu/dcu_p2p_acs.py show
DCU 设备 (4 张):
  DCU[0]  0000:2e:00.0  (vendor=0x1d94 device=0x55b7)
  DCU[1]  0000:33:00.0  (vendor=0x1d94 device=0x55b7)
  DCU[2]  0000:38:00.0  (vendor=0x1d94 device=0x54b7)
  DCU[3]  0000:3b:00.0  (vendor=0x1d94 device=0x54b7)
父链上的桥 (12 个带 ACS):
  0000:00:03.1 0x2a0    0x0000    (none)
  0000:28:00.0 0xf24    0x0000    (none)
  0000:28:04.0 0xf24    0x0000    (none)
  0000:28:08.0 0xf24    0x0000    (none)
  0000:2a:10.0 0xf24    0x0000    (none)
  0000:2d:00.0 0x2a0    0x0000    (none)
  0000:30:00.0 0xf24    0x0000    (none)
  0000:32:00.0 0x2a0    0x0000    (none)
  0000:35:00.0 0xf24    0x0000    (none)
  0000:35:10.0 0xf24    0x0000    (none)
  0000:37:00.0 0x2a0    0x0000    (none)
  0000:3a:00.0 0x2a0    0x0000    (none)

$ systemctl status dcu-p2p-acs
● dcu-p2p-acs.service - Enable PCIe peer-to-peer DMA between Hygon DCUs (clear ACS)
     Loaded: loaded (/etc/systemd/system/dcu-p2p-acs.service; enabled)
     Active: active (exited) since Fri 2026-10-09 07:44:04 UTC; 3h 22min ago
     Process: 2082226 ExecStart=/usr/bin/python3 .../dcu_p2p_acs.py apply (code=exited, status=0/SUCCESS)
```

`p2p_verify.py`（容器内、`HIP_VISIBLE_DEVICES=0,1,2,3`）——**12/12 全部 ~28.3–28.4 GB/s**，
与 `docs/hygon_z100.md` §9.2 的改后值一致（原始输出见 `.tmp-12/p2p-verify-acs_off.txt`）：

```
torch 2.11.0 | devices=4
names: ['Z100L', 'Z100L', 'Z100', 'Z100']
=== can_device_access_peer (P2P 能力矩阵) ===
  -> 12/12 个有向组合支持 P2P
=== 跨卡 copy_ 带宽 (128 MiB f16 x 10) ===
  gpu0 -> gpu1 :   28.37 GB/s      gpu0 -> gpu2 :   28.37 GB/s      gpu0 -> gpu3 :   28.29 GB/s
  gpu1 -> gpu0 :   28.37 GB/s      gpu1 -> gpu2 :   28.41 GB/s      gpu1 -> gpu3 :   28.41 GB/s
  gpu2 -> gpu0 :   28.37 GB/s      gpu2 -> gpu1 :   28.36 GB/s      gpu2 -> gpu3 :   28.39 GB/s
  gpu3 -> gpu0 :   28.36 GB/s      gpu3 -> gpu1 :   28.35 GB/s      gpu3 -> gpu2 :   28.33 GB/s
=== 本卡 D2D 基线(对照) ===
  gpu0 local :  395.75 GB/s   gpu1 local :  398.94 GB/s
  gpu2 local :  287.15 GB/s   gpu3 local :  275.42 GB/s
```

⇒ 状态与预期一致：**ACS 关闭、fabric 直连**。

#### 3.1.2 ACS 开启态（`0x001d`）

切换方式：`sudo -n systemctl stop dcu-p2p-acs`（= `revert`，还原快照里的 `0x001d`）。
**这一步踩到了 `tools/dcu/dcu_p2p_acs.py` 的一个真 bug，见 §2.1**——修好后才切成功。

```
$ sudo -n systemctl stop dcu-p2p-acs
$ systemctl status dcu-p2p-acs      # 第一次：ExecStop exit-code=1（bug 命中）
× dcu-p2p-acs.service - ... (clear ACS)
     Active: failed (Result: exit-code)
     Process: 2336608 ExecStop=/usr/bin/python3 .../dcu_p2p_acs.py revert (code=exited, status=1/FAILURE)
     ... ValueError: too many values to unpack (expected 3)     # ← 见 §2.1

$ sudo -n python3 tools/dcu/dcu_p2p_acs.py revert        # 修好之后
已还原 12 个桥的 ACS Control。

$ sudo -n python3 tools/dcu/dcu_p2p_acs.py show
  0000:00:03.1 0x2a0    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:28:00.0 0xf24    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:28:04.0 0xf24    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:28:08.0 0xf24    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:2a:10.0 0xf24    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:2d:00.0 0x2a0    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:30:00.0 0xf24    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:32:00.0 0x2a0    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:35:00.0 0xf24    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:35:10.0 0xf24    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:37:00.0 0x2a0    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
  0000:3a:00.0 0x2a0    0x001d    SrcValid|ReqRedir|CmpltRedir|UpstreamFwd
```

`p2p_verify.py`（容器内、`HIP_VISIBLE_DEVICES=0,1,2,3`）——**12/12 全部 6.94–6.95 GB/s**：

```
torch 2.11.0 | devices=4
=== can_device_access_peer (P2P 能力矩阵) ===
  -> 12/12 个有向组合支持 P2P
=== 跨卡 copy_ 带宽 (128 MiB f16 x 10) ===
  gpu0 -> gpu1 :    6.95 GB/s   gpu0 -> gpu2 :    6.95 GB/s   gpu0 -> gpu3 :    6.95 GB/s
  gpu1 -> gpu0 :    6.94 GB/s   gpu1 -> gpu2 :    6.95 GB/s   gpu1 -> gpu3 :    6.95 GB/s
  gpu2 -> gpu0 :    6.94 GB/s   gpu2 -> gpu1 :    6.95 GB/s   gpu2 -> gpu3 :    6.95 GB/s
  gpu3 -> gpu0 :    6.95 GB/s   gpu3 -> gpu1 :    6.94 GB/s   gpu3 -> gpu2 :    6.95 GB/s
=== 本卡 D2D 基线(对照) ===
  gpu0 local :  396.81 GB/s   gpu1 local :  404.19 GB/s
  gpu2 local :  279.73 GB/s   gpu3 local :  284.05 GB/s
```

**口径修正（重要）**：`docs/hygon_z100.md` §9.2 记的"改前 **3.12 GB/s**"在本步复现为 **6.95 GB/s**
（≈根端口 PCIe 3.0 x8 的理论 7.9 GB/s，与 §9.1 的分析一致）。
⇒ 本步实测的 fabric 倍率是 **28.4 / 6.95 ≈ 4.1×**（而不是 8–9×）。
两个数字的来源不同（可能 §9.2 那次有并发/协议差异），本文一律以**本步实测的 6.95 GB/s** 作为 A/B 基准。

`can_device_access_peer` 在**两态都是 12/12** —— 再次印证 §9.2 的判读：
"驱动说能 P2P" 与 "fabric 真的直连转发" 是两件事。

### 3.2 NCCL 层：4-rank all-reduce（bf16）两态对比

#### 3.2.1 第一轮（无重复，第一次跑就暴露了两个事实）

`/mnt/game/s12/nccl-acs_off.log`（第一版脚本，每档只测一次）——**同一份代码同一状态下的
run-to-run 抖动高达 74%**（`[1,5120]`：baseline 89.3 us，而 `MIN_NCHANNELS=1` 155.3 us、
`P2P_DISABLE=1` 153.6 us、`RING` 173.4 us）。⇒ **单次测量不足以判定 ACS 效应**，
第二轮改成"每档重复 3 次取 min"（`s12_ar_bench.py` 的 `REPEATS=3`）。

第二条事实来自 `NCCL_DEBUG=INFO` 的拓扑抓取（`.tmp-12/…`，原始在
`/mnt/game/s12/nccl-acs_off-debug.log`）：

```
$ grep -aoE "via [A-Za-z0-9/]+" nccl-acs_off-debug.log | sort | uniq -c
     56 via SHM/direct/direct
$ grep -a "P2P" nccl-acs_off-debug.log | sort -u
[0] NCCL INFO P2P Chunksize set to 131072        # 只有这句话提到 P2P（是参数名，不是 transport）
```

⇒ **RCCL（DTK 26.04 自带）在这台机上对所有尺寸都只用 `SHM/direct/direct`（host 共享内存中转），
根本不走 P2P transport**——`can_device_access_peer` 12/12 YES 也没改变它的选择
（与 `docs/hygon_z100.md` §9.2 的"'驱动说能 P2P' ≠ 'fabric 真的直连转发'"是同一类现象）。

#### 3.2.2 第二轮（重复 3 次取 min；尺寸含 2 MiB）

命令（容器 `z100-s9`，`HIP_VISIBLE_DEVICES=0,1,2,3`）：

```bash
docker exec -e S12_TAG=acs_off z100-s9 bash /src/tools/dcu/s12_nccl_sweep.sh
# 尺寸 5120 / 10240 / 20480 / 40960 / 1048576（元素，bf16）；变体 7 个
```

原始输出（`.tmp-12/nccl-acs_off-summary.txt` = `/mnt/game/s12/nccl-acs_off.log`）：

```
[acs_off] all_reduce bf16 [1,5120] (10240 B): min 100.0 med 110.3 max 133.7 us/call
[acs_off] all_reduce bf16 [1,10240] (20480 B): min 117.7 med 118.1 max 126.1 us/call
[acs_off] all_reduce bf16 [1,20480] (40960 B): min 129.9 med 130.2 max 142.3 us/call
[acs_off] all_reduce bf16 [1,40960] (81920 B): min 162.3 med 162.6 max 163.0 us/call
[acs_off] all_reduce bf16 [1,1048576] (2097152 B): min 2264.1 med 2265.2 max 2273.5 us/call
### rc=0 variant=baseline
… （baseline_rep2 / MIN_NCHANNELS_1 / TREE_1ch / P2P_off_shm / RING_1ch 见下表）
### rc=1 variant=LL128_1ch      ← LL128 协议在这套 RCCL 上直接失败（无输出），两态都会复现
```

| 变体 | 5120 | 10240 | 20480 | 40960 | 1048576 |
|---|---|---|---|---|---|
| baseline | 100.0 | 117.7 | 129.9 | 162.3 | 2264.1 |
| baseline_rep2 | 103.8 | 119.3 | 129.2 | 162.9 | 2297.3 |
| MIN_NCHANNELS=1 | 105.0 | 118.1 | 129.9 | 161.3 | 2252.6 |
| TREE+1ch | 109.7 | 130.0 | 150.5 | 177.5 | 2453.0 |
| P2P_DISABLE=1+1ch | 110.5 | 117.9 | 129.6 | 167.1 | 2223.5 |
| RING+1ch | 90.4 | 118.2 | 129.1 | 161.7 | 2246.2 |
| LL128+1ch | — | — | — | — | —（rc=1） |

（单位 us/call；ACS 关闭态）

**读法**（三条，先记下来，等 §3.7 的 ACS 开启态做完再定论）：

1. **变体之间没有可分性**：`NCCL_MIN_NCHANNELS=1` 与 `P2P_DISABLE=1`（把 P2P 显式关掉）
   与 baseline 落在同一区间（±10%），说明**这些旋钮本来就没改变数据路径**；
   `TREE` 慢一点（+10%），`RING` 在 5120 上偶发更快。
2. **10 KB（=我们的 TP=2 hidden state 尺寸）→ ~100 us/call**，与 S11b 剖析里
   `nccl 5.67 ms / 380 ms`、约 64 次/step 的推算一致（5.67 ms / 64 ≈ 88 us）。
3. **2 MiB → 2.26 ms/call（≈1.33 GB/s 有效）**，远低于 fabric 的 28 GB/s ⇒
   再次说明 RCCL 的数据路径不是 peer DMA（若是，2 MiB 应落在 0.1–0.2 ms）。

#### 3.2.3 ACS 开启态：**与关闭态逐档相同**

同一条命令、同一个容器、同一个会话，只把 fabric 换回 `0x001d`：

```bash
docker exec -e S12_TAG=acs_on z100-s9 bash /src/tools/dcu/s12_nccl_sweep.sh
```

| 变体 | 5120 off→on | 10240 | 20480 | 40960 | 1048576 | 倍率（on/off） |
|---|---|---|---|---|---|---|
| baseline | 100.0 → **85.5** | 117.7 → 118.6 | 129.9 → 129.9 | 162.3 → 162.8 | 2264.1 → 2276.7 | 0.85×–1.01× |
| baseline_rep2 | 103.8 → **113.4** | 119.3 → 118.0 | 129.2 → 130.2 | 162.9 → 161.8 | 2297.3 → 2269.5 | 0.99×–1.09× |
| MIN_NCHANNELS=1 | 105.0 → **86.6** | 118.1 → 118.7 | 129.9 → 128.8 | 161.3 → 162.5 | 2252.6 → 2240.9 | 0.82×–1.01× |
| TREE+1ch | 109.7 → 99.4 | 130.0 → 126.9 | 150.5 → 144.3 | 177.5 → 174.1 | 2453.0 → 2451.9 | 0.96×–1.00× |
| P2P_DISABLE=1+1ch | 110.5 → 111.9 | 117.9 → 117.9 | 129.6 → 130.0 | 167.1 → 164.7 | 2223.5 → 2195.9 | 0.99×–1.01× |
| RING+1ch | 90.4 → 88.5 | 118.2 → 118.3 | 129.1 → 128.8 | 161.7 → 162.8 | 2246.2 → 2272.3 | 0.98×–1.01× |
| LL128+1ch | —（rc=1） | — | — | — | — | 两态都失败 |

（单位 us/call，重复 3 次取 min；`off` = ACS 关闭，`on` = ACS 开启）

**判读（三条，这是本节最重要的结论）**：

1. **两态在所有尺寸、所有旋钮上都在噪声内**（`[1,5120]` 的行间抖动本身就有 85–113 us，
   即 ±15%，比 fabric 带来的任何差异都大；其余三档的倍率全在 0.96–1.01）。
   ⇒ **ACS 的开关对 RCCL all-reduce 没有可测影响**。
2. **原因已经定位**（§3.2.1）：RCCL 在这台机上**只用 `SHM/direct/direct`**，
   数据经 host 共享内存中转，**不产生 peer TLP**，而 ACS 只影响 peer TLP 的路由。
3. **连 `NCCL_P2P_DISABLE=1` 都测不出差别**（与 baseline 同区间）——因为本来就没在用 P2P。
   同样的，`NCCL_MIN_NCHANNELS=1`、`TREE`、`RING` 也没有哪个"吃到收益"：
   **A1 的假设（关 ACS 后某个算法/协议会变快）在本机不成立**。
   唯一的"赢家"是 `RING` 在 5120 上偶发更快（88–90 us vs 100 us），但两态都有，与 ACS 无关。

### 3.3 工作负载：27B W8A8 INT8 TP=2 两态对比

**口径**（与 S9/S11l 逐字一致，除 fabric 状态外没有任何变化）：
`VLLM_DCU_USE_INT8_SDOT4=1`、`--dtype float16`、**无** `--enforce-eager`、
`--tensor-parallel-size 2`、`--max-model-len 4096`、`--max-num-batched-tokens 2048`、
`--gpu-memory-utilization 0.9`、`--disable-custom-all-reduce`、`--skip-mm-profiling`、
`--enable-prefix-caching`、`--mamba-cache-mode align`、卡 `HIP_VISIBLE_DEVICES=0,1`、
默认 `FULL_AND_PIECEWISE` 图模式。请求形状逐字取自 `/mnt/game/port/s9/bench.sh`。

缓存根：`/mnt/game/s12/cache-27b-12/one`（**本步新建的独立子目录**，硬约束 17）；
四个臂共用它，因此 R1 是冷编译（写产物）、R2/R3/R4 都走 AOT 加载（热启动）。

#### 3.3.1 四次启动的机器可读汇总（`.tmp-12/summary.tsv`）

| 名字 | 模型 | TP | SW | maxlen | 卡 | ready | ready_s | AOT 加载 | AOT 重编 | init engine | KV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `S12A1sw1cold` | 27B | 2 | 1 | 4096 | 0,1 | 1 | **1653** | 0 | 0 | 1507.86 s（compile 89.59 s） | 11.53 GiB |
| `S12A2sw1off` | 27B | 2 | 1 | 4096 | 0,1 | 1 | **1543** | **2** | 0 | 1430.47 s（compile 20.52 s） | 11.54 GiB |
| `S12A3sw1off` | 27B | 2 | 1 | 4096 | 0,1 | 见 §3.5 | | | | | |
| `S12A4sw1on` | 27B | 2 | 1 | 4096 | 0,1 | 见 §3.7 | | | | | |

`S12A1sw1cold` 的日志里有 `saved AOT compiled function` ×1（冷编译写下产物），
`S12A2sw1off` 两个 rank 各一条 `Directly load AOT compilation from path`（热加载）——
**AOT 重载链路在 ACS 关闭态同样正常**（S11m 的修复有效）。

#### 3.3.2 两态数字（ACS 关闭 = 本次实测；ACS 开启 = S11l 同口径历史 + §3.7 的同会话复测）

| 指标（S9 协议） | ACS 开启（S11l arm1，历史） | **ACS 关闭（R2，本次）** | 变化 |
|---|---|---|---|
| decode `Median TPOT` | 84.00 ms/token | **84.68 ms/token** | +0.8% |
| decode `Output token throughput` | 11.30 tok/s | **11.25 tok/s** | −0.4% |
| prefill in=3800 `Median TTFT` | 8606.28 ms | **8686.90 ms** | +0.9% |
| conc8 `Output token throughput` | 33.12 tok/s | **33.45 tok/s** | **+1.0%** |
| conc8 `Median TPOT` | 171.42 ms | **162.46 ms** | −5.2%（并发下抖动较大） |
| ttft8k in=8100（≤8192 引擎） | 20959.41 ms → 2.571 ms/prefill-token | 见 §3.8 | — |

原始输出（R2，`/mnt/game/s12/bench-S12A2sw1off-{decode,ttft,conc8}.log`）：

```
============ Serving Benchmark Result ============   # decode
Median TPOT (ms):                        84.68
Median TTFT (ms):                        624.81
Output token throughput (tok/s):         11.25
============ Serving Benchmark Result ============   # ttft (in=3800 out=1)
Median TTFT (ms):                        8686.90
Total input tokens:                      3854
============ Serving Benchmark Result ============   # conc8 (in=512 out=128 np=16 conc=8)
Median TPOT (ms):                        162.46
Output token throughput (tok/s):         33.45
Benchmark duration (s):                  61.22
```

**判读**：三项全部落在 **±1%** 以内（conc8 的 TPOT 抖动本来就有 ±5%）。
⇒ **ACS 从 `0x001d` 清到 `0x0000` 对 27B W8A8 INT8 / TP=2 的 decode、prefill、conc8
都没有可测收益**，与 §3.2 的机制解释完全一致：RCCL 走 `SHM/direct/direct`（host 中转），
根本不产生 peer TLP，所以 fabric 的 ACS 重定向对它无影响。
**这也意味着"关 ACS 让生产也变快、所以我们的 1.458× 不成立"这条担心对 TP=2 不成立**
（对生产 TP=4 如何，见 §3.6 的 vendor 公平基线）。

#### 3.3.3 环境干扰记录（诚实标注）

安装 int8_sdot4 注入层（为 §3.6 的公平基线准备）时，`install_int8_sdot4.sh` 的第 4 步
静态自检 `check_int8_sdot4.py` 会 `import torch`，**该命令没有显式 `HIP_VISIBLE_DEVICES`**
（容器默认全 4 卡），执行时刻为容器时间 19:51（= R2 的启动/编译期）。
它与本轮基准的关系：R2 的 decode/ttft/conc8 全部在 **ready 之后**（12:13–12:16 UTC）测的，
自检在 11:51 UTC，且只做形状/接口静态检查、没有大块分配；
但按硬约束 25 的口径，这是一次**应当避免的卡位污染**，如实记录。
后续所有命令（R3/R4/4B/TP=4/vendor）都恢复显式 `HIP_VISIBLE_DEVICES`。

### 3.4 TP=4 / 4B 缩放曲线

#### 3.4.1 27B W8A8 INT8 TP=4：**复现 S9 的 `oom_kill`**（两次尝试都失败）

**第一次（§3 绕法 2 的严格版）**：新建**无 cgroup 内存上限**的影子容器 `z100-s12`
（`--memory` 不传 ⇒ `Memory=0`、`MemSwap=0`），把我们的 fork 挂到 `/ws:ro` 跑 TP=4。
**在导入期就失败**（不是内存问题）：

```
ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'
```

⇒ 基础镜像自带的是 **vLLM 0.18 的依赖组**，缺我们 fork（0.21 系）需要的
`mistral_common>=1.11`、`openai`（新）、`compressed-tensors==0.15.0.1` 等。
装了 `mistral_common 1.10.0 → 1.11.5`、`compressed-tensors 0.14.0.1 → 0.15.0.1`
之后又撞上 `openai` 太旧 ⇒ **"在无上限容器里跑我们 fork"这条路要先把 0.21 依赖组整体补齐**，
不在本步时间盒内（如实登记为未完成）。

**第二次（改用已有全部依赖的 `z100-s9`，只把内存上限拿掉）**：
`docker update --memory-swap=-1 z100-s9`（`--memory=0` 被 docker 当成"不修改"，`Memory` 仍是 32 GiB；
但 swap 变成无上限，等于多出宿主的 3 GB 交换区）。

```
(Worker_TP0..3) INFO [model_runner.py:406] Model loading took 7.27 GiB memory and 26.3 seconds   # 4×7.27 GiB
$ free -g
               total   used   free   shared  buff/cache   available
Mem:              31     23      0        0           7           7
...
Capturing CUDA graphs (PIECEWISE):  43%|████▎  | 22/51 [07:11<08:45, 18.13s/it]      ← 到此为止
$ cat /sys/fs/cgroup/memory.events
oom_kill 2                                                                          ← cgroup OOM 杀了 2 个进程
$ free -g
Mem:              31     29      0        0           1           0                    ← 宿主可用内存 0
```

⇒ **失败点与 S9 完全一致**（PIECEWISE 捕获中途、宿主内存耗尽、cgroup `oom_kill`），
但这次**明确排除了"cgroup 上限太紧"这个解释**：
① 32 GiB 的 cgroup 上限 ≈ 宿主 31 GiB RAM + 3 GiB swap，本来就不是瓶颈；
② `--memory-swap=-1` 给出了全部可用的交换空间，仍然 OOM；
③ 27B 的权重 29 GB（`du -sh` 实测）在 4 个 rank 上是 4×7.27 GiB **设备**显存，
但加载路径要 mmap/页缓存过 **29 GB 的宿主页**，叠加 4 个进程的匿名内存就必然过 31 GB。
**结论：这台 31 GB 宿主上 27B 的 TP=4 不可行**，与 ACS 无关（两次尝试都是 ACS 关闭态）。
真正的绕法只剩：加真实 swap/扩内存（§3 绕法 4，需用户授权）、W4A16 版 27B（绕法 6，本机无权重）、
或换更小的模型做缩放曲线（下面这条）。

#### 3.4.2 Qwen3.5-4B（8.8 GB）TP=1/2/4 缩放曲线 ← **G5 的替代证据**

口径：`VLLM_DCU_USE_INT8_SDOT4=1`、`--dtype float16`、无 `--enforce-eager`、
`--max-model-len 4096`、`--gpu-memory-utilization 0.9`、默认图模式、
请求形状与 S9 协议一致（decode in=128/out=128/np=2；ttft in=3800/out=1/np=1）；
卡位分别 `0` / `0,1` / `0,1,2,3`；同一缓存根 `/mnt/game/s12/cache-4b`。

| TP | 卡 | ready (s) | KV (GiB) | decode `Median TPOT` | 相对 TP=1 | TP=2→4 增益 | prefill in=3800 `Median TTFT` | 相对 TP=1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 470 | 18.38 | **46.80 ms/token** | 1.00× | — | **7897.46 ms** | 1.00× |
| 2 | 0,1 | 961 | 20.95 | **32.09 ms/token** | **1.46×** | — | **4659.25 ms** | **1.70×** |
| 4 | 0,1,2,3 | 1011 | 23.31 | **29.93 ms/token** | **1.56×** | **1.07×** | **3885.29 ms** | **2.03×** |

原始输出（`/mnt/game/s12/bench-S12B4tp*-{decode,ttft}.log`）：

```
-- bench-S12B4tp1-decode.log   Median TPOT (ms): 46.80   Output token throughput (tok/s): 20.26
-- bench-S12B4tp1-ttft.log     Median TTFT (ms): 7897.46
-- bench-S12B4tp2-decode.log   Median TPOT (ms): 32.09   Output token throughput (tok/s): 29.63
-- bench-S12B4tp2-ttft.log     Median TTFT (ms): 4659.25
-- bench-S12B4tp4-decode.log   Median TPOT (ms): 29.93   Output token throughput (tok/s): 31.73
-- bench-S12B4tp4-ttft.log     Median TTFT (ms): 3885.29
```

**曲线形状本身就是结论（decode 侧强烈饱和）**：

| 步 | decode 增益 | prefill 增益 |
|---|---|---|
| TP=1 → TP=2 | **1.46×** | 1.70× |
| TP=2 → TP=4 | **1.07×** | 1.20× |

⇒ **decode 上 TP=4 相对 TP=2 只有 +7%**，远低于 §3 判据要求的 **≥1.5×**；
prefill 的边际收益（1.20×）也明显衰减。这与"TP=4 只是把权重流切小"的直觉不符，
原因是 4B 是 8.8 GB 的小模型：TP=1 时它已经能占满带宽，切到 4 卡后
**通信/同步与小 tile 的固定开销吃掉了大部分收益**。

**注（4B 是 fp16 GDN 模型，不是 W8A8 INT8）**：它**不走 `sdot4` GEMV 路径**，
所以这条曲线衡量的是**框架层的 TP 缩放**（通信 + attention/GDN + 每卡权重流减半），
**不能直接外推到 27B INT8 的 GEMM 缩放**。外推时必须标注这一点。

**外推（明确标注为推算，非实测）**：4B 上 TP=1→2 是 1.46×、**TP=2→4 只有 1.07×**。
27B 的权重是 4B 的 3.3 倍，切 4 卡后每卡流 7.27 GiB（4B TP=4 是 2.2 GiB），
所以 27B 的 TP=2→4 收益**应当比 4B 更明显**——但即便如此，
按 4B 的边际衰减（1.46→1.07）线性外推，27B 的 TP=2→4 也很难达到 §3 要求的 1.5×；
按"每卡权重流 ∝1/TP 的理想上限 2×、效率打 5–7 折"外推则约 1.1–1.4×。
**两条外推都不支持"切 4 卡"**，而且这条外推**被两个事实进一步削弱**：
(a) 27B 在本机根本起不来，无法验证；
(b) 生产栈在**真机 TP=4** 上的实测是 120.6–132.3 ms/token（§3.6），**明显慢于我们 TP=2 的 84.6**，
说明"TP=4 一定更快"在这套栈上并不成立（TP=4 把 GEMV/GEMM 的 N 维切小之后，
权重流减少的收益被更小的 tile、更多的通信与更差的内核占用率吃掉了）。

#### 3.4.3 "ACS 关闭后 TP=4 相对 TP=2 的收益"与"是否值得切 4 卡"

| 问题 | 答案 | 依据 |
|---|---|---|
| ACS 关闭对 TP=4 的核心收益是什么？ | **只在直接 peer 写（`copy_` 4.1×）上**；RCCL all-reduce 不分 TP 数，全都走 SHM ⇒ **TP=2/TP=4 的通信路径都不受影响** | §3.2（两态逐档相同）、§3.1（p2p_verify 28.4 vs 6.95 GB/s） |
| 我们 fork 的 27B TP=4 能不能测？ | **不能**（两次 `oom_kill`，与 ACS 无关） | §3.4.1 |
| 切 4 卡值得吗？ | **不推荐**（证据从"起不来"升级为"即使能起也不划算"）：① 27B TP=4 在本机起不来（两次 `oom_kill`）；② 4B 真机曲线显示 **TP=2→TP=4 的 decode 增益只有 1.07×**，远低于 §3 的 1.5× 判据；③ 唯一能跑起来的**真机 TP=4**（生产栈）比我们的 **TP=2 慢 1.47–1.55×** | §3.4.1/§3.4.2/§3.6.3 |
| 那真正的杠杆是什么？ | **P2 已经落地（84.6 vs 生产 124.6–131.4）**；剩下的是 decode 里除 GEMM 外的 ~10 ms 与 prefill 的 1.7× 通信/内核开销 | §3.3、§3.6 |

**⇒ 对 S11c（P1：TP 缩放曲线 + 32 GB 内存绕法）的直接影响**：S11c **仍有必要**，
但目标要从"验 TP=4 更快"改为"**先解决宿主内存**（加 swap 或加内存，二选一，都需用户拍板）"，
否则任何 TP=4 实验都注定 `oom_kill`；同时 4B 曲线已经给出框架层的缩放形状（本条已部分完成 S11c）。

### 3.5 冷/热启动电池（ACS 关闭稳定态）

**目标**：S11m 的"3 次启动电池"横跨了 ACS 改动，需在 ACS **关闭**的稳定状态下独立重跑。
**口径**：同缓存根 `/mnt/game/s12/cache-27b-12/one`、同开关 `VLLM_DCU_USE_INT8_SDOT4=1`、
同端口族、同 `tools/dcu/s11l_flip_stats.py` 的 8 prompt × ≤64 token greedy 电池
（`temperature=0`、`top_p=1.0`、`logprobs=true`、`return_token_ids=true`）；
比对工具 `tools/dcu/s11m_flip_matrix.py`。

三次启动：`S12A1sw1cold`（冷编译、写产物）→ `S12A2sw1off`（AOT 加载）→ `S12A3sw1off`（AOT 加载）。

#### 3.5.1 结果

```
$ python3 tools/dcu/s11m_flip_matrix.py /mnt/game/s12/flip-S12A1sw1cold.jsonl \
      /mnt/game/s12/flip-S12A2sw1off.jsonl /mnt/game/s12/flip-S12A3sw1off.jsonl | grep -E '^\[PAIR\]|ALL_IDENTICAL'
[PAIR] S12A1sw1cold vs S12A2sw1off: 314/490 positions differ, max_matched_delta=1.544115
[PAIR] S12A1sw1cold vs S12A3sw1off: 314/490 positions differ, max_matched_delta=1.544115
# ALL_IDENTICAL=False

$ python3 tools/dcu/s11m_flip_matrix.py /mnt/game/s12/flip-S12A2sw1off.jsonl \
      /mnt/game/s12/flip-S12A1sw1cold.jsonl /mnt/game/s12/flip-S12A3sw1off.jsonl | grep -E '^\[PAIR\]'
[PAIR] S12A2sw1off vs S12A1sw1cold: 314/490 positions differ, max_matched_delta=1.544115
[PAIR] S12A2sw1off vs S12A3sw1off: 0/512 positions differ, max_matched_delta=0.251529
```

| 对比 | 位置差 | max_matched_delta | 与 S11m 是否一致 |
|---|---|---|---|
| **冷编译 vs 它自己写下的产物的加载**（A1 vs A2） | **314/490** | 1.544 | ✅ 一致（S11m：318/490、1.4009） |
| 冷编译 vs 第三次热启动（A1 vs A3） | 314/490 | 1.544 | ✅ 一致 |
| **两次热启动之间**（A2 vs A3） | **0/512（逐位一致）** | 0.2515 | 🔶 **token 一致，但 logprob 不一致**（S11m：0/512、Δ=4.8e-5） |

⇒ **S11m §3.5 的两条核心结论在 ACS 关闭态复现**：
① **热启动之间 token ids 逐位一致**（0/512）；
② **"填缓存那次冷编译" ≠ 它自己的加载**（314/490）⇒ 生产必须"先空缓存预热再对外服务"。

#### 3.5.2 与 S11m 的一处**差异**（如实登记）

A2 vs A3 的 token ids 完全一致（门禁通过），但 **logprob 不是逐位一致**：

```
$ # 逐 prompt 的最大 |Δlogprob|（A2 vs A3）
en_fr    n=64 diff=0 maxdelta=2.515e-01     ← 唯一有差异的 prompt
en_index n=64 diff=0 maxdelta=3.338e-06
zh_sky   n=64 diff=0 maxdelta=1.550e-05
其余 5 个 prompt: 0.000e+00

$ # en_fr 上的逐位分布：第 36 位起开始分叉，36–63 位 |Δlogprob| 约 0.001–0.25，token 全部相同
$ python3 - <<'PY'   # 见 §7.6
(36, -6.23445157543756e-05, -5.686121585313231e-05, ' me', ' me')
(37, -0.5363141894340515,   -0.37011823058128357,   ' think', ' think')
(38, -0.8425105214118958,   -0.9389063119888306,    ':', ':')
(40, -0.11726050078868866,  -0.18653905391693115,   'What', 'What')
…
```

S11m 的同口径对照是 **Δmax = 4.8e-5**（S2 vs S3/S4）。本步是 **0.2515**，**大 3 个数量级**。
本步**没有同引擎两遍（flipb）的对照臂**，因此无法判定这 0.25 是"跨进程抖动"还是
"只有 en_fr 这一条 prompt 的贴边位置被放大"（en_fr 恰好是唯一 `finish=stop` 的 prompt：
A1 里它只生成 42 token，A2/A3 里生成 64 token）。
**结论按保守口径写**：**token ids 逐位一致（门禁通过）；logprob 未达到逐位一致，差异点集中在
`en_fr` 一条 prompt 的第 36 位之后**。这条差异不改变"先预热再服务"的上线口径，
但**削弱了"加载路径的数值完全确定"这个更强的说法**，需要 Lead 决定是否要补一条同引擎两遍的对照实验。

### 3.6 公平基线（生产栈 / vendor）

**要替换的口径**：`docs/hygon_z100.md` §8.3 的 **122.47 ms/token** 是
"**ACS 开启 + TP=4 + eager + bf16 + 自研 `int8_sdot4` + GDN torch 替换**"的生产容器实测
（权威原始件 `/home/lyh/server/gfx906_port/bench/results/base/bench_decode.log`，
`env.txt` 只有 `LABEL=base` ⇒ 用 `serve_27b_tp4.sh` 的默认值：
`MAXLEN=262144 GPU_UTIL=0.85 MBT=2048 PREFIX=1 ENFORCE_EAGER=1 CUSTOM_AR=0`）：

```
$ grep -hE "Median TPOT|Output token|Median TTFT|Successful|Total input" \
    /home/lyh/server/gfx906_port/bench/results/base/bench_decode.log
Successful requests:                     2
Total input tokens:                      256
Output token throughput (tok/s):         7.75
Median TTFT (ms):                        954.93
Mean TPOT (ms):                          122.47
Median TPOT (ms):                        122.47
```

#### 3.6.1 复现方式（关键：**不是**"镜像自带 vendor 栈"，而是生产容器里那套打过补丁的栈）

生产容器 `qwen38-vllm` **明令不得 start/kill/exec/commit**，所以只能在**新建的无内存上限影子容器
`z100-s12`** 里从零把同一套栈装回来。实际探明与落地的四件事（全部有原始证据）：

| # | 发现 | 处理 |
|---|---|---|
| 1 | 生产栈 = 镜像自带 vendor vLLM **`0.18.1+das.3266200.dtk2604`**（`docker cp` 只读取出 `vllm/version.py` 确认）+ **容器可写层里的注入层** | 镜像自带的那份 vLLM 就够用（patch 是外挂的） |
| 2 | 注入层**不在**镜像里：`docker diff qwen38-vllm` 与 `/tmp/prod_envs.py` 证明 `envs.py` 里没有 `INT8_SDOT4_*` | 用宿主现成脚本装回：`phaseA/install_phaseA.sh`（`attn_torch.py`+`gdn_torch.py`+`gdn_torch_patch.py`+`zz_gdn_patch.pth`）、`gdn_decode/install_gdn_decode.sh`、`int8_sdot4/install_int8_sdot4.sh` |
| 3 | **镜像自带 triton 编译不出 gfx906**（`error: unsupported target: 'gfx906'` → `RuntimeError: PassManager::run failed`）⇒ 即使 `--enforce-eager`，方案 A 的纯 torch 替换函数仍走 `torch.compile`，需要 gfx906 能编的 triton | 从生产容器**只读**拷出其自建 **triton `3.4.0+gitc817b9b6`**（804 MB，装在 `dist-packages/triton`，原件另存 `triton.bak`）装进影子容器 |
| 4 | 生产 `serve_27b_tp4.sh` 还有两个必需的 env：`TRITON_HIP_LLD_PATH=/opt/dtk/llvm/bin/ld.lld`、`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS/RPC_TIMEOUT` | 逐字抄进 `tools/dcu/s12_vendor_arm.sh` |

装完后的自证（引擎日志，与生产逐条对上）：

```
WARNING [int8_sdot4_patch.py:89] int8_sdot4 patch ACTIVE (gfx906 v_dot4_i32_i8):
   ROCm int8 线性层第 0 位 = TorchSdot4Int8ScaledMMLinearKernel, decode(M<=8) 走 sdot4 GEMV(NT=64) …
GDN torch patch ACTIVE (gfx906): causal_conv1d_fn, causal_conv1d_update, fla_chunk_gated_delta_rule, …
GDN 融合 decode ACTIVE (gfx906): Qwen3_5GatedDeltaNet.forward 的 decode 快路径改用 gdn_decode_ext …
Triton attention patch ACTIVE (gfx906): TritonAttentionImpl.forward + do_kv_cache_update replaced …
INFO [__init__.py:285] Selected TorchSdot4Int8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
INFO [gpu_worker.py:456] Available KV cache memory: 18.42 GiB
INFO [kv_cache_utils.py:1321] Maximum concurrency for 262,144 tokens per request: 4.51x
[vendor] ready=1 after 193s
```

**KV 18.42 GiB / 并发 4.51x / `TorchSdot4Int8ScaledMMLinearKernel`** 与生产容器完全一致 ⇒
这是一次**忠实复现**，不是"另一个 vendor 栈"。

#### 3.6.2 ACS 关闭态实测（TP=4 / eager / MAXLEN 262144）

| 指标（S9 协议） | **生产历史（ACS 开，:8000）** | **本步复现（ACS 关）** | 变化 |
|---|---|---|---|
| decode `Median TPOT` | 122.47 ms/token | **132.28 ms/token** | **+8.0%** |
| decode `Output token throughput` | 7.75 tok/s | 7.18 tok/s | −7.4% |
| decode `Median TTFT`（in=128） | 954.93 ms | 1022.84 ms | +7.1% |
| prefill in=3800 `Median TTFT` | （base 轮未跑 3800，只有 8192） | 11091.28 ms | — |
| conc8 `Output token throughput` | （base 轮未跑） | 22.86 tok/s | — |

⇒ **生产栈在 ACS 关闭下反而慢 8%**。要判定这是"ACS 的真实效应"还是"环境差异"，
必须做同 fabric 状态下的对照 —— 见 §3.7.2（ACS 开启态复现）。
**在拿到那个数字之前，本文不把 132.28 当作"ACS 让生产变慢"的证据。**

#### 3.6.3 由此得到的"真实倍率"（同 fabric 状态、同为 TP=2 vs TP=4 的不同栈）

| 栈 | fabric | decode Median TPOT | 说明 |
|---|---|---|---|
| 本 fork（P2 `sdot4`），TP=2 | ACS 关 | **84.68** | §3.3.2 |
| 生产栈（vendor + 补丁），TP=4 | ACS 关 | **132.28** | §3.6.2 |

⇒ **在同一 fabric 状态下，本 fork 的 TP=2 比生产栈的 TP=4 快 1.56×**（132.28/84.68）。
这比 S11l 跨口径算出来的 "1.458×" **更站得住**，但也说明：
**"1.458×" 里有一部分是 TP 数不同带来的，而不是全部来自 `sdot4`**。

### 3.7 ACS 开启态的复测（同会话 A/B）

#### 3.7.1 我们的工作负载：27B W8A8 INT8 TP=2

**同会话、同缓存根、同开关的 A/B**（R2 = ACS 关，R4 = ACS 开；都是热启动、AOT 加载 ×2）：

| 指标 | **R2（ACS 关）** | **R4（ACS 开）** | 差 |
|---|---|---|---|
| ready | 1543 s | 1553 s | +0.6% |
| AOT 加载 / 重编 | 2 / 0 | 2 / 0 | 一致 |
| `init engine` | 1430.47 s（compile 20.52 s） | 1433.56 s（compile 19.85 s） | 一致 |
| KV cache | 11.54 GiB | 11.54 GiB | 一致 |
| decode `Median TPOT` | **84.68 ms/token** | **84.59 ms/token** | **−0.1%** |
| prefill in=3800 `Median TTFT` | **8686.90 ms** | **8684.11 ms** | **−0.03%** |
| conc8 `Output token throughput` | **33.45 tok/s** | **33.23 tok/s** | **−0.7%** |

⇒ **同会话 A/B 的结论：ACS 开关对我们的 TP=2 工作负载没有任何可测影响（全部 ≤0.7%）**。
这比 §3.3.2 的"S11l 历史 vs 本次"更硬（那一条差 0.8–1.0%，可能含 vintage 差异）。
**"关 ACS 让生产变快、所以我们的 1.458× 不成立"这条担心，对 TP=2 被明确否定。**

#### 3.7.2 生产栈（vendor）：ACS 开 vs 关，各跑两次

| 运行 | fabric | decode `Median TPOT` | decode `Median TTFT` | prefill in=3800 `Median TTFT` | conc8 `Output throughput` | ready |
|---|---|---|---|---|---|---|
| 生产历史（:8000，权威件） | ACS **开** | **122.47** | 954.93 | （未跑 3800） | （未跑） | 170 s |
| VON（本步第 1 次） | ACS 开 | **120.64** | 957.42 | 11144.25 | 22.89 | 193 s |
| VON2（本步第 2 次） | ACS 开 | **128.56** | 1051.34 | 11065.13 | 23.00 | 205 s |
| VOFF（本步第 1 次） | ACS 关 | **132.28** | 1022.84 | 11091.28 | 22.86 | 193 s |
| VOFF2（本步第 2 次） | ACS 关 | **130.50** | 1049.47 | 10955.40 | 22.92 | 205 s |

**读法（保守、如实）**：

1. **复现是忠实的**：ACS 开启下 120.64 / 128.56，把权威基线 **122.47 夹在中间**（−1.5% / +5.0%）。
   ⇒ 这套"影子容器 + 生产补丁 + 生产 triton + 生产 env"确实等价于生产容器。
2. **ACS 关闭的两次数值（130.50、132.28）都落在 ACS 开启两次数值（120.64、128.56）之上**，
   区间不重叠，均值 **131.39 vs 124.60 ⇒ ACS 关闭让生产栈慢 ~5.5%**（取 min 比则慢 8.2%）。
   **但每态只有 2 次、态内抖动已有 6.6%（该栈的 decode 单次只采 254 个 token 间隔），
   所以本文把它定为"提示性（suggestive）而非已证实"**。
3. **受影响的只有单流 decode**：prefill（in=3800）与 conc8 在两态之间**没有差别**
   （ACS 关反而略快 0.7%）。⇒ 若这个效应为真，它作用在 decode 的**每步小消息路径**上，
   而我们的 TP=2 栈在同类路径上**不受影响**（§3.7.1）。
4. **机制上目前没有解释**：RCCL 两态都走 SHM（§3.2），peer TLP 只可能来自
   生产栈自己的 torch/inductor 生成代码或 `gdn_decode_ext`/`sdot4` 里的跨卡访问。
   **需要 Lead 决定是否值得再补 2–4 次重复把它钉死**（每次约 7 min，很便宜）。

#### 3.7.3 同 fabric 状态下的"真实倍率"（本步的核心交付之一）

| 对比 | fabric | 数字 | 倍率 |
|---|---|---|---|
| 本 fork TP=2 **vs** 生产栈 TP=4 | **都在 ACS 关** | 84.68 vs (130.50/132.28) | **1.54×–1.56×** |
| 本 fork TP=2 **vs** 生产栈 TP=4 | 都在 ACS 开 | 84.59 vs (120.64/128.56) | **1.43×–1.52×** |
| 本 fork TP=2 **vs** 权威历史基线 122.47 | 跨口径（历史 ACS 开，本次 ACS 关） | 84.68 vs 122.47 | 1.446×（与 S11l 的 1.458× 吻合） |

⇒ **同 fabric 状态下的真实倍率是 1.43×–1.56×**，与 S11l 跨口径报的 1.458× 落在同一区间。
**结论：S11l 的 "快生产 1.458×" 这条结论经受住了同 fabric 复测**，
但它必须按上表写清"哪两态"——最保守的说法是 **"ACS 关、同 fabric、1.54×"**。

### 3.8 8K prefill（in≈8000）——**未执行 + 原因**

任务要求 prefill 覆盖 in≈3800 与 in≈8000。本步只完成了 **in=3800**（§3.3.2/§3.7.1）。
in≈8100 需要**另一个 `--max-model-len 8192` 的 27B 引擎**（`max-model-len 4096` 装不下 8100 token），
即**每个 fabric 状态各多一次 ~26 min 启动**。

**未执行的原因（时间盒）**：本步的 27B 启动共 4 次（R1–R4，每次 25.7–31.5 min，合计 ~1.8 h），
加 NCCL 两态扫描、vendor 栈 4 次启动、4B 曲线 3 次启动、27B TP=4 两次尝试，
总墙钟已约 4.5 h。**8K 这一档被时间盒挤掉，如实登记为未执行。**

**可用的替代证据**：`docs/result/z100-11l-sdot4-reacceptance-result.md` §3.3 的 `ttft8k`
用的是 **同一个 `--max-model-len 8192` 引擎、in=8100**（≈8K 口径），
`=1` 时 **20959.41 ms → 2.571 ms/prefill-token**（优于生产基线 2.789）。
它是在 **ACS 开启**下测的；由于 §3.7.1 已经证明 prefill 路径对 fabric 状态不敏感（in=3800 差 0.03%），
**"ACS 关闭后 8K prefill 会显著变快"这个假设基本可以排除**，但**该档本身没有在 ACS 关闭下实测**。

### 3.9 本步的"一句话结论"

**ACS 从 `0x001d` 清到 `0x0000` 让跨卡 `copy_` 快 4.1×（6.95 → 28.4 GB/s），
但既没有让 RCCL all-reduce 变快一点（因为它根本不走 P2P，全程 `SHM/direct/direct`），
也没有让我们 TP=2 的工作负载变快一点（同会话 A/B ≤0.7%）；
唯一看起来受影响的，是生产栈的 TP=4 单流 decode ——关掉 ACS 反而慢 ~5.5%（提示性，n=2）。**
⇒ **本步没有推翻 S11l/S11m 的任何结论，但把"关 ACS 之后必须重测"这件事做完了，
并把"生产该切 TP=2 还是 TP=4"从"起不来"升级为"起不来 + 即使能起也不划算"。**

（待填）

## 4. 门禁判定

| # | 门禁（任务书） | 判定 | 证据 |
|---|---|---|---|
| **G1** | **卡位纪律**：每条真机命令显式 `HIP_VISIBLE_DEVICES`；开跑前 `pgrep`+`hy-smi` | ✅ 通过（有 1 次污染，已登记） | §3.0 前置检查（`pgrep` 只有无关的 api-gateway、4 卡 0 MiB）；NCCL/p2p_verify 用 `0,1,2,3`、27B 用 `0,1`、4B 用 `0`/`0,1`/`0,1,2,3`；**全程没有 `unset`**；**唯一污染**：安装 int8_sdot4 时的静态自检未指定卡位（§3.3.3），且它跑在 R2 的就绪期之前 |
| **G2** | 每次 fabric 切换后用 `show` + `p2p_verify` 各确认一次 | ✅ 通过 | §3.1.1（关：12×`0x0000` + 12/12 28.3–28.4 GB/s）、§3.1.2（开：12×`0x001d` + 12/12 6.94–6.95 GB/s） |
| **G3** | NCCL 前后可比：同会话、同容器、同脚本，尺寸 ≥3 档 | ✅ 通过 | §3.2.2/§3.2.3：7 个变体 × 5 档尺寸（5120/10240/20480/40960/1048576）× 两态，重复 3 次取 min |
| **G4** | 工作负载两态：27B W8A8 INT8、`=1`、`float16`、无 `--enforce-eager`、maxlen 4096、TP=2、卡 0/1；decode+prefill+conc8 各一遍 | 🔶 **部分通过** | decode/prefill(in=3800)/conc8 **两态都测了**（§3.3.2、§3.7.1，同会话 A/B）；**in≈8000 未测**（§3.8，未执行 + 原因） |
| **G5** | TP=4：给出"ACS 关闭后 TP=4 相对 TP=2 的收益"与"是否值得切 4 卡" | 🔶 **部分通过** | 27B TP=4 **两次尝试都 `oom_kill`**（§3.4.1，已排除 cgroup 上限这个解释）；改用任务书允许的 4B 曲线：**TP=2→4 的 decode 增益仅 1.07×**（§3.4.2）⇒ 结论"**不推荐切 4 卡**"有两条独立证据；但"27B INT8 上的 TP=4 收益"仍是外推，**未实测** |
| **G6** | 公平基线：vendor 栈同 fabric 状态跑一次，或明确写"未执行+原因" | ✅ **通过（且做了 4 次）** | §3.6：在无内存上限影子容器里**忠实复现生产栈**（vendor 0.18.1+das + phaseA + int8_sdot4 + gdn_decode + 生产 triton + 生产 env），ACS 开 2 次（120.64/128.56，把基线 122.47 夹住）、ACS 关 2 次（130.50/132.28） |
| **G7** | 冷/热电池：ACS 关闭稳定态 3 次启动，热启动逐位一致、冷编译 ≠ 其加载 | ✅ 通过（token 口径） | §3.5：A1(冷) vs A2/A3(热) **314/490**；A2 vs A3 **0/512 逐位一致**；**logprob 口径有一处差异**（`en_fr` 最大 Δ0.2515，S11m 是 4.8e-5），已如实登记 |
| **G8** | 不运行独立的自编译 HIP 可执行程序；阻塞命令带 `timeout` | ✅ 通过 | 全程只有 `python`/`vllm`/`torchrun`(NCCL 基准) 经 python 派发；`hipcc` 只用于**编译** `gdn_decode_ext.so` / `w8a8_sdot4_ext.so`（硬约束 21 允许），**没有执行任何自编译二进制**；27B 启动 5400 s、NCCL 420 s、p2p_verify 900 s 全部带 `timeout` |

**总体判定：G1/G2/G3/G6/G7/G8 通过，G4/G5 部分通过（缺口都已写明"未执行 + 原因"）。**

## 5. 未完成项 / 阻塞与原因

| # | 未完成项 | 原因 | 下一步建议 |
|---|---|---|---|
| 1 | **27B W8A8 INT8 的 TP=4 实测** | 宿主机 31 GiB 内存装不下（29 GB 权重 + 4 进程）；**两次 `oom_kill`**，且已证明不是 cgroup 上限造成的（§3.4.1） | 加真实 swap（如 +16 GB）或加内存；或换 W4A16 版 27B（本机无权重）；**需用户拍板** |
| 2 | **"无上限容器里跑我们 fork"这条绕法** | 基础镜像只有 vLLM 0.18 的依赖组；补齐 `openai`/`mistral_common`/`compressed-tensors` 等多次仍缺（§3.4.1） | 在 `z100-s12` 里按 `requirements/dcu*.txt` 整体装一遍 0.21 依赖组（~20 min），之后 TP=4 实验可复用 |
| 3 | **8K prefill（in≈8000）在 ACS 关闭态** | 需要额外的 `max-model-len 8192` 引擎（每态 ~26 min 启动），被 4.5 h 时间盒挤掉（§3.8） | 用同一个 8192 引擎补一次即可（单次 ~30 min） |
| 4 | **vendor 栈"ACS 关更慢 5.5%"是否属实** | 每态只有 2 次、态内抖动 6.6%（§3.7.2） | 每态再补 2–4 次（每次 ~7 min）把区间收紧 |
| 5 | **A2 vs A3 的 logprob 未逐位一致** | 只有 3 次启动、没有同引擎两遍的对照臂（§3.5.2） | 补一个 `flipb`（同引擎第二遍）即可定性 |
| 6 | **`/home/lyh/server/diag/dcu_p2p_acs.py` 的同名 bug** | 不在本步写作用域（§2.1） | Lead/用户同步该文件 |
| 7 | **custom all-reduce 在 ACS 关闭下是否值得重开** | 未做（时间盒）。这是**唯一**能让 TP=2 吃到 ACS 收益的机制（vLLM 的 custom AR 走直接 peer 写，正是 `copy_` 那条路径），而我们的生产配方一直是 `--disable-custom-all-reduce` | 值得单独立项：一次 27B 启动去掉该 flag，对比 decode（预计 nccl 那 5.7 ms/step 的一部分可省） |

## 6. 对后续步骤的影响

### 6.1 生产该用 TP=2 还是 TP=4
**用 TP=2。** 三条独立证据：
1. 本机 31 GiB 宿主上 27B 的 **TP=4 起不来**（两次 `oom_kill`，且与 cgroup 上限无关）；
2. 4B 真机曲线显示 **TP=2→TP=4 的 decode 增益只有 1.07×**（§3.4.2），远低于 §3 的 1.5× 判据；
3. 唯一能跑的真机 TP=4（生产栈）**比我们的 TP=2 慢 1.43–1.56×**（§3.7.3）。

### 6.2 现在相对生产的**真实倍率**
**同 fabric 状态（同一次开机、同一 fabric 配置）下，我们 TP=2 比生产栈 TP=4 快
1.43×–1.56×**（ACS 开 1.43×–1.52×、ACS 关 1.54×–1.56×，§3.7.3）。
跨口径对比权威基线 122.47 得 1.446×（与 S11l 的 1.458× 吻合）。
⇒ **S11l 的上线结论站得住**，建议部署文档把口径写成"**ACS 关、同 fabric、1.54×**"。

### 6.3 S11c（P1：TP 缩放曲线 + 32 GB 内存绕法）是否还有必要
**仍然必要，但要改目标**：本步已经完成了"4B 的框架层缩放曲线"这一半（§3.4.2），
并把"TP=4 更快"这个前提**证伪**（1.07×）。S11c 剩下的唯一价值是
**"如何在这台 31 GiB 宿主上跑起 27B 的 TP=4"**（加 swap / 加内存 / 换 W4A16），
而这需要用户先拍板资源；**在此之前不建议再花时间在 TP=4 上**。

### 6.4 S11f（P6：FP16 模拟 FP8）是否还有必要
**必要性不受本步影响**。本步只证明 fabric 与并行度不是瓶颈：
decode 已经被 `sdot4` 从 321 ms 压到 84.6 ms，而生产基线 122.47 已被越过；
FP8 是**功能覆盖率**问题（让 FP8 量化在 Z100 上可用），不是本步的性能问题。
⇒ **S11f 按原优先级继续**。

### 6.5 其他被本步改变的前提
1. **"关 ACS 之后所有 TP≥2 基线都要重测"这条已经执行完毕**：结论是**不用重测**——
   NCCL 与我们的 TP=2 工作负载都测不出差别（§3.2、§3.7.1）。历史数字仍然可用，
   唯一需要标注的是"它们是在 ACS 开/关下测的"这一点在生产栈 TP=4 decode 上有 5.5% 的歧义（§3.7.2）。
2. **`docs/hygon_z100.md` §9.2 的"3.12 GB/s"应修正为 6.95 GB/s**（本步实测，§3.1.2），
   fabric 倍率相应从"8–9×"改为"**4.1×**"。
3. **`dcu_p2p_acs.py` 的 `revert` bug 必须先修再用**（§2.1），
   否则"回滚 ACS"这条路是断的（也意味着 `systemd` 的 `ExecStop` 一直在静默失败）。
4. **新增的下一步候选（按性价比排序）**：① 补 vendor 栈重复把 5.5% 钉死（~30 min）；
   ② 测"去掉 `--disable-custom-all-reduce`"（~35 min，唯一可能让 TP=2 吃到 ACS 收益的方向）；
   ③ 补 8K 与 TP=4 的缺口（都需要先解决宿主内存）。

## 7. 复现方式

```bash
# 7.0 通用
cd /home/lyh/z100-vllm
CT=z100-s9          # 本步所有 27B/4B/NCCL 测量都在这个影子容器里
docker exec "$CT" bash -c 'echo ok'      # 确认容器在

# 7.1 fabric 状态切换 + 双向确认（sudo 可用）
sudo -n python3 tools/dcu/dcu_p2p_acs.py show                    # 只读；12 行 0x0000 = 关闭
sudo -n systemctl stop  dcu-p2p-acs                              # 还原 0x001d（= revert）
sudo -n python3 tools/dcu/dcu_p2p_acs.py revert                  # 等价手动调用（修 bug 后可用）
sudo -n systemctl start dcu-p2p-acs                              # 清零 0x0000（幂等）
docker exec -e HIP_VISIBLE_DEVICES=0,1,2,3 -e LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64 \
  "$CT" bash -lc '. /opt/dtk/env.sh >/dev/null 2>&1; export PYTHONPATH=/src; python -c "import torch; exec(open(\"/src/tools/dcu/p2p_verify.py\").read())"'

# 7.2 NCCL 两态扫描（每态 ~6 min）
docker exec -e S12_TAG=acs_off "$CT" bash /src/tools/dcu/s12_nccl_sweep.sh
docker exec -e S12_TAG=acs_on  "$CT" bash /src/tools/dcu/s12_nccl_sweep.sh
python3 tools/dcu/s12_nccl_table.py /mnt/game/s12/nccl-acs_off.log /mnt/game/s12/nccl-acs_on.log

# 7.3 27B 工作负载（每个臂 ~26-32 min）
#   <NAME> <MODEL> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> <TASKS>
bash tools/dcu/s12_run_one.sh S12A1sw1cold /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
     2 1 /mnt/game/s12/cache-27b-12/one 8051 4096 0,1 flip
bash tools/dcu/s12_run_one.sh S12A2sw1off  /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
     2 1 /mnt/game/s12/cache-27b-12/one 8052 4096 0,1 flip,decode,ttft,conc8
bash tools/dcu/s12_run_one.sh S12A3sw1off  /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
     2 1 /mnt/game/s12/cache-27b-12/one 8053 4096 0,1 flip
bash tools/dcu/s12_run_one.sh S12A4sw1on   /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
     2 1 /mnt/game/s12/cache-27b-12/one 8054 4096 0,1 flip,decode,ttft,conc8   # ← ACS 开启态时执行
python3 tools/dcu/s12_bench_table.py /mnt/game/s12/bench-S12A*-*.log

# 7.4 4B TP 缩放曲线（3 个臂，~45 min）
S12_CT="$CT" S12_SRC=/src bash tools/dcu/s12_4b_curve.sh
python3 tools/dcu/s12_bench_table.py /mnt/game/s12/bench-S12B4tp*-*.log

# 7.5 公平基线（生产栈；~7 min/次）
#   前置（一次性）：建无内存上限影子容器 + 装生产补丁 + 拷生产 triton
docker run -d --name z100-s12 --ipc=host --group-add video \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri --security-opt label=disable \
  --shm-size=16g -v /mnt/game:/mnt/game -v /opt/hyhal:/opt/hyhal:ro \
  -v /home/lyh/z100-vllm:/ws:ro \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 sleep infinity
C=z100-s12 bash /home/lyh/server/phaseA/install_phaseA.sh
C=z100-s12 bash /home/lyh/server/gdn_decode/install_gdn_decode.sh
C=z100-s12 SKIP_BUILD=1 bash /home/lyh/server/int8_sdot4/install_int8_sdot4.sh
docker cp /home/lyh/server/phaseA/zz_gdn_patch.pth z100-s12:/usr/local/lib/python3.10/dist-packages/
docker cp /mnt/game/port/s7/pip-wheels/mistral_common-1.11.5-py3-none-any.whl z100-s12:/tmp/
docker cp /mnt/game/port/s8/pip-wheels/compressed_tensors-0.15.0.1-py3-none-any.whl z100-s12:/tmp/
docker exec z100-s12 pip install --no-deps /tmp/mistral_common-1.11.5-py3-none-any.whl /tmp/compressed_tensors-0.15.0.1-py3-none-any.whl
#   **生产 triton（只读拷贝，见 §3.6.1）**：
docker cp qwen38-vllm:/usr/local/lib/python3.10/dist-packages/triton /tmp/prod_triton
docker exec z100-s12 bash -c 'cd /usr/local/lib/python3.10/dist-packages && mv triton triton.image && mkdir triton'
docker cp /tmp/prod_triton/. z100-s12:/usr/local/lib/python3.10/dist-packages/triton
#   跑：
bash tools/dcu/s12_vendor_arm.sh VOFF 4 8060 262144 0,1,2,3      # ACS 关闭态
bash tools/dcu/s12_vendor_arm.sh VON  4 8061 262144 0,1,2,3      # ACS 开启态

# 7.6 冷/热电池比对（纯 stdlib）
python3 tools/dcu/s11m_flip_matrix.py /mnt/game/s12/flip-S12A1sw1cold.jsonl \
        /mnt/game/s12/flip-S12A2sw1off.jsonl /mnt/game/s12/flip-S12A3sw1off.jsonl
# 单独定位 A2 vs A3 的 logprob 差异（§3.5.2 那段）
python3 - <<'PY'
import json
def load(p, pid):
    for line in open(p, encoding='utf-8'):
        r = json.loads(line)
        if r['prompt_id'] == pid: return r
a = load('/mnt/game/s12/flip-S12A2sw1off.jsonl', 'en_fr')
b = load('/mnt/game/s12/flip-S12A3sw1off.jsonl', 'en_fr')
ra, rb = a['rows'], b['rows']
print('tokens equal:', a['token_ids'] == b['token_ids'])
for i in range(len(ra)):
    d = abs((ra[i].get('logprob') or 0) - (rb[i].get('logprob') or 0))
    if d > 1e-6: print(i, ra[i]['logprob'], rb[i]['logprob'], ra[i]['token'])
PY
```

**清理/收尾**（本步结束时的实际状态）：
```bash
docker exec z100-s9 bash -c 'pkill -9 VLLM'         # 无残留进程
docker update --memory-swap=34359738368 z100-s9     # 把 §3.4.1 临时放宽的 swap 上限还原
sudo -n systemctl start dcu-p2p-acs                 # 回到部署态：ACS 关闭
sudo -n python3 tools/dcu/dcu_p2p_acs.py show | grep -c '0x0000    (none)'   # 期望 12
```

## 8. 参考

- [`docs/hygon_z100.md`](../hygon_z100.md) §9（PCIe fabric 与跨卡 P2P：ACS 重定向）、§8.3（性能归因）
- [`docs/海光Z100-性能优化方案.md`](../海光Z100-性能优化方案.md) §1.1–§1.5、§3（P1 并行度与内存绕法）
- [`docs/z100-todolist.md`](../z100-todolist.md) 硬约束 **24**（ACS 与口径）、**25**（卡位纪律）
- 前一档：[`docs/result/z100-11m-aot-restart-reproducibility-result.md`](z100-11m-aot-restart-reproducibility-result.md)、
  [`docs/result/z100-11l-sdot4-reacceptance-result.md`](z100-11l-sdot4-reacceptance-result.md)、
  [`docs/result/z100-09b-full-cudagraph-result.md`](z100-09b-full-cudagraph-result.md)
- 用户既有基准：`/home/lyh/server/diag/{ar_bench.py,ar_sweep.sh,p2p_verify.py,dcu_p2p_acs.py}`
- 生产栈：`/home/lyh/server/serve_27b_tp4.sh`、`/home/lyh/server/{phaseA,gdn_decode,int8_sdot4}/`、
  `/home/lyh/vllmbench/`、权威基线件 `/home/lyh/server/gfx906_port/bench/results/base/`
