# 3 号卡 P2P 14.3 GB/s 的修复手册（x8 链路降宽）

> **状态：2026-11-07 23:42 已修复并实测验证 —— 12/12 全部 28.38 GB/s。**
> 修法是**断电重启**（第 1a 步），链路回到 Gen4 x16；但重启同时引入了
> 一个新的 BDF 漂移问题，见 §1.5，那一步**必须一起做**。
>
> **结论先行**：不是软件问题，是 **`0000:35:10.0 ↔ 0000:39:00.0` 这一跳只协商到
> Gen4 x8**。Gen4 x16 实测 28.43 GB/s、x8 实测 14.27 GB/s，效率同为 89%，
> 比值 1.992 ⇒ 14.3 就是"宽度砍半"的预期值，协议栈没有问题。
>
> **修复 = 让这一跳重新训练回 x16**，做不到就换掉这一段的物理件。
> 本手册按"代价从低到高"排序，每一步都有**明确的判据**和**回滚**。
>
> 关联：[SOL 文档 §1.2](SOL-DECODE-PREFILL-20261107.md) ·
> 巡检工具 [`tools/dcu/pcie_link.py`](../../tools/dcu/pcie_link.py)

---

## 0. 先看清病灶（无需 root，30 秒）

```bash
.venv/bin/python tools/dcu/pcie_link.py
```

当前实测输出（2026-11-07 23:0x，服务在役）：

```
DCU 0000:3b:00.0  16.0 GT/s PCIe x16  ⚠ 降宽
    0000:00:03.1   8.0 GT/s x8   (max x8)    [ 7.88 GB/s]      ← CPU 上行，全球共用
    0000:27:00.0   8.0 GT/s x8   (max x16)   [ 7.88 GB/s]      ← 同上，全球共用
    0000:28:08.0  16.0 GT/s x16  (max x16)   [31.51 GB/s]
    0000:34:00.0  16.0 GT/s x16  (max x16)   [31.51 GB/s]
    0000:35:10.0  16.0 GT/s x8   (max x16)   [15.76 GB/s] ⇐ 瓶颈
    0000:39:00.0  16.0 GT/s x8   (max x16)   [15.76 GB/s] ⇐ 瓶颈
    0000:3a:00.0  16.0 GT/s x16  (max x16)   [31.51 GB/s]
    DPC 事件 3 次: 11-03 15:23 / 11-05 01:27 / 11-06 07:01  (均在 0000:35:10.0)
```

判据：**`narrow_hop_count` 必须是 0**（`--check` 退出码 0）。
另外三张卡（`2e` / `33` / `38`）全程 x16，所以这是**单卡单链路**问题，
不是全机器、不是 ACS、不是 IOMMU、不是 `HSA_FORCE_FINE_GRAIN_PCIE`。

### 0.1 已经排除的（不要再花时间）

| 假设 | 反证 |
|---|---|
| ACS 重定向 | 当前 12 桥 `ACS_Ctl = 0x0011`（`ReqRedir/CmpltRedir` 已清），与健康态一致 |
| `HSA_FORCE_FINE_GRAIN_PCIE` | 服务 env 里就是 `=1`；RCCL 已走 `via P2P/IPC` |
| IOMMU | 内核 `iommu=pt` 生效，且 x8/x16 的效率同为 89% |
| DCU 自己的 PHY | `3b:00.0` 与 `3a:00.0` **都是 x16**；只有中间那一跳 x8 |
| 驱动/RCCL 配置 | 同一次测量里另外 3 张卡满速；12 方向里恰好 6 个慢 = 链路宽度 × 方向数 |
| 走中转载 | 中转载上限 ~15.75 GB/s，实测 28.4 的那 6 个方向证伪 |

### 0.2 一个必须知道的坑

`hy-smi` 的默认输出、`lspci -s 3b:00.0`、`torch.cuda.can_device_access_peer`
（12/12 全 YES）**都看不出问题**——DCU 端点自己报 x16。
只有 `hy-smi -a | grep 'pcie clock level'`（`HCU[3] ... x8`）或
**逐跳走桥的 sysfs** 才能看到。这正是 `pcie_link.py` 存在的原因。

---

## 1. 为什么必须修：影响面

| 场景 | x8（现在） | x16（修好后） | 差 |
|---|---:|---:|---:|
| P2P 大消息 80 MB | 9258 µs | 4641 µs | **2.0×** |
| AR 32 MB 有效带宽 | 8.27 GB/s | ~15.9 GB/s | 1.9× |
| prefill 8k chunk 的 AR 税 | 1.18 s | ~0.59 s | 6.96 s 里 **17% → 8.5%** |
| decode 30 KB × 140 op/步 | ~112 µs/op | ~95–100 µs/op | 每步 **~2 ms（5%）** |
| 随机 SDES 复位 | 3 次/周 | 未知 | **每次杀掉 4 个 worker** |

> **修正 [SOL §4 D3](SOL-DECODE-PREFILL-20261107.md) 的估计**：那里写
> "per-op 112 → 60 µs，−7.3 ms/步"。小消息是 RCCL 固定开销主导，
> 健康时也就 95 µs 量级，**decode 收益 ~5%，不是 15%**。真正的钱在 prefill 大消息。

---

## 2. 修复路径（按代价排序）

### 第 0 步（零风险，先做）：把巡检挂进去

```bash
# 每次采 SOL / 起服务前跑一次；退出码非 0 就在报告里标红
.venv/bin/python tools/dcu/pcie_link.py --check || echo "LINK_DEGRADED"
```

不为修复，只为**不再把 14.3 当常量复用**（10-25 的数据里 12/12 都是 28.4，
说明这中间降级过，而旧脚本没发现）。

---

### 第 1 步（零风险，需要空闲窗口）：重训练 + 冷启动

**先说清楚风险**：PCIe retrain 会短暂断开这条链路。如果当时有训练/推理在跑，
最坏情况就是一次 SDES → DPC → `HCU reset`，把 4 个 worker 打死。
**所以必须在服务停掉、4 卡空闲时做。**

#### 1a. 冷启动（最便宜，也最可能有效）

```bash
# 1) 停服务（stop.sh 会等 worker 退出并清理孤儿进程）
bash stop.sh
hy-smi --showuse          # 确认 4 卡都在 0%

# 2) 完整断电重启（不是 reboot —— 要断 AC，让链路重新训练）
sudo -n shutdown -r now
```

开机后第一件事：

```bash
.venv/bin/python tools/dcu/pcie_link.py --check
```

| 结果 | 含义 | 下一步 |
|---|---|---|
| 退出码 0（全 x16） | 训练残留 / 上电时序问题，**已修复** | **先做 §1.5 的 BDF 漂移检查**，再跳 §3 验 P2P |
| 仍 x8 | 物理层，进 1b | 1b |

> 2026-10-25 的 `sol-20261025.json` 里 12/12 都是 28.4 ⇒ 这条链路**曾经是 x16**。
> 但 11-03 / 11-05 / 11-06 各有一次 SDES，**可能某次复位后就再没训练回 x16**。
> 所以断电冷启动的优先级高于任何硬件更换。
>
> **2026-11-07 23:42 实测：断电重启后 4 卡全部回到 Gen4 x16（降宽跳数 0）。**
> 而且原来那张 x8 的卡（SN `TF2A3400`）换到 `35:00.0` 位置后同样跑满 x16
> ⇒ **卡本身没有坏，问题是那一跳的训练/插接**。所以 1a 就解决了，
> 1b/2a/2b/2c 本轮都不需要执行。

#### 1.5 重启后的必查项：BDF 漂移（本轮踩到了）

**重启会重新枚举 PCI，所有 BDF 都可能变。** 本轮实测：

| | 重启前 | 重启后 |
|---|---|---|
| 卡 0 | `2e:00.0` | `2d:00.0` |
| 卡 1 | `33:00.0` | `30:00.0` |
| 卡 2 | `38:00.0` | `35:00.0` |
| 卡 3（原 x8） | `3b:00.0`（`35:10.0` 下游） | `3b:00.0`（`37:10.0` 下游） |

链上的桥名几乎全换（`2a:10.0`→`2a:00.0`、`35:10.0`→`37:10.0` …）。
后果是**两处按旧 BDF 写死的东西同时失效**：

1. **内核 cmdline** 的 `pci=disable_acs_redir=...` 白名单匹配不上 ⇒ ACS 没禁用；
2. **`dcu-p2p-acs.service`** 当时是 `disabled/inactive` ⇒ 没人再清 ACS。

于是出现一个很迷惑的中间态：**链路已经 x16，但 P2P 只有 11/12 满速**，
`gpu0 -> gpu1` 稳定 **6.97 GB/s**（反向 `gpu1 -> gpu0` 却是 28.38）。
单向重定向回根复合体，正是 ACS `ReqRedir/CmpltRedir` 的招牌签名。
`/usr/bin/python3 tools/dcu/dcu_p2p_acs.py show` 一查就露馅：
12 个桥里**有 5 个是 `0x001d`**。

**处置（本轮已做）**：

```bash
# 1) 清 ACS（工具是动态发现 BDF 的，换卡后不用改代码）
sudo -n .venv/bin/python tools/dcu/dcu_p2p_acs.py show              # 先看
sudo -n cp /var/lib/dcu-p2p-acs.snapshot /var/lib/dcu-p2p-acs.snapshot.bak-$(date +%Y%m%d-%H%M)
sudo -n .venv/bin/python tools/dcu/dcu_p2p_acs.py apply --force-snapshot
#    本轮结果：12/12 桥 0x001d/0x0011 -> 0x0000，之后 12/12 方向 28.37-28.40 GB/s

# 2) 让它开机自启，别再依赖人工记得
sudo -n systemctl enable --now dcu-p2p-acs
systemctl is-enabled dcu-p2p-acs && systemctl is-active dcu-p2p-acs

# 3) 刷新 grub cmdline 里的白名单（这是唯一按 BDF 写死的地方）
.venv/bin/python tools/dcu/pcie_link.py --json   # 拿到当前链上所有桥
#    然后按新 BDF 重写 /etc/default/grub 的 pci=disable_acs_redir=... 再 update-grub
#    本轮已改：/etc/default/grub 备份为 grub.bak-20261107-2351
```

> **为什么两者都要做**：cmdline 是**内核启动时**就生效，service 是**用户态补刀**。
> 只做 service，开机到 service 起来之间有窗口；只做 cmdline，一旦 BDF 漂移就
> 静默失效（且不会有任何告警）。两个都上，任一失效都能被另一个兜住。
>
> **持久建议**：每次重启/换卡后，先跑
> `.venv/bin/python tools/dcu/pcie_link.py --check`（链路）
> 和 `sudo -n .venv/bin/python tools/dcu/dcu_p2p_acs.py show`（ACS 应为 12×`0x0000`），
> 两个都绿了再谈性能数字。

#### 1b. 只重训练这一跳（不重启机器，仍要先停服务）

```bash
# 确认服务已停、卡空闲
pgrep -af 'entrypoints[.]cli' || echo "no service"

# 对链路两端各触发一次 Link Retrain（LnkCtl bit5）
# 注意：这是对 35:10.0 的 downstream 链路做重训练，不是整个 switch
sudo -n setpci -s 35:10.0 CAP_EXP+0x10.w=0x20:0x20
sleep 3
sudo -n setpci -s 39:00.0 CAP_EXP+0x10.w=0x20:0x20
sleep 3

.venv/bin/python tools/dcu/pcie_link.py --check
```

`CAP_EXP+0x10` 是 LnkCtl 寄存器，bit5 = Retrain Link。
`=0x20:0x20` 是"只置这一位"的掩码写，不碰 ASPM/其它位。

| 结果 | 含义 | 下一步 |
|---|---|---|
| 回到 x16 | 训练可恢复 ⇒ **不是坏件** | §3 验证 |
| 仍 x8 | 重训练后仍只能 8 lane ⇒ 8 条 lane 的**物理连接**有问题 | 1c / 2 |

> 如果重训练过程中 dmesg 出现 `DPC: containment event` + `HCU reset`，
> 说明这一跳已经在电气边缘，**立刻停止重试**，直接进 §2 换件排查。
> 反复触发 DPC 是有代价的（每次杀进程），不要连续试。

---

### 第 2 步（需要停机 + 动手）：定位到具体是哪个件

链路三段，**用对调法**一次就能把范围劈成两半：

```
PEX88048 端口 35:10.0  ←→  [连接器/线缆]  ←→  卡 3 转接板上游口 39:00.0  ←→  转接板  ←→  DCU 3b:00.0
        (switch 侧)                                    (卡侧)                 (x16 ✅)
```

**关键观察**：转接板下游（`3a:00.0` → DCU）是 x16，说明**转接板和 DCU 都没问题**；
问题在"switch 端口 ↔ 转接板上游口"这一段。

#### 2a. 对调槽位（最高信息量的一步）

把 3 号卡与 2 号卡（`0000:38:00.0`，链路健康 x16）**互换位置**。
两块转接板的 device id / revision 完全相同（`23b7 rev 01`），可以直接换。

```bash
# 断 AC，换卡，上电，然后：
.venv/bin/python tools/dcu/pcie_link.py
```

| 换完后 | 结论 | 处理 |
|---|---|---|
| 3b 仍是 x8、38 仍 x16 | **跟着槽位走** ⇒ `35:10.0` 端口或它到连接器这段 | 2b |
| 3b 变 x16、38 变 x8 | **跟着卡/转接板走** ⇒ 卡 3 或它的转接板 | 2c |
| 两个都 x16 | 纯连接器接触问题，重插已解决 | §3 |
| 两个都 x8 | switch 端口侧问题 | 2b |

#### 2b. 槽位/连接器侧（若跟着槽位走）

1. 断电，拔插 3 号卡金手指 + 转接板与主板的连接器，检查有无氧化/变形；
2. 用**无水酒精**清洁金手指（PCIe 4.0 对接触电阻很敏感）；
3. 检查这条 lane 的走线/连接器有无物理损伤；
4. 上电复查。仍 x8 ⇒ 该端口 8 条 lane 里有物理损坏，
   需厂商（C-3000 / 海光）按 **Gen4 lane 级故障**报修，附上：
   ```
   tools/dcu/pcie_link.py 的完整输出（含 narrow_hops）
   lspci -vv -s 35:10.0 | grep LnkSta
   journalctl -k -b all | grep 35:10.0   # 3 次 DPC / SDES
   ```

#### 2c. 卡/转接板侧（若跟着卡走）

两块转接板同型号，可**先把卡 3 换到 2 号槽的健康转接板上**做单变量判别；
仍跟着卡走 ⇒ 卡 3 的 PCIe PHY 或 retimer 半坏，走返修。

> 补充证据：3 号卡 SN 是 `TF2A340005081801`，另外三张都是 `TCRF…` 前缀；
> VBIOS 却与 0 号卡相同（`4.51.0003002.303227`）。它是**不同批次**的个体，
> 与"别的卡都 x16 只有它 x8"一致。

---

### 第 3 步（软件侧兜底，不做也行）

在链路修好之前，**不要**用 PCIe 侧的 workaround 去"骗"过这个问题：

- ❌ 把 TP 从 4 降到 2：历史数据里 TP=2 在 27B 上等价于吞吐腰斩
  （`docs/archive/hygon_z100.md` §9：TP=4 graph 49.66 ms/token vs TP=2 84.22），
  用 2 倍的卡换 2 倍带宽，亏。
- ❌ 关 P2P 回 SHM：SHM 在 1 MiB 上比 P2P 慢 8×，等于把问题放大。
- ✅ 真正的兜底是**改算法**：`VLLM_DCU_CUSTOM_AR_MAX_KIB=128`（已采用）意味着
  prefill 大消息走 RCCL、decode 小消息走 push-AR；链路修好后
  **大消息的收益会自动回来**，不需要改任何代码。

---

## 3. 修好之后：怎么确认真的修好了

必须**两层都验**，缺一不可：

```bash
# 1) 链路层（秒级，无副作用）
.venv/bin/python tools/dcu/pcie_link.py --check; echo "exit=$?"   # 期望 0

# 1.5) ACS 层（只读；期望 12 × 0x0000）—— 本轮就是栽在这一层
sudo -n .venv/bin/python tools/dcu/dcu_p2p_acs.py show | grep -c 0x0000   # 期望 12

# 2) P2P 层（要 4 卡空闲，约 40 s）
#    必须用 torch 跨卡 copy_，不要用 hipMemcpyPeer（后者走中转载，只有 6-7 GB/s）
#    注意：直接跑会 ImportError: libgalaxyhip.so.5 —— 要先套生产栈的 LD_LIBRARY_PATH
set -a; while IFS= read -r line; do case "$line" in
  LD_LIBRARY_PATH=*|PATH=*|DTK_ROOT=*|HYHAL_PATH=*|ROCM_PATH=*|HIP_PATH=*|HSA_FORCE_FINE_GRAIN_PCIE=*)
  export "$line";; esac; done < .s60/serve_state_environ.txt; set +a
HIP_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python tools/dcu/p2p_verify.py
```

判据：

| 项 | 期望 | 本轮实测（2026-11-07 23:5x） |
|---|---|---|
| 链路 | 4 卡每跳 Gen4 x16，`--check` = 0 | **✅ 降宽跳数 0** |
| ACS | 12 桥全 `0x0000` | **✅ 12/12**（apply 前有 5 个是 `0x001d`） |
| 12 个方向 | **全部 28.3–28.4 GB/s** | **✅ 28.37–28.40 × 12/12** |
| KFD `p2p_links` | 4 个 node 全 `32000` | **✅ 4/4 × 32000** |
| AR 8 MB | ~500 µs（原 1003） | 待空闲窗口重采 |
| AR 80 MB | ~4641 µs（原 9258） | 待空闲窗口重采 |

> 本轮 P2P 测量是在**服务在役**（4 卡各占 27.6 GiB、`HCU use` 0%）下做的，
> 只占 256 MiB 且 12 个方向数值高度一致（散布 <0.1%），
> 判为不受共卡影响；AR 曲线仍要等空闲窗口。

然后服务级：

```bash
bash start.sh            # 或生产用的 launcher
# 采一轮 SOL，把新常量写回 tools/dcu/sol_20261107.py 的常量表，
# 再 --markdown 重出表 —— 不要手改 doc 里的数字
```

最后**把结果落档**：更新
[`SOL-DECODE-PREFILL-20261107.md`](SOL-DECODE-PREFILL-20261107.md) §1.2 的
12 方向表、§7 的差异表，以及 §4 D3 那行（decode 收益 5% 而不是 15%）。

---

## 4. 如果换件也修不好

那就是这批卡/这个 switch 端口的 Gen4 x16 在**这个机箱里跑不住**。可选：

1. **降级接受**：把 3 号卡当成"半速卡"，TP4 的 ring 每步都过它 ⇒
   所有集合通信按 14.3 算。这时应该做的优化是 **D1 + D2**（GEMV 切片 +
   AR 融合减半），因为它们是**不依赖链路**的，能把 8k 单流从 46.5 压到 31.7 ms。
2. **拓扑上绕开**：把 4 张卡里最慢的那张固定放 rank 3，让 ring 里它
   只承担 1/4 的流量（现在就是这样，收益有限）；或者改成**树形 all-reduce**，
   让慢链路只承担 1/2 流量。
3. **报修定性**：把 §2b 的三份证据交给厂商，要求按
   "Gen4 x16 链路降宽 + 周期 SDES" 做 RMA 判定。

---

## 5. 一页速查

```bash
# 看：30 秒、无 root、可反复跑
.venv/bin/python tools/dcu/pcie_link.py

# 判：全 x16 才通过
.venv/bin/python tools/dcu/pcie_link.py --check && echo LINK_OK

# 修（代价从低到高）
bash stop.sh && sudo -n shutdown -r now            # 1a 冷启动
sudo -n setpci -s 35:10.0 CAP_EXP+0x10.w=0x20:0x20      # 1b 单跳重训练（先停服务）
#                                                        # 2a 对调 2/3 号卡
# 验
HIP_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python tools/dcu/p2p_verify.py
```
