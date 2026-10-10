#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打开海光 DCU 4 卡之间的 PCIe P2P —— 关闭 PCIe fabric 上的 ACS 重定向。

【背景】
hydcu 驱动层的 P2P 默认就是开的(pcie_p2p=Y / PeerDirect),但 PCIe fabric 上
每个桥的 ACS Control 都是 0x001d(SrcValid|ReqRedir|CmpltRedir|UpstreamFwd)。
其中 ReqRedir/CmpltRedir 会把 GPU 之间的 peer TLP 重定向到 CPU 根复合体,
而不是在 PCIe 交换机上直连转发。后果:P2P 带宽被 CPU 上行链路卡住
(本机根端口 00:03.1 只有 PCIe 3.0 x8 ≈ 7.9 GB/s),实测跨卡只有 4~7 GB/s,
而交换机侧明明是 Gen4 x16(应该 ~25 GB/s)。

【本脚本做什么】
把 4 张 DCU 父链上所有带 ACS 扩展能力的桥的 ACS Control 清零
(= 标准 pcie_acs_override 行为),让交换机按地址直接转发 peer 流量。

【用法】(需要 root)
    sudo python3 dcu_p2p_acs.py show            # 只读,列出所有相关桥的当前状态
    sudo python3 dcu_p2p_acs.py apply           # 应用(先写快照,可还原)
    sudo python3 dcu_p2p_acs.py apply --keep-sv # 只清 RR|CR|UF,保留 SrcValid
    sudo python3 dcu_p2p_acs.py apply --force-snapshot  # 强制重写快照
    sudo python3 dcu_p2p_acs.py revert          # 从快照还原

【开机自启】(本机已启用)
    systemctl status dcu-p2p-acs     # oneshot:ExecStart=apply,ExecStop=revert
    systemctl stop  dcu-p2p-acs      # 还原 ACS
    systemctl start dcu-p2p-acs      # 再次清零
    apply 默认**不会**覆盖已存在的快照(同一次开机里重复 apply 会把 old_ctl
    记成 0x0000,让 revert 失效),需要重写时显式加 --force-snapshot。

【注意】
* 改动是"活跃"的 PCIe 配置,重启后失效 —— 需要开机自启(见 README 段)。
* ACS 是 DMA 隔离特性,关掉意味着同 fabric 上的设备可以互相 DMA。
  单租户推理机上没有影响;多租户场景请勿关闭。
* 不依赖 setpci / pciutils,直接读写 /sys/bus/pci/devices/*/config。
"""

import os
import re
import sys
import glob
import struct

HYGON_VENDOR = "0x1d94"          # 海光
ACS_EXT_CAP_ID = 0x000D
SNAPSHOT = "/var/lib/dcu-p2p-acs.snapshot"


# ---------------------------------------------------------------- 发现设备
def pci_cfg_path(bdf):
    return "/sys/bus/pci/devices/%s/config" % bdf


def read_cfg(bdf):
    try:
        with open(pci_cfg_path(bdf), "rb") as f:
            return f.read(4096)
    except OSError:
        return None


def acs_offset(bdf, cfg=None):
    """返回 ACS 扩展能力的偏移(字节),没有则 None。"""
    if cfg is None:
        cfg = read_cfg(bdf)
    if not cfg or len(cfg) < 0x104:
        return None
    off, seen = 0x100, set()
    while 0x100 <= off <= 0xFE0 and off not in seen:
        seen.add(off)
        hdr = struct.unpack_from("<I", cfg, off)[0]
        if (hdr & 0xFFFF) == ACS_EXT_CAP_ID:
            return off
        nxt = (hdr >> 20) & 0xFFF
        if nxt == 0:
            break
        off = nxt
    return None


def acs_ctl(bdf, cfg=None):
    cfg = cfg if cfg is not None else read_cfg(bdf)
    off = acs_offset(bdf, cfg)
    if off is None:
        return None, None
    return off, struct.unpack_from("<H", cfg, off + 6)[0]


def dcus():
    """按 KFD/hy-smi 的编号顺序返回 DCU 的 PCI BDF 列表。"""
    found = []
    for r in glob.glob("/sys/class/drm/renderD*"):
        dev = os.path.realpath(os.path.join(r, "device"))
        bdf = os.path.basename(dev)
        try:
            with open(os.path.join(dev, "vendor")) as f:
                if f.read().strip() != HYGON_VENDOR:
                    continue
        except OSError:
            continue
        m = re.search(r"renderD(\d+)$", r)
        found.append((int(m.group(1)) if m else 0, bdf))
    return [bdf for _, bdf in sorted(found)]


def is_pci_bridge(bdf):
    """class 0x0604xx = PCI-to-PCI bridge。只动桥,绝不动 DCU 端点本身。"""
    try:
        with open("/sys/bus/pci/devices/%s/class" % bdf) as f:
            return f.read().strip().startswith("0x0604")
    except OSError:
        return False


def parents(bdf):
    """从 DCU 设备一路向上到根,返回 [父, 父的父, ...](不含自己)。"""
    chain, p = [], os.path.realpath("/sys/bus/pci/devices/%s" % bdf)
    while True:
        p = os.path.dirname(p)
        m = re.search(r"/([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9])$", p)
        if not m:
            break
        chain.append(m.group(1))
    return chain


def fabric_bridges():
    """返回 {bdf: (acs_off, cur_ctl)} —— 4 张 DCU 父链上所有带 ACS 的 PCI 桥。"""
    result = {}
    for d in dcus():
        for b in parents(d):
            if b in result or not is_pci_bridge(b):
                continue
            off, ctl = acs_ctl(b)
            if off is not None:
                result[b] = (off, ctl)
    return result


# ---------------------------------------------------------------- 位定义
BITS = [(0, "SrcValid"), (1, "TransBlk"), (2, "ReqRedir"), (3, "CmpltRedir"),
        (4, "UpstreamFwd"), (5, "EgressCtrl"), (6, "DirectTrans"),
        (7, "IOReqBlk")]


def decode(ctl):
    on = [n for b, n in BITS if ctl & (1 << b)]
    return "|".join(on) if on else "(none)"


# ---------------------------------------------------------------- 动作
def cmd_show():
    ds = dcus()
    brs = fabric_bridges()
    print("DCU 设备 (%d 张):" % len(ds))
    for i, d in enumerate(ds):
        vendor = open("/sys/bus/pci/devices/%s/vendor" % d).read().strip()
        devid = open("/sys/bus/pci/devices/%s/device" % d).read().strip()
        print("  DCU[%d]  %s  (vendor=%s device=%s)" % (i, d, vendor, devid))
    print("\n父链上的桥 (%d 个带 ACS):" % len(brs))
    print("  %-12s %-8s %-9s %s" % ("BDF", "cap_off", "ACS_Ctl", "enabled bits"))
    for b in sorted(brs, key=lambda x: (x.split(":")[0], x)):
        off, ctl = brs[b]
        print("  %-12s 0x%03x    0x%04x    %s" % (b, off, ctl, decode(ctl)))
    print("\n注: 需要清掉的是 ReqRedir|CmpltRedir(必要时连带 UpstreamFwd)。")


def write_acs(bdf, off, value):
    with open(pci_cfg_path(bdf), "r+b", buffering=0) as f:
        f.seek(off + 6)
        f.write(struct.pack("<H", value))


def cmd_apply(keep_sv=False, force_snapshot=False):
    brs = fabric_bridges()
    if not brs:
        print("没找到任何带 ACS 的桥 —— 可能已经在 P2P 直通状态,或不是 DCU 机器。")
        return 1

    target_mask = 0x0001 if keep_sv else 0x0000
    # 只清掉有能力且当前置位的位;保留硬件不支持的位为 0
    lines = []
    changed = 0
    for b in sorted(brs):
        off, ctl = brs[b]
        new = ctl & target_mask if keep_sv else 0
        lines.append("%s 0x%03x 0x%04x 0x%04x" % (b, off, ctl, new))
        if new != ctl:
            write_acs(b, off, new)
            changed += 1

    # 不要用"已经清干净"的状态覆盖原始快照:本脚本会被 systemd 在每次开机时
    # 调用,而同一次开机里重复 apply 会把 old_ctl 记成 0x0000,使 revert 变成
    # 空操作(实测踩过)。开机后 ACS 总会回到硬件默认值,旧快照依然是对的。
    if os.path.exists(SNAPSHOT) and not force_snapshot:
        print("快照已存在,保留原值: %s(如需重写请加 --force-snapshot)" % SNAPSHOT)
    else:
        with open(SNAPSHOT, "w") as f:
            f.write("# bdf acs_off old_ctl new_ctl\n")
            f.write("\n".join(lines) + "\n")
        print("快照已写入 %s" % SNAPSHOT)
    print("已修改 %d/%d 个桥。" % (changed, len(brs)))
    print("\n  %-12s %-8s %-9s %s" % ("BDF", "cap_off", "before", "after"))
    for b in sorted(brs):
        off, ctl = brs[b]
        new = ctl & target_mask if keep_sv else 0
        print("  %-12s 0x%03x    0x%04x    0x%04x   %s"
              % (b, off, ctl, new, "changed" if new != ctl else ""))
    print("\n现在重跑带宽测试验证;不满意就 revert。")
    return 0


def cmd_revert():
    if not os.path.exists(SNAPSHOT):
        print("没有快照 %s,无法还原。" % SNAPSHOT)
        return 1
    n = 0
    with open(SNAPSHOT) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # `apply` writes "bdf acs_off old_ctl new_ctl" (4 fields); the
            # earliest snapshots only had "bdf acs_off old_ctl" (3 fields).
            # old_ctl is field 2 in both, so accept either -- hard-coding the
            # 3-field unpack made `revert` (and therefore the systemd unit's
            # ExecStop, i.e. the documented rollback) fail with
            # "ValueError: too many values to unpack" on every snapshot the
            # current `apply` writes (measured 2026-10-09).
            parts = line.split()
            if len(parts) < 3:
                print("跳过无法解析的行情: %r" % line)
                continue
            bdf, off, old = parts[0], parts[1], parts[2]
            write_acs(bdf, int(off, 16), int(old, 16))
            n += 1
    print("已还原 %d 个桥的 ACS Control。" % n)
    return 0


def main():
    if os.geteuid() != 0:
        print("需要 root(读写 PCI config space)。请用 sudo 运行。")
        return 1
    args = sys.argv[1:]
    action = args[0] if args else "show"
    if action == "show":
        cmd_show()
        return 0
    if action == "apply":
        return cmd_apply(keep_sv="--keep-sv" in args,
                         force_snapshot="--force-snapshot" in args)
    if action == "revert":
        return cmd_revert()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
