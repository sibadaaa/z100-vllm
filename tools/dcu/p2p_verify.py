#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCU 跨卡 P2P 带宽验证(修正版)。

对比 diag/p2p.py:原版用 torch.cuda.synchronize() 不带参数,只同步当前设备
(默认 cuda:0),所以任何「不涉及 device 0」的 pair 都会测出虚高值
(实测出现过 1324 GB/s 这种不可能的数字)。这里对每张卡分别同步。

用法(宿主机没装 torch,要在容器里跑):
    docker exec -e LD_LIBRARY_PATH=/opt/dtk/dcu... z100-s9 python3 /src/diag/p2p_verify.py
"""
import time

import torch

n = torch.cuda.device_count()
print("torch %s | devices=%d" % (torch.__version__, n))
print("names:", [torch.cuda.get_device_name(i) for i in range(n)])

print("\n=== can_device_access_peer (P2P 能力矩阵) ===")
capable = 0
total = 0
for i in range(n):
    row = []
    for j in range(n):
        if i == j:
            row.append("  .  ")
            continue
        try:
            ok = bool(torch.cuda.can_device_access_peer(i, j))
        except Exception:
            ok = False
        total += 1
        capable += ok
        row.append("%5s" % ("YES" if ok else "no"))
    print("  dev%d " % i + " ".join(row))
print("  -> %d/%d 个有向组合支持 P2P" % (capable, total))

SIZE_MIB = 128
N = SIZE_MIB * 1024 * 1024 // 2          # float16 元素数
ITERS = 10


def sync_all():
    for d in range(n):
        torch.cuda.synchronize(d)


print("\n=== 跨卡 copy_ 带宽 (%d MiB f16 x %d) ===" % (SIZE_MIB, ITERS))
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        try:
            a = torch.empty(N, dtype=torch.float16, device="cuda:%d" % i)
            b = torch.empty(N, dtype=torch.float16, device="cuda:%d" % j)
            for _ in range(3):
                b.copy_(a)
            sync_all()
            t0 = time.time()
            for _ in range(ITERS):
                b.copy_(a)
            sync_all()
            dt = (time.time() - t0) / ITERS
            print("  gpu%d -> gpu%d : %7.2f GB/s" % (i, j, N * 2 / dt / 1e9))
            del a, b
        except Exception as e:
            print("  gpu%d -> gpu%d : FAILED %s: %s"
                  % (i, j, type(e).__name__, str(e)[:70]))

print("\n=== 本卡 D2D 基线(对照) ===")
for d in range(n):
    a = torch.empty(N, dtype=torch.float16, device="cuda:%d" % d)
    b = torch.empty(N, dtype=torch.float16, device="cuda:%d" % d)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize(d)
    t0 = time.time()
    for _ in range(ITERS):
        b.copy_(a)
    torch.cuda.synchronize(d)
    print("  gpu%d local : %7.2f GB/s" % (d, N * 2 / ((time.time() - t0) / ITERS) / 1e9))
    del a, b

print("\n判读:跨卡若接近本卡基线量级 -> 交换机直连 P2P;")
print("      若被压在 ~7 GB/s 附近 -> 仍在走 CPU 上行链路(ACS 未生效)。")
