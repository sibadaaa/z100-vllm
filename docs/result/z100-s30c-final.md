# S30c 终局报告：纯 decode 原生路线（csrc）+ 显存最小化（2026-10-26）

> 施工依据：`docs/design/z100-s30c-pure-native-construction.md`；
> 设计依据：`docs/design/z100-s30c-pure-native-design.md`（D1-D6）。
> 上游基线（勿重推导）：s29-nospec 臂 7469 doc / 96 tok / 3 reps：
> **step p50 = 39.24 ms，TTFT = 739 ms，21.33 t/s**（`.s29/s20_s29-*.json`）。

## 0. 结论速览

| 口径 | 基线 s29-nospec | s30c-full（现役 :8000） | Δ |
|---|---|---|---|
| step p50（=ms/tok，1 tok/step） | 39.24 ms | **29.48 ms**（N=3：29.24/29.48/30.05） | **−9.8 ms（−25%）** |
| 单流 out_tok_per_s（热） | 21.33 t/s | **26.8-27.3 t/s** | +26-28% |
| TTFT（热，7469 doc） | 739 ms | 701-746 ms | ±3% 门内 |
| GPU KV 容量 | 759,104 tokens（dflash 臂） | **1,166,305 tokens** | +407,201（+54%） |
| 权重/秩（dflash vs 原生） | 9.04 GiB | 7.58 GiB | −1.46 GiB（draft 塔释放） |

设计目标 27-29 ms：实测 29.2-30.0 ms。WP2-N/WP3 的 −2.5 ms 未兑现（见 §1 状态表，
两级均按判据兜底不采纳）；WP1-N 的 −9.7 ms 全额兑现并吸收了 host gap。

## 1. WP 状态表

| WP | 判据 | 状态 | 证据 |
|---|---|---|---|
| WP1-N m1_native | per-op ≤40 µs；step p50 −5 ms；流零翻转 | **采纳** | per-op 11.9 µs（17.7×）；e2e −9.5~−9.8 ms（N=2 无仪器化重启 29.44/29.76）；kdiff 12.5-12.9 µs/c、conv 持平；流=基线重启世界（§2）；`docs/result/z100-s30c-gdn-m1-native.md` |
| WP2-N 一级 env | per-op −15%；累计 e2e −1.5 ms | **不采纳**（−2.3~−6.5%） | `.s30c/gemv/sweep_*.jsonl`；`docs/result/z100-s30c-fp8-gemv-m1.md` |
| WP2-N 二级 kernel | （一级 <400 GB/s 时）−15%/−1.5 ms | **不采纳（边际，用户可裁量）** | per-op −6.1%~−12.1%、e2e −1.0~−1.4 ms（N=2：28.32/28.76）；数值与生产位同、流=基线世界；启用 = `VLLM_DCU_FP8_GEMV_STREAM_ROWS=4 VLLM_DCU_FP8_GEMV_M1=1` |
| WP3 host gap | step p50 −1 ms | **落空出表停手** | WP1-N 后 host gap 3.0→≈0（GPU-bound：engine elapsed p50 28.67 < step 29.6）；`docs/result/z100-s30c-host-gap.md` |
| WP4 AR | 记录即可 | 记录 | §4 |
| V1 cutover | 快照差入账；e2e 不回退 | **采纳（延续）** | §3 表；dflash 侧为 s29 存档启动统计 |
| V2 capture 裁剪 | 清单 [1,2,4]；step ±2% | **采纳** | 清单已验；N=3 29.24/29.48/30.05；流=基线世界；⚠️ MAX_SEQS 调高必须 re-add 8/16 |
| V3 可选臂 | 按场景，默认不做 | 未做（留档可开） | UTIL=0.75 / MBT=2048（施工图 §V3 原样可用） |

## 2. 贪心流零翻转门的判读（重要）

96-token temperature=0 流（top5 logprobs，3 reps 会话内位同）：

1. **基线自身跨重启在 pos 2 翻转**：`s29-bm16-r1` vs `r2` 首分歧即 pos 2
   （top1 logprob −1.6009 = −1.6009 的 0-margin 精确平局；96540 vs 115508）——
   s28d"accept lottery"同源，与 s30c 无关。
2. **s30c 各臂的流全部落在基线两个重启世界之内**：m1 r1→r2 世界（96/96 位同）、
   m1 r2→r1 世界、gemv/ctrim 臂同理（`cmp_capture` 输出见 `.s30c/`）。
   未产生基线分布之外的新流。
3. 共同前缀 top1 logprob 偏移 ≤0.0116 nat < S29 BM16 采纳先例 0.0213 nat
   （docs/result/z100-s29-bm16-ab.md）。

**判读**：字面"对新基线 r1 零翻转"不可满足，因为基线跨重启在同一平局处自己
翻转；s30c 的数值扰动（GDN 树形归约的 fp32 ULP 级差异，微测 T=1 fp16 out
位同、T=4 为 2/6144 个 1-ULP 元素）落在基线自身重启方差之内。按设计图
风险表，此项属"S29 WP-C 用户裁决"事项留档；不构成回滚证据。

## 3. 显存快照差（V1，只记实测）

| 项 | dflash 臂（s29 存档启动统计） | s30c 原生臂 | Δ |
|---|---|---|---|
| Model loading / rank | 9.04 GiB | 7.58 GiB | **−1.46 GiB**（draft 5 层 bf16 塔 + embedding 释放） |
| Available KV cache | 15.75 GiB | 18.10 GiB（m1iter 实测）/ 17.51-18.1（ctrim 各重启带内） | **+2.35 GiB** |
| GPU KV cache size | 759,104 tokens | **1,166,305 tokens**（现役 ctrim3b） | **+407,201（+54%）** |
| 262k 并发 | 2.90× | 4.45× | +1.55× |
| hy-smi 四卡 VRAM | 不可追溯（dflash 已先退役，缺当时快照） | 88% ×4（服务中） | — |

注：V2 裁剪的结构性收益（8/16 图池不分配）在 KV token 口径 ≈0（重启间方差
±40k tokens 覆盖了组间差）；step ±2% 门通过，按施工图 V2 采纳并留档
MAX_SEQS 联动注记。dflash 侧 hy-smi 快照因退役时序无法补测，以存档启动统计入账。

## 4. WP4：AR 7.44 ms —— 记录不动

引用 `tools/dcu/perf1_comm_report.md` 结论：TP4 allreduce 为每层 2 次结构性
开销，64.7 µs/op 已低于 90 µs 独立地板；comm 侧杠杆全部关闭（协议/算法调优
无增益、custom AR 结构性受阻、P2P fine-grain 保持开启）；decode 通信份额
10.1-11.1 ms/step（4.6-5.1%）为内在值，其余为 skew-wait。不做实验，不设
NCCL_PROTO。

## 5. 现役现场与回滚

- **现役**：`bash .s30c/restart.sh <tag>`（= `.s30c/launch_native.sh`，m1 臂）
  或 `LAUNCH_VARIANT=ctrim bash .s30c/restart.sh <tag>`（= capture [1,2,4]，**当前 :8000**）。
  env 指纹：`VLLM_GFX906_GDN_DEC_M1=1 CFG=1`，GEMV_M1/ROWS 未设（WP2-N 不采纳）。
- 回滚优先级：
  1. env：`VLLM_GFX906_GDN_DEC_M1=0`（回串行 GDN，一步回滚主收益）；
  2. 臂脚本：换 `tools/dcu/serve_host_dflash2.sh` 系（dflash 退役留档 30.9 ms/tok）；
  3. 二进制：`_C.abi3.so.bak-s30c`（md5 `d44117ae…`）/`_rocm_C.abi3.so.bak-s30c`
     （md5 `f42d0064…`）cp 回后重启。
- md5 台账：`.s30c/wp1_so_md5.log`；新 `_C`=`34d8b074775fa547a6098864f40eff23`，
  新 `_rocm_C`=`f5d279c5b5a47dcfaa443d0bc3eea4a0`（内含默认关闭的 m1 gemv kernel）。
- 指纹：现役臂 `policy: FP` 行见 `.s30c/serve-ctrim3b.log`（M1 四 knob 与
  GEMV_M1 knob 已入 `dcu_policy_fingerprint.py` KNOBS）。

## 6. 构建与工具产物

- 构建流：`docker exec -e MAX_JOBS=5 z100-regress bash -lc 'bash /mnt/game/port/build-sync.sh _C|_rocm_C'`
  （注意：GDN decode kernel 在 **_C**；fp8 gemv 在 **_rocm_C**——施工图 §0.3
  的目标名仅对后者正确）。hipify 拷贝与 ninja 有一次竞态（build1/2 未触发编译），
  重跑 build-sync 即恢复，无需手改产物。
- 新工具：`tools/dcu/s30c_gdn_m1_bench.py`（run/compare/微测/parity）；
  `.s30c/{launch_native,launch_native_ctrim,restart,probe}.sh`。
- 数据：`.s30c/`（npz 微测 parity、sweep jsonl、capture-*.json、kdiff csv、
  s20_s30c-*.json e2e、vram_*.json）；traces `/mnt/game/s30c-prof/`。
