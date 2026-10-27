# Z100 B0 基线冻结（WP0.1，门 G4 输入）

> 设计依据：[`z100-decode-prefill-optimization-design.md`](../design/z100-decode-prefill-optimization-design.md)（DESIGN）
> 施工依据：[`z100-decode-prefill-optimization-construction.md`](../design/z100-decode-prefill-optimization-construction.md)（BUILD）§3.2/§3.3/§4 WP0.1
> 属主：A（Lead）。日期：2026-10-24。运行时：4×Z100/Z100L（gfx906）、DTK **26.04**（`DTK-26.04-DCC2602-0317`）、torch **2.11.0**、Python 3.10、TP=4、DFlash2 k=7、`max_num_seqs=4`、util 0.85、`--enforce-eager`。
> 树：`/home/lyh/z100-vllm`，分支 `dtk`，**HEAD `7078b5fe137feeab3750ee243daeaf172dc439d8`**（未提交改动见 §3）。
> 结果目录：`.s27/p0/`（`.gitignore` 的 `.s*/` 覆盖）；原始 summary JSON 与 serve 日志同目录。

---

## 1. 一句话结论

**B0 基线复现成功**：真实 7469-token 文档、MBT 8192 下 **step p50 = 129.26 ms**（参考 129，落在 129±6 内）、
**TTFT = 8.971 s / prefill = 832.6 t/s**（参考 8.894 s / 839.8，差 **+0.9% / −0.9%**）、acc = 2.667 tok/step（参考 2.84，差 −6%，属 prompt 内容差异）。
**同时发现 BUILD §3.3 的一处口径错误**：`s23_probe.py --prompt-tokens N` 造出的合成 prompt 实际是 **≈4.85×N 个 token**
（`item1 item2 …` 每个词被切成多个 token），所以 `--prompt-tokens 7469` 真跑的是 **36,238 token** 上下文，
必须用 WP0.5 的 `--prompt-file` + token 级截断才能量 "7469"。这条解释了一个表面上的 "step 142.6 vs 129（+10.5%）" 假偏差。

---

## 2. A/B 表（B0 六点，全部 c=1、greedy、`max_tokens=96`、prompt = 同一份真实文档按 token 截断）

Prompt 源：`docs/design/z100-dual-prefill-reconciliation-projection.md`（离线 tokenizer 精确截断；`s20_single_probe.py --prompt-file ... --truncate-tokens N`）。
cold TTFT = 该 prompt 在本服务上的**首个**请求（prefix cache 未命中）；step p50 = 2 次重复的 `srv_step_ms` 中位；
acc = `srv_tok_per_step`（BUILD §1 定义：每 step 被接受/输出的 token 数）；prefill t/s = `prompt_tokens / cold_TTFT`。

| 标签 | prompt tok | MBT | cold TTFT (ms) | prefill t/s | step p50 (ms) | acc (tok/step) | out tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b0-mbt8192-7469` | 7469 | **8192** | **8971** | **832.6** | **129.26** | 2.667 | 10.81 |
| `b0-mbt4096-7469` | 7469 | **4096**（生产默认） | 9042 | 826.0 | 122.35 | 2.595 | 10.75 |
| `b0-mbt2048-7469` | 7469 | 2048 | 9783 | 763.5 | 129.19 | 2.667 | 10.52 |
| `b0-512` | 512 | 2048※ | 770 | 664.9 | 120.20 | 2.57 | 18.2 |
| `b0-1k` | 1024 | 2048※ | 1147 | 892.8 | 129.05 | 3.43 | 20.2 |
| `b0-2048` | 2048 | 2048※ | 2449.5 | 836.1 | 127.45 | 2.64 | 13.5 |

※ 短 prompt 均 < MBT ⇒ 单 chunk，prefill 行为与 MBT 无关（生产 MBT=4096 同口径等价）。这些点是在 MBT=2048 臂上补测的；
MBT 只影响激活 workspace/KV 容量，不影响单 chunk prefill 的算子形状。
`b0-512/1k/2048` 每 run 前加唯一 marker 破坏前缀缓存链（vLLM 无 `/reset_prefix_cache`，dev 端点未开），因此
**cold TTFT 有效、acc 受 marker 影响**（512 两 run acc 2.23/2.91，1024 两 run 一致 3.43，2048 2.53/2.74）。

### 2.1 与 BUILD §3.2「已知参考值」的对照

| 点 | BUILD 参考（prefill t/s / step / acc） | 本测（prefill t/s / step / acc） | 判定 |
|---|---|---|---|
| 7469 @MBT8192 | 839.8 / 129 / 2.84 | 832.6 / 129.26 / 2.667 | **过**（step 129±6；prefill −0.9%；acc −6% = prompt 内容差） |
| 7469 @MBT2048 | 754.0 / 131 / 2.95 | 763.5 / 129.19 / 2.667 | **过**（prefill +1.3%；step −1.4%） |

→ **G4 的 WP0.1 判据满足**：6 个点全部有 `step_ms / acc / TTFT / prefill_tps`；主基线 step 落在 129±6 ms。

### 2.2 口径警示（本次新发现，写入后续所有报告）

`s23_probe.py --prompt-tokens N` 的合成 prompt 是 `item1 item2 …`，服务端 usage 实测：

| `--prompt-tokens`（词数） | 服务端真实 prompt tokens | step p50 (ms) | out tok/s |
|---:|---:|---:|---:|
| 512 | 1940 | 124.58 | 32.7 |
| 1024 | 4013 | 119.15 | 23.8 |
| 2048 | 9133 | 118.02 | 17.7 |
| 4096 | 19373 | 129.40 | 14.1 |
| 7469 | **36238** | 142.60 | 9.1 |

⇒ 历史文档里的 "512-token prompt"（例如 `z100-cudagraph-route-a-20261021.md` §3.1）实际是 ~1940 token；
**跨文档比较 step/TTFT 时必须先确认 prompt 真实 token 数**。本报告起所有 B0/P1 点一律用 `--prompt-file` + `--truncate-tokens`
精确控制 token 数。合成 prompt 表仅保留作"context 扫描"参考（1940…36238 token 时 step 118–142.6 ms，长 ctx 侧上升 11–25 ms，
与 drafter 的 bidirection 注意力随 ctx 增长一致，待 WP1.2 profile 归因）。

---

## 3. env 全文 + 策略指纹 + 树 SHA

**策略指纹（WP0.4，实际生效值）**：`FP 7e70fc2c130b9191`（B0 各臂完全相同；gemv1 臂为 `e472fc80881fd4fe`，见 WP0.2 报告）。
启动日志两行（`tools/dcu/serve_host_dflash2.sh` 已接入 WP0.4 指纹）：

```text
[host-serve] policy: FP 7e70fc2c130b9191
[host-serve] policy: FP_FULL FP16_PREFILL_DEFAULT_MIN_M=4096 FP8_GEMV_MAX_M_W8A16=8 VLLM_DCU_BF16=on ... VLLM_S24_DYNK_LO=16384 VLLM_S24_DYNK_HI=65536
```

实际进程 env 快照（从 `/proc/<pid>/environ` 抓取，逐臂一份）：

- `.s27/p0/b0-mbt8192-7469.env.txt`
- `.s27/p0/b0-mbt4096-7469.env.txt`
- `.s27/p0/b0-mbt2048-7469.env.txt`
- `.s27/p0/gemv1-mbt8192-7469.env.txt`（WP0.2 臂 2）
- `.s27/p0/ngram-mbt8192.env.txt`（WP0.3 臂 2）

B0 生效旋钮（19 个 serve export + env.sh 的 GCN_ARCH，与 WP0.4 报告 §WP0.4-3 逐字一致）：
`VLLM_DCU_BF16=1, VLLM_DCU_FP16_PREFILL=auto, VLLM_DCU_FP8_DECODE_CFG=35, VLLM_DCU_FP8_EMU_FASTPATH=1,
VLLM_DCU_FP8_GEMV_MAX_M=8, VLLM_DCU_FP8_QUANT_FASTPATH=1, VLLM_DCU_FP8_W8A16=1, VLLM_DCU_USE_BF16_GEMV=1,
VLLM_DCU_USE_FP8_EMU=1, VLLM_GFX906_ATTN_FORCE_CAUSAL=0, VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX=16384,
VLLM_GFX906_ATTN_PREFILL_SEGMENTS=4, VLLM_GFX906_ATTN_VERIFY_3D=1, VLLM_GFX906_GDN_CHUNKED_PREFILL=1,
VLLM_GFX906_TRITON_ATTN_PREFILL_3D=1, VLLM_QWEN35_MTP_GDN_TRITON=1, VLLM_S24_DYNK_EMA=0.1,
VLLM_S24_DYNK_HI=65536, VLLM_S24_DYNK_LO=16384, VLLM_DCU_GCN_ARCH=gfx906`。

**树 SHA**：`7078b5fe137feeab3750ee243daeaf172dc439d8`；本会话对产品脚本的改动：
`tools/dcu/serve_host_dflash2.sh`（WP0.3 `SPEC` env 覆盖 + WP0.4 指纹 echo 行，默认行为不变）；
`tools/dcu/s20_single_probe.py`（WP0.5 加法）；新增 `tools/dcu/dcu_policy_fingerprint.py`。`vllm/` 下**无任何改动**。

**采集命令**（每臂；一个臂 = 一次启动 + 一组探测，臂间 stop 再起）：

```bash
cd /home/lyh/z100-vllm && export DTK_ROOT=/mnt/game/host-runtime/dtk2604
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh \
  > .s27/p0/b0-mbt8192-7469.serve.log 2>&1 &
# ready（"Application startup complete"）后：
.venv/bin/python tools/dcu/s23_probe.py --prompt-file docs/design/z100-dual-prefill-reconciliation-projection.md \
  --truncate-tokens 7469 --max-tokens 96 --repeat 2 --tag b0-mbt8192-7469-real --out .s27/p0
```

---

## 4. 判读（对照 BUILD §4 WP0.1 阈值）

| 判据 | 阈值 | 结果 | 判定 |
|---|---|---|---|
| 6 个点全部有 step/acc/TTFT/prefill_tps | 全有 | 6/6 | **过** |
| `b0-mbt8192-7469` step | 129 ± 6 ms | 129.26 | **过** |
| 偏离 >10% ⇒ 停、修栈 | — | 主点 −0.2%（对参考），无需修栈 | **过** |
| 栈身份 G0 | 4 卡空闲、DTK 26.04、torch 2.11、swap ≥16 GiB | 4 卡 0% 显存空闲；swap 19 GiB（4 + 新增 16 GiB） | **过** |

---

## 5. 反证条件

- 若用**用户自己的 7469 文档**（而非本报告用的 PROJ 文档）测得 step 落在 129±6 之外，则 B0 参考点需按该文档重钉——
  本报告只能证明"栈 + 配方"复现了参考值，不能证明 acc 2.84 的 prompt 内容可复现（acc 是 prompt 属性）。
- 若 cold TTFT 的 marker 法引入 >1% 偏差（marker 只有 ~5–8 token），7469 主点不受影响（无 marker）。
- 短上下文点用了 MBT=2048 臂；若有人证明 MBT 对单 chunk prefill 有 >1% 影响，需在 MBT=4096 臂重测。

## 6. 回滚方式

无代码改动可回滚；本工作包只新增测量产物：

```bash
rm -rf /home/lyh/z100-vllm/.s27/p0          # 只删本地产物
```

服务回到"4 卡空闲、无实例"即 B0 的初始状态。swap 扩容回滚：`sudo swapoff /mnt/game/swap-16g && sudo rm /mnt/game/swap-16g`
（**不建议**在 P1 图模式验收前回滚，BUILD §5 WP1.1 要求 swap ≥16 GiB）。
