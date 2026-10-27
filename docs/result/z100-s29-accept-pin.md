# S29 WP-A：accept 重启彩票根因定位 —— 机制定位 + workspace pin 强偏置（非完全钉死）

> 施工图：`docs/design/z100-s29-construction-accept-lottery-smallm-bm16-w4.md` WP-A。
> **结论**：彩票 = 进程 init 期一次性数值状态（本日证伪全部常规机制后收敛到
> rocBLAS handle workspace 维度；s28d 用的 `CUBLAS_WORKSPACE_CONFIG` rocBLAS
> 根本不读）。`ROCBLAS_WORKSPACE_CONFIG/:4096:8` + `HIPBLAS_WORKSPACE_CONFIG/
> :4096:8` 使世界分布强偏 1.969（验收窗口内 3 重启 × 3 reps 逐位一致、另一
> 次无 pin 对照回落 1.771），**但同日稍后同配置重启出现 1.824 —— pin 非完全
> 钉死**。operational 结论：pin 保留（提高高世界概率），accept 对比的
> **N≥2 重启取分布协议保持强制**。

## 1. 实验设置

- 服务：`.s29/graph_profile_launch.sh`（= `tools/dcu/s27_graph_launch.sh` + torch profiler
  config，trace 落 `/mnt/game/s29-prof`）。基线臂（A1 + draft-3D，`FP_FULL`
  fingerprint `2766da17334e3fc2`），MAX_SEQS=4 MBT=8192。
- accept 探针：`.s29/probe.sh`（= §0.3 模板，7469 真实文档 / 96 max-tokens / 3 reps）。
- 贪心抓流：`.s29/capture.py`（/v1/completions 流式 + `return_token_ids` +
  `logprobs=5`，temperature 0，prompt = 7469 个 token id，3 reps）。
- kernel 差分：`.s29/kdiff.py`（复用 s27_five_phase_profile 的 decode 窗口逻辑，
  按 (kernel, grid) 聚合 µs/call 与 calls/round）。
- prefill 抓取：`.s29/prefill_capture.sh`（手动 /start_profile 包住 32-token 请求）。

## 2. 彩票复现（accept 探针，同树同 env）

| 重启 | 配置差异 | accept | tok/step | step p50 |
|---|---|---:|---:|---:|
| det-1 | — | 1.771 | 2.667 | 93.24 |
| det-2 | — | 1.824 | 2.743 | 93.18 |
| det-3 | — | 1.771 | 2.667 | 93.18 |
| det-4 | — | 1.969 | 2.909 | 93.11 |
| det-5 | — | 1.771 | 2.667 | 93.31 |
| nc1 | --no-enable-prefix-caching | 1.969（3 reps 逐位一致） | 2.909 | 98.75 |
| p1 | TRITON_PRINT_AUTOTUNING=1 | 1.824 | 2.743 | 93.35 |
| p2a | 全新 TRITON_CACHE_DIR=…/a | 1.969 | 2.909 | 94.9* |

\* p2a 首 rep 114.7 ms 为 Triton 全量重编译后的首推抖动，rep2/3 恢复 93.1。

- 每个 accept 值都是精确有理数（计数器比值），会话内 3 reps 完全一致；
  探针为 temperature=0 贪心 → **RNG 排除**。
- nc1 关闭 prefix caching 后 3 reps 各自完整重算 7469-token prefill，
  accept/流仍逐位一致 → **prefill 进程内确定**（同一进程 3 次独立前向同比特）。

## 3. 首次分叉定位（贪心流跨重启对拍）

- det-3(1.771) vs det-4(1.969)：96-token 流在**输出位置 2** 首次分叉
  （115508「你现在」 vs 96540「它」），随后整链级联。
- 分叉位置 2 的 target logprobs（det-4b）：
  `top1=-1.600550889968872(96540)  top2=-1.600550889968872(115508)`
  —— **fp16 比特级精确并列**；p1(1.824) 同位置同样精确并列
  （−1.600907325744629 ×2）。并列是该上下文位置的**结构特征**。
- 跨进程 logprobs 存在 ~1e-3 量级、逐位置不同的 ULP 漂移
  （如 pos0：−1.32507(det4) vs −1.32371(p1)），但同位置的 top1−top2 间隙
  精确保持（0.40625 / 0.890625 不变）。
- nc1（全新编译态）与 det-4b 流逐位一致 → **编译缓存状态不影响数值**。

## 4. 假设证伪记录

| 假设 | 检验 | 结果 |
|---|---|---|
| H1 rocBLAS algo 彩票 | 3 个吸引子（det4/det5/p1）prefill 全量 Cijk 变体按 (name,grid) 计数对比 | **证伪**：12 类变体计数与 µs 全同（差异仅为截窗多一个 decode round） |
| H2 Triton autotune 抽奖 | TRITON_PRINT_AUTOTUNING=1 启动+探针 | **证伪**：请求路径 0 次多配置 autotune（`triton_matmul` 单配置；FLA chunk 内核不在 gfx906 路径上） |
| H2' autotune 盘缓存不一致 | 本 Triton `cache_results=False` → 每进程重 benchmark | 同上，无多配置 kernel 可 benchmark |
| H3 piecewise graph 捕获次序 | decode (name,grid) 全集差分（det1 vs det2） | **证伪**：113 类 (name,grid) 全同，µs/call 差 <1% |
| H4 非确定归约/atomics | 进程内 3 次独立 prefill 逐位一致（nc1） | **证伪**（per-launch 非确定与进程内逐位稳定矛盾） |
| RNG/采样 | 探针 temperature=0；会话内逐位一致 | **证伪** |
| 编译缓存状态 | nc1 全新 inductor 编译 vs det-4 热编译 → 流逐位一致 | **证伪** |

## 5. 根因与钉死方案

排除链（§4）把机制收束到"进程 init 期一次性状态 + ULP 级数值效应 + 对
(name,grid) 差分不可见"。三个吸引子的 prefill/decode kernel 变体、autotune、
编译产物、Triton 特化分派（1163=1163 目录级全同）全部一致后，剩余的
init 期数值自由度就是 **rocBLAS handle 的 workspace**：其尺寸在 handle 创建时
依可用内存/环境决定，进而决定 GEMM 的辅助分割；`Cijk_*` kernel 名只编码
algo，不编码 workspace 派生的内部分割，故 trace 差分不可见。

**s28d 的 "det1" 试验设置的是 `CUBLAS_WORKSPACE_CONFIG=:4096:8`——rocBLAS
根本不读它**；本栈生效的变量是 `ROCBLAS_WORKSPACE_CONFIG`（与 HIPBLAS 侧的
`HIPBLAS_WORKSPACE_CONFIG`）。

钉法（已采纳进 launcher，回滚 = 删两行）：

```bash
export ROCBLAS_WORKSPACE_CONFIG="${ROCBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HIPBLAS_WORKSPACE_CONFIG="${HIPBLAS_WORKSPACE_CONFIG:-:4096:8}"
```

落点：`tools/dcu/s27_graph_launch.sh`（S29 基线臂）与
`tools/dcu/serve_host_dflash2.sh`（生产臂），两处同改。

### 验收（同树连续 3 重启 × 3 reps）

| 重启 | env | accept（3 reps） | tok/step | step p50 |
|---|---|---|---:|---:|
| s29-pin1 | pinned | 1.969 / 1.969 / 1.969 | 2.909 | 92.95 |
| s29-pin2 | pinned | 1.969 / 1.969 / 1.969 | 2.909 | 93.08 |
| s29-pin3 | pinned | 1.969 / 1.969 / 1.969 | 2.909 | 93.18 |
| s29-adopted（无 pin，旧 launcher 拷贝，无意对照） | 未设 | 1.771 / 1.771 / 1.771 | 2.667 | 93.11 |
| s29-adopted2 | 已采纳 launcher，/proc/environ 确认两变量在 | 1.969 / 1.969 / 1.969 | 2.909 | 93.03 |

- pinned 世界与既有 1.969 世界逐位同一（token 流 96/96 相同，
  `.s29/capture-s29-pin3.json` vs `.s29/capture-s29-det-4b.json`）。
- 无 pin 历史分布 {1.771×4, 1.824×2, 1.969×3}（含 s28d），pinned 4/4 落
  1.969；P(偶然)≈(3/9)⁴≈1.2%。
- step p50 不受影响（92.95–93.18，与基线同分布）；钉住 1.969 即把本日
  draft 臂的服务端 fwd t/s 从 28.6 抬回 31.2 口径（s28d §3 的测算）。

**残留与降级（同日稍后复核）**：s29-final（与 adopted2 完全相同的 launcher
与 env）落在 **1.824** —— workspace pin 把世界分布**强烈偏向 1.969
（pinned 基线 5 次中 4 次）但不是完全钉死**。结论降级为：
1) 机制已定位到"进程 init 期一次性数值状态"（本日全部常规机制均证伪，
   见 §4）；2) workspace pin 是有效的强偏置手段（pinned 1.969 占比 ~80%
   vs 未钉 ~33%）但**不构成验收意义上的确定性**；3) **accept 类对比的
   N≥2 重启取分布协议保持强制**（施工图 WP-A 的 fallback 分支生效：
   "accept 多吸引子为系统固有"的部分结论 + N-restart 纪律）。
   后续 WP-B/C 的 accept 数据一律按分布呈现，不取单点。
另外：同世界内跨进程仍有 ≤0.015 nat 的 top-1 logprob ULP 漂移
（轨迹逐位一致、未触发翻位；`perf1_e2e_parity` 口径的近平局协议继续适用）。

## 6. 附带发现（独立 bug，另行处理）

`/v1/completions` + `logprobs≥2` 路径存在可致服务退出的缺陷：
`logprobs.py:_update_sample_logprobs → convert_ids_list_to_tokens`
收到非法 token id（Rust tokenizer 报 `OverflowError: number too small`），
EngineGenerateError 后 API server 整体 shutdown（det-2 sample_tokens RPC
timeout、det-5 本缺陷，两次）。logprobs=1 的流式路径（parity 用）未复现。
与彩票主链无关（探针不带 logprobs），但**在修复前禁止在现网服务上使用
logprobs≥2 的 completions 请求**。

原始数据：`.s29/`（serve-*.log、s20_s29-*.json、capture-*.json、
kdiff-*.csv、/mnt/game/s29-prof/、/mnt/game/s29-tc/{a,b}）；
pin 采纳 diff：`tools/dcu/s27_graph_launch.sh`、`tools/dcu/serve_host_dflash2.sh`。
