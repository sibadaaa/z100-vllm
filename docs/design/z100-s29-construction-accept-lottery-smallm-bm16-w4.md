# S29 施工图：accept 彩票根因 → SMALLM 复核 → BM16 → W4 探针（2026-10-25）

> **交给实施 agent 的完整任务书**。按 WP 顺序执行；每个 WP 独立可验收。
> **当前基线**（勿重推导）：`s27_graph_launch.sh` graph-A 配方、7469 真实文档、
> 96 max-tokens 口径下 **step p50 = 93.14 ms**（A1 + draft-3D 已采纳），accept 1.771
> （2.667 tok/step）——但 accept 是重启彩票（见 WP-A）。服务现停在最新采纳臂（:8000）。
> **今日已采纳改动**：`_gfx906_gemv_shape_ok` 加 `n_out<=64→M<=8`（旋钮
> `VLLM_DCU_GEMV_NARROW_MAX_M`，默认 8）；`triton_unified_attention.py` draft 形状
> `(128,4)→BLOCK_M 16` 接入 small-q 3D 路由（旋钮 `VLLM_GFX906_ATTN_VERIFY_3D_DRAFT`，默认 1）。
> 证据：`docs/result/z100-s28-a1-smallm-ab.md`、`docs/result/z100-s28d-draft3d-accept-lottery.md`。

---

## 0. 环境须知（先读，全是踩过的坑）

1. **GPU 实测必须在 DSH 沙箱外**。DSH 会话沙箱挡住 HSA 的 /sys、/dev 访问
   （`hsa_init=4104`、"No CUDA GPUs" 是假象）。两条正路：
   `docker exec fp8-w8a16-gpu bash -c '...'`（容器有 4 GPU + torch 2.11；
   但容器 py3.10 缺 `typing.Never`，跑不了本树的 pytest），或沙箱外的宿主 shell。
   **宿主宿主直跑是唯一能跑 pytest 的路径**。
2. **服务生命周期**（每次重启 ~4.5 min）：
   ```bash
   # 起（后台）：
   cd /home/lyh/z100-vllm
   MAX_SEQS=4 MBT=8192 PORT=8000 bash tools/dcu/s27_graph_launch.sh > .sXX/serve-<arm>.log 2>&1 &
   # 等 ready（最多 ~5 min）：
   for i in $(seq 1 24); do sleep 15; code=$(curl -s -o /dev/null -w "%{http_code}" \
     --max-time 3 http://127.0.0.1:8000/health); [ "$code" = 200 ] && break; done
   # 停（注意正则转义，防止 pkill 匹配到自己的 shell —— 裸字符串会自杀）：
   pkill -f "entrypoints[.]cli"; sleep 12; ps aux | grep "[e]ntrypoints" | wc -l   # 应为 0
   ```
   任何时刻只有一个服务；窗口结束留一个可用服务在 :8000。
3. **探针鉴权**：服务带 API key（`/mnt/game/host-serve/api-key`）。探针必须套壳：
   ```bash
   export DTK_ROOT=/mnt/game/host-runtime/dtk2604
   source tools/dcu/env.sh
   export PYTHONPATH=/home/lyh/z100-vllm/tools/dcu
   .venv/bin/python -c "
   import sys; from s23_common import install_auth; install_auth()
   sys.argv=['s20_single_probe.py','--prompt-file',
     'docs/design/z100-dual-prefill-reconciliation-projection.md',
     '--truncate-tokens','7469','--max-tokens','96','--repeat','3',
     '--tag','<tag>','--out','.sXX']
   exec(open('tools/dcu/s20_single_probe.py').read())"
   ```
4. **环境核验**：每次起服务后必须核对两处——serve log 的
   `[host-serve] policy: FP_FULL ...` 行 + `/proc/<pid>/environ`
   （`tr '\0' '\n' < /proc/$(pgrep -f "entrypoints[.]cli" | head -1)/environ | grep <KNOB>`）。
5. **pytest（宿主）**：
   ```bash
   export PYTHONPATH=/home/lyh/z100-vllm:/mnt/game/triton-gfx906/install
   export HIP_VISIBLE_DEVICES=0   # 或空闲卡
   .venv/bin/python -m pytest tests/... -q
   ```
6. **生产 .so**：`vllm/_rocm_C.abi3.so -> /mnt/game/port/build-sync/temp/_rocm_C.abi3.so`
   （md5 `f42d0064…`）。Python 侧改动重启即生效；csrc 改动走
   `docs/contributing/incremental_build.md` 的构建树同步流程（本项目先不动 csrc）。
7. **度量纪律**（AGENTS.md）：每个改动 真实形状微测 → 数值对照 → 服务 smoke →
   同口径 e2e → 文档；原始 JSON 落 `.sXX/`；结果文档 `docs/result/z100-sXX-*.md`。

---

## WP-A（最高优先）：accept 彩票根因定位

**现象**（`docs/result/z100-s28d-draft3d-accept-lottery.md` §2）：同树多次重启，
accept_len 呈现 {1.96875, 1.8235, 1.7714} 三个吸引子（= tok/step 2.909/2.743/2.667），
服务内 3-5 reps 完全稳定、跨重启漂移 ±6~10%。step p50 跨重启稳定（<0.1 ms）。
`CUBLAS_WORKSPACE_CONFIG=:4096:8` 无效。幅度上钉住 attractor ≈ ±10% t/s，
大于剩余所有 kernel 优化的总和。

**目标**：找到跨重启变化的第一个数值环节，并给出"钉死"方案；验收 = 同树连续 3 次重启
accept 逐位一致。

**步骤**：

1. **采集**（两次重启，同树同 env，各自抓一份稳态 trace）：
   ```bash
   # 服务必须带 profiler 配置启动（见 tools/dcu/s27_five_phase_profile.py docstring）：
   #   --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/mnt/game/s29-prof
   # 每次重启后：
   MAX_SEQS=4 MBT=8192 PORT=8000 bash tools/dcu/s27_graph_launch.sh > .s29/serve-det-<n>.log 2>&1 &
   .venv/bin/python tools/dcu/s27_five_phase_profile.py --phase collect \
     --profile-steps 20 --tag s29-det-<n>   # 驱动真实 7469 请求 + /start_profile
   .venv/bin/python tools/dcu/s27_five_phase_profile.py --phase parse --tag s29-det-<n>
   # 记录该次重启的 accept（probe 3 reps，见 §0.3 模板）
   ```
   重复第 2 次重启。若两次 accept 相同，换 env 微扰（如多设一个无害 env）再采，直到
   拿到**两个不同 accept 吸引子**的 trace 对（历史数据表明概率不低：1.77/1.82/1.97 都出现过）。
2. **逐 kernel diff**（draft 相位优先）：
   ```bash
   # 对两份 trace 按 (kernel 名, grid) 聚合 µs/call 与 calls/round，输出差分表
   .venv/bin/python - <<'EOF'
   # 参考：本会话已用过的模式 —— gzip+json 读 traceEvents，
   # 过滤 ph=='X'、cat kernel，按 name+str(args.grid) 聚合 dur/calls，两文件对拍
   EOF
   ```
   差分表的每一行回答：同 kernel 的 µs/call 或 grid 变了吗？
3. **嫌疑清单**（按可检验顺序）：
   - **H1 rocBLAS algo 彩票**：kernel 名 `Cijk_...` 的**变体后缀**在两次 trace 间是否不同
     （algo 选择随 workspace/内存状态变）？重点：draft lm_head（N=62080×5120, M=7，
     635 MB）、selector 侧 GEMM。若名字变了 → 机制坐实，钉法 = 固定 workspace/algo
     （`HIPBLAS_WORKSPACE_CONFIG` 已试无效 → 试 `HIPBLAS_LT_...`/选定 algo env，
     或把该 GEMM 换成确定性 kernel）。
   - **H2 Triton autotune 彩票**：draft 相位所有 `triton_*` kernel 的 µs/call 分层对比；
     检查 draft 路径上有没有多 config 的 `@triton.autotune`（`triton_matmul` 是单 config，
     嫌疑低；查 `_gdn_mtp_update_kernel`、DFlash conv/selector 的 kernel、
     `triton_red/poi_fused_*` 是 Inductor 产物——它们的 config 固化在编译产物里，
     但**编译缓存 miss 时会重新 autotune**：检查 `VLLM_CACHE_ROOT`
     （/mnt/game/vllm-cache-host）在两次重启间是否有新写入（`find -newer`）。
   - **H3 piecewise graph 捕获次序**：对比两次 trace 的捕获相关 kernel 顺序/数量；
     `--compilation-config` 的 capture sizes [1,2,4,8,16]。
   - **H4 非确定性归约**：draft 相位是否有 atomics 类 kernel（`elementwise_kernel`、
     scatter）——grid 内无序归约在 wave 调度变化时结果抖动。
4. **钉死方案与验收**：按定位结果实施（env/代码），然后**同树连续 3 次重启 × 3 reps**，
   accept 逐位一致 ⇒ WP-A 完成；写 `docs/result/z100-s29-accept-pin.md`。
   若三个假设全部证伪：把 trace 对与差分表存档，结论改为"accept 多吸引子为系统固有"，
   转 WP-B 的 N-restart 协议作为长期度量纪律。

**不要做**：不要试图"选出最好 attractor"（2.909）——目标是确定性，不是调参；
不要在根因不明时继续叠加 accept 类单点结论。

---

## WP-B：SMALLM 复核（WP-A 验收后执行；若 WP-A 判"固有"，用本 WP 的 N-restart 协议）

**背景**：`VLLM_DCU_SMALLM_GEMM=rocblas`（S22 已建派发）使 draft tower 的 matmul
走 rocBLAS。s28 实测 step 101.1→96.46（−4.65 ms）但 accept 1.969→1.771，被否决；
但 accept 彩票发现后该否决证据不足（1.7714 落在午后对照吸引子集合内）。
微测依据：`docs/perf/data/draft-m7-sweep.json`（除 lm_head 外全形状 rocBLAS 快 1.4–5.6×）。

**步骤**：
1. 臂 = {control：SMALLM=off, arm：SMALLM=rocblas}，各 **2 次重启 × 3 reps**
   （用 §0.3 模板，tag `s29-smlm-{ctl,arm}-{r1,r2}`）。每臂每次重启都记录
   accept + step p50 + `docs/perf/data/draft-m7-sweep.json` 对应形状的服务级印证。
2. **判据**：
   - step p50 差值（稳定指标）：预期 −4.5~−6 ms；
   - accept：报分布（每臂 2 值）。若 arm 的两个 accept ≥ control 的两个 accept 的最小值
     且均值不低 ⇒ adopt；若 arm 一致性走低 ⇒ keep rejected，存档。
   - 8-prompt parity（`perf1_e2e_bench.py --mode parity` + `perf1_e2e_parity.py`），
     逐 token 披露（近平局协议见 `docs/perf/ATTN-OPT-20261025.md` §3）。
3. 产出 `docs/result/z100-s29-smallm-recheck.md`；adopt 则把
   `export VLLM_DCU_SMALLM_GEMM=rocblas` 写进 `tools/dcu/s27_graph_launch.sh`
   并同步 `serve_host_dflash2.sh`（两处 diff 关系见其头注），回滚 = 删该行。

---

## WP-C：A2 —— FA verify BLOCK_M=16（等用户 parity 裁决后执行）

**背景**：目标 FA verify（M=8，head 256/GQA6）BLOCK_M=8 → KV 被重读 8 遍，
逻辑带宽 11.6–15.7 GB/s。S24 kernel 扫描（`docs/result/z100-longctx-vram-optimization-result.md`
§12）：BLOCK_M=16 全上下文 **−45~−52%**、32k ITL −7.3%；当时因 tl.dot constexpr 变化
（MMA 求和顺序）出现 1 处近平局翻转被拒。旋钮现成：`VLLM_GFX906_ATTN_DECODE_BLOCK_M=16`。

**步骤**：
1. e2e A/B：臂 = {默认（BM8），BM16}，各 2 重启 × 3 reps（同 WP-B 协议）。
   预期 step −3.5~−4.5 ms（8.22 ms 的 FA 3D 项减半）。
2. **parity 板**（用户裁决项，实施 agent 只产出证据不许自行放行）：
   `perf1_e2e_parity.py` 全量输出翻位清单 + 每个翻位的 top-2 logprob 差；
   协议：翻位仅当 `Δlogprob(top1,top2) < 0.05 nat` 且逐条披露、臂内确定性成立。
3. 过板 → launcher 加 `export VLLM_GFX906_ATTN_DECODE_BLOCK_M=16`（连同 WP-B 的
   改动一次提交）；不过板 → 存档，旋钮保持默认 0。
4. 测试：`tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py` 全绿
   （BM16 是既有旋钮路径，S24 已有 per-op 证据，无需新测试；跑一遍防回归即可）。

---

## WP-D：W4/`v_dot8_i32_i4` GEMV 探针（独立，不需要服务；与 WP-A 并行）

**目标**：验证 `docs/perf/W4-RESEARCH-20261025.md` §0.3 的带宽模型
（W4+dot8 ≈ 600–700 GB/s vs fp8 现网 163），这是 W4 立项的第一道硬闸。

**步骤**：
1. 扩展 `tools/dcu/gemv_probe.cpp`（已有 fp16 lds 探针框架，451 GB/s 结论来源）：
   新增 `w4_dot8` 变体——权重按 u32×8 打包（32 个 4-bit/16 B）、激活 int8
   （`per_token_group_quant` 的输出口径）、`v_dot8_i32_i4` 内循环、fp32 累加后乘
   per-group scale。ISA 前置：扩展 `.s23/probes/isa_gate.py` 确认
   `v_dot8_i32_i4` 在 gfx906 可用（`v_dot4_i32_i8` 已在产线）。
2. 测量：N∈{5120, 8704}、K=5120、M=8，L2-flush，30 iters，对照行 = fp16 `lds_m8`
   （451 GB/s）与生产 fp8 stream（163 GB/s）。数值对照：合成数据 vs fp32 参考，
   记录相对误差（预期 1e-2 量级，W4 特性，不是 bug）。
3. **判据**（写死在 W4-RESEARCH §4 步 1）：实测 ≥450 GB/s ⇒ go（下一步是单层
   量化样张，另立 WP）；<300 GB/s ⇒ 停，存档证伪。
4. 产出：probe 源 + `docs/perf/data/w4-probe-<date>.json` +
   W4-RESEARCH 文档附录一段（数字 + 判定）。**不进生产、不动 launcher**。

---

## 顺序与依赖

```
WP-A（方法论解锁，~0.5-1 天）──→ WP-B（~0.5 天，2+2 重启）──→ WP-C（~0.5 天，需用户板）
WP-D（独立并行，~0.5 天，只要空闲 GPU）
```

- 全部完成后：`docs/result/z100-s29-*.md` 三份 + SOL-GAP §5 追加一行总账。
- 禁止事项：不做本图之外的"顺手优化"；不动 `csrc/`（W4 只到探针为止）；
  不在 accept 彩票未钉死前下任何 accept 单点结论。
