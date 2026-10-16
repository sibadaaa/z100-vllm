# Perf1 task-3 closure — GDN MTP kernel + Triton matmul fallback (gdn-opt)

Date: 2026-10-13. `.so` for every number below: `_C.abi3.so` md5 `6fba05966e153b3d04771d6960c782bb`,
`_rocm_C.abi3.so` md5 `8e98bd60b94ebeb6f69ed8e39d754647` (7e62d7ec for the earlier sweep).

## 1. Result in one line

The GDN MTP decode kernel is **36x faster on the GPU** at a live signature byte-identical to the
standalone probe, but the change is **+9.4 ms/step (about -8% decode throughput) end-to-end** and is
**not shipped**; the mechanism is a **CPU-launch-bound decode step** where the kernel's 33 ms/step of
saved GPU time is not on the critical path while its +190 us/call of host cost is.

## 2. Kernel before/after (real production shape)

Shape: 1 request, T=3 speculative tokens, q/k heads 4, value heads 12 (48/TP4), key=value head dim 128,
fp16 q/k/v, **fp32 state pool (1353 slots = 1.06 GB) in the native layout `[slot][head][v][k]`,
k contiguous** (stride `(196608, 16384, 128, 1)`), accepted int32.
Live signature reported by the in-worker probe: `(T=3, R=1, heads=4, kv_heads=12, KD=128, VD=128,
BV=4, warps=1, grid=(32,12,1), float32)` — identical to the standalone probe.

| path | kernel us/call | event us/call | host us/call |
|---|---|---|---|
| C++ `..._gfx906_mtp_update_kernel` (grid (6,1,1), 256 thr) | **705.2** | 702.0 | 20.3 |
| new Triton kernel (grid (32,12,1), 64 thr) | **19.5** | 112.9 | 119.9 standalone / **210.5-214.9 live** |
| v2 C++ (transposed pool, ineligible in production) | 26.5 | 40.5 | — |

Root cause of the 705 us (falsifies "it is the 6-block grid"): in the native layout consecutive lanes
are 512 B apart, so every warp state load touches 32 cache lines. Same kernel, same grid, with a
v-contiguous view: **26.2 us** (numerics wrong there; perf probe only). `mtp_v2` misses its gate
because production passes `state.stride(2)=128` (`VLLM_GFX906_GDN_MTP_V2_DEBUG=1` →
"GFX906_MTP_V2 MISS: state_s3=1 state_s2=128"). The new kernel keeps the production layout and maps
lanes onto the key axis of a value row.

## 3. Refutation: A-B-A end-to-end (session 6, all arms k=2, MAX_M=4, BF16_GEMV=0, same .so)

| arm | step p50 | decode t/s | tok/step | accept | drafts |
|---|---|---|---|---|---|
| A control | **139.397 ms** | 20.927 | 3.048 | 0.8863 | 255 |
| B `VLLM_QWEN35_MTP_GDN_TRITON=1` (+PROBE+DEBUG) | **150.213 ms** | 19.013 | 3.048 | 0.8582 | 261 |
| C control | **139.963 ms** | 20.203 | 3.048 | 0.8863 | 255 |

A≈C, B is +10.3 ms on the arm p50 (+9.4 ms on the per-repeat medians 148.24/153.67/148.73) ⇒ the
regression is real, single-variable, and not drift. Raw JSON: `/mnt/game/perf1/gdn_aba_{a,b,c}_e2e.json`,
serve log `/mnt/game/perf1/serve_gdn_aba_b.log`.

## 4. Mechanism (live probe + CPU decomposition)

In-worker probe, 800 samples in arm B (`/mnt/game/perf1/serve_gdn_aba_b.log`), independently
reduced by the verifier:

| quantity | value |
|---|---|
| live kernel `gpu_us` | **p50 44.5 us** (only 4/800 samples >500 us, all JIT) |
| live host `host_us` | **mean 224.8 us** (p50 219.2, min 177.8, p90 255.1 — every sample >150 us) |
| `interval_us` (per-layer host gap) | p50 3227.7 us |

* **Accounting closes to 0.02 ms/step**: 224.8 us x 48 = **10.79 ms/step** predicted, **+10.81 ms/step**
  measured (B - A in the same session). Per-layer host 3.23 ms x 48 = 155 ms is the step's real
  critical path length; the control step is 139.4 ms, so the step is at/over the CPU-launch crossover
  and the 33 ms/step of GPU time the kernel removes is not on the critical path.
* The kernel itself is fast **live** too: 44.5 us vs the C++ 705.2 us (16x); the standalone probe
  measured 19.5 us at the same signature, so the pinned worker's larger state pool/TLB and shared L2
  cost about 2.3x, not an order of magnitude.
* The cost is the **launch path**: the live worker pays ~225 us/call of host time where the C++ op pays
  ~20 us; my standalone enqueue loop measured ~120 us, i.e. the pinned worker pays roughly double.
* CPU-only decomposition (`tools/dcu/perf1_gdn_dispatch_cost.py`, 2000 iters): dispatch delta over the
  C++ path **0.12 us/call**, wrapper Python **18.6 us/call** ⇒ everything above the launcher is free.
* Not the cause: shape/grid difference (live grid (32,12,1), R=1), recompilation (single
  specialization, cache pre-warmed; the 4 slow samples are JIT), syncs/`.item()` (none), entry-call
  count (48/step — the draft pass runs the MTP head, not the 48 GDN layers), `torch.empty`/allocator
  (same as the C++ op).
* Fix direction, therefore: **launch path, not kernel math** — a C++ op, a cached launcher with
  precomputed grid/args, or batching the 48 calls.

Instrument bug, self-reported and fixed: the first probe drained the *current* call's stop event
instead of the entry 64 calls old, so its first `gpu_us` (4529 us) was kernel + a full pipeline stall
and is **invalid**; `host_us` is a plain CPU timer and was unaffected. Fix: drain moved into
`start_call` before the slot is overwritten; the verifier's 800-sample run used the fixed version.

Parity at the token level: arm B vs A/C bifurcates once at position 59 (187/256 identical after),
pre-flip mean |dlp| 0.0039 nats — the near-tie class documented in
`docs/archive/result/z100-18e-mtp-parity-failure-and-regression-result.md`, and the reason a kernel
swap cannot be accepted on per-op parity alone.

## 5. Parity evidence (three levels)

* GPU per-op vs the C++ op: max|dout| 7.6e-06, max|dstate| 1.79e-07, **1 of 4608 fp16 outputs differ**
  (the C++ path vs the torch reference itself differs in 2/4608).
* 8-case GPU harness (R=1..3, T=1..4, incl. -1 slots): worst fp16-diff 14/N, max|dstate| 4.8e-07
  (`/mnt/game/perf1/gdn_parity.log`).
* CPU interpreter vs a pure-torch port of the C++ per-token slot logic (10 cases):
  ≤6.0e-08 / 2.4e-07, and **exactly 0.0 on the negative-slot cases** — this found and fixed a real
  divergence (the C++ kernel re-reads the stale slot after a broken slot chain; the kernel now tracks
  `chain_ok` and re-reads only in that case).
* Dispatch smoke test `tools/dcu/perf1_gdn_smoke.py`: 6 modes incl. `VLLM_QWEN35_MTP_GDN_CUSTOM=0`
  still bypassing to the eager bisect path.

## 6. Triton matmul fallback (6.7% of the step): modules and go/no-go

Identification (grid arithmetic with BLOCK_SIZE_N=64, kernel neighbourhood, checkpoint structure):
`(1,1,1)` = GDN `in_proj_ba` per-rank N=24 (48/step, 186.2 us, 8.94 ms/step, 1.3 GB/s);
`(20,1,1)` = `mtp.fc` ColumnParallelLinear per-rank N=1280 (about 2/step, 364 us, 0.74 ms/step);
`(970,1,1)` = `lm_head` vocab shard per-rank N=62080 (about 3.1/step, 1654 us, 5.16 ms/step,
384 GB/s = 59% of the measured 651 GB/s peak).

Measured alternatives (fp16, fp64 rel_err identical across all routes ⇒ no accuracy trade):
`in_proj_ba` GO — M=1..4 133.0/132.5/155.7/155.8 us → 12.5/15.3/19.0/23.5 us with
`bf16_emu_gemv nt=128` (6.6-10.4x, about -6.6 ms/step); `mtp.fc` GO at M≤2 (342.7→58.9, 339.5→87.8)
and at M=3/4 against the alternative this code path actually takes (Triton 415.8/411.0 → 138.6/202.2),
about -0.5 ms/step; `lm_head` GO only at M≤2 (1589.8→893.1, 1588.1→1277.3), tie at M=3, loses at M=4.
**Requires `S11F_BF16_GEMV=1`** (`VLLM_DCU_USE_BF16_GEMV=0` today disables the whole dense-GEMV path)
plus the widened `_gfx906_gemv_shape_ok` in `layers/utils.py`. Suggested cells:
`n_tokens<=4 and n_out<=1280`, plus `n_tokens<=2` above that.
Audit of that change: 42/43 checks pass, one stale test found and fixed
(`tests/kernels/test_gfx906_bf16_gemv.py` compared against a hardcoded `nthreads=64` while the
dispatch now picks 128 for N≤4096); evidence `/mnt/game/perf1/gdn_gemv_audit.log`.

## 7. Negative results / not done

* **Prefill GDN** `..._prefill_parallel_kernel`: 96 calls x 9829 us = **943.6 ms** in the trace,
  grid (1536,1,1) = one block per (seq, head, value) column over 256 tokens with about 6 block
  barriers per token. Out of scope; a chunked-parallel rewrite is a TTFT project.
* **`lm_head` FP8**: not quantized (`modules_to_not_convert` contains `lm_head`; checkpoint BF16
  [248320,5120] → served FP16, 636 MB/rank/call) — the only structural lever is a checkpoint/quant
  change, and at M>=3 the emulated GEMV loses anyway.
* **C++ port of the coalesced-lane kernel**: parked permanently for this regime — it would have the
  same ~20 us host cost as the existing op, so the GPU saving would stay hidden.
* **CompiledKernel fast-launch path**: not attempted; the refutation is now attributed to the launch
  path (225 us/call), so this or a C++ op is the right fix *if* the kernel is ever needed.
* Live kernel duration **is** captured after all (verifier's 800-sample reduction of the fixed probe):
  **p50 44.5 us**, see §4.

## 8. Banked for a launch-overhead-free regime

Per-layer host 2.83 ms x 48 = 136 ms ≈ the step time, and 663 kernel launches/step with
`--enforce-eager` (CUDA graphs disabled). If graph capture or launch-count reduction removes the CPU
bottleneck, the GPU becomes critical again and this kernel's **33 ms/step** (48 x 685 us) should
materialise. The kernel is landed, default-OFF, parity-verified, and the fixed probe
(`VLLM_QWEN35_MTP_GDN_TRITON_PROBE=1`) is ready to measure it in that regime.

## 9. Artefacts

Kernel: `vllm/model_executor/layers/fla/ops/gfx906_gdn_mtp_triton.py`;
dispatch: `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py` (env `VLLM_QWEN35_MTP_GDN_TRITON`,
default OFF; `VLLM_QWEN35_MTP_GDN_CUSTOM=0` still bypasses to eager).
Tools: `tools/dcu/perf1_gdn_{bench,parity,interp,smoke,gpu.sh,warm.py,warmcache.sh,gemv_audit,dispatch_cost}.py`.
Logs: `/mnt/game/perf1/gdn_{sweep,parity,interp,smoke,gemv_audit}.log`, `mm_probe.log`,
`gdn_aba_{a,b,c}_e2e.json`, `serve_gdn_aba_b.log`, `gdn_bench_baseline.log`.
