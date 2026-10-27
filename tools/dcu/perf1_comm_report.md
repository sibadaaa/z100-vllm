# perf1 — communication layer + serving-config levers (task-2, owner `comm-opt`)

Scope: L2 all-reduce cost and L1 serving configuration for Qwen3.8-27B-FP8, TP=4,
`--enforce-eager`, MTP k=2, conc=1, 4× DCU (2× Z100L + 2× Z100, PCIe, P2P on).

Environment during round 1: production service `qwen3.8-27b` resident on :8000
(1 `vllm serve` process), one `verifier` engine arm loading concurrently
(11:00-11:07) ⇒ **round-1 absolutes are contention-inflated and one of its
headline results was a false positive (§1b)**. Round 2 (protocol × prefill
scale) and the staggered chain were re-run uncontended under
`/tmp/perf1_build.lock` at 11:38–11:45; §1b/§1c are the valid numbers.

Reproduce:
```
docker exec z100-vllm bash -c 'flock -w 1800 /tmp/perf1_build.lock bash -c \
  "bash /src/tools/dcu/perf1_comm_matrix2.sh > /mnt/game/perf1/comm_matrix2.log 2>&1"'
docker exec z100-vllm bash -c 'flock -w 1800 /tmp/perf1_build.lock bash -c \
  "bash /src/tools/dcu/perf1_comm_chain.sh > /mnt/game/perf1/comm_chain.log 2>&1"'
```

## 1. All-reduce ladder, round 1 (us/call, 4 ranks, fp16, P2P on)

Min over 2 repeats of the mean of 100 back-to-back calls, min over the 4 ranks
(all four within 6%, so per-rank spread is negligible). Log:
`/mnt/game/perf1/comm_matrix.log`. Sizes are the real per-rank payloads of a
5120-hidden model: 10 KiB = `[1,5120]` f16 (MTP/draft), **30 KiB = `[3,5120]`
f16 (k=2 verify, dominant)**, 120 KiB = `[12,5120]`.

| variant | 2 KiB | 10 KiB | **30 KiB** | 120 KiB | 1 MiB |
|---|---:|---:|---:|---:|---:|
| **base** (RCCL defaults, P2P) | 107.3 | 108.8 | 109.8 | 108.8 | 126.4 |
| `HSA_FORCE_FINE_GRAIN_PCIE=0` (SHM) | 105.2 | 104.3 | 109.7 | 194.1 | **1123.2** |
| `NCCL_ALGO=Tree` | 113.6 | 128.6 | 88.7 | 93.4 | 171.6 |
| **`NCCL_PROTO=LL`** | **83.0** | **73.7** | **74.1** | **73.8** | 172.7 |
| `NCCL_PROTO=Simple` | 106.9 | 103.4 | 128.7 | 104.8 | **120.6** |
| `NCCL_MIN_NCHANNELS=1` | 103.7 | 99.5 | 99.4 | 113.6 | 160.9 |
| `NCCL_LL_THRESHOLD=131072` | 109.9 | 109.5 | 860.9 | 867.4 | 866.1 |

Winner per size: **2 KiB / 10 KiB / 30 KiB / 120 KiB → `NCCL_PROTO=LL`** (74–83 us,
−32% vs default at the decode payloads); **1 MiB → `NCCL_PROTO=Simple`** (120.6).
The default ladder is *flat* from 2 KiB to 120 KiB (107–110 us) ⇒ at decode
payload sizes this platform pays pure latency, not bandwidth.

> **CORRECTION (round 2, uncontended, §1b): the round-1 LL win does NOT
> reproduce.** Under the global lock with no engine arm loading, the default
> configuration measures 84.6–85.5 us at the decode payloads and forced LL
> 85.7–86.9 us — a +1.5% difference, i.e. **no gain**. Round 1's "default 108 vs
> LL 74" was an artifact of the concurrent `verifier` arm loading the engine on
> all four DCUs, which penalised the default (Simple/Ring) path roughly twice as
> much as the forced-LL path. **The recommendation below is withdrawn: keep
> protocol auto.** The rest of the round-1 table (P2P vs SHM, Tree, LL_THRESHOLD,
> MIN_NCHANNELS) is directionally consistent with round 2 for the sizes round 2
> covers, but treat round-1 absolutes as contended.

`HSA_FORCE_FINE_GRAIN_PCIE=1` is **not** a decode lever: 2–30 KiB is identical
with and without it (105–110 us both ways). It is worth 1.8× at 120 KiB and
**8.9× at 1 MiB** (1123 → 126 us), i.e. its value is entirely in prefill/conc>1
(S17: 8K TTFT 1.89×, conc8 +38%). Keep it on; but the decode-side justification
is "no regression", not a win.

Measurement caveat: the ladder times back-to-back `dist.all_reduce` calls on one
stream, so a value below the CPU enqueue cost of `dist.all_reduce` would be
CPU-bound rather than GPU-bound. The default row (107–110 us) is well above that
risk; if the LL row (74 us) is partly CPU-bound its true GPU time is *lower*,
which strengthens the direction but weakens the magnitude. The chain experiment
(§1c) is the GPU-bound check.

### 1b. Round 2 under the lock — clean protocol decision (11:38–11:41)

Same script, `PF1_SIZES` extended to prefill scale, service resident but no arm
loading it (`/v1/models=200`, 1 `vllm serve` process). Logs:
`/mnt/game/perf1/comm_matrix2.log`, `comm_nccl_debug.log`; verdict:
`perf1_comm_verdict.py` → `/mnt/game/perf1/`.

| payload | **default** | LL | LL,Simple | Simple |
|---|---:|---:|---:|---:|
| 10 KiB | **85.5** | 85.7 | 106.2 | 112.5 |
| 30 KiB | **84.6** | 86.3 | 109.0 | 108.3 |
| 120 KiB | **85.0** | 86.9 | 109.4 | 101.3 |
| 1 MiB | **119.4** | 173.3 | 119.5 | 120.4 |
| 4 MiB | **275.5** | 594.5 | 276.0 | 275.8 |
| 20 MiB | **1190.4** | 2796.0 | 1189.5 | n/a¹ |

¹ `r2_Simple`'s 20 MiB row was cut by the runner's `head -12`; it is ≈ default
(the 4 MiB simple row is identical).

* **Default ≈ forced LL at 10–120 KiB** (Δ +1.1/+2.0/+2.2%). No protocol win to
  take. The clean per-call cost is ~85 us, flat from 2 KiB to 120 KiB.
* At prefill scale LL is 1.45× worse at 1 MiB, 2.16× at 4 MiB and **2.35× at
  20 MiB** (+1605.6 us per all-reduce).
* `NCCL_PROTO=LL,Simple` (explicit allowed list) keeps the large sizes at
  default cost but **does not** recover LL's small-message choice (106–109 us at
  the decode sizes) — no free lunch.
* `NCCL_DEBUG=INFO` on the default run shows the tuner picking, for 30 720 bytes,
  `Algo 1 proto 2` = **Ring + Simple**; 8 channels, all `via P2P/IPC`
  (`comm_nccl_debug.log`).

Decision arithmetic (Lead's rule): (a) decode gain required ≤ −25%, measured
**+1.5% → FAIL**; (b) 20 MiB penalty: +1605.6 us/AR × 140 ARs = +224.8 ms per
prefill chunk, ×4 chunks = +899 ms on a 7160 ms TTFT = **+12.6% → FAIL**;
(c) chain below → FAIL. **Verdict: keep protocol auto; do not set `NCCL_PROTO`
in the recipe.**

### 1c. Staggered chain (11:41–11:45): does the win survive a straggler?

`perf1_comm_chain.py`: 50 steps × 140 collectives of 30 KiB, each preceded by a
rank-dependent elementwise burst; ranks 2/3 (Z100) get +300 us in the `s300` rows.
All four ranks report the *same* wall time in every row, which is the expected
lockstep signature (the fast ranks wait, the slow rank sets the pace).

| variant | us per collective (critical) | vs default |
|---|---:|---:|
| default, stagger 0 | 2696.5 | — |
| LL, stagger 0 | 2462.0 | −234.5 (−8.7%) |
| default, stagger 300 | 3338.0 | — |
| LL, stagger 300 | 4084.6 | **+746.6 (+22.4%)** |

Two conclusions, with one caveat. Conclusion: with a straggler present — the
in-service regime — forced LL is *worse*, not better; the LL protocol's chunked
pipeline is more sensitive to rank skew. Verdict per the Lead's rule (c):
keep protocol auto. Caveat: the chain is indicative, not decisive — the burst
calibration differs per run (`iters` 21–25, calibrated 48.6–56.5 us/pass), so
absolute chain times are not comparable across variants at better than ~10%; the
direction of the `s300` row is nevertheless far outside that band.

## 2. The 22% "NCCL" headline is mostly skew-wait, not communication

Offline decomposition of all 4 ranks' traces (atoms pickles,
`/home/lyh/vllm-test/results/atoms/`, analyser `tools/dcu/perf1_comm_skew.py`),
7020 collectives over 50.14 decode steps = 140.0/step:

* Every collective: the four ranks' kernel **END times differ by median 4 us**
  (p90 6 us) while their **START times differ by median 313 us** (p90 649 us).
* Hence per-call kernel duration = arrival wait + a short tail. Split:

| quantity | ms/step | % of 219.6 ms step |
|---|---:|---:|
| intrinsic collective on the critical path (min-over-ranks duration = the last arriver's tail), median 72 / mean 79.5 us per call | **10.1–11.1** | **4.6–5.1%** |
| rank0 spin-wait (of its 49.4 ms/step AllReduce busy time) | **38.2** | 17.4% |
| rank2 spin-wait (of its 30.4 ms/step) | 19.2 | 8.7% |

* Proven, not inferred: `wait_r(i) == max_s gap_s(i) - gap_r(i)` (gap = the
  rank's own compute between two collectives) holds to 0.2%
  (r0: observed 1912.4 ms vs predicted 1915.7 ms).
* The straggler is **not** fixed: argmin of per-collective duration is r0/r1/r2/r3
  = 24.6/25.3/27.7/22.4%. Pairwise interval difference Z100−Z100L: mean +140 us,
  median +5 us, sd 634 us, `corr(gap_r0, gap_r2) = 1.000` ⇒ the stagger is
  ~140 us/collective of systematic card imbalance plus ~600 us sd of jitter.
* Recoverable only by card uniformity (mean-based, upper bound): all-Z100
  **18.9 ms/step (8.6%)**, all-Z100L **38.8 ms/step (17.7%)**; median-based
  (steady state) 7 / 12 ms/step. **No RCCL env can touch this.**

⇒ Report §7.2 #6 ("overlap NCCL with compute") cannot pay either: the fast rank
is idle because the *next* layer's GEMM depends on the collective's result, so
there is nothing to overlap it with. §7.2 #4 (mixed Z100L/Z100) is the real
lever, worth ~3–9% of the step, but it is a deployment/hardware decision.

⇒ Ceiling for any protocol tuning: the in-service tail is 72 us and is already
protocol-independent (Simple floor 73.0, LL floor 71.0 us). LL's e2e ceiling is
therefore `(107−75) us × 140 = 4.5 ms/step (2%)`, and the realistic expectation
is **0–2 ms/step**, not 32% of 49 ms.

## 3. Custom all-reduce — CLOSED (do not spend an arm)

Re-verified in the current tree: `vllm/platforms/rocm.py:1239-1241`
`use_custom_allreduce()` returns True only for `gfx94/gfx95`, so
`ParallelConfig._verify_args` (`vllm/config/parallel.py:1104-1109`) forces
`disable_custom_all_reduce=True` regardless of the CLI flag; and even past that
gate `hipDeviceMallocUncached` returns `hipErrorOutOfMemory` on gfx906, so
`CustomAllreduce.__init__` dies at `meta_ptrs`. S13
(`docs/result/z100-13-custom-allreduce-result.md`) measured the flag e2e at
±0.4% with `Using ['PYNCCL'] all-reduce backends`, and 0/505 bitwise-identical
token ids for two same-vintage load arms. Keeping
`--disable-custom-all-reduce` in the recipe is harmless and honest.

## 4. Serving configuration

### 4.1 `--max-num-batched-tokens`: there is no 2048 clamp

`vllm/config/vllm.py:2461-2486` (`_set_max_num_scheduled_tokens`) only (a) defaults
`max_num_scheduled_tokens = max_num_batched_tokens` and (b) **warns when it is
< 8192**. There is no clamp and no error path for MTP: for `method="mtp"`,
`max_num_new_slots_for_drafting` returns **0** (`vllm/config/speculative.py:1878-1914`,
the eagle-style fallthrough), so `scheduled_token_delta = 0`. The 2048 comes
verbatim from the recipe flag. The warning is a heuristic telling us the
intended operating point is ≥8192.

* Arm (no code change; the knob already exists): `S11F_MBT=8192`.
* Expected: chunked prefill chunk 2048 → 8192 tokens ⇒ one prefill pass per
  8100-token prompt instead of four, with 4× the tokens per attention/conv/GDN
  launch and 4× the M rows per `fp8_emu_gemm_block_kernel` launch (report §7.2
  #2/#5 predict a throughput gain from more rows per pass); **decode unchanged**
  (at conc=1 a step schedules ≤3 tokens, so MBT is irrelevant there). Measure
  TTFT at in≈8100 and conc8 throughput; the §2 baseline to beat is 7443 in →
  7160 ms TTFT (1039 t/s) at MBT=2048.
* Risk/rollback: peak activation memory grows, KV cache shrinks slightly; if
  startup fails or TTFT does not improve, revert to `S11F_MBT=2048`.

### 4.2 Cold-start: `--enforce-eager` hard-disables JIT warmup

`vllm/config/vllm.py:1793-1803`: eager sets `-cc.mode=none`,
`cudagraph_mode=none` **and `kernel_config.enable_jit_warmup=False`** — so there
is no supported flag combination that keeps eager and warms JIT
(`vllm/model_executor/warmup/kernel_warmup.py:170` is then dead code). The only
remedy is an external warmup request before serving traffic.

* Recipe: `tools/dcu/perf1_warmup.py --url http://127.0.0.1:8000 --model qwen3.8-27b`
  — `--measure` first runs the 2048-token prefill twice and prints cold vs warm
  (the A/B in one process), then warms 2048/512/128-token prefill and
  decode+MTP shapes with `max_tokens=8`; outputs are discarded.
* Expected: removes the first-request penalty the report measured at 25–40%
  (p8192: 279 → 1039 t/s, 3.7×). Run once after `/health` before the port is
  opened; cost ~1 request.
* No rollback needed (caches only; `TRITON_CACHE_DIR`/`VLLM_CACHE_ROOT` persist).
* **Methodological warning for every e2e arm in this program**: with
  `--enforce-eager` there is no JIT warmup, so the *first* request after each
  restart is 25–40% slower than steady state (279 vs 1039 t/s at 8k, report §2).
  Any arm comparison that does not warm up first — or that compares one arm's
  first request against another arm's steady-state requests — will attribute
  that penalty to the arm. Either (a) run `perf1_warmup.py` after every restart,
  or (b) discard the first request of every arm and compare steady state only.
  This applies to the control arms as much as to the optimizations.

### 4.3 Knobs evaluated and left alone

* `--enable-prefix-caching`: already on; the 0% hit rate is a workload artifact
  (unique long prompts, conc=1). Multi-turn traffic will hit; do not claim a
  decode gain from it in the report.
* Chunked prefill: always on in V1; its chunk size *is* `max_num_batched_tokens`
  (§4.1). No separate knob.
* `--max-num-seqs`: no effect at conc=1; default is fine — do not touch.
* `--max-model-len`: production runs 262144 vs the recipe's 32768. Not a speed
  lever (KV-cache sizing only); flagged only for config consistency.

### 4.4 `vllm/envs.py`: do not touch

`vllm/envs.py` documents `VLLM_*` variables only; the knobs recommended here
(`NCCL_PROTO`, `NCCL_ALGO`, `HSA_FORCE_FINE_GRAIN_PCIE`) are consumed by
RCCL/ROCr and never read by vLLM. The one nearby family,
`VLLM_ROCM_QUICK_REDUCE_*` (`vllm/envs.py:1503-1520`), is gated behind
`use_custom_allreduce()` (gfx94/gfx95 only) and is therefore dead on gfx906.
⇒ No `envs.py` change is proposed; adding a comment there would document
variables vLLM does not own.

## 5. Verification requirement for any comm arm (numerics)

Any arm that changes a collective's algorithm or protocol must carry a token
parity check (S11l/S13 flip battery), for two different reasons:
`NCCL_ALGO=Tree` **does** change the reduction order (tree vs ring ⇒ different
f16 accumulation) and can flip tokens; `NCCL_PROTO=LL` vs `Simple` is expected
to be bitwise neutral (both are ring reductions in rank order) but has not been
verified on this stack. Cheap insurance: run the S11m flip matrix on the arm and
compare against the same-vintage baseline (the vintage rule still applies: only
same-cache-vintage comparisons are meaningful).

## 5. Falsification notes (dead ends — do not repeat)

1. **Custom all-reduce arm** — closed, §3.
2. **`NCCL_PROTO=LL` as a decode win** — round 1's −32% did NOT reproduce under
   the lock (round 2: default 84.6–85.5 vs LL 85.7–86.9 at 10–120 KiB, +1.5%) and
   forced LL is 2.35× worse at prefill scale (20 MiB: 2796 vs 1190 us) plus 22%
   worse in the staggered chain. **Do not set `NCCL_PROTO`.** The round-1 number
   was contention: a concurrent engine arm loading on all four DCUs penalised the
   default path ~2× more than the forced-LL path. Any comm microbenchmark run in
   this program must be run under the lock, uncontended, or it will produce a
   false positive like this one.
3. **`NCCL_PROTO=LL,Simple`** — keeps large messages at default cost but gives up
   the small-message choice (106–109 us at decode sizes). Not a workaround.
4. **`NCCL_LL_THRESHOLD=131072`** — 861 us/call at 30 KiB (8× worse). Falsified.
5. **`NCCL_ALGO=Tree`** — no consistent win: 88.7 us at 30 KiB but 128.6 at
   10 KiB and 171.6 at 1 MiB. (Also changes the reduction order ⇒ would need a
   parity check.)
6. **`NCCL_MIN_NCHANNELS=1`** — −9% at 10–30 KiB in the contended round 1, but
   round 2 shows the default already at 85 us, so there is nothing to recover;
   and it is worse at 1 MiB (160.9). Not worth a config change.
7. **Turning P2P off for decode** — P2P and SHM are identical at 2–30 KiB
   (105–110 us); the flag's value is at ≥120 KiB. Keep `HSA_FORCE_FINE_GRAIN_PCIE=1`.
8. **"NCCL is 22% of the step, so protocol tuning can win ~2-6 ms/step"** — the
   intrinsic collective is 10.1–11.1 ms/step (4.6–5.1%); the rest is skew-wait
   (§2). With the clean round-2 numbers the protocol knob is worth **0 ms/step**.
9. **Multi-stream / comm-compute overlap** (§7.2 #6) — cannot help here: the
   waiting rank has no independent work (its next kernel consumes the collective
   result). Not worth a code change before the card-skew item is decided.
10. **`--kernel_config.enable_jit_warmup=True`** — silently overridden to False by
    `--enforce-eager` (`vllm/config/vllm.py:1803`). Dead end; use §4.2 instead.

## 6. Recommended patch (Lead review; not applied)

`tools/dcu/s11f_e2e_fp8.sh` — one line, only after the verifier measures it:

```diff
-  --max-model-len "${S11F_MAXLEN:-32768}" --max-num-batched-tokens "${S11F_MBT:-2048}" \
+  --max-model-len "${S11F_MAXLEN:-32768}" --max-num-batched-tokens "${S11F_MBT:-8192}" \
```

Expected: prefill/TTFT gain at long prompts; decode unchanged. Rollback: set
`S11F_MBT=2048` (or revert the line). No other change is proposed yet; the
`NCCL_PROTO` decision waits for round 2 (§1), and the warmup is an operational
step (`perf1_warmup.py`), not a recipe edit.

## 7. Open questions / pending

1. ~~Round 2 (protocol at prefill scale, `LL,Simple`)~~ — **DONE §1b**; verdict
   keep protocol auto.
2. ~~Staggered chain~~ — **DONE §1c**; LL is 22% *worse* with a 300 us straggler.
3. Round 1's absolutes remain contended; round 2 (§1b) is the clean reference for
   the sizes it covers (10 KiB–20 MiB). No further comm measurement is needed.
4. Is the Z100L/Z100 split fixable by pairing (deployment), or only documentable?
   Owner: Lead (report §7.2 #4), with §2 as the quantified ceiling (~7 ms/step
   steady-state, 18.9 ms/step mean-based).
5. Remaining comm-side unknown, deliberately not pursued: the ~10 collectives/step
   above the 130 implied by the model geometry (64 layers × 2 + MTP × 2). Even if
   all were redundant, 10 × 72 us = 0.7 ms/step (0.3%) — below the noise floor of
   any arm here.

## 8. Arm definitions for the verifier (exact commands)

**Arm M — MBT 8192** (independent of the comm question; safe, decidable):
```
S11F_MBT=8192 <same recipe/arm harness as the control>
# measure: TTFT at --random-input-len 8100 --num-prompts 1, conc8 output tok/s,
#          and decode TPOT (must be unchanged). Control = S11F_MBT=2048.
# pass:  TTFT(in=8100) improves by >3% and decode TPOT stays within +-1%.
# fail:  startup OOM / KV cache too small / no TTFT change -> keep 2048.
```

**Arm W — warmup** (operational, no recipe change):
```
docker exec z100-vllm bash -c 'python /src/tools/dcu/perf1_warmup.py \
    --url http://127.0.0.1:8000 --model qwen3.8-27b --measure'
# run right after /v1/models answers 200 and before opening traffic.
# --measure prints the cold vs warm 2048-token prefill in one process: that
# delta IS the penalty the report measured (279 -> 1039 t/s at 8k, section 2).
# pass: cold/warm delta >20% -> adopt as a standard post-start step (cost: 1 request).
# fail: delta <5% -> the penalty is not reachable this way; record as falsified.
# rollback: none (populates TRITON_CACHE_DIR / VLLM_CACHE_ROOT only).
```

**Arm C — `NCCL_PROTO=LL`**: do **not** run until round 2 lands and the Lead's
decision rule is met (decode-size gain ≤ −25% reproduced AND prefill-scale
penalty <5% of TTFT). If it is run:
```
NCCL_PROTO=LL <same arm>   # env-only; rollback = unset
# mandatory with it: S11m flip matrix vs the same-vintage control (numerics).
```

## 9. Unmeasured knobs, with the reason (so they are not re-tried blind)

* `NCCL_MAX_NCHANNELS=1`: `NCCL_MIN_NCHANNELS=1` (the same direction: force few
  channels) measured −9% at 10–30 KiB but is still 34% slower than forced LL and
  worse at 1 MiB; the flat 2 KiB–120 KiB curve at default settings already shows
  channel count is not what sets the per-call cost here.
* `NCCL_BUFFSIZE`: it governs the Simple protocol's chunk size; the default is
  4 MiB and the decode payloads are 10–30 KiB, so no chunking occurs. The
  measured ladder is flat across two orders of magnitude of size, which rules
  out a buffer-size effect for the sizes that matter.
* `NCCL_P2P_LEVEL`: the P2P/SHM A/B (`HSA_FORCE_FINE_GRAIN_PCIE` 0 vs 1) is the
  same decision point and was measured directly (§1): no decode difference.

## 10. Decision for `NCCL_PROTO=LL` (resolved)

The pre-computed arithmetic (20 MiB payload, ~140 per prefill chunk, 5%-of-TTFT
bound) was the right decision frame, and the *scaling* branch is what happened:
LL's penalty grows with size (1 MiB +45%, 4 MiB +116%, 20 MiB **+135%** = +1605.6
us per all-reduce), so the prefill cost is +224.8 ms per chunk and +899 ms on a
7160 ms TTFT (+12.6%), not the optimistic +6.4 ms. Together with the decode sizes
showing no gain (and the chain showing a 22% loss under a straggler), the verdict
is **keep protocol auto**: no `NCCL_PROTO` entry in the serving environment.

Final state of the comm-side levers, in one line each:

| lever | status | number |
|---|---|---|
| NCCL/RCCL protocol & algorithm tuning | **closed, no gain** | default ≈ forced LL at 10–120 KiB (+1.5%); LL 2.35× worse at 20 MiB |
| custom all-reduce | **closed, structurally blocked** | rocm.py gfx94/95 gate; S13 ±0.4% e2e; uncached alloc fails |
| P2P fine-grain | **keep on** | decode neutral; 8.9× at 1 MiB (S17: 8K TTFT 1.89×) |
| comm share of the decode step | **corrected** | 10.1–11.1 ms/step intrinsic (4.6–5.1%), not 49 ms (22%); the rest is skew-wait |
| card heterogeneity (Z100L/Z100) | **the only real lever, owner Lead** | 7 ms/step steady-state, 18.9 ms/step mean-based |
