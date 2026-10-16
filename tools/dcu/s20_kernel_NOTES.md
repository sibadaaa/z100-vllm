# S20 — gfx906 FP8-emulated decode GEMM at M=5..8 (MTP/DFlash verify batch): findings

Scope: the block-scaled FP8 emulated GEMM on the **M=5..8** path (speculative verify
batch, `M = 1 + num_spec_tokens`), which is 40% of a c=1 decode step. What was tried,
what was measured, what landed, and what is left.

All numbers on **DCU0 (HIP_VISIBLE_DEVICES=0)**, `Z100L` @1600 MHz (see §7 for the
clock split — DCU2/3 are 1319 MHz Z100s and read ~20% slower for the same kernel).
Microbenchmarks are `min-of-3x20` (the mprobe's `timeit` takes the best of 3 rounds);
the **canary** is the same run's `8704:5120 M=1` row, target 0.1060-0.1065 ms
(uncontended reference). A canary above ~0.12 ms means the GPU was shared and the
table is provisional.

## 1. The production shape mix at c=1 (measured, not inferred)

`/home/lyh/z100-vllm/.s20/traces/shapes_rank0.json.gz` has `Input Dims` on every
`_rocm_C::fp8_emu_gemm_block` cpu_op; pairing all 5888 cpu_ops with their kernels
(5888/5888 matched) and taking the 21 clean M=8 decode steps gives:

| per rank (TP=4) | N:K | calls/step | median us | ms/step | weight bytes |
|---|---|---|---|---|---|
| mlp gate_up | 8704:5120 | 64 | 463.0 | 29.63 | 44.6 MB |
| mlp down_proj | 5120:4352 | 64 | 310.1 | 19.85 | 22.3 MB |
| GDN in_proj (qkv+z) | 4096:5120 | 48 | 337.6 | 16.20 | 21.0 MB |
| full-attn qkv+gate | 3584:5120 | 16 | 258.1 | 4.13 | 18.4 MB |
| o_proj + GDN out_proj | 5120:1536 | 64 | 124.6 | 7.98 | 7.9 MB |
| **total** | | **256** | 304 (mean) | **77.8** | **6.08 GB** |

* 256 calls/step, not 215: the two independent traces agree on the per-call cost
  (309.5 us/call over 215 calls vs 304 us median over 256) and differ by exactly the
  215/256 = 0.84 step-window undercount of the coarser trace.
* DRAM floor for one decode step = 6.08 GB / 736 GB/s (measured honest read) = **8.3 ms**,
  i.e. the 77.8 ms of fp8 GEMM has ~9x headroom in principle.

### 1.1 Finding: `VLLM_DCU_FP8_GEMV_MAX_M=4` in the production env silently disabled the streaming kernel

The c=1 trace contains **only** `_rocm_C::fp8_emu_gemm_block` cpu_ops and only
`fp8_emu_gemm_block_kernel<...>` kernels (x4064 `<32,64,2,4,Half,1>` + x1824
`<...,4>`, i.e. cfg 5 and cfg 10). The python wrapper routes `M <= _GEMV_MAX_M` to the
**gemv** entry, which is where the S18 shape-aware dispatch (streaming GEMV for
`M*K*2 + M*nkb*4 <= 44 KiB`) lives — so with `MAX_M=4` every M=8 call went to the
tiled GEMM and the streaming path was never consulted. Consequence, measured:
`5120:1536` (A = 24.9 KiB, fits the gate) runs 0.0763 ms through the gemv entry vs
0.1246 ms in the trace = **-48 us x 64 calls = -3.1 ms/step** for a pure-Python env
change (no rebuild). Lesson: an env override that routes around a dispatch silently
disables kernels; check `_GEMV_MAX_M` before believing a kernel is on the hot path.

## 2. Root cause of the M=5..8 cost: 4x wasted rows in an *occupancy-bound* kernel

The decode fallback is the prefill tile (BM=32, BN=64, TM=2, TN=4, MINB=4): the grid
is `(cdiv(N,64), cdiv(M,32))`, so at M=8 it still computes **32 rows** and the
epilogue discards 24 of them — 4x the MACs and 4x the dot2 work that are needed.

Naive modelling says "make the tile exact-M and it gets 4x cheaper" (BM=8 keeps
2.74 instr/dot2 vs 1.91 for BM=32, so ~2.8x fewer instructions). **That model is
wrong**, and the measurement says why:

| tile @ 8704:5120 M=8 | resident threads | warps/CU | ms | GB/s |
|---|---|---|---|---|
| cfg 5/10 legacy (BM=32, TM=2, TN=4, NT=256) | 34,816 | **9.07** | 0.4183 | 107 |
| cfg 11/12 exact-M (BM=8, TN=4, NT=128) | 17,408 | 4.5 | 0.363-0.391 | 114-123 |
| cfg 13 (BM=8, BN=128, TN=8, NT=128) | 8,704 | 2.3 | 0.4439 | 100 |
| cfg 15 (BM=8, TN=8, NT=64) | 8,704 | 2.3 | 0.5201 | 86 |
| cfg 16 (BM=8, TN=2, NT=256) | 34,816 | 9.07 | see §6 | |
| cfg 17 (BM=8, BN=32, TN=2, NT=128) | 34,816 | 9.07 | see §6 | |
| cfg 18 (BM=8, TN=1, NT=512) | 69,632 | 18.1 | see §6 | |

Total resident threads = `(N/BN) * NT = N*BM/(TM*TN)`, so an exact-M tile has half the
threads of the legacy tile at the same TN — and **the runtime tracks warps/CU, not
instruction count**: the legacy tile is issue-bound (1.36e9 lane-instr -> 222 us at
the 6.14e12 lane-instr/s issue peak, measured 418 us) while the exact-M tile runs at
~20% issue efficiency because 9 warps/CU cannot cover the per-chunk global-load +
LDS + barrier latency. That is the whole story of the 1.0-1.2x result in §3.

## 3. Landed change: exact-M decode tiles behind an env knob

`csrc/rocm/fp8_emu_gemm.cu`:
* `FConf::A4/B4` are now **ceil** divisions and the staging helpers guard
  `idx < BM*kFKw4` / `idx < BN*kFKw4`, so a tile may have `BM*kFKw4 < NT` (BM=8 at
  128 threads). Normalized disassembly of the pre-existing cfg 5/cfg 10 kernels is
  **identical** before/after (same instruction count, same mnemonics with addresses
  masked), so the prefill path is untouched.
* new tiles **cfg 11..19** (see the switch in `dispatch_gemm_block_cfg`):
  11 `<8,64,1,4>`, 12 `<8,64,1,4,MINB4>`, 13 `<8,128,1,8>`, 14 `<16,64,2,4>`,
  15 `<8,64,1,8>`, 16 `<8,64,1,2>`, 17 `<8,32,1,2>`, 18 `<8,64,1,1>`, 19 `<8,128,1,2>`.
* `VLLM_DCU_FP8_DECODE_CFG` (A/B knob, read per call so it works without a rebuild):
  unset/`0` = **legacy** cfg 5 (or cfg 10 for N >= 6144); `11..19` force that tile;
  `auto` = the exact-M auto rule. **The default stays legacy** — the Lead owns the
  end-to-end A/B and the default flip.
* `fp8_emu_gemm_block`'s cfg range check widened to 0..19 (it used to reject >10).

### 3.1 Measured (new build `4ce168f1`, canary 0.1065 ms = quiet)

| shape @ M=8 | cfg 0 legacy | auto (cfg 13/11) | best explicit | pytest |
|---|---|---|---|---|
| 8704:5120 | 0.4189 | 0.4432 | 0.3628 cfg12 (1.15x) | 59/59 pass |
| 5120:4352 | 0.2869 | 0.2382 | 0.2728 cfg11 | rel_err 0.00e+00 |
| 4096:5120 | 0.3359 | 0.2795 | 0.3050 cfg11 | (mprobe --check) |
| 3584:5120 | 0.2685 | 0.2486 | 0.2673 cfg0 | |
| 5120:1536 | 0.0763 (stream) | 0.0763 | 0.0763 | |

**Verdict: not a win worth a default flip** (mix-weighted ~6% at best, and the auto
rule of §3.1 was worse than legacy on the dominant shape). Kept as opt-in machinery.

## 4. Why the "stream at M=8" family does not deliver either

Standalone HIP prototypes (`tools/dcu/s20_probe.cu`, no rebuild needed) established the
device limits and the wall for stream-style M=8 kernels:

| probe | result |
|---|---|
| `v_dot2_f32_f16` peak (ILP>=8) | 4.72e12 dot2/s = 9.4 T-MAC/s |
| honest global read | 750 GB/s |
| W-stream + widen + 64 dot2 per chunk, A in registers (`abl0`) | **736 GB/s (DRAM bound)** |
| same + A read from LDS per (m, chunk) | 149 GB/s (A0->A8 axis: 693/310/199/149) |
| K-chunked stream, A slice staged in LDS per chunk (k3) | 122 GB/s |
| K-chunked stream, A in registers per chunk (k6/k8) | 35-115 GB/s |

The A operand costs `M*2` bytes of LDS traffic per W byte (16:1 at M=8) against a
12.3 TB/s LDS peak, i.e. a hard ceiling of ~770 GB/s of W even at 100% LDS efficiency,
and every staged variant needs a `__syncthreads` per K chunk whose latency cannot be
hidden at 8-16 warps/CU. **Pitfall worth recording:** a prototype without
`__launch_bounds__` is silently capped at 64 VGPR and spills (private segment 150-650
B) — several of the early numbers in this file were spill artefacts, not kernel
properties. Add `__launch_bounds__(NT)` before believing any prototype.

## 5. What is left

1. **TN=2 tiles (cfg 16..19)** — the falsifiable prediction from the warps/CU model:
   `8N/2 = 34,816` threads = 18 warps/CU with 2.99 instr/dot2 => 87 us at the issue
   peak, i.e. 0.17-0.20 ms on 8704:5120 (2.1-2.4x) if the occupancy fix works.
   Measured result: see §6.
2. `VLLM_DCU_FP8_GEMV_MAX_M=8` (pure env, -3.1 ms/step via 5120:1536 on the stream path).
3. A K-split of the streaming kernel would put 3584:5120 (A = 57 KiB) on the stream
   path too; at M=4 that shape is 0.0865 ms vs 0.3442 tiled, so the split-K variant is
   the only remaining large idea for the big shapes. Not attempted.
4. Not attempted, deliberately: any INT8 `v_dot4_i32_i8` requantisation (anti-goal).
5. **Split-K streaming, rejected on measured rates**: with A sliced by K/2 the M=8
   stream path fits LDS again (M*K/2*2 = 40 KiB at K=5120), but the stream path
   measured at M=8 is only ~103 GB/s (5120:1536) because of the 16:1 A-LDS traffic, so
   the two K halves plus a 1.1 MB fp32 scratch would land at ~0.43 ms for 8704:5120 —
   no better than the 0.419 ms tiled fallback it would replace. Only worth revisiting
   if the A-traffic problem is solved first (e.g. A in registers with row reuse).
6. **kFBk = 64/128 for the decode tiles** (halve/quarter the per-chunk barrier count)
   is the natural follow-up if the TN=2 build shows a residual latency floor: the
   staging cost per W byte is unchanged, only the number of staging rounds falls. It
   needs `kFBk/kFKw4/kFKp/kFLogKw4` templated through the shared GEMM helpers, i.e. a
   real (if mechanical) change to code the prefill path also compiles, plus a
   `block_k % BK == 0` guard so the scale-fold cadence stays exact.

## 6. TN=2 tiles (cfg 16..19): prediction vs measurement — the shipped result

### 6.1 Prediction (made before the build, kept verbatim as the falsification target)

Model (per W byte, wavefront instructions; one warp instruction covers 64 lanes):

| tile | instr/thread/chunk | warps/block | wf-instr per 2048 W bytes | total wf-instr 8704:5120 | us @100% issue |
|---|---|---|---|---|---|
| cfg 5/10 legacy (BM=32,TN=4) | 245 / 128 dot2 | 4 | 980 | 21.3e6 | 222 |
| cfg 16 (BM=8,TN=2,NT=256) | 101 / 32 dot2 | 4 | 404 | 8.8e6 | 91.7 |
| cfg 18 (BM=8,TN=1,NT=512) | 62 / 16 dot2 | 8 | 496 | 10.8e6 | 113 |

Resident threads = `(N/BN)*NT = N*BM/(TM*TN)` = 34,816 at N=8704 for both the legacy
tile and cfg 16 (that is the whole point of TN=2), so the model predicted cfg 16 keeps
the legacy's measured 53% issue efficiency and lands at **41.4% of the legacy time on
every shape** (mix 77.8 -> 32.0 ms/step, band 26-32).

### 6.2 Measurement (Lead's 4-arm A/B, one contiguous window, 224 engine steps/arm)

| shape @ M=8 | legacy -> cfg16 | ratio |
|---|---|---|
| 8704:5120 | 0.4189 -> 0.242 ms | 1.73x |
| 5120:4352 | 0.2869 -> 0.167 | 1.72x |
| 4096:5120 | 0.3359 -> 0.198 | 1.70x |
| 3584:5120 | 0.2685 -> 0.180 | 1.49x |
| mix (256 calls/step) | 74.6 -> **41.6 ms/step** | **1.79x** |

Every shape landed at **58-67% of legacy**, not 41.4%: direction and mechanism
confirmed (occupancy transfers, 4x wasted rows removed), magnitude ~30% optimistic.
The residual is *not* identified by measurement yet; the strongest candidate is the
TN=2 B-fragment LDS layout: the B row stride in LDS is `NG*kFKp` = 32*17 = 544 dwords
= 0 (mod 32) banks, i.e. every lane of a B-read instruction hits one bank (32-way
conflict), against 16-way at the legacy NG=16. Fixing it means a k-major B tile or a
conflict-free padded stride, or halving the number of staging rounds with kFBk=64
(§5.6). cfg 17/18/19 were in the same build and are the intended discriminators.

### 6.3 End-to-end effect (the honest number, corrected)

The Lead shipped `.so 03ac92eabf0d1ad6ac635e6a1529a9c0` with
`VLLM_DCU_FP8_GEMV_MAX_M=8 VLLM_DCU_FP8_DECODE_CFG=16` (the only way to reach the
tiles, see §6.4). Engine-level c=1, 512-token prompts:

| arm | .so | env | step p50 |
|---|---|---|---|
| B1 | 8e98bd60 | prod MAX_M=4 | 211.9 ms |
| T1 | 03ac92ea | MAX_M=8, cfg 0 | 218.8 |
| T2 | 03ac92ea | MAX_M=8, cfg 16 | **203.9** |
| B2 | 8e98bd60 | prod MAX_M=4 | 204.4 |
| final production | 03ac92ea + cfg 16 | | **188.5**, 8.0 tok/step, accept 7.00 |

* The two **control** arms differ by 7.5 ms (3.6%), so per-arm p50 deltas of that size
  are not resolvable; pooled across the two available windows cfg 16 wins by ~5%
  (213.6 -> 199.1 and 208.1 -> 203.9). Parity (`p0=37576f01 p1=a8eb3bd7 p2=470b91a1`)
  is byte-identical in every cfg 16 arm of both windows -> the tiles are deterministic.
* **Corrected headline: cfg 16 is worth -2% to -7% of a ~200 ms step**, even though it
  removes ~33 ms from the fp8 GEMM time. The step also contains ~75 ms of all-reduce
  (83-88% arrival spin-wait on this 2x1600 MHz + 2x1319 MHz box) and ~55 ms of launch
  bubbles, so a 33 ms kernel saving is largely hidden. "The operator is slow" and "the
  step is slow" are different claims; measure both.

### 6.4 Dispatch wart (follow-up, needs a build)

`VLLM_DCU_FP8_DECODE_CFG` is consulted only by the **gemv** entry, so on the ordinary
`fp8_emu_gemm_block` path that production actually uses (cfg-0 auto rule) the knob does
nothing — which is why the shipped recipe needs `VLLM_DCU_FP8_GEMV_MAX_M=8` as well,
and why the free win of §1.1 (5120:1536 -> streaming, 0.1246 -> 0.0763 ms) is bundled
into the treatment arm instead of being independently testable. If another build
happens: make `fp8_emu_gemm_block`'s cfg-0 rule (or `run_block`'s gemv branch) consult
`VLLM_DCU_FP8_DECODE_CFG` when `M <= 8`, then the knob works alone.

## 7. Device heterogeneity

`pp_dpm_sclk`: 2x Z100L @1600 MHz (DCU0/1) and 2x Z100 @1319 MHz (DCU2/3); the same
kernel reads 0.418 ms on DCU0/1 and 0.504 ms on DCU2/3 at M=8. State the device for
every number; TP=4 means the slow cards decide the step time.

## 8. Commands

```bash
# build + install (serialised; install overwrites the live .so -> back it up first)
docker exec z100-vllm bash -lc 'md5sum /src/vllm/_rocm_C.abi3.so; cp -n /src/vllm/_rocm_C.abi3.so /mnt/game/perf1/_rocm_C.$(md5sum /src/vllm/_rocm_C.abi3.so | cut -d" " -f1).so'
docker exec z100-vllm bash -c 'flock /tmp/perf1_build.lock bash -c "cp /src/csrc/rocm/fp8_emu_gemm.cu /src/csrc/rocm/fp8_emu_gemm.h /mnt/game/port/gfx906-vllm-211/csrc/rocm/ && bash /mnt/game/port/build-211.sh _rocm_C && bash /mnt/game/port/install-211.sh _rocm_C"'
# correctness gate (90 tests on the cfg 11..19 build)
docker exec z100-vllm bash -lc '<prelude>; HIP_VISIBLE_DEVICES=0 python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q'
# per-shape sweep + mix-weighted ms/step
docker exec z100-vllm bash -lc 'bash /src/tools/dcu/s20_cfg_sweep.sh "0 auto 16 17 18 19"'
# standalone roofline/prototype probe (hipcc, seconds)
docker exec z100-vllm bash -lc 'cd /tmp && hipcc -O3 --offload-arch=gfx906 -o /tmp/s20_probe /src/tools/dcu/s20_probe.cu && HIP_VISIBLE_DEVICES=0 /tmp/s20_probe abl 8704 5120 8 20'
```

## 9. Evidence

* Source-vs-binary note: the shipped `.so 03ac92ea` was built from
  `csrc/rocm/fp8_emu_gemm.cu` *before* a comment-only edit (the stale warps/CU
  figures in the cfg 16..19 comment were corrected to 4.5 vs 9.07 afterwards, and the
  `S20 6` cross-reference added). No code, constant or template argument changed, so
  codegen is identical; a rebuild would still change the md5 through debug line info.
* `.so` chain in `/mnt/game/perf1/`: `8e98bd60b94ebeb6f69ed8e39d754647` (pre-S20 live /
  rollback target), `04353713a87cde74e5881f8d94468bff` (1st S20 build), 
  `4ce168f1778870e0b0b0eed665f04937` (auto-parse + cfg-range fix),
  `03ac92eabf0d1ad6ac635e6a1529a9c0` (TN=2/1 tiles cfg 16..19, 90/90 pytest).
* `/mnt/game/s20/`: build logs, `sweep_cfg_*.json`, mprobe JSON per cfg.
* `/home/lyh/z100-vllm/tools/dcu/s20_probe.cu`, `s20_cfg_sweep.sh`, this file.

## 10. Follow-ups closed by the next session (2026-10-15)

### 10.1 §6.4 dispatch wart is fixed (rebuilt + verified)

`fp8_emu_gemm_block`'s cfg-0 rule now consults `decode_gemm_cfg(N)` when
`M <= kMsMax && block_k % kFBk == 0`, so `VLLM_DCU_FP8_DECODE_CFG` takes effect
on the production `M > VLLM_DCU_FP8_GEMV_MAX_M` dispatch **on its own** —
reaching the exact-M tiles no longer requires also setting
`VLLM_DCU_FP8_GEMV_MAX_M=8` (which silently re-routed M <= 4 as well).  With the
env unset the rule is byte-identical to the old one, so the default dispatch is
unchanged.

Verification (`s18_fp8_block_mprobe.py --shapes 8704:5120 --ms 8 --modes gemm`,
i.e. the `fp8_emu_gemm_block` entry, DCU0, `MAX_M=4` in every arm):

| arm | ms | GB/s |
|---|---|---|
| `DECODE_CFG` unset | 0.4180 | 107 |
| `DECODE_CFG=16` | **0.2415** | **185** |
| `DECODE_CFG=0` | 0.4178 | 107 |

.so `67b0bba36c541ab6a68145edd7f049ce` (2026-10-15 08:26).  The replaced build
`03ac92ea…` is still backed up at
`/mnt/game/perf1/_rocm_C.03ac92eabf0d1ad6ac635e6a1529a9c0.so` (from the previous
round; `/mnt/game/perf1` itself is root-owned, so copy it from inside the
container).  Correctness gate on the new .so:
`tests/kernels/quantization/test_gfx906_fp8_emu.py` **123 passed**.  Rollback:
`VLLM_DCU_FP8_DECODE_CFG=0` (behavioural, no rebuild) or restore that .so.

### 10.2 §5.1/§6.2 "TN=2 B-fragment 32-way bank conflict" is **falsified**

The hypothesis was that the TN=2 B-tile's row stride `NG*kFKp = 32*17 = 544`
dwords is `0 (mod 32)` and therefore every B-fragment read conflicts 32 ways.
That confuses the *row-to-row* stride with the *lane-varying* term:

* the read is `Bs[S][(tx + NG*j)*kFKp + u]`; for this tile `NG = 32` = warp size,
  so `tx` is the lane id and `ty` is warp-uniform ⇒ the only lane-varying address
  term is `lane * kFKp = lane * 17` dwords, and `gcd(17, 32) = 1` maps the 32
  lanes onto 32 distinct banks in each phase;
* `544` is a compile-time immediate shared by all lanes, so it cannot conflict;
* the A-fragment read is warp-uniform (`ty` fixed) ⇒ LDS broadcast, which is free.

ISA artifact (`.hip_fatbin` extracted from the installed `.so`, `llvm-objdump
--mcpu=gfx906`, kernel `fp8_emu_gemm_block_kernel<8,64,1,2,Half,1>` = cfg 16):
the fragment reads are `ds_read2_b32 v[..], v17 offset1:1` (two dwords per
instruction, consecutive `u`), the j=1 row is at `v17 + 0x880` bytes = +544 dwords
= exactly `NG*kFKp` (so the compiler kept the source layout), and the
lane-varying base is the 17-stride row pointer.  LDS census for cfg 16:
`ds_read2_b32` ×72, `ds_read_u16` ×36 (LUT staging), `ds_write2_b32` ×9.

⇒ Do **not** spend a build on a conflict-free B layout.  The residual gap between
the measured 1.7-1.8x and the 2.4x warps/CU model in §6.2 must come from
something else (issue efficiency, per-chunk staging serialisation, barrier
latency), and the discriminators for that are `kFBk = 64/128` (§5.6), which
remains open.

