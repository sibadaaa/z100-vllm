# perf1 / L3 — gfx906 FP8-emulated decode path: measured results (kernel-opt)

All numbers on DCU0 (Z100L, gfx906), block-scaled shape (block_n=block_k=128), K=5120,
fp16 output, 20 iters after 5 warmups, `--check` = max relative error against an fp32
reference built from the same fp8 operands.

SO versions
- baseline (pre-work): `fbfd45edb00713594793151a3e9ca76a` — `/mnt/game/perf1/_rocm_C.baseline_fbfd45ed.so`
- stream v1 + GEMM cfg 5..10 + M<=4 dispatch: `7e62d7ecf0569249416e8b85a048c07f` — `_rocm_C.stream_7e62.so`
- + M>4 -> GEMM fallback, rows target 120, cfg-0 auto rule: `8e98bd60b94ebeb6f69ed8e39d754647` — live 12:17
  (staged, NOT built: `kStreamMinBlocks` 120 -> 100)

## 1. Decode (streaming GEMV), us, rel_err = 0.00e+00 on every stream row

| N | M=1 | M=2 | M=3 | M=4 | M=5 | M=8 |
|---|---|---|---|---|---|---|
| 1280 | 25.5 | 29.7 | 36.9 | 51.9 | 260.1 | 245.3 |
| 3584 | 45.9 | 55.9 | 63.8 | 101.1 | 247.2 | 247.0 |
| 4096 | 58.5 | 71.5 | 80.4 | 121.9 | 320.5 | 321.1 |
| 5120 | 61.5 | 74.5 | 84.1 | 126.5 | 321.9 | 321.5 |
| 8704 | 106.0 | 120.9 | 149.9 | 220.1 | 417.2 | 417.1 |
| 4352 | 60.0 | 73.9 | 81.3 | 121.7 | 320.9 | 321.3 |

Comparators at the same shapes: old GEMV 92/235/302/494/256 us (M=1) and 704 us-3.62 ms at
M=8; tiled GEMM M-flat at 245-709 us (cfg 0). Best measured GB/s 445 (N=8704, M=1), i.e.
72% of the probe's 621 GB/s honest-stream reference and ~4.5x the old path.
M=5/8 rows are the new fallback (stream does not fit the 44 KiB LDS gate at K=5120):
1115-3622 us (old GEMV) -> 245-417 us.

## 2. Cheap levers on the tiled GEMM (same build)

`__launch_bounds__(NT, MINB)` VGPR cap + 256-thread tiles, all cfg switchable:

| cfg | tile | N=1280 | N=3584 | N=4096 | N=5120 | N=8704 |
|---|---|---|---|---|---|---|
| 0/1 (old) | 32x64 TM4TN4, 128 thr | 343 | 352 | 356 | 363 | 709 |
| 5 | 32x64 TM2TN4, 256 thr | 245 | 247 | 320 | 321 | 532 |
| 10 | cfg 5 @ 4 blocks/CU | 262 | 271 | 344 | 329 | **414** |
| 3 | 64x128 TM4TN8 | 546 | 545 | 545 | 547 | 1094 |
| 7/8 | cfg 1 @ 4 / 6 blocks per CU | 656 / 1047 | 663 / 1063 | 691 / 1135 | 699 / 1140 | 809 / 1376 |

cfg 0 auto rule now: cfg 5, or cfg 10 when `N >= 6144` or (`M > 32` and `N >= 4096`);
explicit cfg 1..10 still overrides.

## 3. Prefill (M=16..129), cfg-0 auto before/after — no regression anywhere

| N | M=16 | M=32 | M=64 | M=129 |
|---|---|---|---|---|
| 1280 old -> new | 420.6 -> 283.9 | 422.0 -> 285.9 | 447.4 -> 278.6 | 448.1 -> 356.9 |
| 4352 old -> new | 405.5 -> 372.7 | 436.8 -> 361.9 | 896.8 -> 461.6 | 1626.9 -> 813.7 |
| 8704 old -> new | 771.4 -> 454.4 | 879.1 -> 462.7 | 1349.4 -> 748.0 | 2975.2 -> 1540.5 |

## 4. rows target A/B (7e62 target 60 vs 8e98 target 120, back-to-back, same state)

4352 -17.5% / -17.8% / -19.7% (M=1/3/4); 4096 -19.2 / -18.7 / -16.4; 5120 -16.2 / -15.3 / -15.2;
8704 ~0 (identical geometry); 1280 -1..-7; **3584 +6.0 / +8.3 / +15.7**.

### STAGED, MEASURED, NOT LANDED

```cpp
// Measured variant (NOT shipped): 100 puts every production N in the
// 112-160 row-group sweet spot (N=3584 back to 43.3/58.9/87.4 us at
// M=1/3/4) while keeping the -16..-20% on 4352/4096/5120, at the cost of
// a rebuild + full re-verification; left for the next build window.
constexpr int kStreamMinBlocks = 120;   // shipped / verified value
```

`kStreamMinBlocks = 100` restores N=3584 to the old-policy 43.3 / 58.9 / 87.4 us (rows=32,
112 row-groups) while keeping the 4352/4096/5120 gains; it puts every production N in the
measured 112-160 row-group sweet spot.  It costs a rebuild plus a full re-verification cycle
for a ~0.17 ms/step gain (N=3584 is 17 of 260 calls/step), so it is left for the next build
window.  The shipped .so is frozen at `8e98bd60b94ebeb6f69ed8e39d754647` and was compiled
from a tree whose `csrc/rocm/fp8_emu_gemm.cu` md5 is `235bf28e5e30dfd96605a221e26268e3`;
the workspace copy differs from that file by comment lines only (the note above), so codegen
is identical.

## 4b. Interaction with `num_speculative_tokens` (k): a real cliff at k=4

Verify runs M = k+1.  The stream path holds while `M*K*2 + M*nkb*4 <= ~44 KiB`:

| k | M | path at K=5120 | time (N=4352 / 4096 / 8704) |
|---|---|---|---|
| 0 | 1 | stream | 60.0 / 58.5 / 106.0 |
| 2 (today) | 3 | stream | 81.3 / 80.4 / 149.9 |
| 3 | 4 | stream (41.6 KiB, just fits) | 121.7 / 121.9 / 220.1 |
| **4** | **5** | **GEMM fallback (52 KiB > gate)** | **320.9 / 320.5 / 417.2** |

So k=3 -> k=4 roughly doubles the verify call (1.9-2.6x) instead of the ~1.2x the stream path
would have cost; the cliff is LDS-capacity-driven, not algorithmic, and it moves with K: for
K <= 2560 (e.g. the attention projections) M up to 8 still fits and stays on the stream path.
Any recipe that raises the speculation depth past 3 should be measured end to end, not
extrapolated.

## 5. Policy shipped in csrc/rocm/fp8_emu_gemm.cu

```
gemv entry point (M <= 8):
  stream v1  iff K%16==0 && w_stride%16==0 && block_k%16==0 && block_n%8==0
             && M*K*2 + M*nkb*4 <= min(44 KiB, deviceLDS - 1 KiB)
  else if M > 4  -> tiled GEMM (cfg as in section 2)
  else           -> old GEMV
env: VLLM_DCU_FP8_GEMV_STREAM = 0 old GEMV | 1 stream v1 (default) | 2 stream v2 (rejected:
     1280/M1 47.5 vs 26.9 us, and M>=3 degrades sharply)
     VLLM_DCU_FP8_GEMV_PREF = 1 | 3 (default) W pipeline depth
python: VLLM_DCU_FP8_GEMV_MAX_M stays the coarse override (1 = old production; unset/8 =
     route M<=8 to this entry point and let the shape-aware rule decide; 4 is equivalent).
```

## 6. Commands

```
# build + install (serialised; install is a separate step from build-211.sh)
docker exec z100-vllm bash -c 'flock /tmp/perf1_build.lock bash -c "cp /src/csrc/rocm/fp8_emu_gemm.cu /src/csrc/rocm/fp8_emu_gemm.h /mnt/game/port/gfx906-vllm-211/csrc/rocm/ && bash /mnt/game/port/build-211.sh _rocm_C && bash /mnt/game/port/install-211.sh _rocm_C"'
# correctness
docker exec z100-vllm bash -c '<prelude>; HIP_VISIBLE_DEVICES=0 python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q'   # 46 passed
# tables
bash tools/dcu/perf1_kernel_measure.sh <tag> gemv-s1 gemv-s2 gemv-old gemm prefill
bash tools/dcu/perf1_kernel_postbuild.sh <tag>
bash tools/dcu/perf1_kernel_ab.sh          # .so-swap A/B, restores + md5-checks
```

## 7. Evidence

`/mnt/game/perf1/`: perf1_ab_{new,old}_{gemv,prefill}.log, perf1_post1_{pytest,gemv,prefill}.log,
perf1_s1_{gemvs1,gemvs2,gemvold,gemmcfgs,prefill,prefill_before,prefill_cfg}.log,
perf1_s1_s18_before_after.log, perf1_kernel_mprobe_stream{0,1,2}_prefunset.json,
`_rocm_C.{baseline_fbfd45ed,stream_7e62,stream_8e98}.so`.
Host copies: tools/dcu/perf1_kernel_{s1,ab}_*.log, tools/dcu/perf1_pytest.log.

## 8. Workspace files

Modified: `csrc/rocm/fp8_emu_gemm.cu` (stream v1/v2 kernels, GEMM cfg 5..10 + MINB, M<=4
dispatch + M>4 GEMM fallback, cfg-0 auto rule, one staged constant kStreamMinBlocks=100),
`tests/kernels/quantization/test_gfx906_fp8_emu.py` (decode-range coverage; 46 pass).
Added: `tools/dcu/perf1_kernel_mprobe.py`, `perf1_kernel_measure.sh`,
`perf1_kernel_postbuild.sh`, `perf1_kernel_ab.sh`, `perf1_kernel_NOTES.md`, evidence logs.
Untouched: `csrc/rocm/fp8_emu_gemm.h`, `csrc/rocm/torch_bindings.cpp`,
`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py`, the prefill path.

A/B knobs: `VLLM_DCU_FP8_GEMV_STREAM=0` old GEMV | `1` stream v1 (shipped default) |
`2` stream v2 (rejected); `VLLM_DCU_FP8_GEMV_PREF=1|3` W pipeline depth (default 3);
`VLLM_DCU_FP8_GEMV_MAX_M` coarse override (1 = old production behaviour).
