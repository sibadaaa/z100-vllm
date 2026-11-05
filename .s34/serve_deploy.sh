#!/usr/bin/env bash
# S34 production deployment (2026-10-27): s33 ctrim + WP2 push-AR + native MTP k=2.
# Same kernel/env stack as the s33 no-spec arm; the ONLY functional change is
# --speculative-config '{"method":"mtp","num_speculative_tokens":2}'.
# Measured same-protocol (runs/specbench/REPORT.md): greedy 51.6 vs 36.9 t/s,
# sampled 40.6 vs 34.7 t/s; warm TTFT 0.72 -> 2.4 s.
#
# Start/stop:  .s34/start.sh / .s34/stop.sh   (wraps systemctl z100-vllm-mtp2)
# Logs:        journalctl -u z100-vllm-mtp2 -f
# Rollback:    bash .s34/rollback.sh          (back to the s33 no-spec arm)
#
# Start/stop unit:  systemctl start|stop|restart z100-vllm-mtp2
# Rollback:         comment out the three VLLM_DCU_* exports below (RCCL path),
#                   or systemctl start z100-vllm-s33 (mutual Conflicts stops this arm).
set -u
cd /home/lyh/z100-vllm

export VLLM_DCU_CUSTOM_AR=1
export VLLM_DCU_CUSTOM_AR_ALLOC=finegrained
export VLLM_CUSTOM_ALLREDUCE_ALGO=oneshot
# S34-WP3: GEMV stack (adopted, docs/result/z100-s34-wp3-gemv-stack.md).
export VLLM_DCU_FP8_GEMV_STREAM_ROWS=4
export VLLM_DCU_FP8_GEMV_M1=1
# S34-WP2: AR size gate 128 KiB (adopted, docs/result/z100-s34-wp2-size-gate.md).
# Default 64 = S33 behavior; 256 validated equal in service but 128 is the
# microbench sweet spot (push +15.4% @120 KiB, break-even @256 KiB).
export VLLM_DCU_CUSTOM_AR_MAX_KIB=128
# 2026-10-30: MTP disabled at operator request (decode-stage re-profile run).
# Kernel/env stack unchanged (fingerprint e9f6b1ff41e682d6); the ONLY change is
# no --speculative-config (SPEC_OFF implemented in launch_mtp2.sh, DEPLOY.md
# contract).  Rollback: comment out the next line + systemctl restart
# z100-vllm-mtp2 (root) to restore MTP k=2 (~51.6 greedy t/s, warm TTFT 2.4 s).
# S46 (ADOPTED 2026-11-01): MTP k=2 recommissioned on the S44/S45 adopted
# stack (AR fusion hybrid + W8A16 silu-mul fold).  SPEC_OFF commented out per
# the documented contract -> launch_mtp2.sh boots the MTP k=2 default via CLI.
# Same-day ABAB (runs/s46/e2e-s46-mtp-{B1,A1,B2,A2}.json): W1 med 51.35/52.72
# (B) vs 38.44/39.16 (A) = +33.6%/+34.6% paired (gate +29%); W2 med 40.06/44.49
# vs 37.72/37.66 (B-pooled 42.4 >= 41.3 floor, strict paired +12.4%); TTFT
# 2.44-2.47 s (S34 cost band); board first_div=None.  Long-ctx same-protocol:
# 50k +179% (24.8 vs 8.9 tps); 128k -16% (15.2 vs 18.0; S37 +28% NOT
# reproduced on this stack, soft gate, recorded only).  Details:
# .s46/mtp-adopt.md.  Rollback: uncomment the next line + systemctl restart
# z100-vllm-mtp2 (env-level, zero .so/code change).
# export SPEC_OFF=1
# S39-B1 probe (TEMPORARY, remove after the verdict): one-time log of
# custom-AR refusals to decide the nccl ring fallback root cause.
# S38 A1 (temporary ABAB toggle): fuse the DCU push-AR epilogue with the
# decoder add-rmsnorm (Gemma (1+w) convention) via
# vllm.all_reduce_fused_rmsnorm.  A arms: uncomment; B arms: comment out.
# S38 A1 (temporary ABAB toggle): fuse the DCU push-AR epilogue with the
# decoder add-rmsnorm (Gemma (1+w) convention) via
# vllm.all_reduce_fused_rmsnorm.  NOT ADOPTED in S38 round 1: the attn-only
# variant is bit-perfect (parity 8/8) but the two-exit variant corrupts
# under piecewise compile+capture (eager is clean) -- see .s38/a1-close.md.
# A arms: uncomment the two exports; B arms (current): keep commented.
# export VLLM_DCU_CUSTOM_AR_FUSION=1
# export VLLM_DCU_CUSTOM_AR_FUSION_MLP=0
# S44-A4 (ADOPTED 2026-11-01): attn-exit push-AR + add-rmsnorm fusion (hybrid
# arm, MLP exit unfused).  B/A/B/A e2e: W1 41.73/40.70 (B) vs 40.03/39.59 (A)
# t/s (+4.25%/+2.81% paired, both >= +2% gate); norm class-ledger 1.635 ->
# 0.392 ms/step; TTFT delta within intra-A-arm spread.  The two-exit (full)
# variant is FAIL (garbage under piecewise capture, S38 corruption reproduced:
# docs/perf/data/s44-a0-debug.txt) and MUST stay off (FUSION_MLP=0).
# Rollback: comment the two exports + systemctl restart (env-off = bitwise
# pre-S44 path; .so unchanged, md5 54ccf2e2).
export VLLM_DCU_CUSTOM_AR_FUSION=1
export VLLM_DCU_CUSTOM_AR_FUSION_MLP=0
# S45-N2a (ADOPTED 2026-11-01): fold the MLP SiluAndMul activation into the
# W8A16 down_proj wholea_m1 GEMV's whole-A staging (M == 1 decode arm).
# Bit-identical: torch.equal parity max|d|=0 (.s45/n2a_parity.py), service
# token stream identical to the unfused arm (board first_div=None), the
# mul_silu kernel leaves the ledger (-47 launches/step).  Requires the
# rebuilt _rocm_C (md5 6dee7fad; backup _rocm_C.s44-before.so = f72bb39d).
# Rollback: comment the export + systemctl restart (env-off = bitwise
# pre-S45 path; with the old .so also restored, fully pre-S45).
# S49-A1 (2026-11-03) REVERTED: the smul fold's whole-A staging reads g|u,
# doubling per-launch A traffic to N*K (= the W bytes); measured down GEMV
# 69.8us/319 GB/s vs plain 41.8us/534 (graph trace + card-3 window bench,
# output bit-identical either way).  See runs/fulldec/graph_recalibration.md
# section 2 and docs/design/z100/s49/.  Re-enable only behind the S49
# L2-smul kernel variant (per-slice A read, no whole-row staging).
# export VLLM_DCU_W8A16_SILU_MUL=1
# NOTE (S45 audit, DO NOT SET): VLLM_QWEN35_EMPTY_CORE_ATTN_OUT=1 forks the
# greedy stream -- gdn_attention_core leaves elements of core_attn_out
# unwritten and downstream reads them; the torch.zeros is load-bearing
# (upstream PR 28182).  See .s45/s45-adopt.md §2.
# S40-B1 (ADOPTED 2026-10-31): route the lm_head shape class
# (M == 1, N >= 2048, K % 128 == 0, fp16 dense GEMV) to the streaming
# fp16_gemv_stream_m1 kernel: 797 vs 709 GB/s (+12.3%, -0.10 ms/step;
# docs/perf/data/s40-b2-on.txt, docs/perf/data/s40-b0-verdict.md).
# Rollback: comment the export + systemctl restart (e2e ABAB: no regression;
# A-vs-B logprob 0.0107 nat, 0 token mismatch; trace 808.6 vs 899.3 us).
export VLLM_DCU_FP16_GEMV_STREAM=1
# S53-ATTN (2026-11-04, user-approved production): gfx906 decode split-KV
# attention inner-loop rebuild (HIP dot2, csrc/rocm/attnd_inner.hip) --
# microbench 141 GB/s dedup @109k = 2.35x in-service Triton; numerics
# 12/12 vs SDPA fp32 (|d| <= 1.4e-4); env-gated dispatch, supported subset
# only (num_seqs 1 / causal / fp16 / head 256 / GQA6), else Triton bit-path.
# Rollback: comment this export + systemctl restart z100-vllm-mtp2 (the
# dispatch reads the env live; the Triton path is bit-identical to pre-S53).
export VLLM_GFX906_ATTN_DECODE_HIP_INNER=1
# S41-A1 (ABAB in progress, 2026-10-31): wholea adoption candidate --
# route M==1 fp8 GEMV to fp8_emu_gemv_wholea_m1_kernel (bit-identical to m1,
# .s41/a0 dump 7/7 torch.equal green; probe window 643 vs m1-form 582).
# B arms: keep uncommented.  A arms: comment out + restart.
# Rollback: comment the export (bit-identical, .so rollback not needed).
export VLLM_DCU_FP8_GEMV_WA=1
export MAX_SEQS=4
export MBT=8192
export PORT=8000
# HOST defaults to 0.0.0.0 inside the launcher; set HOST=127.0.0.1 to lock down.

exec bash .s34/launch_mtp2.sh
