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
export SPEC_OFF=1
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
# S40-B1 (ADOPTED 2026-10-31): route the lm_head shape class
# (M == 1, N >= 2048, K % 128 == 0, fp16 dense GEMV) to the streaming
# fp16_gemv_stream_m1 kernel: 797 vs 709 GB/s (+12.3%, -0.10 ms/step;
# docs/perf/data/s40-b2-on.txt, docs/perf/data/s40-b0-verdict.md).
# Rollback: comment the export + systemctl restart (e2e ABAB: no regression;
# A-vs-B logprob 0.0107 nat, 0 token mismatch; trace 808.6 vs 899.3 us).
export VLLM_DCU_FP16_GEMV_STREAM=1
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
