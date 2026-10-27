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
export MAX_SEQS=4
export MBT=8192
export PORT=8000
# HOST defaults to 0.0.0.0 inside the launcher; set HOST=127.0.0.1 to lock down.

exec bash .s34/launch_mtp2.sh
