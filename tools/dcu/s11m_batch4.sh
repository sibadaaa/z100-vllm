#!/usr/bin/env bash
# S11m batch 4: third =0 warm start (Z3) -> is the =0 load path reproducible too
# (Z2 == Z3), the way the =1 load path is (S2 == S3 == S4)?
set -u
WS=/home/lyh/z100-vllm
R="$WS/tools/dcu/s11m_run_one.sh"
CR=/mnt/game/vllm-cache-11m
CT=z100-s9
busy() { docker exec $CT bash -c 'ps -eo args | grep -c "[i]mport torch" || true' | tail -1; }
wait_clean() { while [ "$(busy)" -ne 0 ]; do echo "[batch4] waiting engine exit $(date -Is)"; sleep 20; done; }
while ! grep -q "\[batch3\] all done" "$WS/.tmp-11m/batch3.log" 2>/dev/null; do
  echo "[batch4] waiting batch3 $(date -Is)"; sleep 30
done
wait_clean
echo "[batch4] batch3 done, starting Z3 $(date -Is)"
S11M_DEVICES=0,1 READY_TIMEOUT=2700 bash "$R" Z3sw0hit 0 "$CR/zero" 8053 4096 flip
wait_clean
echo "[batch4] all done $(date -Is)"
