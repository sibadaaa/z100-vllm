#!/usr/bin/env bash
# S11m batch 3: does `benchmark_combo_kernel=false` collapse the cold/warm
# token difference seen in stage1 (S1 cold vs S2/S3 warm)?
#
#   tools/dcu/s11m_batch3.sh
#
#   W1 =1 + benchmark_combo_kernel=false, empty cache root -> cold compile + tokens
#   W2 =1 + benchmark_combo_kernel=false, same cache root  -> warm load   + tokens
#
# Then: W1 vs W2 (does the flag make cold == warm?) and W2 vs S2/S3 (is the
# flag's warm variant the same as the default warm variant?).
set -u
WS=/home/lyh/z100-vllm
R="$WS/tools/dcu/s11m_run_one.sh"
P="$WS/tools/dcu/s11m_probe_fresh.sh"
CR=/mnt/game/vllm-cache-11m
CT=z100-s9
FLAG="-cc.inductor_compile_config.benchmark_combo_kernel=false"

busy() { docker exec $CT bash -c 'ps -eo args | grep -c "[i]mport torch" || true' | tail -1; }
wait_clean() {
  while [ "$(busy)" -ne 0 ]; do echo "[batch3] waiting engine exit $(date -Is)"; sleep 20; done
}
while ! grep -q "\[batch2\] all done" "$WS/.tmp-11m/batch2.log" 2>/dev/null; do
  echo "[batch3] waiting batch2 $(date -Is)"; sleep 30
done
wait_clean
echo "[batch3] batch2 done, starting $(date -Is)"

echo "[batch3] ==== F0b FORCE_AOT_LOAD=1, INFO (DEBUG would flip inductor asserts -> new AOT key) $(date -Is)"
S11M_FORCE_AOT_LOAD=1 S11M_DEVICES=0,1 bash "$P" F0bforceload "$CR/one"
wait_clean

echo "[batch3] ==== W1 =1 combo=false cold $(date -Is)"
S11M_DEVICES=0,1 READY_TIMEOUT=2700 bash "$R" W1sw1nocombo 1 "$CR/combo" 8051 4096 flip $FLAG
wait_clean

echo "[batch3] ==== W2 =1 combo=false warm $(date -Is)"
S11M_DEVICES=0,1 READY_TIMEOUT=2700 bash "$R" W2sw1nocombo 1 "$CR/combo" 8052 4096 flip $FLAG
wait_clean

echo "[batch3] all done $(date -Is)"
