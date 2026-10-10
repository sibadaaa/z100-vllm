#!/usr/bin/env bash
# S11m batch 2: after the three =1 restarts of stage1.
#
#   tools/dcu/s11m_batch2.sh
#
#   F0  =1 FORCE_AOT_LOAD=1 + DEBUG against the stage1 cache -> the load path
#       must not throw (no silent fallback), and the traced-file list shows up.
#   F2  =1 benchmark_combo_kernel=false into an empty cache -> shows the flag
#       lands in compilation_config -> a different AOT key (fresh warm-up cost).
#   S4  =1 cache hit + flip,flipb -> third warm restart for the token gate and a
#       within-engine repeatability check on this tree.
#   Z1  =0 uncached (fills /mnt/game/vllm-cache-11m/zero) + flip
#   Z2  =0 cache hit + flip   -> =0 three-phase timing and a cross-engine
#       reproducibility control (is the stack cross-process nondeterministic
#       with the switch OFF too?)
set -u
WS=/home/lyh/z100-vllm
P="$WS/tools/dcu/s11m_probe_fresh.sh"
R="$WS/tools/dcu/s11m_run_one.sh"
CR=/mnt/game/vllm-cache-11m
CT=z100-s9

busy() { docker exec $CT bash -c 'ps -eo args | grep -c "[i]mport torch" || true' | tail -1; }
wait_clean() {
  while [ "$(busy)" -ne 0 ]; do echo "[batch2] waiting engine exit $(date -Is)"; sleep 20; done
}
while true; do
  rows=$(wc -l < "$WS/.tmp-11m/summary.tsv" 2>/dev/null || echo 0)
  b=$(busy)
  if [ "${rows:-0}" -ge 3 ] && [ "${b:-1}" -eq 0 ]; then break; fi
  echo "[batch2] waiting stage1: rows=$rows busy=$b $(date -Is)"
  sleep 30
done
echo "[batch2] stage1 done $(date -Is)"

echo "[batch2] ==== F0 FORCE_AOT_LOAD + DEBUG $(date -Is)"
S11M_FORCE_AOT_LOAD=1 S11M_LOGLEVEL=DEBUG S11M_DEVICES=0,1 \
  bash "$P" F0forceload "$CR/one"
wait_clean

echo "[batch2] ==== F2 benchmark_combo_kernel=false, empty cache $(date -Is)"
docker exec $CT bash -c "rm -rf $CR/freshB; mkdir -p $CR/freshB"
S11M_DEVICES=0,1 bash "$P" F2freshnocombo "$CR/freshB" \
  -cc.inductor_compile_config.benchmark_combo_kernel=false
wait_clean

echo "[batch2] ==== S4 =1 cache hit, flip+flipb $(date -Is)"
S11M_DEVICES=0,1 READY_TIMEOUT=2700 bash "$R" S4sw1hit 1 "$CR/one" 8048 4096 flip,flipb
wait_clean

echo "[batch2] ==== Z1 =0 uncached/fill $(date -Is)"
S11M_DEVICES=0,1 READY_TIMEOUT=2700 bash "$R" Z1sw0fill 0 "$CR/zero" 8049 4096 flip
wait_clean

echo "[batch2] ==== Z2 =0 cache hit $(date -Is)"
S11M_DEVICES=0,1 READY_TIMEOUT=2700 bash "$R" Z2sw0hit 0 "$CR/zero" 8050 4096 flip
wait_clean

echo "[batch2] all done $(date -Is)"
