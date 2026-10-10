#!/usr/bin/env bash
# S11m: the three cheap follow-up probes, run after the stage1 (=1) restarts.
#
#   tools/dcu/s11m_probes_stage.sh
#
#   F0  =1 + VLLM_FORCE_AOT_LOAD=1 + DEBUG, against the *existing* stage1 cache
#       -> proves the load path no longer throws (FORCE turns any silent
#          fallback into a hard error) and prints the traced-file list.
#   F1  =1 into an EMPTY cache root -> second independent fresh compile of the
#       same config; used to compare generated-code fingerprints with S1.
#   F2  =1 + benchmark_combo_kernel=false into an EMPTY cache root -> shows that
#       the flag lands in `compilation_config` -> a DIFFERENT AOT key, i.e. it
#       buys nothing unless the cache is warmed again.
set -u
WS=/home/lyh/z100-vllm
P="$WS/tools/dcu/s11m_probe_fresh.sh"
CR=/mnt/game/vllm-cache-11m
CT=z100-s9

busy() { docker exec $CT bash -c 'ps -eo args | grep -c "[i]mport torch"' || echo 1; }
wait_clean() {
  while [ "$(busy)" -ne 0 ]; do echo "[probes] waiting for engine exit $(date -Is)"; sleep 20; done
}
fresh_dir() { docker exec $CT bash -c "rm -rf $1; mkdir -p $1"; }

# Wait for stage1 (three restarts) to be fully done: 3 summary rows + no engine
# left running inside the container.
while true; do
  rows=$(wc -l < "$WS/.tmp-11m/summary.tsv" 2>/dev/null || echo 0)
  b=$(busy)
  if [ "${rows:-0}" -ge 3 ] && [ "${b:-1}" -eq 0 ]; then break; fi
  echo "[probes] waiting stage1: rows=$rows busy=$b $(date -Is)"
  sleep 30
done
echo "[probes] stage1 done, starting probes $(date -Is)"

echo "[probes] ==== F0 FORCE_AOT_LOAD + DEBUG $(date -Is)"
S11M_FORCE_AOT_LOAD=1 S11M_LOGLEVEL=DEBUG S11M_DEVICES=0,1 \
  bash "$P" F0forceload "$CR/one"
wait_clean

echo "[probes] ==== F1 fresh compile, default inductor config $(date -Is)"
fresh_dir "$CR/freshA"
S11M_DEVICES=0,1 bash "$P" F1freshdefault "$CR/freshA"
wait_clean

echo "[probes] ==== F2 fresh compile, benchmark_combo_kernel=false $(date -Is)"
fresh_dir "$CR/freshB"
S11M_DEVICES=0,1 bash "$P" F2freshnocombo "$CR/freshB" \
  -cc.inductor_compile_config.benchmark_combo_kernel=false
wait_clean

echo "[probes] all done $(date -Is)"
