#!/usr/bin/env bash
# S11m: run the 27B W8A8 restarts needed by the S11m gate.
#
#   tools/dcu/s11m_matrix.sh <stage>
#
# stage = stage1  -> =1 with a fresh cache root: fill (uncached) + 2 cache hits
#         stage2  -> =0 with a fresh cache root: fill (uncached) + 2 cache hits
#         stage3  -> =1 cache hit with benchmark_combo_kernel=False (workaround)
#         all     -> stage1 stage2 stage3
#
# One fresh cache root per switch value, under /mnt/game/vllm-cache-11m/, so the
# three phases (uncached / cache-fill / cache-hit) are not polluted by any
# pre-existing cache.  Every run executes the same greedy token battery
# (tools/dcu/s11l_flip_stats.py, 8 prompts x 64 tokens) so token ids can be
# compared across restarts.
set -u
WS=/home/lyh/z100-vllm
RUN="$WS/tools/dcu/s11m_run_one.sh"
CR=/mnt/game/vllm-cache-11m
STAGE="${1:?usage: s11m_matrix.sh <stage1|stage2|stage3|all>}"

run_one() { # name sw cache port maxlen tasks extra
  echo "[matrix] ==== $1 (sw=$2 cache=$3) $(date -Is)"
  S11M_DEVICES=0,1 READY_TIMEOUT=2700 bash "$RUN" "$@"
  echo "[matrix] ==== $1 done rc=$? $(date -Is)"
}

do_stage1() {
  run_one S1sw1fill 1 "$CR/one"  8041 4096 flip
  run_one S2sw1hit  1 "$CR/one"  8042 4096 flip
  run_one S3sw1hit  1 "$CR/one"  8043 4096 flip
}
do_stage2() {
  run_one Z1sw0fill 0 "$CR/zero" 8044 4096 flip
  run_one Z2sw0hit  0 "$CR/zero" 8045 4096 flip
  run_one Z3sw0hit  0 "$CR/zero" 8046 4096 flip
}
do_stage3() {
  run_one S4sw1nocombo 1 "$CR/one" 8047 4096 flip \
    -cc.inductor_compile_config.benchmark_combo_kernel=False
}

case "$STAGE" in
  stage1) do_stage1 ;;
  stage2) do_stage2 ;;
  stage3) do_stage3 ;;
  all)    do_stage1; do_stage2; do_stage3 ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "[matrix] $STAGE complete $(date -Is)"
