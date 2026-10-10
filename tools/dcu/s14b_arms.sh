#!/usr/bin/env bash
# S14b: sequential arm driver.  Runs one engine per arm (independent engines,
# explicit HIP_VISIBLE_DEVICES, artifacts under /mnt/game/s14b).
#
#   tools/dcu/s14b_arms.sh A2 A3 A4 ...
#
# Every arm is fully specified in ARM SPECS below so the exact command line of
# each recorded number is reproducible from this single file.  The s14_* /
# s13_* / s11* scripts are NOT touched; this driver only calls the s14b_* copies.
set -u
cd "$(dirname "$0")/../.." || exit 1
M27=/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8
M4=/mnt/game/models/Qwen3.5-4B

run() { # NAME MODEL TP SW CACHE PORT MAXLEN DEVICES TASKS EAGER CGMODE CGSIZES ENVS SERVED UTIL MBT GTOK FTOK
  local name=$1 model=$2 tp=$3 sw=$4 cache=$5 port=$6 maxlen=$7 devices=$8 tasks=$9
  local eager=${10} cgmode=${11} cgsizes=${12} envs=${13} served=${14} util=${15} mbt=${16}
  local gtok=${17} ftok=${18}
  echo "[driver] ===== $name start $(date -Is) ====="
  echo "[driver] $name model=$model tp=$tp sw=$sw cache=$cache port=$port maxlen=$maxlen devices=$devices eager=$eager cg=$cgmode/$cgsizes served=$served util=$util mbt=$mbt envs='$envs'"
  env S14B_CT=z100-s14 S14B_EAGER="$eager" S14B_CG_MODE="$cgmode" S14B_CG_SIZES="$cgsizes" \
      S14B_ENVS="$envs" S14B_SERVED="$served" S14B_UTIL="$util" S14B_MBT="$mbt" \
      S14B_GREEDY_TOKENS="$gtok" S14B_FLIP_TOKENS="$ftok" \
      S14B_PROMPTS=/mnt/game/s14b/prompts.txt READY_TIMEOUT="${S14B_RT:-2400}" \
      tools/dcu/s14b_run_one.sh "$name" "$model" "$tp" "$sw" "$cache" "$port" "$maxlen" "$devices" "$tasks" \
      > "/mnt/game/s14b/hostlogs/orch-$name.log" 2>&1
  local rc=$?
  echo "[driver] ===== $name done rc=$rc $(date -Is) ====="
  grep -E "ready=|summary:" "/mnt/game/s14b/hostlogs/orch-$name.log" | tail -3
}

# ---- arm specs -----------------------------------------------------------
# A1  : TP=2 graph, S14 P2 vintage (default FULL_AND_PIECEWISE, 51 buckets)
# A2  : TP=2 eager, S14 P2e vintage  -> the G1 pair with A1
# A3  : TP=2 compile, cudagraph_mode=NONE (inductor without any cudagraph)
# A4  : TP=2 graph PIECEWISE + 4 capture buckets, packed decode ON
#       (S14 P4g recipe: cheap capture; baseline for A5/A7)
# A5  : A4 + VLLM_QWEN35_PACKED_DECODE=0
# A6  : TP=2 eager + VLLM_QWEN35_PACKED_DECODE=0 (paired control for A5)
# A7  : A4 + VLLM_QWEN35_CHUNK_DECODE=0
# A8  : A4 + inductor combo_kernels/benchmark_combo_kernel OFF (task book's
#       explicit "config control" for mechanism B; changes the AOT key -> new cache)
# B1  : 4B TP=1 eager, card 3
# B2  : 4B TP=1 graph default cg, card 3
# B3  : 4B TP=1 compile, cudagraph_mode=NONE, card 3
for arm in "$@"; do
case "$arm" in
  A1) run A1 "$M27" 2 1 /mnt/game/s14/cache-27b-p2 8522 262144 0,1 promptlp,greedy,flip 0 "" "" "" qwen3.8-27b 0.85 2048 256 128 ;;
  A2) run A2 "$M27" 2 1 /mnt/game/s14/cache-27b-p2e 8523 262144 0,1 promptlp,greedy,flip 1 "" "" "" qwen3.8-27b 0.85 2048 256 128 ;;
  A3) run A3 "$M27" 2 1 /mnt/game/s14b/cache-27b-p2-cgnone 8524 262144 0,1 promptlp,greedy 0 NONE "" "" qwen3.8-27b 0.85 2048 256 128 ;;
  A4) run A4 "$M27" 2 1 /mnt/game/s14b/cache-27b-p2-pw4 8525 262144 0,1 promptlp,greedy 0 PIECEWISE "1 2 4 8" "" qwen3.8-27b 0.85 2048 256 128 ;;
  A5) run A5 "$M27" 2 1 /mnt/game/s14b/cache-27b-p2-pw4-nopacked 8526 262144 0,1 promptlp,greedy 0 PIECEWISE "1 2 4 8" "VLLM_QWEN35_PACKED_DECODE=0" qwen3.8-27b 0.85 2048 256 128 ;;
  A6) run A6 "$M27" 2 1 /mnt/game/s14b/cache-27b-p2e-nopacked 8527 262144 0,1 promptlp,greedy 1 "" "" "VLLM_QWEN35_PACKED_DECODE=0" qwen3.8-27b 0.85 2048 256 128 ;;
  A7) run A7 "$M27" 2 1 /mnt/game/s14b/cache-27b-p2-pw4-nochunk 8528 262144 0,1 promptlp,greedy 0 PIECEWISE "1 2 4 8" "VLLM_QWEN35_CHUNK_DECODE=0" qwen3.8-27b 0.85 2048 256 128 ;;
  A8) S14B_SERVE_EXTRA='-cc.inductor_compile_config={"combo_kernels":false,"benchmark_combo_kernel":false}' \
      run A8 "$M27" 2 1 /mnt/game/s14b/cache-27b-p2-pw4-nocombo 8529 262144 0,1 promptlp,greedy 0 PIECEWISE "1 2 4 8" "" qwen3.8-27b 0.85 2048 256 128 ;;
  B1) run B1 "$M4" 1 1 /mnt/game/s14b/cache-4b-eager 8531 32768 3 promptlp,greedy 1 "" "" "" qwen3.5-4b 0.90 2048 96 64 ;;
  B2) run B2 "$M4" 1 1 /mnt/game/s14b/cache-4b-graph 8532 32768 3 promptlp,greedy 0 "" "" "" qwen3.5-4b 0.90 2048 96 64 ;;
  B3) run B3 "$M4" 1 1 /mnt/game/s14b/cache-4b-cgnone 8533 32768 3 promptlp,greedy 0 NONE "" "" qwen3.5-4b 0.90 2048 96 64 ;;
  *) echo "[driver] unknown arm $arm" ;;
esac
done
echo "[driver] all done $(date -Is)"
