#!/usr/bin/env bash
# perf1 baseline / A-B suite (run INSIDE the container, under the build lock):
#
#   docker exec z100-vllm bash -c 'flock /tmp/perf1_build.lock \
#       bash /src/tools/dcu/perf1_e2e_session.sh --arm control [KV...] \
#       --arm k3 S11F_MTP_TOKENS=3 --arm control_restore' \
#       > /mnt/game/perf1/arm_session.log 2>&1
#
# One invocation = one arm:
#   1. snapshot the currently running service (cmdline + env) for the record,
#   2. stop it and start the requested arm (control = production recipe,
#      no --profiler-config) via perf1_arm.sh,
#   3. cold-start probe (first request after a fresh start, p8192),
#   4. perf sweep p128/512/2048/8192, g96, 3 repeats, ignore_eos, conc 1,
#   5. anchor run p512 g64 (matches vllm-test/results/final_verify_*.json),
#   6. greedy parity capture (/v1/completions, return_token_ids + logprobs).
#
# Artifacts land in /mnt/game/perf1/<arm>_*.json.  The arm service is left
# RUNNING at the end; the suite only ever stops the service it is replacing.
# The _rocm_C.abi3.so md5 is recorded before and after every arm and embedded
# in the JSON; a change invalidates the arm (a rebuild slipped in).
set -u
ARM="${1:?usage: perf1_e2e_suite.sh <arm-name> [KEY=VAL ...]}"
shift || true
PERF1_DIR="${PERF1_DIR:-/mnt/game/perf1}"
OUT="$PERF1_DIR"

# Start from a known-clean knob set: docker exec does not inherit the service
# env, but be explicit so no arm is accidentally the result of stray exports.
unset S11F_FP8_EMU S11F_BF16_GEMV S11F_MAXLEN S11F_UTIL S11F_MBT S11F_EAGER \
      S11F_DTYPE S11F_SERVED S11F_CACHE S11F_LOGLEVEL S11F_DEVICES S11F_TRITON_CACHE \
      S11F_MTP_TOKENS S11F_MTP_GEMM S11F_P2P VLLM_DCU_USE_FP8_EMU \
      VLLM_DCU_USE_BF16_GEMV VLLM_DCU_FP8_GEMV_MAX_M VLLM_GFX906_TRITON_CAUSAL_CONV1D \
      VLLM_QWEN35_MTP_GDN_DEBUG VLLM_QWEN35_MTP_CONV_DEBUG 2>/dev/null || true

cd /src || exit 1
mkdir -p "$OUT"
ALOG="$OUT/arm_${ARM}.log"

# Measurement shape (overridable so a consolidated multi-arm set stays cheap):
#   PERF1_MODE=anchor -> p512/g64 anchor protocol only (~40 s client: 3 perf
#                        reps + 1 greedy parity capture), for attribution arms;
#   PERF1_MODE=full   -> cold probe + 4-length sweep + parity + anchor.
# Individual PERF1_* vars still win over the mode defaults.
if [ "${PERF1_MODE:-full}" = "anchor" ]; then
  LENGTHS="${PERF1_LENGTHS:-512}"; MAXTOK="${PERF1_MAX_TOKENS:-64}"
  REPEAT="${PERF1_REPEAT:-3}"; COLD_LEN="${PERF1_COLD_LENGTH:-0}"
  COLD_MAX="${PERF1_COLD_MAX_TOKENS:-32}"
  PARITY_LENGTHS="${PERF1_PARITY_LENGTHS:-512}"
  PARITY_MAX="${PERF1_PARITY_MAX_TOKENS:-64}"
  PARITY_REPEAT="${PERF1_PARITY_REPEAT:-1}"; ANCHOR="${PERF1_ANCHOR:-0}"
else
  LENGTHS="${PERF1_LENGTHS:-128,512,2048,8192}"; MAXTOK="${PERF1_MAX_TOKENS:-96}"
  REPEAT="${PERF1_REPEAT:-3}"; COLD_LEN="${PERF1_COLD_LENGTH:-8192}"
  COLD_MAX="${PERF1_COLD_MAX_TOKENS:-32}"
  PARITY_LENGTHS="${PERF1_PARITY_LENGTHS:-512,2048}"
  PARITY_MAX="${PERF1_PARITY_MAX_TOKENS:-64}"
  PARITY_REPEAT="${PERF1_PARITY_REPEAT:-2}"; ANCHOR="${PERF1_ANCHOR:-1}"
fi

echo "=== [suite] arm=$ARM $(date -Is) overrides: $* ===" | tee -a "$ALOG"
SO0=$(md5sum /src/vllm/_rocm_C.abi3.so | cut -d' ' -f1)
echo "[suite] so_md5_start=$SO0" | tee -a "$ALOG"
{
  echo "arm=$ARM date=$(date -Is)"
  echo "so_md5=$SO0 so_target=$(readlink -f /src/vllm/_rocm_C.abi3.so)"
  echo "git_head=$(git -C /src rev-parse HEAD 2>/dev/null)"
  echo "--- git status (relevant trees) ---"
  git -C /src status --porcelain -- csrc vllm tools/dcu 2>/dev/null | head -40
  echo "--- source md5 ---"
  md5sum /src/csrc/rocm/fp8_emu_gemm.cu \
         /src/csrc/rocm/gfx906_emu_common.cuh \
         /src/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py \
         /src/vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py \
         /src/vllm/envs.py 2>/dev/null
} > "$OUT/arm_${ARM}_code_state.txt"

{
  bash tools/dcu/perf1_arm.sh snapshot "pre_${ARM}"
  bash tools/dcu/perf1_arm.sh start "$ARM" "$@"
} 2>&1 | tee -a "$ALOG"
bash tools/dcu/perf1_arm.sh status 2>&1 | tee -a "$ALOG"
grep -q "arm $ARM live" "$ALOG" || { echo "[suite] arm $ARM failed to start" | tee -a "$ALOG"; exit 1; }

# Override-survival guard.  The recipe launcher is allowed to compute defaults,
# but it must never silently overwrite a knob the arm explicitly asked for.
# Observed 2026-10-13: tools/dcu/s11f_e2e_fp8.sh exported
# VLLM_DCU_FP8_GEMV_MAX_M=1 whenever MTP was on, so four nominally different
# arms all ran the production dispatch and produced four identical results.
# Every KEY=VAL this arm passed is checked against the live worker environment.
for kv in "$@"; do
  key="${kv%%=*}"; want="${kv#*=}"
  got=$(sed -n "s/^${key}=//p" "$OUT/arm_${ARM}_environ.txt" 2>/dev/null | head -1)
  if [ "$got" != "$want" ]; then
    {
      echo "[suite] FATAL override did not survive the recipe: asked $key=$want, effective '$key=${got:-<unset>}'"
      echo "[suite] see $OUT/arm_${ARM}_environ.txt -- the arm is VOID"
    } | tee -a "$ALOG"
    exit 6
  fi
  echo "[suite] override OK: $key=$got" | tee -a "$ALOG"
done

# Derived-knob guard: an arm phrased against the wrapper's S11F_* namespace must
# also see the derived VLLM_*/HSA_* value in the worker environment.  Observed
# 2026-10-13: /mnt/game/port/serve_fp8_8000.sh pins S11F_BF16_GEMV=0, so an arm
# asking S11F_BF16_GEMV=1 never enabled the dense GEMV even after the launcher
# was fixed one level down.  '*' means "must be set to something".
derived_expect() {
  case "$1=$2" in
    S11F_FP8_EMU=1) echo "VLLM_DCU_USE_FP8_EMU=1" ;;
    S11F_FP8_EMU=0) echo "VLLM_DCU_USE_FP8_EMU=0" ;;
    S11F_BF16_GEMV=1) echo "VLLM_DCU_USE_BF16_GEMV=1" ;;
    S11F_BF16_GEMV=0) echo "VLLM_DCU_USE_BF16_GEMV=0" ;;
    S11F_MTP_GEMM=1) echo "VLLM_DCU_FP8_GEMV_MAX_M=1" ;;
    S11F_MTP_GEMM=0) echo "VLLM_DCU_FP8_GEMV_MAX_M=*" ;;
    S11F_P2P=1) echo "HSA_FORCE_FINE_GRAIN_PCIE=1" ;;
    S11F_P2P=0) echo "HSA_FORCE_FINE_GRAIN_PCIE=0" ;;
    S11F_DEVICES=*) echo "HIP_VISIBLE_DEVICES=$2" ;;
  esac
}
for kv in "$@"; do
  key="${kv%%=*}"; want="${kv#*=}"
  exp=$(derived_expect "$key" "$want")
  [ -n "$exp" ] || continue
  dkey="${exp%%=*}"; dwant="${exp#*=}"
  dgot=$(sed -n "s/^${dkey}=//p" "$OUT/arm_${ARM}_environ.txt" 2>/dev/null | head -1)
  if [ "$dwant" = "*" ]; then
    [ -n "$dgot" ] || dgot=""
    [ -n "$dgot" ] && { echo "[suite] derived OK: $key=$want -> $dkey=$dgot" | tee -a "$ALOG"; continue; }
  elif [ "$dgot" = "$dwant" ]; then
    echo "[suite] derived OK: $key=$want -> $dkey=$dgot" | tee -a "$ALOG"
    continue
  fi
  {
    echo "[suite] FATAL derived knob mismatch: asked $key=$want (=> $dkey=$dwant), effective '$dkey=${dgot:-<unset>}'"
    echo "[suite] see $OUT/arm_${ARM}_environ.txt -- the arm is VOID"
  } | tee -a "$ALOG"
  exit 6
done

# Optional warm-start arm: warm the prefill shapes before the first measured
# request, so the client's "cold_start" row becomes the post-warmup penalty.
WARM_FLAG=""
if [ "${PERF1_WARMUP:-0}" = "1" ]; then
  echo "[suite] PERF1_WARMUP=1 -> warming prefill shapes before measurement" | tee -a "$ALOG"
  python3 -u tools/dcu/perf1_warmup.py --url "http://127.0.0.1:${PERF1_PORT:-8000}" \
    --model "${S11F_SERVED:-qwen3.8-27b}" \
    --lens "${PERF1_WARMUP_LENS:-8192,2048,512,128}" 2>&1 | tee -a "$ALOG"
  WARM_FLAG="--warmup"
fi

# Decode-path warm-up: one discarded chat request with real decode steps.  The
# prefill warm-up above does not reach decode-only kernels (e.g. gdn-opt's
# Triton GDN MTP kernel JITs on the first decode step, 10-40 s, which would land
# in the first measured request and look like a regression).
if [ "${PERF1_DECODE_WARMUP:-0}" = "1" ]; then
  echo "[suite] PERF1_DECODE_WARMUP=1 -> discarded 16-token decode request" | tee -a "$ALOG"
  python3 -u tools/dcu/perf1_e2e_bench.py \
    --arm "${ARM}_decodewarm" --so-md5 "$SO0" --tag warmup --mode perf \
    --lengths 128 --max-tokens 16 --repeat 1 \
    --out "$OUT" --out-name "warmup_discard_${ARM}.json" 2>&1 | tee -a "$ALOG"
fi

if [ "${PERF1_PARITY_ONLY:-0}" = "1" ]; then
  echo "[suite] PERF1_PARITY_ONLY=1 -> parity capture only (no cold probe/sweep)" | tee -a "$ALOG"
  python3 -u tools/dcu/perf1_e2e_bench.py \
    --arm "$ARM" --so-md5 "$SO0" --tag "e2e_${ARM}" --mode parity \
    --parity-lengths "$PARITY_LENGTHS" --parity-max-tokens "$PARITY_MAX" \
    --parity-repeat "$PARITY_REPEAT" \
    --out "$OUT" --out-name "${ARM}_e2e.json" 2>&1 | tee -a "$ALOG"
  rc_main=${PIPESTATUS[0]}
  rc_anchor=0
else
  python3 -u tools/dcu/perf1_e2e_bench.py \
    --arm "$ARM" --so-md5 "$SO0" --tag "e2e_${ARM}" --mode both \
    --lengths "$LENGTHS" --max-tokens "$MAXTOK" --repeat "$REPEAT" \
    --cold-length "$COLD_LEN" --cold-max-tokens "$COLD_MAX" $WARM_FLAG \
    --parity-lengths "$PARITY_LENGTHS" --parity-max-tokens "$PARITY_MAX" \
    --parity-repeat "$PARITY_REPEAT" \
    --out "$OUT" --out-name "${ARM}_e2e.json" 2>&1 | tee -a "$ALOG"
  rc_main=${PIPESTATUS[0]}

  if [ "$ANCHOR" = "1" ]; then
    python3 -u tools/dcu/perf1_e2e_bench.py \
      --arm "$ARM" --so-md5 "$SO0" --tag "e2e_${ARM}" --mode perf \
      --lengths 512 --max-tokens 64 --repeat "${PERF1_ANCHOR_REPEAT:-3}" \
      --out "$OUT" --out-name "${ARM}_anchor_p512_g64.json" 2>&1 | tee -a "$ALOG"
    rc_anchor=${PIPESTATUS[0]}
  else
    echo "[suite] PERF1_ANCHOR=0 -> skipping the p512 g64 anchor" | tee -a "$ALOG"
    rc_anchor=0
  fi
fi

SO1=$(md5sum /src/vllm/_rocm_C.abi3.so | cut -d' ' -f1)
{
  echo "so_md5_before=$SO0"
  echo "so_md5_after=$SO1"
  [ "$SO0" = "$SO1" ] && echo "so_stable=yes" || echo "so_stable=NO -- arm invalid, a rebuild swapped the .so"
} > "$OUT/arm_${ARM}_so.txt"
echo "[suite] arm=$ARM done $(date -Is) rc_main=$rc_main rc_anchor=$rc_anchor so_stable=$([ "$SO0" = "$SO1" ] && echo yes || echo NO)" | tee -a "$ALOG"
bash tools/dcu/perf1_arm.sh status 2>&1 | tee -a "$ALOG"
[ "$SO0" = "$SO1" ] || exit 3
exit $(( rc_main != 0 || rc_anchor != 0 ))
