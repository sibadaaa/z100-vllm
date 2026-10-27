#!/usr/bin/env bash
# Kernel / GDN / combined arm set (Lead-approved list, 2026-10-13).
#
# Run INSIDE the container, holding the global GPU lock for the whole set:
#   docker exec z100-vllm bash -c 'flock -w 3600 /tmp/perf1_build.lock \
#       bash /src/tools/dcu/perf1_armset_kernel.sh' > /mnt/game/perf1/arm_session3.log 2>&1
#
# Sequence: stop the running service -> repo numeric tests (need the whole DCU)
# -> attribution arms on the anchor protocol -> one full sweep for the recipe
# candidate (with warm-up) -> leave production running on the newest .so.
#
# Knobs (env): ARMSET_GEMV_MAX_M (default 4), ARMSET_K (MTP depth, default 3),
# ARMSET_COMM (extra KEY=VAL for the combined arm, default empty),
# ARMSET_SKIP_TESTS=1 to skip pytest, ARMSET_SKIP_RESTORE=1 to skip the last arm.
set -u
cd /src || exit 1

GEMV_MAX_M="${ARMSET_GEMV_MAX_M:-4}"
K="${ARMSET_K:-3}"
COMM="${ARMSET_COMM:-}"
STAGE="${ARMSET_STAGE:-all}"
OUT=/mnt/game/perf1

echo "=== [armset] start $(date -Is) stage=$STAGE gemv_max_m=$GEMV_MAX_M k=$K comm='$COMM' ==="
echo "=== [armset] so_md5=$(md5sum /src/vllm/_rocm_C.abi3.so | cut -d' ' -f1) ==="
python3 -u tools/dcu/perf1_verify_build.py --out "$OUT/verify_build_${STAGE}.json" \
  --label "stage=$STAGE" || \
  echo "=== [armset] WARN: staged build tree does not match the workspace sources ==="

if [ "${ARMSET_SKIP_TESTS:-0}" != "1" ]; then
  echo "=== [armset] stopping the service and running the repo numeric tests ==="
  bash tools/dcu/perf1_arm.sh stop || true
  bash tools/dcu/perf1_verify_tests.sh "$OUT/verify_pytest_kernel.log" "$@" || \
    echo "=== [armset] WARN pytest failed (see $OUT/verify_pytest_kernel.log) ==="
fi

if [ "$STAGE" = "kernel" ]; then
  # Stage 1: new-binary sanity + same-session cross-build reference + the
  # streaming-kernel dispatch arms.  GDN/combined stay for stage 2.
  # shellcheck disable=SC2086
  PERF1_PREFLIGHT_WAIT="${PERF1_PREFLIGHT_WAIT:-900}" bash tools/dcu/perf1_e2e_session.sh \
    --arm control_new anchor \
    --arm kern_draft VLLM_DCU_FP8_GEMV_MAX_M=1 anchor \
    --arm kern_verify VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" anchor \
    --arm kern_verify_old VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" VLLM_DCU_FP8_GEMV_STREAM=0 anchor \
    --arm restore:control_prod
elif [ "$STAGE" = "final" ]; then
  # Stage 2 (after the GDN fix and kernel-opt's final rebuild): attribution on
  # the FINAL .so + the recipe candidate with a full sweep (TTFT/prefill per
  # prompt length, p8192 included) and the warm-start variant.
  #   * ctl_preperf1 = pre-perf1 dispatch (S11F_MTP_GEMM=1 -> GEMV_MAX_M=1) on
  #     the SAME binary/session, so the dispatch win is not a cross-build
  #     comparison; it is also the canary (a failure aborts the set).
  #   * decodewarm on the gdn/combined arms forces the Triton JIT before any
  #     measured request.
  # shellcheck disable=SC2086
  PERF1_PREFLIGHT_WAIT="${PERF1_PREFLIGHT_WAIT:-900}" bash tools/dcu/perf1_e2e_session.sh \
    --arm ctl_preperf1 S11F_MTP_GEMM=1 anchor \
    --arm kern_verify VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" anchor \
    --arm kern_verify_old VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" VLLM_DCU_FP8_GEMV_STREAM=0 anchor \
    --arm gemv_route S11F_BF16_GEMV=1 S11F_MTP_GEMM=1 anchor \
    --arm gdn VLLM_QWEN35_MTP_GDN_TRITON=1 anchor decodewarm \
    --arm combined VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" VLLM_QWEN35_MTP_GDN_TRITON=1 \
          S11F_BF16_GEMV=1 S11F_MTP_TOKENS="$K" $COMM full warmup decodewarm \
    --arm restore:control_prod
else
# shellcheck disable=SC2086
PERF1_PREFLIGHT_WAIT="${PERF1_PREFLIGHT_WAIT:-900}" bash tools/dcu/perf1_e2e_session.sh \
  --arm kern_draft VLLM_DCU_FP8_GEMV_MAX_M=1 anchor \
  --arm kern_verify VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" anchor \
  --arm kern_verify_old VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" VLLM_DCU_FP8_GEMV_STREAM=0 anchor \
  --arm gdn VLLM_QWEN35_MTP_GDN_TRITON=1 anchor \
  --arm combined VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" VLLM_QWEN35_MTP_GDN_TRITON=1 \
        S11F_MTP_TOKENS="$K" $COMM full warmup \
  --arm restore:control_prod
fi

echo "=== [armset] done $(date -Is) ==="
bash tools/dcu/perf1_arm.sh status
