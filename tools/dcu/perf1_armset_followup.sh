#!/usr/bin/env bash
# Follow-up verification session (after stage 2), one lock window.
#
#   docker exec z100-vllm bash -c 'flock -w 3600 /tmp/perf1_build.lock \
#     bash /src/tools/dcu/perf1_armset_followup.sh' \
#     > /mnt/game/perf1/arm_session6.log 2>&1
#
# Arms and why they exist:
#   ctl_preperf1_256 / kern_verify_256 : 256-token greedy parity pair (divergence
#       rate vs length for the shipped dispatch change), parity_only.
#   gemv_route  : isolated dense-GEMV routing (VLLM_DCU_USE_BF16_GEMV=1) with the
#       fp8 dispatch pinned to pre-perf1, so it is single-variable vs
#       ctl_preperf1.  The first attempt was voided: the production wrapper pins
#       S11F_BF16_GEMV=0, so the arm must speak VLLM_DCU_USE_BF16_GEMV.
#   gdn_aba_a/b/c : A-B-A of the GDN Triton kernel on the EXACT kern_verify
#       config (MAX_M=4, BF16 GEMV off, k=2, anchor) because the original
#       +10.3% step result compared arms 11 minutes and 3 restarts apart.  B
#       carries VLLM_QWEN35_MTP_GDN_TRITON=1 + _TRITON_PROBE=1 + _DEBUG=1 so the
#       worker logs per-call host_us / gpu_us / interval_us.
#   combined_v2 : recipe candidate = MAX_M=4 + dense-GEMV routing + k=3, full
#       4-length sweep incl. p8192.  GDN triton deliberately absent.
#   restore:control_prod : leaves :8000 on the production recipe.
set -u
cd /src || exit 1
OUT=/mnt/game/perf1
GEMV_MAX_M="${ARMSET_GEMV_MAX_M:-4}"
K="${ARMSET_K:-3}"

export PERF1_PARITY_MAX_TOKENS="${PERF1_PARITY_MAX_TOKENS:-256}"
export PERF1_PARITY_REPEAT="${PERF1_PARITY_REPEAT:-2}"

echo "=== [followup] start $(date -Is) so=$(md5sum /src/vllm/_rocm_C.abi3.so | cut -d' ' -f1) ==="
echo "=== [followup] parity max tokens=$PERF1_PARITY_MAX_TOKENS reps=$PERF1_PARITY_REPEAT ==="
python3 -u tools/dcu/perf1_verify_build.py --out "$OUT/verify_build_followup.json" \
  --label followup || \
  echo "=== [followup] WARN: staged build tree does not match the workspace sources ==="

if [ "${ARMSET_SKIP_TESTS:-0}" != "1" ]; then
  echo "=== [followup] stopping the service and running the repo numeric tests ==="
  bash tools/dcu/perf1_arm.sh stop || true
  bash tools/dcu/perf1_verify_tests.sh "$OUT/verify_pytest_followup.log" || \
    echo "=== [followup] WARN pytest failed (see $OUT/verify_pytest_followup.log) ==="
fi

PERF1_PREFLIGHT_WAIT="${PERF1_PREFLIGHT_WAIT:-900}" bash tools/dcu/perf1_e2e_session.sh \
  --arm ctl_preperf1_256 S11F_MTP_GEMM=1 parity_only \
  --arm kern_verify_256 VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" parity_only \
  --arm gemv_route VLLM_DCU_USE_BF16_GEMV=1 S11F_MTP_GEMM=1 anchor \
  --arm gdn_aba_a VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" anchor \
  --arm gdn_aba_b VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" VLLM_QWEN35_MTP_GDN_TRITON=1 \
        VLLM_QWEN35_MTP_GDN_TRITON_PROBE=1 VLLM_QWEN35_MTP_GDN_DEBUG=1 anchor decodewarm \
  --arm gdn_aba_c VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" anchor \
  --arm combined_v2 VLLM_DCU_FP8_GEMV_MAX_M="$GEMV_MAX_M" VLLM_DCU_USE_BF16_GEMV=1 \
        S11F_MTP_TOKENS="$K" full warmup decodewarm \
  --arm restore:control_prod

echo "=== [followup] done rc=$? $(date -Is) ==="
bash tools/dcu/perf1_arm.sh status
