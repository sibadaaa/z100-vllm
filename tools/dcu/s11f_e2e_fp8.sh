#!/usr/bin/env bash
# S11f end-to-end: serve an FP8 checkpoint on Z100 (gfx906) through the emulated
# W8A8 FP8 kernel (csrc/rocm/fp8_emu_gemm.cu).
#
# Container-side launcher (the repo is mounted read-only at /src), so:
#
#   docker exec -d z100-vllm bash /src/tools/dcu/s11f_e2e_fp8.sh \
#       <MODEL_DIR> <TP> <PORT> [EXTRA...]
#
# Log: ${S11F_LOG:-/tmp/serve_fp8.log}.  The line that proves the new kernel is
# live is `Selected Gfx906Fp8ScaledMMLinearKernel for <module>` (vLLM logs it
# once per module at layer construction).
#
# Environment (all optional):
#   S11F_FP8_EMU    1|0  VLLM_DCU_USE_FP8_EMU    (default 1 = the arm under test;
#                        0 is the control: per-tensor FP8 must be refused)
#   S11F_BF16_GEMV  1|0  VLLM_DCU_USE_BF16_GEMV  (default 0: keep this arm clean)
#   S11F_DEVICES         HIP_VISIBLE_DEVICES     (default 0,1,2,3)
#   S11F_MAXLEN          --max-model-len         (default 32768)
#   S11F_UTIL            --gpu-memory-utilization(default 0.85)
#   S11F_MBT             --max-num-batched-tokens(default 2048)
#   S11F_EAGER      1|0  add --enforce-eager      (default 1; graph mode changes
#                        the token stream on this target -- S14b -- so keep the
#                        first arm eager and treat graph as a separate arm)
#   S11F_DTYPE           --dtype                 (default float16, production)
#   S11F_SERVED          --served-model-name     (default qwen3.8-27b-fp8)
#   S11F_LOG             log path                (default /tmp/serve_fp8.log)
#   S11F_CACHE           VLLM_CACHE_ROOT         (default .../vllm-cache-11f/e2e)
#   S11F_SRC             source tree             (default /src)
set -u
MODEL=${1:?usage: s11f_e2e_fp8.sh <MODEL_DIR> <TP> <PORT> [EXTRA...]}
TP=${2:?usage: s11f_e2e_fp8.sh <MODEL_DIR> <TP> <PORT> [EXTRA...]}
PORT=${3:?usage: s11f_e2e_fp8.sh <MODEL_DIR> <TP> <PORT> [EXTRA...]}
shift 3

. /opt/dtk/env.sh >/dev/null 2>&1
. "${S11F_SRC:-/src}/tools/dcu/env.sh" >/dev/null 2>&1
# RCCL P2P (S16/S17): without this the DTK/hydcu runtime answers
# hipDeviceAttributeFineGrainSupport=0, RCCL's p2pCanConnect() rejects every
# peer pair and all channels fall back to SHM.  With it: 112x P2P/IPC, 8K TTFT
# 1.89x, conc8 output +38%, greedy tokens/logprobs bitwise identical.
# Set S11F_P2P=0 (or HSA_FORCE_FINE_GRAIN_PCIE=0) to roll back to SHM.
export HSA_FORCE_FINE_GRAIN_PCIE="${S11F_P2P:-${HSA_FORCE_FINE_GRAIN_PCIE:-1}}"
export PYTHONPATH="${S11F_SRC:-/src}:/mnt/game/triton-gfx906/install"
export VLLM_CACHE_ROOT="${S11F_CACHE:-/mnt/game/vllm-cache-11f/e2e}"
export TRITON_CACHE_DIR="${S11F_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${S11F_DEVICES:-0,1,2,3}"
export VLLM_DCU_USE_FP8_EMU="${S11F_FP8_EMU:-1}"
# perf1 (2026-10-13): the dense fp16/bf16 decode GEMV now defaults ON.  The mm
# probe (tools/dcu/perf1_gdn_gpu.sh, /mnt/game/perf1/mm_probe.log) measured it
# against the Triton fallback this recipe actually runs: GDN in_proj_ba (N=24)
# 155.7 -> 19.0 us at M=3, MTP mtp.fc (N=1280) 415.8 -> 139.5 us, unquantized
# lm_head at M=1 1589.8 -> 902.8 us, i.e. ~8 ms/step of decode.
# Precedence: an explicit VLLM_DCU_USE_BF16_GEMV wins (needed for A/B arms),
# then S11F_BF16_GEMV, then 1.  Rollback: S11F_BF16_GEMV=0.
export VLLM_DCU_USE_BF16_GEMV="${VLLM_DCU_USE_BF16_GEMV:-${S11F_BF16_GEMV:-1}}"
export VLLM_LOGGING_LEVEL="${S11F_LOGLEVEL:-INFO}"
export PYTHONUNBUFFERED=1

EAGER=""
[ "${S11F_EAGER:-1}" = "1" ] && EAGER="--enforce-eager"

# S18 MTP (speculative decoding with the checkpoint's built-in MTP head).
# Default OFF: --speculative-config is only added when S11F_MTP_TOKENS is set.
#
# perf1 (2026-10-13): the M = 1+k verify pass now stays on the GEMV path.  The
# new streaming decode kernel (csrc/rocm/fp8_emu_gemm.cu) measures 2.5-9x faster
# than the tiled GEMM at the real decode shapes (M=3: 99.7 us vs 355 us at
# N=5120/K=5120, 37.6 vs 344 at N=1280), so VLLM_DCU_FP8_GEMV_MAX_M now defaults
# to 4 (= k=3 verify) and an explicit value from the environment always wins.
#   * rollback to the pre-perf1 dispatch: S11F_MTP_GEMM=1  -> GEMV_MAX_M=1
#   * k=4+ (M=5..8) falls back to the tiled GEMM inside the kernel when the
#     streaming path's LDS budget does not fit, so raising this is safe.
S11F_MTP_TOKENS="${S11F_MTP_TOKENS:-0}"
S11F_MTP_GEMM="${S11F_MTP_GEMM:-0}"
if [ "$S11F_MTP_GEMM" = "1" ]; then
  export VLLM_DCU_FP8_GEMV_MAX_M=1
else
  export VLLM_DCU_FP8_GEMV_MAX_M="${VLLM_DCU_FP8_GEMV_MAX_M:-4}"
fi
SPEC=""
[ "$S11F_MTP_TOKENS" != "0" ] && SPEC="--speculative-config {\"model\":\"${MODEL}\",\"method\":\"mtp\",\"num_speculative_tokens\":${S11F_MTP_TOKENS}}"

cd "${S11F_SRC:-/src}" || exit 1
echo "[s11f-e2e] model=$MODEL tp=$TP port=$PORT devices=$HIP_VISIBLE_DEVICES" \
     "fp8_emu=$VLLM_DCU_USE_FP8_EMU bf16_gemv=$VLLM_DCU_USE_BF16_GEMV eager=${S11F_EAGER:-1}" \
     "maxlen=${S11F_MAXLEN:-32768} host=$(hostname) date=$(date -Is)"
echo "[s11f-e2e] HSA_FORCE_FINE_GRAIN_PCIE=$HSA_FORCE_FINE_GRAIN_PCIE (1 = RCCL P2P)"
echo "[s11f-e2e] mtp_tokens=$S11F_MTP_TOKENS mtp_gemm=$S11F_MTP_GEMM (VLLM_DCU_FP8_GEMV_MAX_M=${VLLM_DCU_FP8_GEMV_MAX_M:-8})"
echo "[s11f-e2e] cache=$VLLM_CACHE_ROOT log=${S11F_LOG:-/tmp/serve_fp8.log}"

exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  "$MODEL" \
  --dtype "${S11F_DTYPE:-float16}" --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "${S11F_UTIL:-0.85}" --disable-custom-all-reduce \
  --skip-mm-profiling --enable-prefix-caching --mamba-cache-mode align \
  --max-model-len "${S11F_MAXLEN:-32768}" --max-num-batched-tokens "${S11F_MBT:-2048}" \
  --served-model-name "${S11F_SERVED:-qwen3.8-27b-fp8}" --host 127.0.0.1 --port "$PORT" \
  $EAGER $SPEC "$@"
