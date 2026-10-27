#!/usr/bin/env bash
# S11l: run the 7 fork gfx906 test files under VLLM_DCU_USE_INT8_SDOT4=<SW>.
#
#   docker exec z100-s9 bash /src/tools/dcu/s11l_tests.sh <0|1> <outdir>
#
# Container-side script.  Runs from /src (workspace tree, read-only in the
# container), keeps every log in <outdir> (which is expected to live under
# /mnt/game) and prints one machine-readable summary line per file.
set -u
SW="${1:?usage: s11l_tests.sh <0|1> <outdir>}"
OUT="${2:?usage: s11l_tests.sh <0|1> <outdir>}"
mkdir -p "$OUT"

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b
export VLLM_DCU_USE_INT8_SDOT4="$SW"
export HIP_VISIBLE_DEVICES="${S11L_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
cd /src || exit 1

echo "[s11l-tests] sw=$SW devices=$HIP_VISIBLE_DEVICES out=$OUT"
# Hard rule 8: import torch (and triton) before vllm in this image.
python - <<'PY'
import torch  # noqa: F401
import triton  # noqa: F401
import vllm
from vllm import envs
print("[s11l-tests] vllm=%s sdot4_env=%s" % (vllm.__file__, envs.VLLM_DCU_USE_INT8_SDOT4))
PY

FILES="
tests/kernels/attention/test_triton_unified_attention_gfx906.py
tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py
tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py
tests/kernels/mamba/test_fused_sigmoid_gating.py
tests/kernels/mamba/test_fused_recurrent_packed_decode.py
tests/platforms/test_hygon_dtk.py
tests/kernels/test_gfx906_packed_silu_fp16.py
"
for t in $FILES; do
  name="$(basename "$t" .py)"
  log="$OUT/sw${SW}-${name}.log"
  echo "[s11l-tests] RUN $t -> $log"
  python -m pytest -q --noconftest -p no:cacheprovider "$t" >"$log" 2>&1
  rc=$?
  summary="$(grep -E '^[0-9]+ (passed|failed)|passed|failed|error' "$log" | tail -1)"
  echo "S11L_TEST sw=$SW file=$t rc=$rc summary='${summary}'"
done
echo "[s11l-tests] DONE sw=$SW"
