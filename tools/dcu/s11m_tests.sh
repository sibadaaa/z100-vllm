#!/usr/bin/env bash
# S11m: run the S9 A6 test matrix (7 gfx906 files, 146 passed) *plus* the
# S11l substitute file, under VLLM_DCU_USE_INT8_SDOT4=<SW>.
#
#   docker exec <ct> bash /src/tools/dcu/s11m_tests.sh <0|1> <outdir>
#
# The 7 files are the S9 §3.6 list (which S11l replaced one of); the extra
# tests/kernels/test_gfx906_packed_silu_fp16.py is the S11l substitute, so both
# "146" (S9 set) and "149" (union) can be reported from one run.
#
# Container-side script.  S11M_DEVICES picks the GPU (default 3: the 27B arms
# own 0,1).
set -u
SW="${1:?usage: s11m_tests.sh <0|1> <outdir>}"
OUT="${2:?usage: s11m_tests.sh <0|1> <outdir>}"
mkdir -p "$OUT"

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export VLLM_CACHE_ROOT=${S11M_CACHE:-/mnt/game/vllm-cache-11m/tests}
export VLLM_DCU_USE_INT8_SDOT4="$SW"
export HIP_VISIBLE_DEVICES="${S11M_DEVICES:-3}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
cd /src || exit 1

echo "[s11m-tests] sw=$SW devices=$HIP_VISIBLE_DEVICES out=$OUT host=$(hostname)"
# Hard rule 8: import torch (and triton) before vllm in this image.
python - <<'PY'
import torch  # noqa: F401
import triton  # noqa: F401
import vllm
from vllm import envs
print("[s11m-tests] vllm=%s sdot4_env=%s" % (vllm.__file__, envs.VLLM_DCU_USE_INT8_SDOT4))
PY

# S9 §3.6 A6 list (the "146" set)
S9_FILES="
tests/kernels/attention/test_triton_unified_attention_gfx906.py
tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py
tests/kernels/mamba/test_fused_recurrent_packed_decode.py
tests/kernels/mamba/test_fused_sigmoid_gating.py
tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py
tests/model_executor/test_qwen3_5_gdn.py
tests/platforms/test_hygon_dtk.py
"
# S11l used this one instead of test_qwen3_5_gdn.py
EXTRA_FILES="
tests/kernels/test_gfx906_packed_silu_fp16.py
"

total_s9=0
total_all=0
for t in $S9_FILES $EXTRA_FILES; do
  name="$(basename "$t" .py)"
  log="$OUT/sw${SW}-${name}.log"
  echo "[s11m-tests] RUN $t -> $log"
  python -m pytest -q --noconftest -p no:cacheprovider "$t" >"$log" 2>&1
  rc=$?
  summary="$(grep -oE '[0-9]+ (passed|failed|error)' "$log" | tr '\n' ' ')"
  np=$(grep -oE '[0-9]+ passed' "$log" | tail -1 | grep -oE '[0-9]+' || echo 0)
  echo "S11M_TEST sw=$SW file=$t rc=$rc passed=$np summary='${summary}'"
  total_all=$(( total_all + np ))
  case " ${S9_FILES//$'\n'/ } " in *" $t "*) total_s9=$(( total_s9 + np ));; esac
done
echo "S11M_TOTAL sw=$SW s9_7files=${total_s9} union_8files=${total_all}"
echo "[s11m-tests] DONE sw=$SW"
