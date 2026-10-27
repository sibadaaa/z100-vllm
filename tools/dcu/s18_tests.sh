#!/usr/bin/env bash
# S18: run the MTP-related test set on gfx906 (GPU-free-ish: kernel tests need a
# visible device, so pass -e HIP_VISIBLE_DEVICES=0 via S18_T_DEVICES).
#
#   docker exec z100-s18 bash /src/tools/dcu/s18_tests.sh <outdir> [extra pytest args]
#
# Prints one machine-readable summary line per file, same convention as
# tools/dcu/s11l_tests.sh (which is the S9/S11 gate runner).
set -u
OUT="${1:?usage: s18_tests.sh <outdir> [pytest args...]}"
shift || true
mkdir -p "$OUT"

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-s18
export HIP_VISIBLE_DEVICES="${S18_T_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
cd /src || exit 1

echo "[s18-tests] devices=$HIP_VISIBLE_DEVICES out=$OUT extra=$*"
python - <<'PY'
import torch  # noqa: F401
import triton  # noqa: F401
import vllm
print("[s18-tests] vllm=%s device=%s" % (vllm.__file__, torch.cuda.get_device_name(0)))
PY

FILES="
tests/kernels/mamba/test_causal_conv1d.py
tests/kernels/mamba/test_fused_sigmoid_gating.py
tests/kernels/mamba/test_gdn_fused_mtp.py
tests/model_executor/test_qwen3_5_gdn.py
tests/model_executor/test_qwen3_5_mtp_gguf.py
tests/model_executor/model_loader/test_mtp_validation.py
tests/models/test_qwen3_5_mtp_config.py
tests/v1/spec_decode/test_mtp.py
"
for t in $FILES; do
  name="$(basename "$t" .py)"
  log="$OUT/${name}.log"
  echo "[s18-tests] RUN $t -> $log"
  python -m pytest -q --noconftest -p no:cacheprovider "$t" "$@" >"$log" 2>&1
  rc=$?
  summary="$(grep -E 'passed|failed|error|no tests ran' "$log" | tail -1)"
  echo "S18_TEST file=$t rc=$rc summary='${summary}'"
done
echo "[s18-tests] DONE"
