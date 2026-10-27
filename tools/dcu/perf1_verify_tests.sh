#!/usr/bin/env bash
# task-5: run the repo numeric tests for the gfx906 FP8-emulation path.
#
#   docker exec z100-vllm bash /src/tools/dcu/perf1_verify_tests.sh \
#       /mnt/game/perf1/verify_pytest_<change>.log [extra pytest args]
#
# The tests allocate DCU memory, so the :8000 service must be DOWN first
# (perf1_arm.sh stop) and restarted afterwards -- the session driver does that
# around this script.  Records the .so md5 and any concurrent service so the
# result is tied to an exact build.
set -u
OUT="${1:?usage: perf1_verify_tests.sh <log> [pytest args...]}"
shift || true
cd /src || exit 1

# The DTK environment is required: without it `import torch` dies with
# "librocm_smi64.so.2: cannot open shared object file" (pytest exit code 4 on
# conftest import), which looks like a test failure but is an env failure.
. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export VLLM_DCU_GCN_ARCH="${VLLM_DCU_GCN_ARCH:-gfx906}"

{
  echo "=== perf1 verify tests $(date -Is) ==="
  echo "so_md5=$(md5sum /src/vllm/_rocm_C.abi3.so | cut -d' ' -f1) $(readlink -f /src/vllm/_rocm_C.abi3.so)"
  echo "service_procs=$(pgrep -c -f 'vllm.entrypoints.cli.main' 2>/dev/null || echo 0)"
  echo "--- pytest tests/kernels/quantization/test_gfx906_fp8_emu.py $* ---"
} > "$OUT"

python3 -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q "$@" >> "$OUT" 2>&1
rc=$?
{
  echo "--- pytest rc=$rc ---"
  echo "service_procs_after=$(pgrep -c -f 'vllm.entrypoints.cli.main' 2>/dev/null || echo 0)"
} >> "$OUT"
echo "[verify-tests] rc=$rc log=$OUT"
exit $rc
