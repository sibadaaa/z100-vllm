#!/usr/bin/env bash
# Single MBT arm (Lead request): does --max-num-batched-tokens 8192 help?
#
# The production wrapper pins `export S11F_MBT=2048`, so the arm cannot override
# it through the normal chain.  This script uses /mnt/game/perf1/recipe_mbt.sh,
# which is that same wrapper with exactly ONE line changed
# (`export S11F_MBT=2048` -> `export S11F_MBT="${S11F_MBT:-2048}"`); the diff is
# printed into the log for the record.  S11F_MTP_GEMM=1 pins the fp8 dispatch to
# pre-perf1 so the arm isolates the MBT change from the kernel/dispatch change.
set -u
cd /src || exit 1
echo "=== [mbt] start $(date -Is) so=$(md5sum /src/vllm/_rocm_C.abi3.so | cut -d' ' -f1) ==="
echo "=== [mbt] recipe diff vs production wrapper (must be the MBT line only) ==="
diff /mnt/game/port/serve_fp8_8000.sh /mnt/game/perf1/recipe_mbt.sh && echo "(no diff?!)" || true

PERF1_RECIPE=/mnt/game/perf1/recipe_mbt.sh PERF1_PREFLIGHT_WAIT=900 \
  bash tools/dcu/perf1_e2e_session.sh \
    --arm mbt8192 S11F_MBT=8192 S11F_MTP_GEMM=1 anchor

echo "=== [mbt] restoring the production wrapper on :8000 ==="
bash tools/dcu/perf1_arm.sh start control_prod
echo "=== [mbt] done $(date -Is) ==="
bash tools/dcu/perf1_arm.sh status
