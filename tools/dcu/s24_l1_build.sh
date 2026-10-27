#!/bin/bash
# S24 L1 build + gates (one GPU-lock window, service stopped):
#   1. ninja build _rocm_C with the cfg 36-39 prefill wide-BK tiles
#   2. ISA gate (v_dot2 present, no mfma/cmac) on the new .so
#   3. numeric gate: cfg 36-39 bitwise == cfg 0 on all per-card shapes
#   4. perf: cfg 36-39 vs 0/3/5/10 at M=4096/8192 on all per-card shapes
#   5. pytest gate (fp8 emu suites)
# Leaves the BASELINE .so promoted and the service DOWN afterwards
# (the caller then runs the e2e window or restores).
set -u
W=/home/lyh/z100-vllm
LOCK=$W/.s23/gpu.lock
OUT=$W/.s24/w2/L1
BASE=d209af9585aaf0049394ced4a3c3211b
mkdir -p "$OUT"
exec 9>"$LOCK"
flock 9
echo "[s24 $(date +%H:%M:%S)] lock held; stopping service"
systemctl --user stop z100-vllm-host.service
for i in $(seq 1 60); do
  pid=$(systemctl --user show -p MainPID --value z100-vllm-host.service)
  [ "$pid" = "0" ] && break
  sleep 1
done
sleep 8

echo "[s24 $(date +%H:%M:%S)] ninja build _rocm_C"
docker exec z100-regress bash -c 'cd /src && ninja -C /mnt/game/port/build-sync/temp _rocm_C' \
  > "$OUT/build.log" 2>&1
rc=$?
echo "[s24 $(date +%H:%M:%S)] build rc=$rc"
tail -3 "$OUT/build.log"
[ $rc -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

NEW=$(docker exec z100-regress bash -lc "md5sum /mnt/game/port/build-sync/temp/_rocm_C.abi3.so" | cut -d' ' -f1)
echo "new_so_md5=$NEW" | tee "$OUT/so.md5"
docker exec z100-regress bash -lc "cp -p /mnt/game/port/build-sync/temp/_rocm_C.abi3.so /mnt/game/perf1/_rocm_C.$NEW.so" \
  || { echo "BACKUP FAILED"; exit 1; }
echo "$NEW" > "$OUT/so-new.txt"

echo "[s24 $(date +%H:%M:%S)] ISA gate"
docker exec z100-regress bash -c "cd /src && .venv/bin/python .s23/probes/isa_gate.py /mnt/game/perf1/_rocm_C.$NEW.so" \
  > "$OUT/isa-gate.log" 2>&1
rc=$?
tail -5 "$OUT/isa-gate.log"
echo "isa_rc=$rc"

echo "[s24 $(date +%H:%M:%S)] numeric + perf gate (service down, DCU3)"
docker exec z100-regress bash -c "cd /src && bash .s24/probes/l1_wide2_run.sh" \
  > "$OUT/l1-check.log" 2>&1
rc=$?
grep -E "l1-numeric. SUMMARY|\"verdict\": \"FAIL\"" "$OUT/l1-check.log" | tail -3
grep -E "\"tag\": \"l1-perf\"" "$OUT/l1-check.log" | head -20
echo "check_rc=$rc"

echo "[s24 $(date +%H:%M:%S)] pytest gate"
docker exec z100-regress bash -c "cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
  bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
  export HIP_VISIBLE_DEVICES=3; export VLLM_DCU_USE_FP8_EMU=1; \
  .venv/bin/python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q'" \
  > "$OUT/pytest.log" 2>&1
rc=$?
tail -3 "$OUT/pytest.log"
echo "pytest_rc=$rc"

# restore baseline .so so the next window starts from a known state
docker exec z100-regress bash -lc "cp -p /mnt/game/perf1/_rocm_C.$BASE.so /mnt/game/port/build-sync/temp/_rocm_C.abi3.so"
echo "[s24 $(date +%H:%M:%S)] baseline .so restored; service left DOWN"
