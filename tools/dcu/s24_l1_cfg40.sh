#!/usr/bin/env bash
# S24 R4 follow-up: cfg40 (BM16/BN32 @ BK64) build + gates.
# The combined R3/R4 window ran against the 2026-10-19 19:00 build-tree
# snapshot (cfg 36-39 only); this window syncs the repo csrc into the
# root-owned build tree, rebuilds, and re-runs the L1 gates so cfg 40 is
# actually measured, then restores the baseline .so and production.
set -u
W=/home/lyh/z100-vllm
LOCK=$W/.s23/gpu.lock
OUT=$W/.s24/w2/L1
UNIT=z100-vllm-host.service
LOG=/mnt/game/host-serve/logs/serve-host.log
KEY=/mnt/game/host-serve/api-key
BASE=d209af9585aaf0049394ced4a3c3211b

exec 9>"$LOCK"
flock 9
echo "[s24 $(date +%H:%M:%S)] lock held; stopping service"
systemctl --user stop "$UNIT"
for i in $(seq 1 60); do
  [ "$(systemctl --user show -p MainPID --value "$UNIT")" = "0" ] && break
  sleep 1
done
sleep 8

echo "[s24 $(date +%H:%M:%S)] syncing repo csrc into the build tree"
docker exec z100-regress bash -c "cp /src/csrc/rocm/fp8_emu_gemm.cu \
  /mnt/game/port/build-sync/temp/csrc/rocm/fp8_emu_gemm.cu" || { echo SYNC_FAILED; exit 1; }

echo "[s24 $(date +%H:%M:%S)] ninja build _rocm_C (cfg 36-40)"
docker exec z100-regress bash -c "cd /src && ninja -C /mnt/game/port/build-sync/temp _rocm_C" \
  > "$OUT/build-cfg40.log" 2>&1
rc=$?
tail -3 "$OUT/build-cfg40.log"
[ $rc -ne 0 ] && { echo BUILD_FAILED; systemctl --user start "$UNIT"; exit 1; }

NEW=$(docker exec z100-regress bash -lc "md5sum /mnt/game/port/build-sync/temp/_rocm_C.abi3.so" | cut -d' ' -f1)
echo "new_so_md5=$NEW" | tee "$OUT/so-cfg40.md5"
docker exec z100-regress bash -lc "cp -p /mnt/game/port/build-sync/temp/_rocm_C.abi3.so /mnt/game/perf1/_rocm_C.$NEW.so" \
  || { echo BACKUP_FAILED; exit 1; }

echo "[s24 $(date +%H:%M:%S)] ISA gate"
docker exec z100-regress bash -c "cd /src && .venv/bin/python .s23/probes/isa_gate.py /mnt/game/perf1/_rocm_C.$NEW.so" \
  > "$OUT/isa-gate-cfg40.log" 2>&1
echo "isa_rc=$?"; tail -3 "$OUT/isa-gate-cfg40.log"

echo "[s24 $(date +%H:%M:%S)] numeric + perf gate (cfg40 live, service down)"
docker exec z100-regress bash -c "cd /src && bash .s24/probes/l1_wide2_run.sh" \
  > "$OUT/l1-check-cfg40.log" 2>&1
grep -E "l1-numeric. SUMMARY|\"verdict\"" "$OUT/l1-check-cfg40.log" | tail -3
grep -E "\"tag\": \"l1-perf\"" "$OUT/l1-check-cfg40.log" | head -12

echo "[s24 $(date +%H:%M:%S)] pytest gate"
docker exec z100-regress bash -c "cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
  bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
  export HIP_VISIBLE_DEVICES=3; export VLLM_DCU_USE_FP8_EMU=1; \
  .venv/bin/python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q'" \
  > "$OUT/pytest-cfg40.log" 2>&1
tail -3 "$OUT/pytest-cfg40.log"

echo "[s24 $(date +%H:%M:%S)] restoring baseline .so + production"
docker exec z100-regress bash -lc "cp -p /mnt/game/perf1/_rocm_C.$BASE.so /mnt/game/port/build-sync/temp/_rocm_C.abi3.so"
systemd-run --user --unit=$UNIT --collect \
  --property=WorkingDirectory="$W" \
  --property=StandardOutput=append:$LOG \
  --property=StandardError=append:$LOG \
  --property=TimeoutStopSec=30 \
  --setenv=HOST=0.0.0.0 \
  /bin/bash "$W/tools/dcu/serve_host_dflash2.sh"
ok=0
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' -H "Authorization: Bearer $(cat "$KEY")" http://127.0.0.1:8000/v1/models)
  [ "$code" = "200" ] && ok=$((ok + 1)) || ok=0
  [ "$ok" -ge 2 ] && break
  sleep 3
done
[ "${ok:-0}" -ge 2 ] && echo "production ready (baseline .so, promoted launcher)" || echo "SERVICE NOT READY"
echo "[s24 $(date +%H:%M:%S)] cfg40 window done"
