#!/usr/bin/env bash
# S24 R3/R4 combined stopped-engine window, take 2.
#   1. stop service; sync repo csrc into the build tree; rebuild (cfg 36-40)
#   2. ISA gate + l1_wide2 numeric/perf gate (incl. cfg 40) + pytest
#   3. restore baseline .so
#   4. L3.1 verify/decode split-KV attention scan (service down, clean GPU)
#   5. L7 int8 sdot4 research probe (clean GPU)
#   6. restart production (promoted launcher: MBT=4096 + DYNK) + smoke
set -u
W=/home/lyh/z100-vllm
LOCK=$W/.s23/gpu.lock
OUT=$W/.s24/w2
UNIT=z100-vllm-host.service
LOG=/mnt/game/host-serve/logs/serve-host.log
KEY=/mnt/game/host-serve/api-key
BASE=d209af9585aaf0049394ced4a3c3211b
mkdir -p "$OUT/L31" "$OUT/L7"

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
  > "$OUT/L1/build-cfg40.log" 2>&1
rc=$?
tail -3 "$OUT/L1/build-cfg40.log"
[ $rc -ne 0 ] && { echo BUILD_FAILED; exit 1; }
NEW=$(docker exec z100-regress bash -lc "md5sum /mnt/game/port/build-sync/temp/_rocm_C.abi3.so" | cut -d' ' -f1)
echo "new_so_md5=$NEW" | tee "$OUT/L1/so-cfg40.md5"
docker exec z100-regress bash -lc "cp -p /mnt/game/port/build-sync/temp/_rocm_C.abi3.so /mnt/game/perf1/_rocm_C.$NEW.so" \
  || { echo BACKUP_FAILED; exit 1; }

echo "[s24 $(date +%H:%M:%S)] ISA gate"
docker exec z100-regress bash -c "cd /src && .venv/bin/python .s23/probes/isa_gate.py /mnt/game/perf1/_rocm_C.$NEW.so" \
  > "$OUT/L1/isa-gate-cfg40.log" 2>&1
echo "isa_rc=$?"; tail -3 "$OUT/L1/isa-gate-cfg40.log"

echo "[s24 $(date +%H:%M:%S)] l1 numeric+perf gate (cfg40 live)"
docker exec z100-regress bash -c "cd /src && bash .s24/probes/l1_wide2_run.sh" \
  > "$OUT/L1/l1-check-cfg40.log" 2>&1
grep -E "\"verdict\"" "$OUT/L1/l1-check-cfg40.log" | tail -2
grep -E "\"tag\": \"l1-perf\"" "$OUT/L1/l1-check-cfg40.log" | grep -oE '"cfg": 40[^}]*}|"best": [^}]*}' | head -14

echo "[s24 $(date +%H:%M:%S)] pytest gate"
docker exec z100-regress bash -c "cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
  bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
  export HIP_VISIBLE_DEVICES=3; export VLLM_DCU_USE_FP8_EMU=1; \
  .venv/bin/python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q'" \
  > "$OUT/L1/pytest-cfg40.log" 2>&1
tail -2 "$OUT/L1/pytest-cfg40.log"

echo "[s24 $(date +%H:%M:%S)] restoring baseline .so"
docker exec z100-regress bash -lc "cp -p /mnt/game/perf1/_rocm_C.$BASE.so /mnt/game/port/build-sync/temp/_rocm_C.abi3.so"

echo "[s24 $(date +%H:%M:%S)] L3.1 verify scan (DCU3, service down)"
docker exec z100-regress bash -c "cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
  bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
  export HIP_VISIBLE_DEVICES=3; \
  .venv/bin/python .s24/probes/l31_verify_scan.py \
      --out /src/.s24/w2/L31/verify-scan.json'" \
  > "$OUT/L31/verify-scan.log" 2>&1
echo "l31_rc=$?"; tail -3 "$OUT/L31/verify-scan.log"

echo "[s24 $(date +%H:%M:%S)] L7 int8 sdot4 probe (DCU3, service down)"
docker exec z100-regress bash -c "cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
  bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
  export HIP_VISIBLE_DEVICES=3; \
  .venv/bin/python .s24/probes/l7_sdot4_probe.py \
      --out /src/.s24/w2/L7/sdot4-probe.json'" \
  > "$OUT/L7/sdot4-probe.log" 2>&1
echo "l7_rc=$?"; tail -3 "$OUT/L7/sdot4-probe.log"

echo "[s24 $(date +%H:%M:%S)] restarting production launcher (promoted defaults)"
systemd-run --user --unit=$UNIT --collect \
  --property=WorkingDirectory="$W" \
  --property=StandardOutput=append:$LOG \
  --property=StandardError=append:$LOG \
  --property=TimeoutStopSec=30 \
  --setenv=HOST=0.0.0.0 \
  /bin/bash "$W/tools/dcu/serve_host_dflash2.sh" || { echo "START FAILED"; exit 1; }
ok=0
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' -H "Authorization: Bearer $(cat "$KEY")" http://127.0.0.1:8000/v1/models)
  [ "$code" = "200" ] && ok=$((ok + 1)) || ok=0
  [ "$ok" -ge 2 ] && break
  sleep 3
done
[ "${ok:-0}" -ge 2 ] || { echo "SERVICE NOT READY"; exit 1; }
"$W/.venv/bin/python" "$W/tools/dcu/s23_smoke.py" --out "$OUT/restart-smoke.json" \
  > "$OUT/restart-smoke.log" 2>&1 && echo "restart smoke OK" || echo "SMOKE FAILED"
echo "[s24 $(date +%H:%M:%S)] combined window done"
