#!/usr/bin/env bash
# S24 R3/R4 combined stopped-engine window:
#   1. s24_l1_build.sh: stop service -> rebuild _rocm_C (cfg 36-40) -> ISA
#      gate -> l1_wide2 numeric+perf gate -> pytest -> restore baseline .so
#   2. L3.1 verify/decode split-KV attention scan (.s24/probes/l31_verify_scan.py)
#   3. L7 int8 sdot4 research probe (.s24/probes/l7_sdot4_probe.py)
#   4. restart the production launcher (promoted defaults: MBT=4096 + DYNK)
#      and smoke
set -u
W=/home/lyh/z100-vllm
LOCK=$W/.s23/gpu.lock
OUT=$W/.s24/w2
UNIT=z100-vllm-host.service
LOG=/mnt/game/host-serve/logs/serve-host.log
KEY=/mnt/game/host-serve/api-key
mkdir -p "$OUT/L31" "$OUT/L7"

exec 9>"$LOCK"
flock 9
echo "[s24 $(date +%H:%M:%S)] lock held; L1 build window begins"

bash "$W/tools/dcu/s24_l1_build.sh"
rc=$?
echo "[s24 $(date +%H:%M:%S)] l1_build rc=$rc (continuing: .so is baseline)"

echo "[s24 $(date +%H:%M:%S)] L3.1 verify scan (DCU3, service down)"
docker exec z100-regress bash -c "cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
  bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
  export HIP_VISIBLE_DEVICES=3; \
  .venv/bin/python .s24/probes/l31_verify_scan.py \
      --out /src/.s24/w2/L31/verify-scan.json'" \
  > "$OUT/L31/verify-scan.log" 2>&1
echo "l31_rc=$?"; tail -4 "$OUT/L31/verify-scan.log"

echo "[s24 $(date +%H:%M:%S)] L7 int8 sdot4 probe (DCU3)"
docker exec z100-regress bash -c "cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
  bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
  export HIP_VISIBLE_DEVICES=3; \
  .venv/bin/python .s24/probes/l7_sdot4_probe.py \
      --out /src/.s24/w2/L7/sdot4-probe.json'" \
  > "$OUT/L7/sdot4-probe.log" 2>&1
echo "l7_rc=$?"; tail -4 "$OUT/L7/sdot4-probe.log"

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
