#!/usr/bin/env bash
# S24 R3: L3.1 verify/decode split-KV attention scan, stopped-engine window.
set -u
W=/home/lyh/z100-vllm
LOCK=$W/.s23/gpu.lock
UNIT=z100-vllm-host.service
LOG=/mnt/game/host-serve/logs/serve-host.log
KEY=/mnt/game/host-serve/api-key

exec 9>"$LOCK"
flock 9
echo "[s24 $(date +%H:%M:%S)] lock held; stopping service"
systemctl --user stop "$UNIT"
for i in $(seq 1 60); do
  [ "$(systemctl --user show -p MainPID --value "$UNIT")" = "0" ] && break
  sleep 1
done
sleep 8

echo "[s24 $(date +%H:%M:%S)] L3.1 verify scan (DCU3, service down)"
docker exec z100-regress bash -c "cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
  bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
  export HIP_VISIBLE_DEVICES=3; \
  .venv/bin/python .s24/probes/l31_verify_scan.py \
      --out /src/.s24/w2/L31/verify-scan.json'" \
  > "$W/.s24/w2/L31/verify-scan.log" 2>&1
echo "l31_rc=$?"; tail -3 "$W/.s24/w2/L31/verify-scan.log"

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
[ "${ok:-0}" -ge 2 ] && echo "production ready" || echo "SERVICE NOT READY"
echo "[s24 $(date +%H:%M:%S)] l31 window done"
