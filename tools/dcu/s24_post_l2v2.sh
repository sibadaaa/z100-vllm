#!/bin/bash
# S24 post-L2 v2: the down_proj per-card shape mscan (mechanism test for the
# cfg3 e2e inversion) + fixed L6 replays + warm 32k probe.
set -u
W=/home/lyh/z100-vllm
LOCK=$W/.s23/gpu.lock
OUT=$W/.s24/w1/L2
mkdir -p "$OUT" "$W/.s24/w1/L6"
exec 9>"$LOCK"
flock 9
echo "[s24 $(date +%H:%M:%S)] lock held; stopping service for down_proj mscan"
systemctl --user stop z100-vllm-host.service
for i in $(seq 1 60); do
  pid=$(systemctl --user show -p MainPID --value z100-vllm-host.service)
  [ "$pid" = "0" ] && break
  sleep 1
done
sleep 8
# per-card (TP=4) shapes the engine actually issues, incl. down_proj
# 1280:17408 whose nkb=136 > kQMaxNkb=64 bypasses the fused decode tile;
# M=8 rows answer whether forced cfg3 re-tiles the VERIFY GEMM.
echo "[s24 $(date +%H:%M:%S)] mscan per-card decode+down_proj shapes"
docker exec z100-regress bash -c "cd /src && bash .s24/probes/prefill_gemm_cfg_mscan_run.sh \
  8,512,8192 1280:17408,2048:5120" > "$OUT/mscan-percard-dcu3.log" 2>&1
echo "[s24 $(date +%H:%M:%S)] mscan rc=$?; starting default config"
systemd-run --user --unit=z100-vllm-host --collect \
  --property=WorkingDirectory="$W" \
  --property=StandardOutput=append:/mnt/game/host-serve/logs/serve-host.log \
  --property=StandardError=append:/mnt/game/host-serve/logs/serve-host.log \
  --property=TimeoutStopSec=30 --setenv=HOST=0.0.0.0 \
  /bin/bash "$W/tools/dcu/serve_host_dflash2.sh" > /dev/null
for i in $(seq 1 200); do
  code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' \
    -H "Authorization: Bearer $(cat /mnt/game/host-serve/api-key)" \
    http://127.0.0.1:8000/v1/models)
  [ "$code" = "200" ] && break
  sleep 3
done
echo "[s24 $(date +%H:%M:%S)] service ready; L6 replay grow"
"$W/.venv/bin/python" "$W/tools/dcu/s24_prefix_replay.py" --mode grow \
  --turns 8 --rounds 2 --doc-tokens 6000 --tag l6_grow --out "$W/.s24/w1/L6" \
  > "$W/.s24/w1/L6/replay-grow.log" 2>&1
grep -E "SUMMARY" "$W/.s24/w1/L6/replay-grow.log"
echo "[s24 $(date +%H:%M:%S)] L6 replay independent"
"$W/.venv/bin/python" "$W/tools/dcu/s24_prefix_replay.py" --mode independent \
  --turns 8 --rounds 2 --doc-tokens 6000 --tag l6_indep --out "$W/.s24/w1/L6" \
  > "$W/.s24/w1/L6/replay-indep.log" 2>&1
grep -E "SUMMARY" "$W/.s24/w1/L6/replay-indep.log"
echo "[s24 $(date +%H:%M:%S)] 32k probe x2 (WARM, default config)"
"$W/.venv/bin/python" "$W/tools/dcu/s24_longctx.py" --lens 32768 --reps 2 \
  --seed0 888001 --tag warm_default --out "$OUT" > "$OUT/warm-probe.log" 2>&1
grep -E "SUMMARY" "$OUT/warm-probe.log"
echo "[s24 $(date +%H:%M:%S)] done; default config left running"
