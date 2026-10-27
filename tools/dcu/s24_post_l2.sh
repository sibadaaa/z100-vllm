#!/bin/bash
# S24 post-L2-window sequence (one GPU-lock session):
#   1. stop service, mscan the FFN shapes the plan's 5-shape list missed
#      (34816:5120 gate_up, 5120:17408 down, + o/qkv-side shapes)
#   2. start the default config (fresh server), probe 32k x2  -> FRESH state
#   3. L6 prefix replay (traffic), probe 32k x2 again         -> WARM state
#   4. leave the default config running
set -u
W=/home/lyh/z100-vllm
LOCK=$W/.s23/gpu.lock
OUT=$W/.s24/w1/L2
mkdir -p "$OUT"
exec 9>"$LOCK"
echo "[s24 $(date +%H:%M:%S)] waiting for GPU lock"
flock 9
echo "[s24 $(date +%H:%M:%S)] lock held; stopping service for FFN mscan"
systemctl --user stop z100-vllm-host.service
for i in $(seq 1 60); do
  pid=$(systemctl --user show -p MainPID --value z100-vllm-host.service)
  [ "$pid" = "0" ] && break
  sleep 1
done
sleep 8
echo "[s24 $(date +%H:%M:%S)] mscan FFN shapes (service down)"
docker exec z100-regress bash -c "cd /src && bash .s24/probes/prefill_gemm_cfg_mscan_run.sh \
  512,1024,2048,4096,8192 34816:5120,5120:17408,5120:4096,5120:5120" \
  > "$OUT/mscan-ffn-dcu3.log" 2>&1
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
echo "[s24 $(date +%H:%M:%S)] service ready (fresh); 32k probe x2 (FRESH)"
"$W/.venv/bin/python" "$W/tools/dcu/s24_longctx.py" --lens 32768 --reps 2 \
  --seed0 777001 --tag fresh_default --out "$OUT" > "$OUT/fresh-probe.log" 2>&1
grep -E "SUMMARY" "$OUT/fresh-probe.log"
echo "[s24 $(date +%H:%M:%S)] L6 prefix replay: grow mode"
"$W/.venv/bin/python" "$W/tools/dcu/s24_prefix_replay.py" --mode grow \
  --turns 8 --rounds 2 --doc-tokens 6000 --tag l6_grow --out "$W/.s24/w1/L6" \
  > "$W/.s24/w1/L6/replay-grow.log" 2>&1
grep -E "SUMMARY" "$W/.s24/w1/L6/replay-grow.log"
echo "[s24 $(date +%H:%M:%S)] L6 prefix replay: independent mode"
"$W/.venv/bin/python" "$W/tools/dcu/s24_prefix_replay.py" --mode independent \
  --turns 8 --rounds 2 --doc-tokens 6000 --tag l6_indep --out "$W/.s24/w1/L6" \
  > "$W/.s24/w1/L6/replay-indep.log" 2>&1
grep -E "SUMMARY" "$W/.s24/w1/L6/replay-indep.log"
echo "[s24 $(date +%H:%M:%S)] 32k probe x2 (WARM)"
"$W/.venv/bin/python" "$W/tools/dcu/s24_longctx.py" --lens 32768 --reps 2 \
  --seed0 888001 --tag warm_default --out "$OUT" > "$OUT/warm-probe.log" 2>&1
grep -E "SUMMARY" "$OUT/warm-probe.log"
echo "[s24 $(date +%H:%M:%S)] done; default config left running"
