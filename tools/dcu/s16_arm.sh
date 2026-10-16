#!/usr/bin/env bash
# S16: launch one Z100 27B TP=4 serving arm for the RCCL fine-grain-PCIe A/B.
#
#   tools/dcu/s16_arm.sh <tag> <hsa_value> <port> [maxlen]
#
# Only HSA_FORCE_FINE_GRAIN_PCIE differs between arms; everything else comes
# from tools/dcu/serve_z100_tp4.sh (TP=4, graph PIECEWISE + 4 buckets, INT8
# sdot4, maxlen 8192 by default to keep the host 31 GB RAM safe).  Logs and
# RCCL channel dumps land in /mnt/game/s16/.
set -uo pipefail
TAG=$1; HSA=$2; PORT=$3; MAXLEN=${4:-8192}; EAGER=${5:-0}
CT=${S16_CT:-z100-s14}
OUT=/mnt/game/s16
mkdir -p "$OUT"

docker exec -i "$CT" bash -c "cat > /tmp/_s16_${TAG}.sh" <<EOF
#!/usr/bin/env bash
export HSA_FORCE_FINE_GRAIN_PCIE=$HSA
export MAXLEN=$MAXLEN PORT=$PORT EAGER=$EAGER UTIL=0.85 MBT=2048 TP=4
export CACHE=/mnt/game/serve/vllm-cache
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT
export NCCL_DEBUG_FILE=$OUT/rccl-$TAG.%h.%p.log
echo "[\$TAG] HSA_FORCE_FINE_GRAIN_PCIE=\$HSA_FORCE_FINE_GRAIN_PCIE port=$PORT maxlen=$MAXLEN"
bash /src/tools/dcu/serve_z100_tp4.sh
EOF

docker exec -d "$CT" bash -c "bash /tmp/_s16_${TAG}.sh > $OUT/serve-$TAG.log 2>&1"
echo "[s16] arm=$TAG hsa=$HSA port=$PORT maxlen=$MAXLEN -> $OUT/serve-$TAG.log"

t0=$(date +%s)
started=0
while true; do
  if grep -q "Application startup complete" "$OUT/serve-$TAG.log" 2>/dev/null; then
    echo "[s16] arm=$TAG READY after $(( $(date +%s) - t0 )) s"; break
  fi
  el=$(( $(date +%s) - t0 ))
  # Give `docker exec -d` time to fork before treating a missing PID as death.
  if [ "$el" -gt 60 ] && [ "$started" = "1" ] &&
     ! docker exec "$CT" pgrep -f "entrypoints.cli.main" >/dev/null 2>&1; then
    echo "[s16] arm=$TAG DIED after ${el} s (see log)"; exit 1
  fi
  if [ "$el" -gt 20 ]; then started=1; fi
  if [ "$el" -gt 3600 ]; then
    echo "[s16] arm=$TAG TIMEOUT"; exit 1
  fi
  sleep 15
done
