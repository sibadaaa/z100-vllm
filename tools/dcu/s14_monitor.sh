#!/usr/bin/env bash
# S14: host-side resource monitor for one engine run.
# Samples host memory + the container cgroup every $INTERVAL seconds so the
# "no cgroup limit -> no oom_kill" claim has a time series, not just endpoints.
#
#   tools/dcu/s14_monitor.sh <LABEL> <DURATION_S> [INTERVAL_S]
# -> /mnt/game/s14/hostlogs/<LABEL>-monitor.log
set -u
LABEL="$1"; DUR="${2:-7200}"; IV="${3:-15}"
CT="${S14_CT:-z100-s14}"
OUT="/mnt/game/s14/hostlogs/${LABEL}-monitor.log"
mkdir -p "$(dirname "$OUT")"
{
  echo "# label=$LABEL ct=$CT start=$(date -Is) interval=${IV}s duration=${DUR}s"
  echo "# cols: time free_total free_used free_free free_avail swap_used cgroup_current cgroup_max oom oom_kill nVLLM maxvramMiB"
} > "$OUT"
t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt "$DUR" ]; do
  read -r tot used free avail < <(free -m | awk '/^Mem:/{print $2, $3, $4, $7}')
  swapu=$(free -m | awk '/^Swap:/{print $3}')
  cg=$(docker exec "$CT" bash -c 'cat /sys/fs/cgroup/memory.current; cat /sys/fs/cgroup/memory.max; awk "/^oom /{print \$2}" /sys/fs/cgroup/memory.events; awk "/^oom_kill /{print \$2}" /sys/fs/cgroup/memory.events' 2>/dev/null | tr '\n' ' ')
  nv=$(pgrep -c VLLM 2>/dev/null || echo 0)
  vr=$(hy-smi --showmeminfo vram 2>/dev/null | awk -F'[: ]+' '/Used Memory/{s+=$NF} END{print s+0}')
  echo "$(date -Is) $tot $used $free $avail ${swapu:-0} ${cg:-NA NA NA NA} $nv $vr" >> "$OUT"
  sleep "$IV"
done
echo "# end=$(date -Is)" >> "$OUT"
