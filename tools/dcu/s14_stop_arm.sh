#!/usr/bin/env bash
# S14: stop one arm's engine cleanly and verify all device memory is released.
# Hard constraint 4: the vLLM workers overwrite their `comm` with
# `VLLM::Worker_TP` and blank their /proc/<pid>/cmdline, so `pkill -f python3`
# cannot see them -- `pkill VLLM` is the only reliable sweep.  Only our own
# engines are ever live on this host during S14 (the production container
# qwen38-vllm is Exited(137) and is never touched).
#
#   tools/dcu/s14_stop_arm.sh <NAME>
set -u
CT="${S14_CT:-z100-s14}"
NAME="${1:?usage: s14_stop_arm.sh <NAME>}"
LOGDIR="${S14_LOGDIR:-/mnt/game/s14/hostlogs}"

live_vllm() {
  local p st
  for p in $(pgrep VLLM 2>/dev/null); do
    st=$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)
    [ "$st" = "Z" ] && continue
    echo "pid=$p state=$st rss_kb=$(awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null)"
  done
}

echo "[stop14] $(date -Is) name=$NAME ct=$CT"
echo "[stop14] live VLLM before:"; live_vllm
if docker exec "$CT" test -f "/tmp/s14_${NAME}.pid" 2>/dev/null; then
  docker exec "$CT" bash -c "kill \$(cat /tmp/s14_${NAME}.pid) 2>/dev/null" || true
  sleep 20
  docker exec "$CT" bash -c "kill -9 \$(cat /tmp/s14_${NAME}.pid) 2>/dev/null" || true
fi
if [ -f "$LOGDIR/$NAME.wrapper.pid" ]; then
  wp=$(cat "$LOGDIR/$NAME.wrapper.pid")
  kill "$wp" 2>/dev/null || true
  sleep 3
  kill -9 "$wp" 2>/dev/null || true
fi
pkill VLLM 2>/dev/null || true
for i in 1 2 3 4 5 6; do
  n=$(live_vllm | wc -l)
  [ "$n" = "0" ] && break
  sleep 10
  pkill VLLM 2>/dev/null || true
done
echo "[stop14] live VLLM after ($(live_vllm | wc -l) live, rest are historical zombies):"
live_vllm
echo "[stop14] resident zombies (harmless, no memory): $(pgrep -c VLLM 2>/dev/null || echo 0)"
echo "[stop14] vram after:"; hy-smi --showmeminfo vram | grep -E "Used Memory"
echo "[stop14] free after: $(free -m | tr '\n' ' ')"
echo "[stop14] cgroup memory.events:"
docker exec "$CT" bash -c 'cat /sys/fs/cgroup/memory.events; echo "current: $(cat /sys/fs/cgroup/memory.current)"'
echo "[stop14] done $(date -Is)"
