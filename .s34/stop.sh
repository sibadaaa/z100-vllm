#!/usr/bin/env bash
# Stop the S34 MTP k=2 arm (z100-vllm-mtp2).  The unit's KillMode=control-group
# tears down the TP workers with it; orphan workers are cleaned defensively.
#
#   bash .s34/stop.sh
#
# NOTE: this leaves port 8000 free but serves NOTHING.  To go back to the
# no-spec s33 arm instead, use .s34/rollback.sh.
set -u
SUDO=""
[ "$(id -u)" != "0" ] && SUDO="sudo -n"

$SUDO systemctl stop z100-vllm-mtp2.service || true
for i in $(seq 1 30); do
  pgrep -f "entrypoints[.]cli" >/dev/null || break
  sleep 2
done
pkill -9 -f 'VLLM[:]:Worker' 2>/dev/null || true
sleep 3
if pgrep -f "entrypoints[.]cli" >/dev/null; then
  echo "[stop] WARNING: a vllm server process is still running:"
  pgrep -af "entrypoints[.]cli"
else
  echo "[stop] stopped; no vllm server running on the host"
fi
