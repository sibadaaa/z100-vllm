#!/usr/bin/env bash
# One-step rollback to the s33 no-spec arm (z100-vllm-s33, FP e9f6b1ff41e682d6).
# The mutual Conflicts= relation stops z100-vllm-mtp2 automatically.
#
#   bash .s34/rollback.sh
set -u
cd /home/lyh/z100-vllm
SUDO=""
[ "$(id -u)" != "0" ] && SUDO="sudo -n"

$SUDO systemctl disable z100-vllm-mtp2.service >/dev/null 2>&1 || true
$SUDO systemctl enable z100-vllm-s33.service >/dev/null 2>&1 || true
$SUDO systemctl start z100-vllm-s33.service || exit 1

echo "[rollback] waiting for /health (s33 arm cold start ~4.5-6 min)..."
c=""
i=0
for i in $(seq 1 180); do
  sleep 5
  c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && break
done
if [ "$c" != "200" ]; then
  echo "[rollback] ERROR not healthy; journal tail:"
  $SUDO journalctl -u z100-vllm-s33 --since "-15 min" --no-pager | tail -25
  exit 1
fi
echo "[rollback] health 200; policy line:"
$SUDO journalctl -u z100-vllm-s33 --since "-10 min" --no-pager | grep -m1 "policy: FP"
echo "[rollback] done (back on the s33 no-spec arm; MTP arm disabled)"
