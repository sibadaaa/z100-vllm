#!/usr/bin/env bash
# Start the S34 MTP k=2 production arm on port 8000.
# The s33 no-spec arm (z100-vllm-s33) is stopped automatically via the unit's
# mutual Conflicts= relation.
#
#   bash .s34/start.sh             # start + wait for /health + verify config
#   bash .s34/start.sh --no-wait   # just kick the unit and return
set -u
cd /home/lyh/z100-vllm
SUDO=""
[ "$(id -u)" != "0" ] && SUDO="sudo -n"

$SUDO systemctl start z100-vllm-mtp2.service || exit 1
if [ "${1:-}" = "--no-wait" ]; then
  echo "[start] unit kicked (--no-wait); check journalctl -u z100-vllm-mtp2 -f"
  exit 0
fi

echo "[start] waiting for /health (MTP arm cold start ~6-8 min)..."
# Verify the :8000 listener is THIS unit's main process: right after `start`,
# a still-draining previous server can answer 200 (false positive).
pid=$(systemctl show z100-vllm-mtp2 -p MainPID --value)
c=""
lp=""
i=0
for i in $(seq 1 180); do
  sleep 5
  c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8000/health || true)
  lp=$(ss -tlnpH "sport = :${PORT:-8000}" 2>/dev/null | grep -oP "pid=\K[0-9]+" | head -1)
  [ "$c" = "200" ] && [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$lp" = "$pid" ] && break
done
if [ "$c" != "200" ] || [ "$lp" != "$pid" ]; then
  echo "[start] ERROR not healthy (http=$c listener=$lp expected=$pid) after ~$((i*5))s"
  echo "[start] last journal lines:"
  $SUDO journalctl -u z100-vllm-mtp2 --since "-15 min" --no-pager | tail -25
  exit 1
fi
echo "[start] health 200 after ~$((i*5))s (listener pid $lp = unit main pid)"
echo "[start] main pid: $(systemctl show z100-vllm-mtp2 -p MainPID --value)"
echo "[start] --- spec line (expect num_speculative_tokens: 2) ---"
$SUDO journalctl -u z100-vllm-mtp2 --since "-15 min" --no-pager \
  | grep -m1 "spec=" || echo "NO SPEC LINE"
echo "[start] --- policy line (expect FP e9f6b1ff41e682d6) ---"
$SUDO journalctl -u z100-vllm-mtp2 --since "-15 min" --no-pager \
  | grep -m1 "policy: FP" || echo "NO POLICY LINE"
echo "[start] done"
