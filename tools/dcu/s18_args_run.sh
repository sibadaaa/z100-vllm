#!/usr/bin/env bash
# S18: run one arm with the args-instrumentation injected, fire a single short
# request, then collect and clean up.
#
#   docker exec -d z100-s18 bash /src/tools/dcu/s18_args_run.sh <SLOT> <TP> <K> <MAXLEN> <LABEL>
#
# Installs the site-packages .pth, starts the engine through s18_mtp_arm.sh,
# waits for /health, sends ONE greedy request (the zh_sky prompt, 6 tokens),
# kills the engine and removes the .pth again.  The args log lands in
# /mnt/game/s18/s18_args-<LABEL>.log.
set -u
SLOT=${1:?usage: s18_args_run.sh <SLOT> <TP> <K> <MAXLEN> <LABEL>}
TP=${2:?}; K=${3:?}; MAXLEN=${4:?}; LABEL=${5:?}
PORT=${S18_PORT:-8001}
TOP=/mnt/game/s18
PTH=/usr/local/lib/python3.10/dist-packages/zz_s18_args.pth
ARGS_LOG="$TOP/s18_args-${LABEL}.log"

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH="/src:/mnt/game/triton-gfx906/install"
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache

kill_engine() {
  pkill -9 -f 'entrypoints.cli.main' 2>/dev/null
  pkill -9 -f 'VLLM::' 2>/dev/null
  pkill -9 VLLM 2>/dev/null
  sleep 4
}

echo "=== s18 args-run label=$LABEL slot=$SLOT tp=$TP k=$K maxlen=$MAXLEN"
kill_engine
cp -f /src/tools/dcu/s18_args_patch.py "$TOP/s18_args_patch.py"
rm -f "$ARGS_LOG"
echo "import sys; sys.path.insert(0, \"$TOP\"); import s18_args_patch" > "$PTH"
echo "  [pth] installed"

S18_ARGS_LOG="$ARGS_LOG" S18_ARGS_N="${S18_ARGS_N:-12}" \
  S18_LAYER_LOG="${S18_LAYER_LOG:-}" S18_LAYER_N="${S18_LAYER_N:-256}" \
  S18_STATE_LOG="${S18_STATE_LOG:-}" S18_STATE_N="${S18_STATE_N:-400}" \
  nohup bash /src/tools/dcu/s18_mtp_arm.sh "$SLOT" "$TP" "$PORT" "$K" \
  >"$TOP/argsarm-${LABEL}.out" 2>&1 &
T0=$(date +%s)
READY=0
for _ in $(seq 1 40); do pgrep -f 'entrypoints.cli.main' >/dev/null && break; sleep 2; done
while [ $(( $(date +%s) - T0 )) -lt 900 ]; do
  if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then READY=1; break; fi
  pgrep -f 'entrypoints.cli.main' >/dev/null || { echo "  [wait] engine exited early"; break; }
  sleep 5
done
echo "  [wait] ready=$READY startup=$(( $(date +%s) - T0 ))s"
echo "  [pth-check] $(grep -c 's18-args' "$ARGS_LOG" 2>/dev/null || echo 0) install lines"

if [ "$READY" = "1" ]; then
  python - <<'PY' 2>&1 | tail -5
import json, urllib.request
import os
prompt = os.environ.get("S18_REQ_PROMPT", "用一句话介绍杭州。")
ntok = int(os.environ.get("S18_REQ_TOKENS", "6"))
body = json.dumps({"model": "qwen3.8-27b",
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": ntok, "temperature": 0.0}).encode()
req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.loads(r.read().decode())
print("[req] ok, text:", repr(out["choices"][0]["message"]["content"][:160]))
PY
  echo "  [req] rc=$?"
fi
kill_engine
rm -f "$PTH"
echo "  [pth] removed ($( [ -f "$PTH" ] && echo STILL_THERE || echo gone ))"
echo "=== collected args (first 20 lines):"
head -20 "$ARGS_LOG" 2>/dev/null || echo "  (no args log)"
echo "S18_ARGS_RUN_DONE label=$LABEL ready=$READY"
