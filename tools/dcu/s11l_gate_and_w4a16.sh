#!/usr/bin/env bash
# S11l: end-to-end regressions that do not need the 27B arm, run on card 3
# (the 27B arms own cards 0/1):
#
#   docker exec z100-s9 env VLLM_DIR=/src VLLM_DCU_USE_INT8_SDOT4=1 \
#       bash /src/tools/dcu/s11l_gate_and_w4a16.sh
#
#   1) s8b_gate.sh   (kill-list / FP8 / W4A16 selection / _C_stable identity)
#   2) W4A16 tinyllama, one greedy chat request
set -u
CARD="${S11L_CARD:-3}"
export S8B_HIP_VISIBLE_DEVICES="$CARD"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/mnt/game/vllm-cache-27b}"
export S8B_OUT_DIR="${S8B_OUT_DIR:-/mnt/game/s11l/s8b}"
export VLLM_DIR="${VLLM_DIR:-/src}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$S8B_OUT_DIR"
echo "[s11l-gw] card=$CARD VLLM_DIR=$VLLM_DIR sdot4=${VLLM_DCU_USE_INT8_SDOT4:-<default>}"

echo "===== 1) s8b_gate.sh ====="
bash "${VLLM_DIR}/tools/dcu/s8b_gate.sh"
GATE_RC=$?
echo "[s11l-gw] s8b_gate rc=$GATE_RC"

echo "===== 2) W4A16 tinyllama e2e ====="
. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT_W4A16:-/mnt/game/vllm-cache-w4a16}"
export HIP_VISIBLE_DEVICES="$CARD"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
PORT="${S11L_W4A16_PORT:-8004}"
cd /src || exit 1
python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  /mnt/game/models/tinyllama-w4a16-group128 \
  --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.3 \
  --served-model-name tinyllama-w4a16 --host 127.0.0.1 --port "$PORT" \
  > /tmp/s11l-w4a16.log 2>&1 &
SRV=$!
echo "[s11l-gw] w4a16 pid=$SRV port=$PORT"
for i in $(seq 1 90); do
  if grep -q "Application startup complete" /tmp/s11l-w4a16.log; then break; fi
  if grep -qE "Traceback|RuntimeError" /tmp/s11l-w4a16.log; then break; fi
  sleep 10
done
grep -nE "Selected |W4A16|gptq|Application startup|Traceback|Error" /tmp/s11l-w4a16.log | tail -20
python - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
payload = {
    "model": "tinyllama-w4a16",
    # Same request shape as the S9 archive (Section 3.11.4): /v1/completions,
    # greedy, 32 tokens, prompt "The capital of France is".
    "prompt": "The capital of France is",
    "max_tokens": 32, "temperature": 0.0,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.load(resp)
        print("[s11l-gw] W4A16 HTTP", resp.status,
              "usage=", json.dumps(body.get("usage", {})))
        print("[s11l-gw] W4A16 text=",
              json.dumps(body["choices"][0]["text"][:200], ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    print("[s11l-gw] W4A16 request FAILED:", repr(exc))
PY
kill "$SRV" 2>/dev/null || true
sleep 10
kill -9 "$SRV" 2>/dev/null || true
echo "[s11l-gw] done gate_rc=$GATE_RC"
