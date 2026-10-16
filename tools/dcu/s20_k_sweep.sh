#!/usr/bin/env bash
# S20: speculation-depth sweep at c=1.  Reuses the s19 arm launcher with a
# different k (the production launcher hardcodes k=7), so the only variable is
# `num_speculative_tokens`.
#
#   bash tools/dcu/s20_k_sweep.sh <k> [tag] [extra env KEY=VAL ...]
#
# Why this lever: at c=1 the step cost is dominated by terms that are flat in
# the verify batch size M = k+1 (all-reduce, launch, most of the FFN GEMM at
# M <= 8 falls back to tiles whose cost is M-flat), while every extra accepted
# position is a whole token.  Measured per-position acceptance at k=7 falls off
# 0.973 (pos 0) -> 0.703 (pos 6), so positions 7..k are not free but they are
# not exhausted either.  The decision needs tok/step and step ms measured, not
# extrapolated: only the product matters.
set -u
K=${1:?usage: s20_k_sweep.sh <k> [tag] [KEY=VAL ...]}
TAG=${2:-k$K}
shift 2 2>/dev/null || shift $#
EXTRA="$*"
C=z100-vllm
W=/home/lyh/z100-vllm
BENCH=$W/.bench/ab20
log() { echo "[ksweep $(date +%H:%M:%S)] $*"; }
mkdir -p "$BENCH"

log "0) stop service"
docker exec $C bash -lc 'pkill -9 -f "vllm.entrypoints.cli.main"; sleep 1; pkill -9 -f s19_dflash' 2>/dev/null
for i in $(seq 1 40); do
  V=$(hy-smi 2>/dev/null | awk '/^[0-9]/ {gsub("%","",$6); s+=$6} END {print s+0}')
  [ "$V" -lt 20 ] && { log "VRAM released (${V}%)"; break; }
  sleep 3
done

log "1) render arm script (k=$K, EXTRA='$EXTRA')"
{
  echo '#!/usr/bin/env bash'
  echo 'set -u'
  echo 'export S19_MAXLEN=262144'
  echo 'export S19_LOGLEVEL=INFO'
  echo 'export S19_DRAFT_DTYPE=bfloat16'
  echo 'export S19_DRAFT_KV_DTYPE=bfloat16'
  echo "export S19_LOG='/mnt/game/dflash/serve-k${K}.log'"
  echo 'export VLLM_DCU_BF16=1'
  echo 'export VLLM_QWEN35_MTP_GDN_TRITON=1'
  echo 'export VLLM_GFX906_GDN_CHUNKED_PREFILL=1'
  [ -n "$EXTRA" ] && printf '%s\n' "$EXTRA" | tr ' ' '\n' | sed '/^$/d; s/^/export /'
  echo "exec bash /src/tools/dcu/s19_dflash_arm.sh fp8 4 8000 /mnt/game/models/Qwen3.8-27B-DFlash2-ms ${K} \\"
  echo "  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \\"
  echo "  --enable-prompt-tokens-details \\"
  echo "  --override-generation-config '{\"temperature\":0.7,\"repetition_penalty\":1.05}' \\"
  echo "  --default-chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \\"
  echo "  --host 0.0.0.0"
} | docker exec -i $C bash -c "cat > /tmp/arm_k${K}.sh"
docker exec $C bash -lc "head -12 /tmp/arm_k${K}.sh"

log "2) start (k=$K)"
docker exec -d $C bash /tmp/arm_k${K}.sh
bash /home/lyh/vllm-test/scripts/arm_ctl.sh wait || { log "FAILED ready"; exit 1; }
sleep 5

log "3) c=1 probe with accept curve (k=$K)"
python3 "$W/tools/dcu/s20_accept_profile.py" --prompt-tokens 256 --max-tokens 256 \
  --repeat 3 2>&1 | tail -22

log "4) standard c=1 probe (k=$K)"
python3 "$W/tools/dcu/s20_single_probe.py" --prompt-tokens 512 --max-tokens 128 \
  --repeat 4 --tag "k${K}" --out "$BENCH" 2>&1 | tail -4
log "done k=$K"
