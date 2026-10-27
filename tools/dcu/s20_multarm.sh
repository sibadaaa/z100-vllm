#!/usr/bin/env bash
# S20: multi-arm c=1 comparison inside ONE window (no other GPU/compiler work).
#
#   bash tools/dcu/s20_multarm.sh <so-md5> [TAG=ENV1,ENV2 ...] <so-md5> [TAG=...] ...
#
# Arms are specified as "<so-md5-or-path>[#TAG][@K=val[,K=val...]]"; the last
# argument may be `--probe-args ...` style extras passed to s20_ab_probe.py via
# S20_PROBE_ARGS.  Every arm restarts the production recipe, prints the
# installed .so md5 before and after, and runs tools/dcu/s20_ab_probe.py with a
# long window so the comparison has the statistical power to resolve <5%.
#
# Example (baseline / free-config / exact-M tiles / baseline again):
#   bash tools/dcu/s20_multarm.sh \
#     8e98bd60b94ebeb6f69ed8e39d754647#B \
#     03ac92eabf0d1ad6ac635e6a1529a9c0#T1@VLLM_DCU_FP8_GEMV_MAX_M=8,VLLM_DCU_FP8_DECODE_CFG=0 \
#     03ac92eabf0d1ad6ac635e6a1529a9c0#T2@VLLM_DCU_FP8_GEMV_MAX_M=8,VLLM_DCU_FP8_DECODE_CFG=16 \
#     8e98bd60b94ebeb6f69ed8e39d754647#B2
set -u
C=z100-vllm
W=/home/lyh/z100-vllm
BENCH=$W/.bench/ab20
LOG=$W/.s20/logs/multarm_$(date +%Y%m%d-%H%M%S).log
mkdir -p "$BENCH" "$(dirname "$LOG")"
PROBE_ARGS="${S20_PROBE_ARGS:---requests 18 --max-tokens 128 --prompt-tokens 512 --parallel 1}"
log() { echo "[multarm $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

resolve() { # $1 = md5 or path
  if [ -f "$1" ]; then echo "$1"; else echo "/mnt/game/perf1/_rocm_C.$1.so"; fi
}

vram_sum() { hy-smi 2>/dev/null | awk '/^[0-9]/ {gsub("%","",$6); s+=$6} END {print s+0}'; }

install_so() { # $1 = md5 or path -> echoes installed md5
  local SRC; SRC=$(resolve "$1")
  docker exec $C bash -lc "test -f '$SRC'" || { log "FAIL: $SRC missing"; return 1; }
  docker exec $C bash -lc "
    set -e
    TGT=/mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so
    CUR=\$(md5sum \$TGT | cut -d' ' -f1)
    [ -f /mnt/game/perf1/_rocm_C.\$CUR.so ] || cp \$TGT /mnt/game/perf1/_rocm_C.\$CUR.so
    cp '$SRC' \$TGT
    md5sum \$TGT | cut -d' ' -f1"
}

for SPEC in "$@"; do
  SO=${SPEC%%#*}
  REST=${SPEC#*#}
  if [ "$REST" = "$SPEC" ]; then TAG=arm; else
    TAG=${REST%%@*}; ENVS=${REST#*@}
  fi
  [ "$ENVS" = "$REST" ] && ENVS=""

  log "=========== ARM $TAG (so=$SO envs='$ENVS') ==========="
  docker exec $C bash -lc 'pkill -9 -f "vllm.entrypoints.cli.main"; sleep 1; pkill -9 -f s19_dflash' 2>/dev/null
  for i in $(seq 1 40); do [ "$(vram_sum)" -lt 20 ] && break; sleep 3; done

  MD5=$(install_so "$SO") || exit 1
  log "$TAG installed md5=$MD5"

  {
    echo '#!/usr/bin/env bash'; echo 'set -u'
    [ -n "$ENVS" ] && printf '%s\n' "$ENVS" | tr ',' '\n' | sed '/^$/d; s/^/export /'
    echo 'exec bash /src/tools/dcu/s19_dflash_e2e_fp8.sh'
  } | docker exec -i $C bash -c "cat > /tmp/arm_multi_${TAG}.sh"
  docker exec -d $C bash /tmp/arm_multi_${TAG}.sh
  bash /home/lyh/vllm-test/scripts/arm_ctl.sh wait >>"$LOG" 2>&1 || { log "$TAG FAILED ready"; continue; }
  sleep 5

  log "$TAG md5 before probe: $(docker exec $C bash -lc 'md5sum /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so | cut -d" " -f1')"
  if [ "${S20_SWEEP:-0}" = "1" ]; then
    python3 "$W/tools/dcu/s20_conc_sweep.py" --tag "$TAG" --concurrency "${S20_CONC:-1,4,16}" \
      --prompt-tokens "${S20_PT:-512}" --max-tokens "${S20_MT:-128}" --out "$BENCH" 2>&1 | tee -a "$LOG" | grep -E "^\[c=|saved"
  else
    python3 "$W/tools/dcu/s20_ab_probe.py" --tag "$TAG" --out "$BENCH" $PROBE_ARGS 2>&1 | tee -a "$LOG" | grep -E "ABRESULT|saved"
  fi
  log "$TAG md5 after probe:  $(docker exec $C bash -lc 'md5sum /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so | cut -d" " -f1')"
  python3 /home/lyh/vllm-test/scripts/parity_probe.py 2>&1 | tail -2 | tee -a "$LOG"
done

log "all arms done; log=$LOG"
log "restoring production (unchanged recipe, current disk .so)"
docker exec $C bash -lc 'pkill -9 -f "vllm.entrypoints.cli.main"; sleep 1; pkill -9 -f s19_dflash' 2>/dev/null
for i in $(seq 1 40); do [ "$(vram_sum)" -lt 20 ] && break; sleep 3; done
docker exec -d $C bash /src/tools/dcu/s19_dflash_e2e_fp8.sh
bash /home/lyh/vllm-test/scripts/arm_ctl.sh wait >>"$LOG" 2>&1 || log "WARN production restart timeout"
log "done"
