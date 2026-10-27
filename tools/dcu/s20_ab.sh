#!/usr/bin/env bash
# S20 A/B driver: swap the _rocm_C .so + restart the production service, then
# measure c=1 with s20_single_probe.py.  Both arms run in the SAME window so
# host-side drift cancels; the ratio is the only claimable result.
#
#   bash tools/dcu/s20_ab.sh <tag> <so-md5-or-path> [extra env KEY=VAL ...]
#
# Examples:
#   bash tools/dcu/s20_ab.sh base 8e98bd60b94ebeb6f69ed8e39d754647
#   bash tools/dcu/s20_ab.sh exactm 4ce168f1778870e0b0b0eed665f04937
#
# The .so is looked up as /mnt/game/perf1/_rocm_C.<md5>.so and copied over
# /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so (which /src/vllm/_rocm_C.abi3.so
# symlinks to).  The previous .so is backed up first.
set -u
TAG=${1:?usage: s20_ab.sh <tag> <so-md5|path> [KEY=VAL ...]}
SO=${2:?need an .so md5 or absolute path}
shift 2 || true
EXTRA="$*"
C=z100-vllm
W=/home/lyh/z100-vllm
BENCH=$W/.bench/ab20
log() { echo "[ab20 $(date +%H:%M:%S)] $*"; }
mkdir -p "$BENCH"

# resolve the target .so
if [ -f "$SO" ]; then SRC=$SO; else
  SRC="/mnt/game/perf1/_rocm_C.${SO}.so"
fi
docker exec $C bash -lc "test -f '$SRC'" || { log "FAIL: $SRC not found"; exit 2; }
NEWMD5=$(docker exec $C bash -lc "md5sum '$SRC' | cut -d' ' -f1")

log "0) stop service"
docker exec $C bash -lc 'pkill -9 -f "vllm.entrypoints.cli.main"; sleep 1; pkill -9 -f s19_dflash' 2>/dev/null
for i in $(seq 1 40); do
  V=$(hy-smi 2>/dev/null | awk '/^[0-9]/ {gsub("%","",$6); s+=$6} END {print s+0}')
  [ "$V" -lt 20 ] && { log "VRAM released (${V}%)"; break; }
  sleep 3
done

log "1) back up current .so and install $NEWMD5"
docker exec $C bash -lc '
  set -e
  TGT=/mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so
  CUR=$(md5sum "$TGT" | cut -d" " -f1)
  [ -f "/mnt/game/perf1/_rocm_C.$CUR.so" ] || cp "$TGT" "/mnt/game/perf1/_rocm_C.$CUR.so"
  cp "'"$SRC"'" "$TGT"
  echo "installed: $(md5sum "$TGT" | cut -d" " -f1)"'

log "2) start production recipe (EXTRA='$EXTRA')"
# The knobs have to be exported *before* the launcher runs: s19_dflash_e2e_fp8.sh
# reads them as ${VAR:-default} and s19_dflash_arm.sh re-derives from the same
# names, so exporting inside a wrapper that then `exec bash launcher` works only
# if the wrapper is the parent shell (it is) -- but an inline `VAR=1 bash` would
# not survive the extra exec, hence the explicit export list.
if [ -n "$EXTRA" ]; then
  { echo '#!/usr/bin/env bash'
    echo 'set -u'
    printf '%s\n' "$EXTRA" | tr ' ' '\n' | sed '/^$/d; s/^/export /'
    echo 'exec bash /src/tools/dcu/s19_dflash_e2e_fp8.sh'
  } | docker exec -i $C bash -c 'cat > /tmp/arm20_'"$TAG"'.sh'
  docker exec $C bash -lc "cat /tmp/arm20_${TAG}.sh"
  docker exec -d $C bash /tmp/arm20_${TAG}.sh
else
  docker exec -d $C bash /src/tools/dcu/s19_dflash_e2e_fp8.sh
fi
bash /home/lyh/vllm-test/scripts/arm_ctl.sh wait || { log "FAILED ready"; exit 1; }
sleep 5

log "3) c=1 probe ($TAG)"
python3 "$W/tools/dcu/s20_single_probe.py" --prompt-tokens 512 --max-tokens 128 \
  --repeat 4 --tag "ab20_${TAG}" --out "$BENCH" 2>&1 | tail -6

log "4) parity check ($TAG)"
python3 /home/lyh/vllm-test/scripts/parity_probe.py 2>&1 | tail -4

log "5) md5 of installed .so: $(docker exec $C bash -lc 'md5sum /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so | cut -d" " -f1')"
log "done $TAG"
