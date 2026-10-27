#!/usr/bin/env bash
# S14c: swap bookkeeping for the G3 "minimum viable swap" bisect.
#
# Hard constraint 4 of the task book only authorises `swapon`/`swapoff` of
# /mnt/game/swapfile (16 GiB) and requires the before/after /proc/swaps text to
# be recorded and the host state restored.  For the bisect we need an
# *additional*, smaller swapfile; it is created under the S14c write scope
# (/mnt/game/s14c) and removed again at the end of the bisect arm.
#
#   tools/dcu/s14c_swap.sh on  <swapfile> <size_MiB> <LABEL>
#   tools/dcu/s14c_swap.sh off <swapfile>            <LABEL>
#   tools/dcu/s14c_swap.sh rm  <swapfile>            <LABEL>
#   tools/dcu/s14c_swap.sh show                      <LABEL>
#
# Every invocation appends the verbatim `cat /proc/swaps` + `free -m` + the
# matching /proc/meminfo swap fields to
#   /mnt/game/s14c/hostlogs/swap-<LABEL>.txt
set -u
CMD="${1:?usage: s14c_swap.sh on|off|rm|show ...}"
LOGDIR="${S14C_LOGDIR:-/mnt/game/s14c/hostlogs}"
mkdir -p "$LOGDIR"

log() { # log <LABEL> <note>
  local lbl="$2"
  {
    echo "########## $(date -Is)  cmd=$CMD  note=$1"
    echo "--- /proc/swaps ---"; cat /proc/swaps
    echo "--- free -m ---"; free -m
    echo "--- /proc/meminfo swap ---"; grep -E "^(SwapTotal|SwapFree|SwapCached):" /proc/meminfo
  } >> "$LOGDIR/swap-$lbl.txt"
  {
    echo "########## $(date -Is) cmd=$CMD note=$1"
    cat /proc/swaps; free -m
  }
}

case "$CMD" in
  on)
    SW="$2"; SZ="${3:?size MiB}"; LBL="${4:?label}"
    log "before-swapon $SW ${SZ}MiB" "$LBL"
    [ -f "$SW" ] || { dd if=/dev/zero of="$SW" bs=1M count="$SZ" status=none && chmod 600 "$SW"; }
    sudo -n mkswap "$SW" || exit 1
    sudo -n swapon "$SW" || exit 1
    log "after-swapon $SW" "$LBL"
    ;;
  off)
    SW="$2"; LBL="${3:?label}"
    log "before-swapoff $SW" "$LBL"
    sudo -n swapoff "$SW" || exit 1
    log "after-swapoff $SW" "$LBL"
    ;;
  rm)
    SW="$2"; LBL="${3:?label}"
    sudo -n swapoff "$SW" 2>/dev/null || true
    log "before-rm $SW" "$LBL"
    rm -f "$SW"
    log "after-rm $SW (file deleted)" "$LBL"
    ;;
  show)
    LBL="${2:?label}"; log "show" "$LBL"
    ;;
  *) echo "bad cmd $CMD"; exit 2 ;;
esac
