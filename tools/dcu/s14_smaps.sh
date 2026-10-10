#!/usr/bin/env bash
# S14 task item (3): host-side memory composition of the live TP workers.
#
# Answers the question that decides whether the swap-based graph-mode TP=4
# retry (P4g) is worth attempting at all:
#   * RssAnon ~= the per-rank weight shard  -> weights are anonymous and touched
#     every token -> swap would page weights out -> P4g is already refuted;
#   * RssAnon << the per-rank weight shard  -> the host anonymous footprint is a
#     transient compile/graph-pool peak -> swap only bridges start-up -> P4g ok.
#
# Container processes are visible in the host /proc, so this runs on the host and
# is read-only.  `/proc/<pid>/smaps_rollup` needs ptrace over a root-owned
# process; it is attempted with `sudo -n` and its absence is recorded rather
# than worked around.
#
#   tools/dcu/s14_smaps.sh <LABEL>
# -> /mnt/game/s14/hostlogs/smaps-<LABEL>.txt  (also echoed)
set -u
LABEL="${1:-snap}"
CT="${S14_CT:-z100-s14}"
OUTDIR="${S14_LOGDIR:-/mnt/game/s14/hostlogs}"
OUT="$OUTDIR/smaps-$LABEL.txt"
mkdir -p "$OUTDIR"
# S14_SMAPS_APPEND=1 -> append (time series); default truncates (single snapshot)
TEEFLAG=""
if [ "${S14_SMAPS_APPEND:-0}" = "1" ]; then TEEFLAG="-a"; else : > "$OUT"; fi
{
  echo "# label=$LABEL ct=$CT date=$(date -Is)"
  echo "# pgrep -a VLLM:"
  pgrep -a VLLM
  echo "# live (non-zombie) VLLM pids:"
  for p in $(pgrep VLLM 2>/dev/null); do
    st=$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)
    [ "$st" = "Z" ] && continue
    echo "=== pid=$p comm=$(cat /proc/$p/comm 2>/dev/null) state=$st pinfo=$(awk '{print $1}' /proc/$p/stat 2>/dev/null) ==="
    echo "[status]"
    grep -E "^(VmPeak|VmSize|VmRSS|VmHWM|RssAnon|RssFile|RssShmem|VmData|VmStk|VmExe|VmLib|VmPTE|VmSwap|Threads):" "/proc/$p/status" 2>/dev/null
    echo "[smaps_rollup]"
    if sudo -n cat "/proc/$p/smaps_rollup" 2>/dev/null; then
      :
    else
      echo "  (unavailable: $(sudo -n true 2>&1 | head -1) -- smaps_rollup requires ptrace over a root-owned process)"
      echo "  (RssAnon/RssFile from /proc/status above carry the same split)"
    fi
    echo "[loaded vllm extension .so]  # which _C is actually used (the S7c fix lives in _C_stable_libtorch)"
    sudo -n grep -oE "/[^ ]*\.abi3\.so" "/proc/$p/maps" 2>/dev/null | sort -u
    echo "[md5 of loaded extensions]"
    for so in $(sudo -n grep -oE "/[^ ]*\.abi3\.so" "/proc/$p/maps" 2>/dev/null | sort -u); do
      md5sum "$so" 2>/dev/null
    done
    echo "[env of interest]  # NOTE: vLLM workers putenv after exec, so /proc/<pid>/environ"
    echo "                   #       can be stale/incomplete for workers; the APIServer pid is authoritative"
    sudo -n cat "/proc/$p/environ" 2>/dev/null | tr '\0' '\n' | grep -E "^(VLLM_DCU_USE_INT8_SDOT4|HIP_VISIBLE_DEVICES|VLLM_CACHE_ROOT|VLLM_USE_V1)="
  done
  echo "# free -m:"; free -m
  echo "# /proc/meminfo (head):"; grep -E "^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree|Cached|Buffers):" /proc/meminfo
  echo "# /proc/swaps:"; cat /proc/swaps
  echo "# hy-smi --showmeminfo vram:"; hy-smi --showmeminfo vram
} 2>&1 | tee $TEEFLAG "$OUT"
