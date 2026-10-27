#!/usr/bin/env bash
# S14c G1: high-resolution host-memory x time tracer for ONE engine run.
#
# Why this exists (S14 only sampled `free -m` + cgroup at 15 s and took a single
# smaps snapshot after the fact): G1 needs *phase-resolved* attribution, i.e.
# "how many host bytes does each phase of the startup hold, and which process /
# which mapping does the peak belong to".
#
# Emits a line-oriented trace (awk-friendly):
#   S <iso> <epoch> mem_total_kb mem_free_kb mem_avail_kb cached_kb shmem_kb \
#       swap_total_kb swap_free_kb cg_current cg_max oom_kill n_worker n_engine \
#       vram_used_MiB load1
#   P <epoch> <role> <pid> rssanon_mib rssfile_mib vmswap_mib heap_mib \
#       unnamed_mib named_anon_mib stack_mib other_anon_mib anon_total_mib vmsize_mib
#   M <epoch> <iso> <marker-tail>          # phase marker seen in the engine log
#
# role: W0..W3 = vLLM TP workers (from comm VLLM::Worker_TP<N>), E = APIServer/
#       EngineCore python process.
# heap/unnamed/... come from a full /proc/<pid>/smaps aggregate
# (tools/dcu/s14c_smaps_agg.awk); their sum equals RssAnon.  `heap` is glibc's
# main arena (brk); every *secondary* glibc arena is an unnamed mmap and lands in
# `unnamed` -- which is exactly the column MALLOC_ARENA_MAX moves.
#
#   tools/dcu/s14c_memtrace.sh <LABEL> <DURATION_S> [INTERVAL_S] [ENGINE_LOG]
# S14C_MEMTRACE_APPEND=1 -> append instead of truncating (used when the tracer is
# restarted mid-arm without restarting the engine; a "# resumed" line marks it).
# -> /mnt/game/s14c/hostlogs/<LABEL>-memtrace.log
set -u
LABEL="${1:?usage: s14c_memtrace.sh <LABEL> <DURATION_S> [INTERVAL_S] [ENGINE_LOG]}"
DUR="${2:-7200}"
IV="${3:-5}"
ELOG="${4:-/mnt/game/s14c/hostlogs/${LABEL}.log}"
CT="${S14C_CT:-${S14_CT:-z100-s14}}"
OUTDIR="${S14C_LOGDIR:-/mnt/game/s14c/hostlogs}"
OUT="$OUTDIR/${LABEL}-memtrace.log"
AGG="$(dirname "$0")/s14c_smaps_agg.awk"
mkdir -p "$OUTDIR"
if [ "${S14C_MEMTRACE_APPEND:-0}" = "1" ]; then
  echo "# resumed=$(date -Is)" >> "$OUT"
else
  : > "$OUT"
fi

{
  echo "# label=$LABEL ct=$CT start=$(date -Is) interval=${IV}s duration=${DUR}s engine_log=$ELOG"
  echo "# cols_S: time epoch mem_total_kb mem_free_kb mem_avail_kb cached_kb shmem_kb swap_total_kb swap_free_kb cg_current cg_max oom_kill n_worker n_engine vram_used_MiB load1"
  echo "# cols_P: time epoch role pid rssanon_mib rssfile_mib vmswap_mib heap_mib unnamed_mib named_anon_mib stack_mib other_anon_mib anon_total_mib vmsize_mib"
  echo "# cols_M: time epoch marker"
  echo "# swapon_state: $(cat /proc/swaps | tr '\n' '|')"
} >> "$OUT"

marker_re='Model loading took|Loading weights took|torch.compile took|Graph capturing finished|Available KV cache memory|GPU KV cache size|init engine \(profile|Application startup complete|Capturing CUDA graphs|EngineCore|Out of memory|oom-kill|RuntimeError|Traceback|maximum concurrency|Using .* KV cache layout|Selected .*LinearKernel|warmup'
seen="${S14C_MEMTRACE_SEEN:-0}"   # S14C_MEMTRACE_SEEN: seed when resuming mid-arm

# one sample of one process -> "P <epoch> <role> <pid> ..." line
emit_p() {
  local ep="$1" role="$2" p="$3"
  local ra rf vs vm
  read -r ra rf vs vm < <(awk '/^RssAnon:/{a=$2}/^RssFile:/{f=$2}/^VmSwap:/{s=$2}/^VmSize:/{v=$2}END{print a+0,f+0,s+0,v+0}' "/proc/$p/status" 2>/dev/null)
  local agg
  agg=$(sudo -n awk -f "$AGG" "/proc/$p/smaps" 2>/dev/null)
  [ -z "$agg" ] && agg="0 0 0 0 0 0 0"
  echo "P $ep $role $p $((ra/1024)) $((rf/1024)) $((vs/1024)) $agg $((vm/1024))" >> "$OUT"
}

t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt "$DUR" ]; do
  iso=$(date -Is); ep=$(date +%s)
  # ---- host-wide ----
  read -r mtotal mfree mavail mcached mshmem < <(awk '/^MemTotal:/{t=$2}/^MemFree:/{f=$2}/^MemAvailable:/{a=$2}/^Cached:/{c=$2}/^Shmem:/{s=$2}END{print t,f,a,c,s}' /proc/meminfo)
  read -r stotal sfree < <(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{print t,f}' /proc/meminfo)
  # ---- cgroup of the engine container ----
  cg=$(docker exec "$CT" bash -c 'cat /sys/fs/cgroup/memory.current; cat /sys/fs/cgroup/memory.max; awk "/^oom_kill /{print \$2}" /sys/fs/cgroup/memory.events' 2>/dev/null | tr '\n' ' ')
  # ---- device memory ----
  vr=$(hy-smi --showmeminfo vram 2>/dev/null | awk -F'[: ]+' '/Used Memory/{s+=$NF} END{print s+0}')
  load1=$(awk '{print $1}' /proc/loadavg)

  nw=0; ne=0
  # ---- per-process: TP workers ----
  for p in $(pgrep VLLM 2>/dev/null); do
    st=$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)
    [ "$st" = "Z" ] && continue
    comm=$(cat "/proc/$p/comm" 2>/dev/null)
    role="W?"
    case "$comm" in
      VLLM::Worker_TP*) role="W${comm#VLLM::Worker_TP}";;
      VLLM*) role="W$comm";;
    esac
    emit_p "$ep" "$role" "$p"
    nw=$((nw+1))
  done
  # ---- per-process: APIServer / EngineCore ----
  for p in $(pgrep -f "vllm.entrypoints.cli.main" 2>/dev/null); do
    st=$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)
    [ "$st" = "Z" ] && continue
    comm=$(cat "/proc/$p/comm" 2>/dev/null)
    case "$comm" in VLLM*) continue;; esac
    emit_p "$ep" "E" "$p"
    ne=$((ne+1))
  done
  echo "S $iso $ep $mtotal $mfree $mavail $mcached $mshmem $stotal $sfree ${cg:-NA NA NA} $nw $ne ${vr:-0} $load1" >> "$OUT"

  # ---- phase markers from the engine log (new lines only) ----
  if [ -f "$ELOG" ]; then
    cur=$(wc -l < "$ELOG" 2>/dev/null || echo 0)
    if [ "$cur" -gt "$seen" ]; then
      tail -n +$((seen+1)) "$ELOG" 2>/dev/null | grep -aE "$marker_re" | while IFS= read -r ln; do
        echo "M $ep $(date -Is) ${ln:0:220}" >> "$OUT"
      done
      seen=$cur
    fi
  fi

  sleep "$IV"
done
echo "# end=$(date -Is)" >> "$OUT"
