#!/usr/bin/env bash
# S14c G1 (mapping level): top anonymous VMAs by Rss for every live vLLM process.
#
# The phase table says *how much* of the anonymous footprint is `[heap]` (glibc
# main arena) vs unnamed mmap.  This says *what shape* the unnamed part has,
# which is what decides whether MALLOC_ARENA_MAX can touch it:
#   * glibc secondary arenas are big `rw-p` anonymous mmaps in ~64 MiB chunks
#     (HEAP_MAX_SIZE) that grow;
#   * CPython obmalloc arenas are 1 MiB (or 256 KiB) anonymous mmaps;
#   * a few huge multi-GB anonymous mappings would mean a single big allocator.
#
#   tools/dcu/s14c_anon_top.sh <LABEL> [N]
# -> /mnt/game/s14c/hostlogs/anon-top-<LABEL>.txt
set -u
LABEL="${1:?usage: s14c_anon_top.sh <LABEL> [N]}"
N="${2:-10}"
OUTDIR="${S14C_LOGDIR:-/mnt/game/s14c/hostlogs}"
OUT="$OUTDIR/anon-top-$LABEL.txt"
mkdir -p "$OUTDIR"
{
  echo "# label=$LABEL date=$(date -Is)"
  for p in $(pgrep VLLM 2>/dev/null); do
    st=$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)
    [ "$st" = "Z" ] && continue
    echo "=== worker pid=$p comm=$(cat /proc/$p/comm 2>/dev/null) ==="
    grep -E '^(RssAnon|RssFile|VmSize):' "/proc/$p/status"
    echo "-- top $N anonymous VMAs by Rss (addr range, size MiB, rss MiB, first flag) --"
    sudo -n awk -v n="$N" '
      /^[0-9a-fA-F]+-[0-9a-fA-F]+ /{
        if (have) { print rss" "size" "addr" "flags }
        addr=$1; flags=$2; size=0; rss=0; have=1
        if (NF>=6) have=0                 # named mapping -> not anonymous
        next
      }
      /^Size:/{ size=$2 }
      /^Rss:/{ rss=$2 }
      /^VmFlags:/{ if (have && rss>0) print rss/1024" "size/1024" "addr" "flags }
      END{}
    ' "/proc/$p/smaps" | sort -rn | head -"$N"
    echo "-- bucketed Rss by VMA size class (anonymous only, MiB) --"
    sudo -n awk '
      /^[0-9a-fA-F]+-[0-9a-fA-F]+ /{ addr=$1; size=0; rss=0; have=(NF<6)?1:0; next }
      /^Size:/{ size=$2 }
      /^Rss:/{ rss=$2 }
      /^VmFlags:/{
        if (have && rss>0) {
          mib=size/1024; cls = (mib>=1024?"ge1G":(mib>=64?"64M-1G":(mib>=1?"1M-64M":"lt1M")));
          s[cls]+=rss/1024; c[cls]++
        }
      }
      END{ for (k in s) printf "   %-9s %8.0f MiB  (%d VMAs)\n", k, s[k], c[k] }
    ' "/proc/$p/smaps" | sort
  done
  echo "=== /proc/meminfo ==="; grep -E "^(MemTotal|MemFree|MemAvailable|Cached|Shmem):" /proc/meminfo
  echo "=== free -m ==="; free -m
} 2>&1 | tee "$OUT"
