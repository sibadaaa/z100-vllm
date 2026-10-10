#!/usr/bin/env bash
# S14 task item (3), time-series variant: sample the live workers' RssAnon/RssFile
# every IV seconds for N samples into one file, so the answer to "is the ~7 GiB
# per worker anonymous memory the weight shard (swap would evict it) or a
# transient compile/graph pool (swap only bridges start-up)?" is backed by a
# series covering idle -> first request -> steady decode, not one snapshot.
#
#   tools/dcu/s14_smaps_loop.sh <LABEL> <N> [IV_SECONDS]
# -> /mnt/game/s14/hostlogs/smaps-<LABEL>.txt   (appended)
set -u
LABEL="${1:?usage: s14_smaps_loop.sh <LABEL> <N> [IV]}"
N="${2:-20}"
IV="${3:-30}"
HERE="$(cd "$(dirname "$0")" && pwd)"
for _ in $(seq 1 "$N"); do
  S14_SMAPS_APPEND=1 bash "$HERE/s14_smaps.sh" "$LABEL" >/dev/null 2>&1
  sleep "$IV"
done
echo "[s14_smaps_loop] label=$LABEL samples=$N iv=${IV}s done $(date -Is)"
