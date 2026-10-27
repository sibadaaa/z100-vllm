#!/usr/bin/env bash
# S14c: block until one arm's orchestrator finishes, then print the G1 report +
# the G2 numbers (peak/decode) for that arm in one shot.
#
#   tools/dcu/s14c_wait.sh <NAME> [TIMEOUT_S]
#
# Prints: decode TPOT summary, ready time, oom_kill delta, min MemAvailable,
# max swap used, the phase table, and the global-peak composition.
set -u
NAME="${1:?usage: s14c_wait.sh <NAME> [TIMEOUT_S]}"
TIMEOUT="${2:-3600}"
LOGDIR="${S14C_LOGDIR:-/mnt/game/s14c/hostlogs}"
ORCH="$LOGDIR/orch-$NAME.log"
MT="$LOGDIR/$NAME-memtrace.log"
t0=$(date +%s)
while true; do
  if [ -f "$ORCH" ] && grep -q "\[orch14c\] done name=$NAME" "$ORCH"; then break; fi
  if [ $(( $(date +%s) - t0 )) -gt "$TIMEOUT" ]; then echo "!! TIMEOUT waiting for $NAME"; break; fi
  sleep 15
done
echo "############ $NAME : orchestrator tail ############"
tail -14 "$ORCH" | sed 's/^/  /'

echo "############ $NAME : G2 numbers ############"
for f in "$LOGDIR/$NAME-bench-decode.log"; do
  [ -f "$f" ] || continue
  echo "-- $f"
  grep -aE "Successful requests|Benchmark duration|Output token throughput|Mean TPOT|Median TPOT|P90 TPOT|P99 TPOT|Median TTFT" "$f" | sed 's/^/  /'
done
echo "-- ready / oom_kill / AOT"
grep -aE "\[orch14c\] (ready=|summary:)" "$ORCH" | sed 's/^/  /'
echo "-- free / swaps at end"
grep -aE "\[orch14c\] (free_after_stop|swaps_after_stop)" "$ORCH" | sed 's/^/  /'

echo "############ $NAME : G1 extremes ############"
if [ -f "$MT" ]; then
  awk '
    /^S /{
      n++;
      avail=$6/1048576; sw=($9-$10)/1048576; cg=$11/1073741824;
      if (m==""||avail<m) {m=avail; me=$2}
      if (sw>swm) {swm=sw; swe=$2}
      if (cg>cgm) cgm=cg;
    }
    END{printf "  samples=%d\n  min MemAvailable = %.2f GiB @ %s\n  max swap_used   = %.2f GiB @ %s\n  max cgroup.current = %.2f GiB\n", n, m, me, swm, swe, cgm}
  ' "$MT"
  python3 "$(dirname "$0")/s14c_phase_report.py" "$MT" --curve-every 999 2>&1 | sed 's/^/  /'
else
  echo "  (no memtrace)"
fi
