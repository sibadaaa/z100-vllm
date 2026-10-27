#!/usr/bin/env bash
# S14g: address-layout fingerprint of the live engine (runs INSIDE the container).
#
#   docker exec <CT> bash /src/tools/dcu/s14g_maps.sh
#
# For every process that has the DTK HIP runtime mapped (this DTK ships it as
# libgalaxyhip.so.*; libamdhip64 is accepted too) it prints the personality flag
# (bit 0x40000 == ADDR_NO_RANDOMIZE), the profile bases of the libraries that
# matter and a short md5 over the (start-address, object) pairs.  Two runs whose
# fingerprints are equal had the same address layout; this is the ASLR
# instrument for H1.
set -u
PAT='libgalaxyhip|libamdhip64|libtorch_hip'
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  maps="/proc/$pid/maps"
  [ -r "$maps" ] || continue
  grep -qaE "$PAT" "$maps" || continue
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-90)
  pers=$(cat "/proc/$pid/personality" 2>/dev/null)
  n=$(wc -l < "$maps")
  peak=$(awk '/VmPeak/{print $2}' "/proc/$pid/status" 2>/dev/null)
  fp=$(awk '/libgalaxyhip|libamdhip64|libtorch_python|libtorch_cpu|libtorch_hip|libpython3|\[heap\]|\[stack\]|libc\.so/ {print $1, $NF}' "$maps" | md5sum | cut -c1-16)
  echo "== pid=$pid personality=$pers n_maps=$n vmpeak_kB=${peak:-NA} fp=$fp"
  echo "   exe=$(readlink /proc/$pid/exe 2>/dev/null) cmd=$cmd"
  head -2 "$maps" | sed 's/^/   first: /'
  awk '/libgalaxyhip|libamdhip64|libtorch_python|libtorch_cpu|libtorch_hip|libpython3|\[heap\]/ && !seen[$0]++ {print "   map: " $1, $NF}' "$maps" | head -8
done
