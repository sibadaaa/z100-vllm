#!/usr/bin/env bash
# S11m: move host-side evidence directories into /mnt/game (container-side view).
#
#   docker exec z100-s9 bash /src/tools/dcu/s11m_tidy_evidence.sh <src> <dest> [...]
#
# Copies <src> (a directory under /src = the workspace) into <dest> under
# /mnt/game, then verifies every file's md5 matches.  Deletion of the workspace
# copy is done by the caller only after MD5_ALL_MATCH is printed.
set -u
if [ $# -lt 2 ]; then echo "usage: $0 <src> <dest> [more pairs...]" >&2; exit 2; fi
rc=0
while [ $# -ge 2 ]; do
  src="$1"; dst="$2"; shift 2
  echo "[tidy] $src -> $dst"
  mkdir -p "$dst"
  cp -a "$src"/. "$dst"/
  n_src=$(find "$src" -type f | wc -l)
  n_dst=$(find "$dst" -type f | wc -l)
  ( cd "$src" && find . -type f -exec md5sum {} \; | sort -k2 ) > /tmp/tidy-a.md5
  ( cd "$dst" && find . -type f -exec md5sum {} \; | sort -k2 ) > /tmp/tidy-b.md5
  if diff -q /tmp/tidy-a.md5 /tmp/tidy-b.md5 >/dev/null; then
    echo "[tidy] MD5_ALL_MATCH $src files=$n_src -> $dst files=$n_dst"
  else
    echo "[tidy] MD5_MISMATCH $src (src=$n_src dst=$n_dst)"; diff /tmp/tidy-a.md5 /tmp/tidy-b.md5 | head
    rc=1
  fi
done
echo "[tidy] done rc=$rc"
exit $rc
