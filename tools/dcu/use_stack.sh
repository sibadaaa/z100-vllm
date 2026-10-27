#!/usr/bin/env bash
# Switch / inspect the four compiled vLLM extensions per stack.
#
# There are two independent trees, on purpose:
#   * /home/lyh/z100-vllm                 -> the DTK stack (this tree is /src in
#                                            z100-regress; its vllm/*.so are
#                                            symlinks into the DTK build dir)
#   * /mnt/game/port/gfx906-vllm-rocm101  -> the ROCm 10.1 + torch 2.16 stack
#                                            (PYTHONPATH of z100-vllm-rocm101-rt)
#
# `dtk` only touches the first tree, `rocm101` only touches the second, so
# switching one never breaks the other.  The .so files are gitignored build
# artefacts; switching never needs a git operation.
#
# Usage:
#   tools/dcu/use_stack.sh status     # identity of both trees (default)
#   tools/dcu/use_stack.sh dtk        # relink repo vllm/*.so -> DTK artefacts
#   tools/dcu/use_stack.sh rocm101    # relink copy vllm/*.so -> ROCm 10.1 artefacts
set -euo pipefail

DTK_TREE=${DTK_TREE:-/home/lyh/z100-vllm}
ROCM101_TREE=${ROCM101_TREE:-/mnt/game/port/gfx906-vllm-rocm101}
DTK_SO_DIR=${DTK_SO_DIR:-/mnt/game/port/build-sync/temp}
# The validated ROCm 10.1 set is the *installed* copy (cmake --install rewrites
# the runtime path deterministically; the build-tree artefacts have a different
# rpath and a different md5).  Keep an immutable snapshot of the installed
# set, which is what every L0-L4 result was produced with:
#   md5 9302ece8/_C  ffc176af/_C_stable_libtorch  7b4185c6/_moe  a250420b/_rocm_C
ROCM101_SO_DIR=${ROCM101_SO_DIR:-/mnt/game/port/gfx906-vllm-rocm101/installed-so}
MODULES=(_C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C)

_identity() {
  local tree=$1 label=$2 head
  head=$(git -C "$tree" rev-parse --short HEAD 2>/dev/null || echo "(no git)")
  echo "== $label  tree=$tree  HEAD=$head"
  local m
  for m in "${MODULES[@]}"; do
    local so="$tree/vllm/$m.abi3.so"
    if [ -e "$so" ]; then
      printf '  %-24s %s  -> %s\n' "$m" "$(md5sum "$so" | cut -d' ' -f1)" \
        "$(readlink -f "$so")"
    else
      printf '  %-24s MISSING\n' "$m"
    fi
  done
}

_relink() {
  local tree=$1 dir=$2 label=$3 m
  for m in "${MODULES[@]}"; do
    local src="$dir/$m.abi3.so"
    if [ ! -f "$src" ]; then
      echo "ERROR: $label artefact missing: $src" >&2
      exit 1
    fi
    ln -sfn "$src" "$tree/vllm/$m.abi3.so"
  done
  echo "relinked $tree/vllm/*.so -> $dir ($label)"
  _identity "$tree" "$label"
}

case "${1:-status}" in
  status)
    _identity "$DTK_TREE" "DTK"
    echo
    _identity "$ROCM101_TREE" "ROCm 10.1 + torch 2.16"
    ;;
  dtk) _relink "$DTK_TREE" "$DTK_SO_DIR" "DTK" ;;
  rocm101) _relink "$ROCM101_TREE" "$ROCM101_SO_DIR" "ROCm 10.1 + torch 2.16" ;;
  *)
    echo "usage: $0 [status|dtk|rocm101]" >&2
    exit 2
    ;;
esac
