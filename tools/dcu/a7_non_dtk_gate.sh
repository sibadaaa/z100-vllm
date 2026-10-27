#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# S11a / A7 evidence collector: run the non-DTK probe matrix against one source
# tree and dump one `.jsonl` per probe configuration under /mnt/game/port/a7.
# Must be executed *inside* the DTK container (it sources /opt/dtk/env.sh).
#
#   docker exec z100-a7      bash -lc 'source .../a7_non_dtk_gate.sh dtk /mnt/game/port/gfx906-vllm-211 canonical'
#   docker exec z100-a7-nodtk bash -lc '... a7_non_dtk_gate.sh sim /mnt/game/port/a7-head head'
#
# `dtk` keeps the DTK environment; `sim` hides every DTK marker the runtime
# predicate looks at (DTKROOT / HYHAL_PATH / /opt/dtk/.info/rocm_version, the
# latter hidden by a bind mount over the file) and points ROCM_PATH at the
# non-existent /opt/rocm, i.e. the "DTK is not installed" condition.
#
# A7_CARD selects the GPU (default 3) so the probes stay on one card.
set -u

MODE="${1:?usage: a7_non_dtk_gate.sh <dtk|sim> <tree> <tag>}"
TREE="${2:?usage: a7_non_dtk_gate.sh <dtk|sim> <tree> <tag>}"
TAG="${3:?usage: a7_non_dtk_gate.sh <dtk|sim> <tree> <tag>}"
PROBE=/src/tools/dcu/a7_non_dtk_probe.py
OUT=/mnt/game/port/a7
CARD="${A7_CARD:-3}"
mkdir -p "$OUT"

# shellcheck disable=SC1091
source /opt/dtk/env.sh >/dev/null 2>&1

if [ "$MODE" = "sim" ]; then
  unset DTKROOT HYHAL_PATH VLLM_DCU_GCN_ARCH VLLM_DCU_BF16
  export ROCM_PATH=/opt/rocm
else
  # env.sh exports the gfx906 override when the KFD topology is readable.
  # Drop it so the DTK baseline exercises the real device probe instead of
  # echoing the override back.
  unset VLLM_DCU_GCN_ARCH
fi

export HIP_VISIBLE_DEVICES="$CARD"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH="$TREE${PYTHONPATH:+:$PYTHONPATH}"
cd "$TREE" || exit 1

section() {
  local name="$1"
  shift
  echo "### ${TAG}.${name}: $*"
  # stdout carries both vLLM log lines and the JSON datum: keep them apart.
  "$@" 2>&1 | tee "${OUT}/${TAG}.${name}.log" | grep '^{' >"${OUT}/${TAG}.${name}.jsonl"
  cat "${OUT}/${TAG}.${name}.jsonl"
}

echo "[a7] mode=${MODE} tag=${TAG} tree=${TREE} card=${CARD}"
echo "[a7] tree_rocm_py_md5=$(md5sum "${TREE}/vllm/platforms/rocm.py" | cut -d' ' -f1)"
python3 -c 'import torch; print("[a7] torch", torch.__version__)' 2>/dev/null

section env python3 "$PROBE" env
section plugin-detect python3 "$PROBE" plugin --drop-visible-devices
section plugin-vis python3 "$PROBE" plugin
section rocm-real python3 "$PROBE" rocm
section rocm-fakeamdsmi python3 "$PROBE" rocm --fake-amdsmi-arch 'gfx942:sramecc+:xnack-'
section rocm-fakeamdsmi-errors python3 "$PROBE" rocm \
  --fake-amdsmi-arch 'gfx942' --fake-amdsmi-fail uuid,mem,topo
section rocm-noamdsmi python3 "$PROBE" rocm --block-amdsmi --torch-arch 'gfx906:sramecc+:xnack-'
# Same but with the torch fallback left real and no GPU visible: both trees
# must die with the identical torch exception from _get_gcn_arch().
section rocm-noamdsmi-nogpu env HIP_VISIBLE_DEVICES= \
  python3 "$PROBE" --stub-device-name FakeGPU rocm --block-amdsmi

for policy in auto 0 1; do
  export VLLM_DCU_BF16="$policy"
  section "rocm-gfx906-bf16-${policy}" python3 "$PROBE" rocm --stub-arch gfx906
done

unset VLLM_DCU_BF16
section rocm-gfx942 python3 "$PROBE" rocm --stub-arch gfx942
section rocm-gfx90a python3 "$PROBE" rocm --stub-arch gfx90a
echo "[a7] done tag=${TAG}"
