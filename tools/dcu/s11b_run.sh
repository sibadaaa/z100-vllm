#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# S11b (P0): launch the 27B W8A8 INT8 TP=2 decode profiler inside z100-s9.
#
#   bash /mnt/game/port/s11b/s11b_run.sh <default|none> <out-dir> [extra args]
#
# Cards 0,1 (TP=2) per the task书; no --enforce-eager (hard rule 13).
set -euo pipefail
MODE="${1:?mode: default|none}"
OUT="${2:?out dir}"
shift 2 || true

source /opt/dtk/env.sh
export S8_HIP_VISIBLE_DEVICES="${S11B_DEVICES:-0,1}"
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b
# Spawn (not the default fork): with fork the 27B WorkerProc dies in
# init_device() with "CUDA driver initialization failed" on this DTK stack.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
cd "$VLLM_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# Optional: record the GEMM M/N/K distribution of each int8 Triton kernel as a
# TSV (workers write one line per call).
export S11B_SHAPE_DIR="$OUT"
mkdir -p "$OUT"
exec python /mnt/game/port/s11b/s11b_profile_27b.py \
  --mode "$MODE" --out "$OUT" --steps 32 --max-model-len 4096 "$@"
