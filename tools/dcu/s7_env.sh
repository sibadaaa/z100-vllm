#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# S7 (memory / KV cache / graph) runtime environment on Hygon Z100 (gfx906).
# Source it, do not execute it:
#
#     source tools/dcu/s7_env.sh
#
# Layers the target stack on top of tools/dcu/env.sh:
#   * torch 2.11.0+das.opt1.dtk2604 + gfx906-only vLLM extensions
#     (/mnt/game/port/gfx906-vllm-211)
#   * self-built triton 3.6.0+git7c56a5e4 (/mnt/game/triton-gfx906/install),
#     REQUIRED on gfx906: the DTK wheels cannot emit gfx906 code.
#     The same build carries the S6 SFU hazard fix (s_nop slots), without
#     which triton softmax silently computes wrong values.
#
# Keep TRITON_CACHE_DIR on /mnt/game: the root filesystem has ~48 GB.

_dcu_s7_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${_dcu_s7_dir}/env.sh"

export VLLM_DIR="${VLLM_DIR:-/mnt/game/port/gfx906-vllm-211}"
export TRITON_INSTALL="${TRITON_INSTALL:-/mnt/game/triton-gfx906/install}"

# Order matters: the self-built triton, the fork copy and the site shim must win.
# ${_dcu_s7_dir} carries sitecustomize.py, which works around hard rule 8 in the
# model-registry subprocess and the missing DTK torchvision for torch 2.11.
export PYTHONPATH="${TRITON_INSTALL}:${VLLM_DIR}:${_dcu_s7_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_OVERRIDE:-/mnt/game/triton-gfx906/cache}"

# One card only; the production container owns the other three when it runs.
# tools/dcu/env.sh already exported "0,1,2,3", so re-assign unconditionally
# unless the caller asked for something else first (S7_HIP_VISIBLE_DEVICES).
export HIP_VISIBLE_DEVICES="${S7_HIP_VISIBLE_DEVICES:-0}"

echo "[dcu-s7] VLLM_DIR=${VLLM_DIR}"
echo "[dcu-s7] TRITON_INSTALL=${TRITON_INSTALL} TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "[dcu-s7] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"

unset _dcu_s7_dir
