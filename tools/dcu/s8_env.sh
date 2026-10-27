#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# S8 (attention / quantization path audit) runtime environment on Hygon Z100
# (gfx906). Source it, do not execute it:
#
#     source tools/dcu/s8_env.sh
#
# This is tools/dcu/s7_env.sh with one deliberate difference: the S8 audit runs
# on card 1 (HIP_VISIBLE_DEVICES=1), because the Lead owns card 0 and another
# agent owns card 2. Everything else (target stack, self-built gfx906 triton,
# TRITON_CACHE_DIR on /mnt/game, the usercustomize.py shim) is inherited
# unchanged from S7 so the two steps observe the same runtime.
#
# Hard rule 13: never pass --enforce-eager on gfx906 (logits become NaN).

_dcu_s8_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${_dcu_s8_dir}/s7_env.sh"

# s7_env.sh already picked HIP_VISIBLE_DEVICES from S7_HIP_VISIBLE_DEVICES.
# Re-assign unconditionally so a caller that exported HIP_VISIBLE_DEVICES
# directly still lands on the S8 card.
export HIP_VISIBLE_DEVICES="${S8_HIP_VISIBLE_DEVICES:-1}"

# torch.compile / inductor artifacts must not land in the container layer
# (the DTK image's / has ~48 GB). VLLM_CACHE_ROOT defaults to ~/.cache/vllm.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/mnt/game/port/s8/vllm-cache}"

echo "[dcu-s8] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
echo "[dcu-s8] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"

unset _dcu_s8_dir
