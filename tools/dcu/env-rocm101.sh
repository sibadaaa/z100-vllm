#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Environment contract for running vLLM on Hygon DCU (Z100 / gfx906) with the
# self-built AMD ROCm 10.1 stack:
#
#   * ROCm 10.1 / TheRock layout under /opt/rocm/core-10.1
#   * torch 2.16.0.dev20261016+rocm10.1 (self-built with the SFU-patched clang)
#   * triton 3.6.0 (rocm101 build) consumed through PYTHONPATH, never pip-installed
#
# Source it, do not execute it:
#
#     source tools/dcu/env-rocm101.sh
#
# This is the rocm101 counterpart of tools/dcu/env.sh (DTK). The two stacks can
# coexist because nothing here touches the DTK tree.
#
# Assumed runtime container (docs/z100-torch216-rocm101-adapt-plan.md §4.1):
#   z100-vllm-rocm101-rt, image z100-rocm10.1:24.04-r642, /mnt/game rw, and the
#   repository mounted read-only at /src.
#
# Everything below is overridable: ROCM_PATH, ROCM_SDK_SITE,
# HIP_VISIBLE_DEVICES, VLLM_DCU_GCN_ARCH, VLLM_DCU_BF16, TRITON_CACHE_DIR,
# VLLM_CACHE_ROOT, VLLM_ROCM_USE_SKINNY_GEMM, VLLM_USE_AOT_COMPILE.

if [ -z "${BASH_SOURCE[0]:-}" ]; then
  echo "tools/dcu/env-rocm101.sh must be sourced from bash" >&2
  return 1 2>/dev/null || exit 1
fi

# 1. ROCm 10.1 / TheRock. Upstream HIP reports the Z100 as GFX090106 unless the
#    runtime is told which gfx version the device implements; without
#    HSA_OVERRIDE_GFX_VERSION the cards are skipped as "unrecognized".
export ROCM_PATH="${ROCM_PATH:-/opt/rocm/core-10.1}"
export HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"

# 2. ROCm SDK 10.1 python wheels. The torch wheel's RUNPATH points at
#    ${SP}/_rocm_sdk_devel/lib (needed to resolve librccl et al.), and the ROCm
#    python packages live in ${SP}/_rocm_sdk_{core,libraries,devel}.
#    NOTE: the devel wheel ships its payload as rocm_sdk_devel/_devel.tar, so the
#    tree has to be extracted once after pip-installing it:
#        tar xf ${SP}/rocm_sdk_devel/_devel.tar -C ${SP}
SP="${ROCM_SDK_SITE:-/usr/local/lib/python3.12/dist-packages}"

# 3. Libraries. /mnt/game/rccl-gfx906/lib is the locally rebuilt gfx906 RCCL
#    (plan §6.1-P0.5) and must come first so the process loads that .so instead
#    of the SDK librccl under _rocm_sdk_devel/lib, whose device code has no
#    gfx906 support. While P0-R is still in flight the directory may not exist;
#    a missing entry in LD_LIBRARY_PATH is harmless.
#    The SDK wheel libs then come before the image libs, as in the original
#    plan P0.3 / torch install-verify.sh (verified working on this machine).
#    NOTE: the first-allocation hang seen on 2026-10-17 was NOT caused by this
#    ordering (see README-finding.md); it was PYTORCH_HIP_ALLOC_CONF
#    expandable_segments, handled in §5.
export LD_LIBRARY_PATH="/mnt/game/rccl-gfx906/lib:${SP}/_rocm_sdk_libraries/lib:${SP}/_rocm_sdk_core/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.0.6}"
# Fine-grained PCIe memory is required for RCCL P2P on Z100 (see env.sh §9).
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"

# 4. torch / RCCL build-time choices (both mandatory):
#    * USE_LIBUV=0: the wheel was built with USE_TENSORPIPE=0, i.e. without libuv;
#      unset it and the distributed store dies with "use_libuv was requested but
#      PyTorch was built without libuv support".
#    * RCCL_USE_AMD_SMI_LIB=1: Hygon vendor 0x1d94 is mapped through ARSMI,
#      otherwise RCCL fails with "ARSMI_lib: <BDF> device index not found".
export USE_LIBUV="${USE_LIBUV:-0}"
export RCCL_USE_AMD_SMI_LIB="${RCCL_USE_AMD_SMI_LIB:-1}"

# 5. vLLM DCU knobs.
#    VLLM_DCU_GCN_ARCH is pinned so that platform code does not have to
#    initialize a HIP context (nor trust amdsmi, which reports gfx916).
export VLLM_DCU_GCN_ARCH="${VLLM_DCU_GCN_ARCH:-gfx906}"
# gfx906 bf16 is opened by decision D6 / §9.1-Q1; auto keeps the runtime probe.
export VLLM_DCU_BF16="${VLLM_DCU_BF16:-auto}"
# gfx906 has no MFMA, so the ROCm skinny GEMM kernels serve no purpose here.
export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-0}"
# Bring-up period: keep AOT compile off (torch >= 2.10 enables it by default).
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-0}"

# DO NOT set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True here. On this
# stack (torch 2.16 + ROCm 10.1 + hycu driver, gfx906) the expandable-segments
# allocator hangs forever on the FIRST allocation in an infinite
# drm_syncobj_timeline_wait (HIP VMM path; see
# /mnt/game/port/rocm101-runtime/README-finding.md). Measured A/B 2026-10-17:
# set => hang at `torch.ones(4, device="cuda")`; unset => pass. The DTK-era
# advice does not apply to ROCm 10.1. If it is ever needed again, set it
# explicitly for a single experiment, never in this shared env contract.
# export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_HIP_ALLOC_CONF

# 6. Python path: the writable vLLM working copy (build products land there),
#    then the triton 3.6.0 rocm101 build (never pip-installed, so PYTHONPATH is
#    the only way to consume it), then the SDK amdsmi 27.1.0.
#    amdsmi is NOT a pip package: _rocm_sdk_devel/share/amd_smi contains the
#    importable `amdsmi` tree. Verify with:
#        python3 -c "import amdsmi; print(amdsmi.amdsmi_get_lib_version())"
#    Do not install PyPI amdsmi==7.0.2: wrong version/ABI for ROCm 10.1.
VLLM_ROCM101_REPO="/mnt/game/port/gfx906-vllm-rocm101"
TRITON_ROCM101_ROOT="/mnt/game/triton-z100-rocm101"
export PYTHONPATH="${VLLM_ROCM101_REPO}:${TRITON_ROCM101_ROOT}/install:${SP}/_rocm_sdk_devel/share/amd_smi${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_ROCM101_ROOT}/cache/jit}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/mnt/game/vllm-cache-rocm10}"

# 7. Run-time defaults carried over from the DTK stack.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM=false
# (no PYTORCH_HIP_ALLOC_CONF here: expandable_segments hangs this stack, see §5)
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"

# Do NOT unset the image-provided ROCBLAS_TENSILE_LIBPATH. ROCm 10.1 ships the
# full gfx906 Tensile directory (${ROCM_PATH}/lib/rocblas/library/gfx906) and the
# image ENV already points there; unlike the DTK stack, setting it is correct
# here. (The DTK-era "leave unset" advice does not apply to ROCm 10.1.)
#
# Import order: always `import torch` before `import vllm` (plan §7-L0.7).

echo "[rocm101] ROCM_PATH=${ROCM_PATH} SP=${SP}"
echo "[rocm101] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE}"
echo "[rocm101] vllm-copy=${VLLM_ROCM101_REPO} triton=${TRITON_ROCM101_ROOT}/install amdsmi=${SP}/_rocm_sdk_devel/share/amd_smi"
echo "[rocm101] VLLM_DCU_GCN_ARCH=${VLLM_DCU_GCN_ARCH} VLLM_DCU_BF16=${VLLM_DCU_BF16} VLLM_ROCM_USE_SKINNY_GEMM=${VLLM_ROCM_USE_SKINNY_GEMM} VLLM_USE_AOT_COMPILE=${VLLM_USE_AOT_COMPILE}"
echo "[rocm101] TRITON_CACHE_DIR=${TRITON_CACHE_DIR} VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
