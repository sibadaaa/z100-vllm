#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Environment for building and running vLLM on Hygon DCU (Z100 / gfx906) with
# DTK. Source it, do not execute it:
#
#     source tools/dcu/env.sh
#
# Everything below is overridable: DTK_ROOT, HIP_VISIBLE_DEVICES,
# VLLM_DCU_GCN_ARCH, VLLM_DCU_BF16, TRITON_CACHE_DIR,
# VLLM_ROCM_USE_SKINNY_GEMM, VLLM_CACHE_ROOT.

if [ -z "${BASH_SOURCE[0]:-}" ]; then
  echo "tools/dcu/env.sh must be sourced from bash" >&2
  return 1 2>/dev/null || exit 1
fi

DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
if [ ! -f "${DTK_ROOT}/env.sh" ]; then
  echo "[dcu] DTK not found at ${DTK_ROOT}; set DTK_ROOT to override" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
. "${DTK_ROOT}/env.sh"

# 1. Prefer a librocm_smi64 that actually exports rsmi_init. Some DTK images
#    list a stub under /opt/hyhal/lib first, which makes `import torch` fail
#    with "undefined symbol: rsmi_init".
for _dcu_dir in "${DTK_ROOT}/.hyhal/rocm_smi/lib" "${DTK_ROOT}/lib" "${DTK_ROOT}/lib64"; do
  for _dcu_lib in "${_dcu_dir}"/librocm_smi64.so*; do
    [ -e "${_dcu_lib}" ] || continue
    if command -v nm >/dev/null 2>&1 &&
      nm -D "${_dcu_lib}" 2>/dev/null | grep -q " T rsmi_init"; then
      case ":${LD_LIBRARY_PATH}:" in
        *":${_dcu_dir}:"*) ;;
        *) export LD_LIBRARY_PATH="${_dcu_dir}:${LD_LIBRARY_PATH}" ;;
      esac
      _dcu_smi="${_dcu_dir}"
      break 2
    fi
  done
done
unset _dcu_dir _dcu_lib

# 2. C++ standard headers. DTK's clang looks for a gcc installation matching
#    its own defaults (gcc 12 on DTK 24.04) while the host may only ship gcc
#    11 headers, which fails with "'cmath' file not found".
for _dcu_cxx in /usr/include/c++/*; do
  [ -d "${_dcu_cxx}" ] || continue
  for _dcu_inc in "${_dcu_cxx}" "/usr/include/x86_64-linux-gnu/c++/$(basename "${_dcu_cxx}")"; do
    [ -d "${_dcu_inc}" ] || continue
    case ":${CPLUS_INCLUDE_PATH}:" in
      *":${_dcu_inc}:"*) ;;
      *) export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:+${CPLUS_INCLUDE_PATH}:}${_dcu_inc}" ;;
    esac
  done
done
unset _dcu_cxx _dcu_inc

# 3. GCN arch: read the KFD topology instead of trusting DTK's amdsmi, which
#    can report gfx18446744073709551615. Setting this also avoids
#    initializing a HIP context while importing vLLM (Ray workers).
#    gfx_target_version 90106 == gfx906 == Z100.
if [ -z "${VLLM_DCU_GCN_ARCH:-}" ]; then
  _dcu_ver=""
  _dcu_uniform=1
  for _dcu_props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ -e "${_dcu_props}" ] || continue
    _dcu_v="$(sed -n 's/^gfx_target_version //p' "${_dcu_props}" 2>/dev/null)"
    # CPU nodes report gfx_target_version 0.
    [ -n "${_dcu_v}" ] && [ "${_dcu_v}" != "0" ] || continue
    if [ -n "${_dcu_ver}" ] && [ "${_dcu_v}" != "${_dcu_ver}" ]; then
      _dcu_uniform=0
    fi
    _dcu_ver="${_dcu_v}"
  done
  if [ "${_dcu_uniform}" = "1" ] && [ "${_dcu_ver}" = "90106" ]; then
    export VLLM_DCU_GCN_ARCH=gfx906
  fi
  unset _dcu_ver _dcu_uniform _dcu_props _dcu_v
fi

# 4. Triton: DTK ships ld.lld under its LLVM tree.
export TRITON_HIP_LLD_PATH="${TRITON_HIP_LLD_PATH:-${DTK_ROOT}/llvm/bin/ld.lld}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"

# 5. Do NOT export ROCBLAS_TENSILE_LIBPATH. Measured on Z100 / DTK 26.04: the
#    gfx906 Tensile directory only ships the *lazy* library
#    (${DTK_ROOT}/lib/rocblas/library/TensileLibrary_lazy_gfx906.dat), and
#    setting this variable makes rocBLAS load the legacy non-lazy
#    TensileLibrary.dat instead, which does not exist here. Tensile host init
#    then fails and takes the whole process down:
#        rocBLAS error: Could not initialize Tensile host: std::bad_alloc
#        Aborted (core dumped)      # SIGABRT (134) / SIGSEGV (139)
#    The DTK default location is already compiled into rocBLAS and works, so
#    leave the variable unset; advanced users can still export it themselves.

# 6. gfx906 has no MFMA, so the ROCm skinny GEMM kernels serve no purpose
#    here and are not selected; keep them off until the gfx906 build filter
#    in CMakeLists.txt is in place.
export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-0}"

# 7. Visible devices: only HIP/CUDA visibility vars matter, and they also make
#    vLLM select the ROCm platform without probing amdsmi.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"

# 8. torch.compile / inductor artifacts. vLLM defaults VLLM_CACHE_ROOT to
#    ~/.cache/vllm, which lands in the container writable layer; one smoke run
#    of a small model was measured at 2.1 GB (hard rule 17). Keep it on
#    /mnt/game when that mount exists. Pre-set VLLM_CACHE_ROOT to override.
if [ -z "${VLLM_CACHE_ROOT:-}" ] && [ -d /mnt/game ]; then
  export VLLM_CACHE_ROOT=/mnt/game/vllm-cache
fi

# 9. RCCL P2P requires fine-grained PCIe memory. On Z100 (hydcu/gfx906) the
#    runtime reports hipDeviceAttributeFineGrainSupport=0 and fails every
#    hipDeviceMallocFinegrained allocation with hipErrorOutOfMemory unless this
#    variable is set; RCCL's p2pCanConnect() hard-requires hasFineGrain and
#    otherwise falls back to SHM. RCCL itself warns "Missing
#    HSA_FORCE_FINE_GRAIN_PCIE=1 from environment" at startup.
#    Measured (4 ranks on the same PCIe switch, docs/result/z100-16-*):
#      via P2P=56 / via SHM=0   (was via P2P=0 / via SHM=56)
#      1 MiB all-reduce 1143 -> 143 us, 4 MiB 4388 -> 318 us
#      <=64 KiB unchanged; pinned H2D/D2H, peer copy_ and GEMM numerics unchanged
#    Set HSA_FORCE_FINE_GRAIN_PCIE=0 to reproduce the old SHM-only behaviour.
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"

echo "[dcu] DTK_ROOT=${DTK_ROOT} ($(cat "${DTK_ROOT}/.info/rocm_version" 2>/dev/null || echo unknown))"
echo "[dcu] HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE} (RCCL P2P gate; 0=SHM-only)"
echo "[dcu] rocm_smi=${_dcu_smi:-<DTK default>} VLLM_DCU_GCN_ARCH=${VLLM_DCU_GCN_ARCH:-<auto>}"
echo "[dcu] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} VLLM_DCU_BF16=${VLLM_DCU_BF16:-auto}"
echo "[dcu] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-<vLLM default ~/.cache/vllm>}"
echo "[dcu] verify with: python3 tools/dcu/check_bf16.py"
