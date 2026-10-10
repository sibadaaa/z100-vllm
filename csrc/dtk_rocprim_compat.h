// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Compatibility shim for the Hygon DCU Toolkit (DTK) HIP toolchain.
//
// DTK 26.04 ships rocPRIM (`rocprim/types/uninitialized_array.hpp`:
// `uninitialized_array::emplace`) which calls the libstdc++ placement-new
// declared in `/usr/include/c++/11/new` from a `__host__ __device__`
// function:
//
//     return *new(ptr) T(std::forward<Args>(args)...);
//
// The DTK clang 18 resource-dir wrapper `<clang>/include/cuda_wrappers/new`
// is meant to add the `__device__` placement-new overloads, but it is picked
// up while `__device__` is not yet defined and then deliberately disarms
// itself (`#if !defined(__device__) ... #undef <guard>`), so only the
// host-only libstdc++ overload is ever visible.  Every TU that instantiates
// `rocprim::uninitialized_array::emplace` - e.g. anything that pulls in a
// hipcub `BlockLoad`/`BlockExchange` such as
// `csrc/mamba/mamba_ssm/selective_scan_fwd.cu` - therefore fails to compile
// for gfx906 with
//
//     error: reference to __host__ function 'operator new' in
//            __host__ __device__ function
//
// The overloads below are exactly the ones the upstream ROCm toolchain
// already provides (see `cuda_wrappers/new`): placement new is an
// address-initialising operation, so returning the passed pointer, and doing
// nothing on delete, is the identical semantics.  Nothing is relaxed, no
// check is skipped, and code generation is unaffected.
//
// This header is injected with `-include` only for DTK builds (see the
// `VLLM_DTK_ROCPRIM_COMPAT` block in CMakeLists.txt); CUDA and community
// ROCm builds never see it.
//
// C++ requires `operator new`/`operator delete` to be declared at global
// scope, so this header cannot live in a namespace.
#ifndef VLLM_CSRC_DTK_ROCPRIM_COMPAT_H_
#define VLLM_CSRC_DTK_ROCPRIM_COMPAT_H_

#if defined(__HIP__) || defined(__HIPCC__)

#include <cstddef>
#include <new>

#include <hip/hip_runtime.h>

__device__ [[gnu::noinline]] void* operator new(std::size_t,
                                                void* __p) noexcept {
  return __p;
}
__device__ [[gnu::noinline]] void operator delete(void*, void*) noexcept {}

__device__ [[gnu::noinline]] void* operator new[](std::size_t,
                                                  void* __p) noexcept {
  return __p;
}
__device__ [[gnu::noinline]] void operator delete[](void*, void*) noexcept {}

#endif  // defined(__HIP__) || defined(__HIPCC__)

#endif  // VLLM_CSRC_DTK_ROCPRIM_COMPAT_H_
