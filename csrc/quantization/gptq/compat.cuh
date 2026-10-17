/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _compat_cuh
#define _compat_cuh

#ifdef USE_ROCM
#include <type_traits>
#include <utility>

// Probe for an unqualified `atomicAdd(half2*, half2)` reaching this namespace;
// the static_assert at the bottom of this file uses it to confirm the
// `__DTK_ARCH__` decision below against the overload set the compiler sees.
// A preprocessor test cannot do the probing itself (`::value` is not valid in
// `#if`), and a version test cannot be trusted: DTK 26.04 reports HIP 6.3 while
// its <hip/amd_detail/amd_hip_fp16.h> declares `__half2 atomicAdd(__half2*,
// __half2)` unconditionally.
template <typename T, typename = void>
struct vllm_gptq_has_native_half2_atomic_add : std::false_type {};
template <typename T>
struct vllm_gptq_has_native_half2_atomic_add<
    T,
    std::void_t<decltype(atomicAdd(std::declval<T*>(), std::declval<T>()))>>
    : std::true_type {};
#endif  // USE_ROCM

// `__DTK_ARCH__` is defined by DTK's <ATen/dtk_macros.h>, which DTK's hipify
// injects into every generated file, i.e. it is present in every DTK HIP
// translation unit. DTK's amd_hip_fp16.h declares both `__half atomicAdd(
// __half*, __half)` and `__half2 atomicAdd(__half2*, __half2)` at global scope
// unconditionally, so on DTK neither emulation below may exist: each one is an
// equally good candidate for the unqualified `atomicAdd(out, val)` calls in
// this namespace and overload resolution would be ambiguous.
#if defined(__CUDA_ARCH__) || defined(USE_ROCM)
  #if !defined(__DTK_ARCH__)
    #ifdef USE_ROCM
      // Community ROCm provides the `half2` overload from 6.2 on. There is no
      // feature-test macro for it, so gate on the documented version; a wrong
      // guess fails the static_assert below rather than miscompiling.
      #if defined(HIP_VERSION) && \
          (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) >= 602
        #define VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED 0
      #else
        #define VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED 1
      #endif
    #else
      // CUDA: the native overloads arrived with sm_60 (half2) / sm_70 (half).
      #if __CUDA_ARCH__ < 600
        #define VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED 1
      #else
        #define VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED 0
      #endif
    #endif
  #else
    #define VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED 0
  #endif
#else
  #define VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED 0
#endif

// HIP 7.0+ (upstream ROCm 10.x) declares the native `half` overload as well, so
// on that toolchain no emulation may be visible either; DTK's headers behave
// like the modern ones. Older community ROCm still needs the `half` emulation.
#if defined(USE_ROCM) && defined(HIP_VERSION) && HIP_VERSION >= 70000000
  #define VLLM_GPTQ_HALF_ATOMIC_ADD_EMULATED 0
#elif defined(__DTK_ARCH__)
  #define VLLM_GPTQ_HALF_ATOMIC_ADD_EMULATED 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700
  #define VLLM_GPTQ_HALF_ATOMIC_ADD_EMULATED 1
#else
  #define VLLM_GPTQ_HALF_ATOMIC_ADD_EMULATED 0
#endif

namespace vllm {
namespace gptq {
// atomicAdd for half types, to support CC < 7.x

__device__ __forceinline__ void atomicAdd_half(half* address, half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = __hadd(hsum, val);
    hsum = __half_raw(tmpres);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

// atomicAdd for half2 types

__device__ __forceinline__ void atomicAdd_half2(half2* address, half2 val) {
  unsigned int* address_as_ui = (unsigned int*)address;
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    half2 old_val = *((half2*)&old);
    half2 new_val = __hadd2(old_val, val);
    old = atomicCAS(address_as_ui, assumed, *((unsigned int*)&new_val));
  } while (assumed != old);
}

#if defined(__CUDA_ARCH__) || defined(USE_ROCM)
  #if __CUDA_ARCH__ < 700 || defined(USE_ROCM)

    #if VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED

__device__ __forceinline__ void atomicAdd(half* address, half val) {
  atomicAdd_half(address, val);
}

__device__ __forceinline__ void atomicAdd(half2* address, half2 val) {
  atomicAdd_half2(address, val);
}

    #else

      #if VLLM_GPTQ_HALF_ATOMIC_ADD_EMULATED
__device__ __forceinline__ void atomicAdd(half* address, half val) {
  atomicAdd_half(address, val);
}
      #endif

    #endif

  #endif
#endif

}  // namespace gptq
}  // namespace vllm

#ifdef USE_ROCM
namespace vllm {
namespace gptq {
// Confirms the `__DTK_ARCH__` decision above against the overload set the
// compiler actually sees. With `VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED == 1` the
// emulation is visible here too; with 0 the toolchain's own overload must be.
static_assert(
    vllm_gptq_has_native_half2_atomic_add<half2>::value,
    "no atomicAdd(half2*, half2) is visible although the emulation was "
    "compiled out by VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED");
}  // namespace gptq
}  // namespace vllm
#endif  // USE_ROCM

#endif
