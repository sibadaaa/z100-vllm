# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.rocm import on_gfx9, on_gfx906
from vllm.utils.flashinfer import (
    flashinfer_bf16_mm,
    is_flashinfer_cutedsl_bf16_gemm_supported,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=3,
            num_warps=2,
        ),
    ]


def get_heuristics():
    return {"BLOCK_SIZE_M": lambda args: min(16, triton.next_power_of_2(args["M"]))}


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.heuristics(values=get_heuristics())
@triton.jit
def triton_matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # NOTE(gfx906): match the output tensor's dtype (fp16 or bf16) instead of
    # hardcoding fp16 -- a bf16 drafter behind an fp16 target runs this path.
    c = accumulator.to(c_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], (
        "Incompatible dimensions"
    )  # NOTE(gfx906): b.shape inv
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype, "triton_matmul requires matching input dtypes"
    M, K = a.shape
    N, K = b.shape  # NOTE(gfx906): b.shape inv
    # Allocates output in the input dtype (fp16 target, bf16 drafter).
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    triton_matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(1),
        b.stride(0),  # NOTE(gfx906): b.stride inv
        c.stride(0),
        c.stride(1),  #
    )
    return c


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    linear_input = x if x.dtype == weight.dtype else x.to(weight.dtype)
    return torch.nn.functional.linear(linear_input, weight, bias)


_FlashInferBf16RuntimeCheck = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None], bool
]


@dataclass(frozen=True)
class _FlashInferBf16Backend:
    flashinfer_backend: str
    is_supported: Callable[[], bool]
    can_implement: _FlashInferBf16RuntimeCheck


def _can_use_flashinfer_cutedsl_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return False
    if x.ndim < 1 or weight.ndim != 2:
        return False
    if (
        not x.is_cuda
        or not weight.is_cuda
        or x.device != weight.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        return False

    k = x.shape[-1]
    n = weight.shape[0]
    if (
        k <= 0
        or n <= 0
        or x.numel() == 0
        or weight.shape[1] != k
        or k % 128 != 0
        or x.data_ptr() % 32 != 0
        or weight.data_ptr() % 32 != 0
    ):
        return False
    return bias is None or (
        bias.is_cuda
        and bias.device == x.device
        and bias.dtype == torch.bfloat16
        and bias.ndim == 1
        and bias.shape[0] == n
        and bias.is_contiguous()
    )


_FLASHINFER_BF16_BACKENDS = {
    "flashinfer_cutedsl": _FlashInferBf16Backend(
        flashinfer_backend="cute-dsl",
        is_supported=is_flashinfer_cutedsl_bf16_gemm_supported,
        can_implement=_can_use_flashinfer_cutedsl_bf16,
    ),
}


def _get_flashinfer_bf16_backend(vllm_backend: str) -> _FlashInferBf16Backend:
    backend_spec = _FLASHINFER_BF16_BACKENDS.get(vllm_backend)
    if backend_spec is None:
        supported = ", ".join(sorted(_FLASHINFER_BF16_BACKENDS))
        raise ValueError(
            f"Unsupported vLLM FlashInfer BF16 backend {vllm_backend!r}; "
            f"supported backends: {supported}"
        )
    return backend_spec


def cuda_flashinfer_bf16_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pdl: bool,
    vllm_backend: str,
) -> torch.Tensor:
    backend_spec = _get_flashinfer_bf16_backend(vllm_backend)
    if not backend_spec.can_implement(x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias)

    k = x.shape[-1]
    n = weight.shape[0]
    x_2d = x.view(-1, k)
    out_2d = flashinfer_bf16_mm(
        x_2d,
        weight.t(),
        bias,
        pdl,
        backend_spec.flashinfer_backend,
    )
    return out_2d.view(*x.shape[:-1], n)


def cuda_flashinfer_bf16_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pdl: bool,
    vllm_backend: str,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def cuda_flashinfer_bf16_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    vllm_backend: str,
    pdl: bool,
) -> torch.Tensor:
    return torch.ops.vllm.cuda_flashinfer_bf16_gemm(
        x,
        weight,
        bias,
        pdl,
        vllm_backend,
    )


direct_register_custom_op(
    op_name="cuda_flashinfer_bf16_gemm",
    op_func=cuda_flashinfer_bf16_gemm_impl,
    fake_impl=cuda_flashinfer_bf16_gemm_fake,
)


def use_aiter_triton_gemm(n, m, k, dtype):
    from vllm.platforms.rocm import on_gfx1250

    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    if on_gfx1250():
        return True

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
    )


def wvsplitkrc_dispatch(n: int, k: int, m: int, cu_count: int) -> tuple[int, bool]:
    """Pick the K-shard split for wvSplitKrc and say whether the shape fits.

    Mirrors wvSplitKrc() in csrc/rocm/skinny_gemms.cu, which is also where the
    shard cap is explained. Both must pick the same chunkk or the workspace
    check here bounds the wrong k_rnd.

    Returns:
        The CHUNKK the kernel will dispatch with, and whether the CU budget and
        split-K workspace admit the shape at all.

    """
    # Next ^2 of n
    N_p2 = 1 << (n - 1).bit_length()
    # How many of 4 waves in a group can work on same 16 Ms at same time?
    # This reduces the Ms each group works on, i.e. increasing the CUs needed.
    GrpsShrB = min(N_p2 // 16, 4)
    # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile), and each
    # working on a 512-shard of K, how many CUs would we need?
    CuNeeded = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512) * GrpsShrB

    CHUNKK2_MAX_SHARDS = 11
    shards_chunkk2 = (k + 256 - 1) // 256  # 256-wide shards
    chunkk = (
        2
        if (
            N_p2 != 16
            and CuNeeded * 2 <= cu_count
            and shards_chunkk2 <= CHUNKK2_MAX_SHARDS
        )
        else 1
    )

    # Deterministic reduction stores one fp32 partial per (M, N, k-shard); all
    # of them must fit the split-K workspace.
    k_rnd = (k + 512 // chunkk - 1) // (512 // chunkk)
    fits = N_p2 * m * k_rnd <= 128 * 1024 * 12 and CuNeeded <= cu_count
    return chunkk, fits


_GFX906_GEMV_THREADS = 64
# perf1 (2026-10-13) mm probe: 128 threads wins on narrow N (in_proj_ba N=24 at
# M=1: 12.5 vs 16.6 us; mtp.fc N=1280/K=10240 at M=3: 139.5 vs 174.0), while 64
# threads wins on the wide unquantized lm_head (N=62080 at M=1: 893 vs 902 us,
# M=3: 1812 vs 1879 us), so the count follows the output width.
_GFX906_GEMV_THREADS_NARROW = 128
_GFX906_GEMV_NARROW_MAX_N = 4096
_GFX906_GEMV_AVAILABLE = hasattr(torch.ops, "_rocm_C") and hasattr(
    torch.ops._rocm_C, "bf16_emu_gemv"
)
# On gfx11 a weight row stride that is a multiple of 2048 B makes the multi-row global
# loads of a GEMM collide on the L2/MALL channel hash. Offsetting the stride by one
# 128 B cache line dodges the collision and keeps each row cache-line aligned.
def cache_aliasing_pad_elems(k: int, element_size: int) -> int:
    """Row-stride pad, in elements, that stops an ``[n, k]`` weight aliasing on gfx11,
    or 0 when the stride does not alias.

    Args:
        k: Number of elements per weight row.
        element_size: Size of one element in bytes.

    Returns:
        The number of elements to add to the row stride.

    """
    if (k * element_size) % 2048 != 0:
        return 0
    return 128 // element_size


def maybe_pad_weight_avoid_cache_aliasing(weight: torch.Tensor) -> torch.Tensor:
    """Offset a dense weight's row stride so it stops aliasing.

    Returns a view with the original shape whose row stride is one cache line
    wider, or ``weight`` unchanged when the pad does not apply.
    """
    from vllm.platforms.rocm import on_gfx11

    if not (current_platform.is_rocm() and on_gfx11()):
        return weight
    if weight.dim() != 2 or not weight.is_contiguous():
        return weight
    pad = cache_aliasing_pad_elems(weight.shape[1], weight.element_size())
    if pad == 0:
        return weight
    return torch.nn.functional.pad(weight, (0, pad))[..., :-pad]


def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    if envs.VLLM_BATCH_INVARIANT:
        from vllm.model_executor.determinism.batch_invariant import (
            linear_batch_invariant,
        )

        return linear_batch_invariant(x, weight, bias)

    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx11, on_gfx950


def _gfx906_gemv_threads(n_out: int) -> int:
    """Threads per block for the emulated dense GEMV (measured, see above)."""
    if n_out <= _GFX906_GEMV_NARROW_MAX_N:
        return _GFX906_GEMV_THREADS_NARROW
    return _GFX906_GEMV_THREADS


def _gfx906_gemv_shape_ok(n_tokens: int, n_out: int, dtype: torch.dtype) -> bool:
    """Shapes where the emulated GEMV beats rocBLAS (measured, see
    ``tools/dcu/s11f_bf16_gemv_probe.py``).

    Measured speedups over `F.linear` (Z100, K=5120): a single-token decode
    wins everywhere (1.3-5.7x, 590-625 GB/s vs rocBLAS's 215-235 GB/s), bf16
    small-N wins too (N=512: 5.7x at M=1, 2.1x at M=4), while M >= 8 becomes
    LDS/compute bound and *loses* to rocBLAS (0.2-0.9x).  The rule below only
    enables the cells that were measured to win, so a non-qualifying call keeps
    the previous rocBLAS behaviour unchanged.

    perf1 re-measured the same cells against the *Triton* matmul this recipe
    actually runs (fp16, `tools/dcu/perf1_gdn_gpu.sh` mm probe):

    * narrow N wins at every decode M -- GDN ``in_proj_ba`` (N=24, K=5120):
      12.5/15.3/19.0/23.5 us for M=1..4 vs 133.0/132.5/155.7/155.8, i.e. 6.6-10.6x;
      MTP ``mtp.fc`` (N=1280, K=10240): 58.9/87.8/139.5/202.2 vs
      342.7/339.5/415.8/411.0;
    * the unquantized ``lm_head`` (N=62080, K=5120) wins at M<=2 (M=1: 902.8 vs
      1589.8 us; M=2: 1412.0 vs 1588.1), ties at M=3 (1876.0 vs 1856.5) and
      loses at M=4 (2416.5 vs 1873.6), so the wide-N allowance stops at M=2.

    A1 (2026-10-25, ``tools/dcu/a1_ba_gemv_probe.py``): the k=7 verify step
    runs ``in_proj_ba`` at M=8, which fell through to ``triton_matmul``
    (188.4 us at N=24, ~10.4 ms/step over 48 layers).  The emulated GEMV
    (nt=128) measured 10.2/16.7/25.8/31.9/32.4/36.5/40.7 us for M=1..8 at the
    same shape, i.e. 4.6-11.6x over Triton and a tie with rocBLAS at M=8
    (42.3 us); fp16 rel err 2.5e-4 of |max|, identical to rocBLAS/Triton
    (fp16 output quantization dominates).  Narrow N (<=64, the ``in_proj_ba``
    class) is therefore allowed up to M=8, which is also the kernel's M cap
    (``kMsMax``).  Wider N keeps the old rule.
    """
    if n_out <= 64:
        return n_tokens <= _gfx906_gemv_narrow_max_m()
    if n_out <= 1280:
        return n_tokens <= 4
    if n_tokens <= 2:
        return True
    return False


@functools.lru_cache(maxsize=1)
def _gfx906_gemv_narrow_max_m() -> int:
    """A/B override for the narrow-N GEMV rule (A1, 2026-10-25).

    Default 8 = the A1 rule (kernel cap ``kMsMax``); setting
    ``VLLM_DCU_GEMV_NARROW_MAX_M=4`` restores the pre-A1 behaviour without a
    code change, the same pattern as the other gfx906 A/B knobs.  Cached
    because the service reads it once per arm restart.
    """
    raw = os.environ.get("VLLM_DCU_GEMV_NARROW_MAX_M", "").strip()
    try:
        n = int(raw) if raw else 8
    except ValueError:
        n = 8
    return max(0, min(n, 8))


def _gfx906_dense_gemv_enabled() -> bool:
    """Whether the gfx906 dense decode GEMV is enabled.

    Uses the shared family switch so that ``VLLM_DCU_USE_GFX906_EMU=1`` turns
    this kernel on together with the INT8/FP8 scaled-MM kernels, while an
    explicit ``VLLM_DCU_USE_BF16_GEMV=0`` still wins.
    """
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_common import (
        gfx906_emu_enabled,
    )

    return gfx906_emu_enabled("VLLM_DCU_USE_BF16_GEMV")


# S22: rocBLAS vs the single-config Triton matmul for the small-M band.
# `_GFX906_ROCBLAS_MAX_N` is the widest N (out features) where rocBLAS was
# measured to win at M >= 4; above it the Triton path keeps the smallest M.
_GFX906_ROCBLAS_MAX_N = 8192


def _gfx906_smallm_rocblas_enabled() -> bool:
    """Whether small-M unquantized GEMMs prefer rocBLAS over Triton (S22)."""
    raw = os.environ.get("VLLM_DCU_SMALLM_GEMM", "").strip().lower()
    return raw in ("rocblas", "1", "true", "yes", "on")


def _gfx906_bf16_route() -> int:
    """Map VLLM_DCU_BF16_ROUTE to the kernel's route selector."""
    route = envs.VLLM_DCU_BF16_ROUTE
    if route == "f32":
        return 1
    if route == "f16":
        return 2
    return 0  # auto: fp16 -> dot2, bf16 -> fp32 (exact)


def _register_gfx906_dense_gemv() -> None:
    """Expose the gfx906 dense GEMV as a `torch.library.custom_op` (once).

    Same reason as `gfx906_sdot4.py`: a raw ``TORCH_LIBRARY`` op leaks an
    ``Eq(dim, hint)`` guard on the symbolic token dimension at its call site
    and aborts vLLM's compile with ``ConstraintViolationError``; a
    ``custom_op`` with a ``register_fake`` keeps it an opaque in-graph call.
    """
    if hasattr(torch.ops.vllm, "bf16_emu_mm"):
        return

    @torch.library.custom_op("vllm::bf16_emu_mm", mutates_args=())
    def bf16_emu_mm(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.ops._rocm_C.bf16_emu_gemv(
            x,
            weight,
            bias,
            _gfx906_gemv_threads(weight.shape[0]),
            _gfx906_bf16_route(),
            False,
        )

    @bf16_emu_mm.register_fake
    def _(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return x.new_empty((x.shape[0], weight.shape[0]))


if current_platform.is_rocm():
    _register_gfx906_dense_gemv()


def rocm_unquantized_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    x_view = x.reshape(-1, x.size(-1))
    n = x_view.shape[0]
    m = weight.shape[0]
    k = weight.shape[1]

    cu_count = num_compute_units()

    _, fits_wvsplitkrc = wvsplitkrc_dispatch(n, k, m, cu_count)

    skinny_operands_compatible = weight.is_contiguous() and (
        bias is None or bias.is_contiguous()
    )

    use_skinny_reduce_counting = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and on_gfx950()
        and x.dtype in [torch.float16, torch.bfloat16]
        and x.dim() == 2
        and (
            10 <= n <= 128
            and k % 8 == 0
            and k > 512
            and m % 16 == 0
            and fits_wvsplitkrc
            and skinny_operands_compatible
        )
    )

    if use_skinny_reduce_counting:
        x_view = x.reshape(-1, x.size(-1)).contiguous()
        return ops.wvSplitKrc(x_view, weight, cu_count, bias)

    if use_aiter_triton_gemm(n, m, k, x.dtype):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        return gemm_a16w16(x, weight, bias)

    skinny_gemm_enabled = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and (on_gfx9() or on_gfx1x())
        # build (gfx9/gfx11 ISA); fall back to torch GEMM there.
        # TODO GFX1250: Include once skinny GEMM is supported on gfx1250
        and x.dtype in [torch.float16, torch.bfloat16]
        and bias is None
        and m % 4 == 0
        and n == 1
        and k <= 8192
        and k % 8 == 0
    )

    if skinny_gemm_enabled:
        out = ops.LLMM1(weight, x_view, 4)
        return out.view(*x.shape[:-1], weight.shape[0])

    # gfx906 (Z100): LLMM1 is excluded from the gfx906 build (MFMA ISA) and
    # `on_gfx9()` is False here -- upstream defines it as the CDNA2/CDNA3 parts
    # (gfx90a/gfx942/gfx950), *not* as "any gfx9" -- so the `n <= 16` Triton
    # matmul below is taken rather than skipped.  Decode-shaped layers would
    # therefore land on Triton or rocBLAS, which move ~235 GB/s of weights for
    # a [1 x 5120] @ [17408 x 5120] bf16 GEMV (24% of HBM) and ~24 GB/s for
    # small-N shapes.  The emulated GEMV (csrc/rocm/bf16_emu_gemv.cu) streams
    # the same weights with coalesced loads and an LDS reduction, so it must be
    # tested *before* the Triton branch: measured 2.3-8.3x over Triton at the
    # shapes below (`tools/dcu/s11f_bf16_gemv_probe.py`).  Opt in with
    # VLLM_DCU_USE_BF16_GEMV=1; on failure/unavailability the previous path is
    # used unchanged.
    if (
        on_gfx906()
        and _gfx906_dense_gemv_enabled()
        and _GFX906_GEMV_AVAILABLE
        and x.dtype == weight.dtype
        and x.dtype in (torch.float16, torch.bfloat16)
        and k >= 2
        and k % 2 == 0
        and _gfx906_gemv_shape_ok(n_tokens=n, n_out=m, dtype=x.dtype)
        and x_view.is_contiguous()
        and weight.is_contiguous()
    ):
        out = torch.ops.vllm.bf16_emu_mm(x_view, weight, bias)
        return out.view(*x.shape[:-1], weight.shape[0])

    # NOTE: upstream's cliff is 16. Raising it to 32 (to catch the DFlash2 k=7
    # draft tower at M = 8 x concurrency = 32) was measured on Z100 and is a
    # *regression*: decoding at c=4 went 184 -> 261 ms/step because the target
    # model's large-N linears lose more on the Triton small-M path than the
    # draft tower gains. Leave at 16.
    #
    # S22 (2026-10-16, tools/dcu/s22_draft_gemm_probe.py) measured the same
    # branch the other way round, at the DFlash2 draft tower's real shapes and
    # M = 1..32: the single-config `triton_matmul` below is 1.8-2.9x slower than
    # rocBLAS (`F.linear`) at M = 8 on N = 5120/6144/4096/1536 (2.45x on
    # 5120:17408) and 1.9-4.6x slower at M = 16/32 on every shape, while it wins
    # only on the widest N at the smallest M (34816:5120 at M <= 8: 3.12 vs
    # 4.59 ms).  Decode of the FP8 target runs 81.6 such calls per step
    # (29.8 ms/step, 13.2% of the step, see OPTIMIZATION_RESULTS.md 12.6b), so
    # the M >= 4 small-N band is worth rerouting.  Opt in with
    # VLLM_DCU_SMALLM_GEMM=rocblas; the default keeps upstream's choice.
    small_m_rocblas = (
        on_gfx906()
        and _gfx906_smallm_rocblas_enabled()
        and ((n >= 4 and m <= _GFX906_ROCBLAS_MAX_N) or n >= 16)
    )
    if n <= 16 and not on_gfx9() and not small_m_rocblas:
        out = triton_matmul(x_view, weight).view(*x.shape[:-1], weight.shape[0])
        if bias is not None:
            out = out + bias
        return out

    # wvSplitK allows row strides. Perf impact is only validated on gfx11.
    weight_ok_for_wvsplitk = weight.is_contiguous() or (
        on_gfx11() and weight.stride(1) == 1
    )

    use_wvsplitk = (
        skinny_gemm_enabled
        and weight_ok_for_wvsplitk
        and (bias is None or bias.is_contiguous())
        and (m == 1 or m > 8)
        and 0 < n <= 5
    )

    use_llmm1 = (
        skinny_gemm_enabled
        and weight.is_contiguous()
        and bias is None
        and m % 4 == 0
        and n == 1
        and k <= 8192
    )

    if use_wvsplitk:
        # The skinny kernels assume contiguous K elements. A shape-preserving
        # reshape can retain a transposed activation's non-contiguous strides.
        # Note: Only build that view inside the branches that consume it.
        x_view = x.reshape(-1, x.size(-1)).contiguous()
        out = ops.wvSplitK(weight, x_view, cu_count, bias)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    if use_llmm1:
        x_view = x.reshape(-1, x.size(-1)).contiguous()
        out = ops.LLMM1(weight, x_view, 4)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    if rocm_aiter_ops.is_tgemm_enabled():
        from aiter.tuned_gemm import tgemm

        return tgemm.mm(x, weight, bias)

    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # dispatch_unquantized_gemm() picks this backend for any ROCm build,
    # regardless of the tensor's actual device; the custom op below is only
    # registered for the CUDA/HIP dispatch key, so route CPU tensors (e.g.
    # from unit tests that build layers without moving them to the GPU)
    # through the device-agnostic fallback instead of hard-crashing.
    if not x.is_cuda:
        return default_unquantized_gemm(layer, x, weight, bias)
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm",
    op_func=rocm_unquantized_gemm_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)


@functools.cache
def warmup_rocm_skinny_gemm_workspaces(device: torch.device) -> None:
    """Eagerly allocate wvSplitKrc's per-device split-K workspace pool.

    wvSplitKrc partitions one per-device allocation into ``kWvSlots`` slots
    (csrc/rocm/skinny_gemms.cu) and hands each stream one on first use, so that
    two streams never share the split-K partials and counters.

    The pool is otherwise created lazily on the first qualifying GEMM
    (csrc/rocm/skinny_gemms.cu), which can be the first real request — after
    the KV cache backing buffer exists. If it landed in that segment's rounding
    tail, it would pin the entire segment at engine shutdown; it could also land
    inside a cudagraph capture, where it would be taken from the graph's private
    pool and its zero-fill would become a replayed graph node.
    """
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return
    try:
        x = torch.zeros(16, 1024, dtype=torch.bfloat16, device=device)
        weight = torch.zeros(32, 1024, dtype=torch.bfloat16, device=device)
        ops.wvSplitKrc(x, weight, num_compute_units())
    except Exception:
        logger.debug("wvSplitKrc workspace warmup failed", exc_info=True)


# Above this weight size, oneDNN's onednn_mm consistently matches or beats
# the SGL AMX kernel once M grows past decode-sized batches, and is within
# noise of it at decode-sized M -- so larger weights default to oneDNN
# rather than SGL. 1 MiB comfortably covers MoE router/gate weights (e.g.
# (2048, 128) .. (2880, 32) bf16/fp16, 180-720 KiB) while staying well below
# any dense qkv/o_proj/gate_up/down/lm_head projection in practice. This
# threshold is derived from bf16/fp16 unquantized dense-GEMM benchmarks only,
# so it does not apply to the int8 scaled_mm path below.
_CPU_SGL_GEMM_MAX_WEIGHT_BYTES = 1 * 1024 * 1024


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    if not torch.cpu._is_amx_tile_supported() or dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.int8,
    ):
        return False
    if dtype == torch.float16 and not torch.cpu._is_amx_fp16_supported():
        # AMX-BF16/INT8 (amx_tile) and AMX-FP16 are separate CPU ISA
        # extensions -- e.g. Sapphire/Emerald Rapids expose the former but
        # not the latter -- and can_use_brgemm<at::Half> (gemm.h) always
        # attempts brgemm for fp16 regardless of M, so this needs its own
        # capability check rather than piggybacking on amx_tile.
        return False
    if dtype == torch.int8:
        # int8_scaled_mm_with_quant requires the packed weight to stay int8
        # (gemm_int8.cpp); convert_weight_packed's N < TILE_N fallback
        # returns a float32 tensor instead (gemm.cpp), which would trip
        # that check, so N must be a full TILE_N tile here.
        return k % 32 == 0 and n % 16 == 0
    if n * k * dtype.itemsize > _CPU_SGL_GEMM_MAX_WEIGHT_BYTES:
        return False
    if n < 16:
        # convert_weight_packed transposes to fp32 instead of VNNI-packing
        # when N < TILE_N (gemm.cpp), and weight_packed_linear detects that
        # (via the packed weight's dtype) and routes to its fp32/brgemm
        # fallback kernel -- no N/K alignment required in that regime.
        return True
    return k % 32 == 0 and n % 16 == 0


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    # skip for missing layers
    if layer.weight.is_meta:
        layer.cpu_linear = torch.nn.functional.linear
        return

    # Skip CPU GEMM dispatch for non-2D weights (e.g. MoE 3D expert weights).
    # These layers are handled by their own specialized methods.
    if layer.weight.ndim != 2:
        # This is not a linear layer.
        # For now it should be a causal_conv1d op or MoE 3D expert weights.
        # Causal-conv weights use [dim, 1, width].
        # The C++ causal_conv1d kernels support aarch64 with bf16 and
        # x86 CPUs that use VDPBF16PS (no AMX tiles). So the
        # weight prepack applies to aarch64 with bf16 and any AVX-512BF16 CPU,
        # not just AMX (e.g. AMD Zen5/Turin).
        is_causal_conv1d_weight = (
            layer.weight.ndim == 3
            and layer.weight.size(1) == 1
            and layer.weight.size(2) == 4
            and layer.weight.dtype == torch.bfloat16
        )
        is_arm_bf16 = (
            current_platform.get_cpu_architecture() == CpuArchEnum.ARM
            and torch.cpu.get_capabilities().get("bf16", False)
        )
        if (
            (is_arm_bf16 or torch.cpu._is_avx512_bf16_supported())
            and is_causal_conv1d_weight
            and hasattr(ops, "causal_conv1d_weight_pack")
        ):
            # prepack conv weight
            unpacked = (
                layer.weight.view(
                    layer.weight.size(0),
                    layer.weight.size(2),
                )
                .contiguous()
                .clone()
            )
            # Stash the un-packed (dim, width) weight so the speculative-decode
            # and unsupported-layout fallbacks can still use ordinary weights.
            layer._cpu_unpacked_conv_weight = unpacked
            layer.weight.data = ops.causal_conv1d_weight_pack(unpacked)
        return

    N, K = layer.weight.size()
    dtype = layer.weight.dtype

    # Zen CPU path: zentorch_linear_unary with optional eager weight prepacking.
    if current_platform.is_zen_cpu() and hasattr(
        torch.ops.zentorch, "zentorch_linear_unary"
    ):
        zen_weight = layer.weight.detach()
        is_prepacked = False

        if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
            torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
        ):
            zen_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                zen_weight
            )
            is_prepacked = True

        layer.cpu_linear = lambda x, weight, bias, _p=is_prepacked: (
            torch.ops.zentorch.zentorch_linear_unary(
                x, zen_weight, bias, is_weight_prepacked=_p
            )
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(
                torch.empty(0, dtype=dtype, device=layer.weight.device),
                requires_grad=False,
            )
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
            is_prepacked,
        )
        return

    # Small weights (e.g. MoE router/gate projections, where N is the expert
    # count rather than a hidden-size-scaled dimension) never reach oneDNN's
    # compute-bound regime, no matter how large the batch gets: SGL's lower
    # per-call dispatch overhead wins consistently across the full measured
    # M range. Larger dense projections (qkv/o_proj/gate_up/down/lm_head)
    # cross over to favoring oneDNN once batch size grows past decode-sized
    # M, so they keep using oneDNN below.
    if check_cpu_sgl_kernel(N, K, dtype):
        # For small size GEMM, packed_weight might be float32
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            layer.bias.data = layer.bias.to(torch.float32)
        layer.cpu_linear = lambda x, weight, bias: ops.weight_packed_linear_cpu(
            x,
            packed_weight,
            N,
            bias,
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(
                torch.empty(0, dtype=dtype, device=layer.weight.device),
                requires_grad=False,
            )
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using sgl-kernel weight_packed_linear"
        )
        return

    if (
        ops._supports_onednn
        and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
    ):
        try:
            origin_weight = layer.weight
            handler = ops.create_onednn_mm(origin_weight.t(), 32)
            layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias)
            if remove_weight:
                layer.weight = torch.nn.Parameter(
                    torch.empty(0, dtype=dtype, device=layer.weight.device),
                    requires_grad=False,
                )
            logger.debug_once("CPU unquantized GEMM dispatch: using oneDNN onednn_mm")
            return
        except RuntimeError as e:
            logger.warning_once(
                "Failed to create oneDNN linear, fallback to torch linear."
                f" Exception: {e}"
            )

    # fallback case
    layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
        x, weight, bias
    )
    logger.debug_once(
        "CPU unquantized GEMM dispatch: using torch.nn.functional.linear (fallback)"
    )


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)


def dispatch_unquantized_gemm(
    linear_backend: str = "auto",
) -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    elif not current_platform.is_cuda():
        return default_unquantized_gemm

    backend_spec = _FLASHINFER_BF16_BACKENDS.get(linear_backend)
    if backend_spec is None:
        return default_unquantized_gemm

    if not backend_spec.is_supported():
        logger.warning_once(
            "--linear-backend=%s requested FlashInfer mm_bf16 backend %r, "
            "but it is unavailable on the current hardware or environment; "
            "using automatic selection for unquantized linear layers.",
            linear_backend,
            backend_spec.flashinfer_backend,
        )
        return default_unquantized_gemm

    logger.info_once(
        "Using FlashInfer %s for eligible unquantized BF16 GEMMs.",
        backend_spec.flashinfer_backend,
    )
    return functools.partial(
        cuda_flashinfer_bf16_gemm,
        vllm_backend=linear_backend,
        pdl=current_platform.is_arch_support_pdl(),
    )
