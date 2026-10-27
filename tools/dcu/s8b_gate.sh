#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# S8b gate: the gfx906 kill-list plus the S8b dependency / cache conventions,
# as executable checks. Run it *inside* a DTK container, because it needs the
# gfx906-only vLLM tree and the self-built gfx906 triton (s8_env.sh sets both):
#
#     docker exec z100-s8 bash /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_gate.sh
#
# Card: S8b measured on card 3; override with S8B_HIP_VISIBLE_DEVICES.
#
# Hard rule 13: never pass --enforce-eager on gfx906.
#
# Exit code 0 iff every check passed.

set -u

_dcu_s8b_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export S8_HIP_VISIBLE_DEVICES="${S8B_HIP_VISIBLE_DEVICES:-3}"
# shellcheck disable=SC1091
. "${_dcu_s8b_dir}/s8_env.sh"

cd "${VLLM_DIR}" || exit 1

OUT_DIR="${S8B_OUT_DIR:-/mnt/game/port/s8b}"
mkdir -p "${OUT_DIR}" 2>/dev/null || true

echo "===== S8b gate $(date -Is) ====="
echo "host python : $(python -V 2>&1)"
echo "VLLM_DIR    : ${VLLM_DIR}"

echo
echo "----- 1) kill-list / convention checks -----"
python - <<'PY'
import glob
import sys


def check(name, ok, detail=""):
    print(f"CHECK {name:34s} {'PASS' if ok else 'FAIL'} {detail}")
    return bool(ok)


results = []

import torch  # noqa: E402  (must precede `import vllm`, hard rule 8)
import vllm.envs as envs  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.platforms import current_platform, rocm  # noqa: E402
from vllm.platforms.rocm import (  # noqa: E402
    get_cdna_version,
    on_gfx906,
    on_gfx9,
    use_rocm_custom_paged_attention,
)

print(f"  device = {torch.cuda.get_device_name(0)} "
      f"{torch.cuda.get_device_properties(0).gcnArchName}")

results.append(check(
    "on_gfx906/on_gfx9/cdna_version",
    on_gfx906() and not on_gfx9() and get_cdna_version() == 0,
    f"on_gfx906={on_gfx906()} on_gfx9={on_gfx9()} cdna_version={get_cdna_version()}",
))

# 2) skinny GEMM: runtime probe plus the env backstop.
results.append(check(
    "ROCM_SKINNY_GEMM_ENABLED",
    ops.ROCM_SKINNY_GEMM_ENABLED is False,
    f"ROCM_SKINNY_GEMM_ENABLED={ops.ROCM_SKINNY_GEMM_ENABLED} "
    f"env={envs.VLLM_ROCM_USE_SKINNY_GEMM!r}",
))
results.append(check(
    "VLLM_ROCM_USE_SKINNY_GEMM=0",
    not bool(envs.VLLM_ROCM_USE_SKINNY_GEMM),
    f"env read = {envs.VLLM_ROCM_USE_SKINNY_GEMM!r}",
))

# 3) _rocm_C registration surface: the upstream paged_attention (which stays
#    unreachable on gfx906, see the next check) plus the fork's own gfx906
#    kernels.  The two w8a8_sdot4 ops are the P2 additions (S11e) and are
#    opt-in behind VLLM_DCU_USE_INT8_SDOT4=1, so they belong on the allow-list:
#    the point of this check is "no *other* ROCm kernel sneaks in", not "the
#    surface is frozen at the S8b revision".  (S11l: the S8b text asserted
#    `== [paged_attention]`, which became stale as soon as S11e registered the
#    two sdot4 ops -- it failed 14/1 regardless of the sdot4 switch.)
names = sorted(
    s.name for s in torch._C._jit_get_all_schemas() if s.name.startswith("_rocm_C::")
)
_ALLOWED_ROCM_C = [
    "_rocm_C::paged_attention",
    "_rocm_C::w8a8_sdot4_gemm",
    "_rocm_C::w8a8_sdot4_gemv",
]
results.append(check(
    "_rocm_C ops == allow-list(paged_attention,sdot4_*)",
    names == _ALLOWED_ROCM_C,
    f"{names}",
))
results.append(check(
    "use_rocm_custom_paged_attention=False",
    current_platform.is_rocm()
    and not use_rocm_custom_paged_attention(torch.float16, 128, 16, 4, 8192, 0, "auto"),
    "gfx906 early return",
))

# 4) aiter: found but unsupported, and every sub-switch short-circuited to None.
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops  # noqa: E402

aiter_switches = {
    n: getattr(rocm_aiter_ops, n)()
    for n in (
        "is_enabled",
        "is_linear_enabled",
        "is_mha_enabled",
        "is_mla_enabled",
        "is_fused_moe_enabled",
        "is_triton_unified_attn_enabled",
    )
}
results.append(check(
    "is_aiter_found_and_supported=False",
    is_aiter_found_and_supported() is False,
    f"switches={aiter_switches}",
))

# 5) flash-attn unavailable.
from vllm.attention.utils.fa_utils import (  # noqa: E402
    is_flash_attn_varlen_func_available,
)

results.append(check(
    "flash-attn unavailable",
    is_flash_attn_varlen_func_available() is False,
    "",
))

# 6) hipBLASLt: no gfx906 target code in the DTK image.
_hipblaslt = glob.glob("/opt/dtk/lib/hipblaslt/library/*gfx906*")
_hipblaslt_all = sorted(
    p.rsplit("/", 1)[-1] for p in glob.glob("/opt/dtk/lib/hipblaslt/library/*.co")
)
results.append(check(
    "no gfx906 hipBLASLt kernels",
    not _hipblaslt,
    f"gfx906={_hipblaslt} library={_hipblaslt_all}",
))

# 7) FP8: per-tensor / per-channel must be refused loudly; block-scaled is the
#    only accepted path. Hard rule 18 (S9): on gfx906 it is emulated in software
#    and ~36x slower than fp16, so it must warn once and still be *selected*
#    (not refused). Numerics verified in S8b; see the archive.
import logging  # noqa: E402

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel  # noqa: E402
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    GroupShape,
    QuantKey,
    ScaleDesc,
)

_fp8 = current_platform.fp8_dtype()


def _key(shape, static=False):
    return QuantKey(dtype=_fp8, scale=ScaleDesc(torch.float32, static, shape))


def _select(akey, wkey):
    try:
        return type(init_fp8_linear_kernel(
            activation_quant_key=akey,
            weight_quant_key=wkey,
            input_dtype=torch.float16,
            out_dtype=torch.float16,
            weight_shape=(256, 512),
        )).__name__
    except ValueError as exc:
        return f"ValueError: {str(exc).splitlines()[0]}"


_fp8_warn: list[str] = []


class _WarningCapture(logging.Handler):
    def emit(self, record):
        _fp8_warn.append(record.getMessage())


logging.getLogger(
    "vllm.model_executor.kernels.linear.scaled_mm.triton"
).addHandler(_WarningCapture())

with set_current_vllm_config(VllmConfig()):
    _block = _select(_key(GroupShape(1, 128)), _key(GroupShape(128, 128)))
    _per_tensor_dyn = _select(_key(GroupShape.PER_TOKEN), _key(GroupShape.PER_TENSOR))
    _per_tensor_static = _select(
        _key(GroupShape.PER_TENSOR, static=True), _key(GroupShape.PER_TENSOR)
    )
    _per_channel = _select(_key(GroupShape.PER_TOKEN), _key(GroupShape.PER_CHANNEL))

results.append(check(
    "FP8 block-scaled -> Triton",
    _block == "TritonFp8BlockScaledMMKernel",
    f"selected {_block} (numerics verified, see s8b_fp8_block_probe.py)",
))
results.append(check(
    "FP8 block-scaled warns once on gfx906",
    any("no native FP8 support" in _m for _m in _fp8_warn),
    f"{len(_fp8_warn)} warning(s): "
    + (f"{_fp8_warn[0][:80]}..." if _fp8_warn else "<none>"),
))
for _label, _got in (
    ("FP8 per-tensor (dynamic) refused", _per_tensor_dyn),
    ("FP8 per-tensor (static) refused", _per_tensor_static),
    ("FP8 per-channel refused", _per_channel),
):
    results.append(check(_label, _got.startswith("ValueError"), _got[:110]))

# 8) cache root convention (hard rule 17).
import os  # noqa: E402

_cache_root = os.environ.get("VLLM_CACHE_ROOT", "")
results.append(check(
    "VLLM_CACHE_ROOT under /mnt/game",
    _cache_root.startswith("/mnt/game"),
    f"VLLM_CACHE_ROOT={_cache_root or '<unset>'}",
))

# 9) compressed-tensors >= 0.15.0.1 (hard rule 16).
import importlib.metadata as _md  # noqa: E402

try:
    _ct = _md.version("compressed-tensors")
except Exception as exc:  # pragma: no cover
    _ct = f"<{type(exc).__name__}>"
results.append(check(
    "compressed-tensors >= 0.15.0.1",
    tuple(int(x) for x in _ct.split(".")[:3]) >= (0, 15, 0) if _ct[0].isdigit() else False,
    f"installed {_ct}",
))

print()
print(f"CHECK_SUMMARY pass={sum(results)} fail={len(results) - sum(results)}")
sys.exit(0 if all(results) else 1)
PY
RC1=$?
if [ "${RC1}" -ne 0 ]; then
  echo "----- 1) FAILED (rc=${RC1}) -----"
else
  echo "----- 1) OK -----"
fi

echo
echo "----- 2) FP8 block-scaled numeric probe (full sweep) -----"
python "${_dcu_s8b_dir}/s8b_fp8_block_probe.py" 2>&1 | tee "${OUT_DIR}/s8b-gate-fp8-probe.log" | tail -12
RC2=${PIPESTATUS[0]}
echo "----- 2) rc=${RC2} (log ${OUT_DIR}/s8b-gate-fp8-probe.log) -----"

echo
echo "----- 3) W4A16 kernel-selection test -----"
python -c "import torch, pytest, sys; sys.exit(pytest.main(['-q','--no-header','--noconftest','-p','no:cacheprovider','tests/kernels/quantization/test_w4a16_kernel_selection.py']))" \
  2>&1 | tee "${OUT_DIR}/s8b-gate-w4a16-sel.log" | tail -6
RC3=${PIPESTATUS[0]}
echo "----- 3) rc=${RC3} (log ${OUT_DIR}/s8b-gate-w4a16-sel.log) -----"

echo
echo "----- 4) informational: _C_stable_libtorch identity (hard rule 15) -----"
_stable="${VLLM_DIR}/vllm/_C_stable_libtorch.abi3.so"
if [ -f "${_stable}" ]; then
  echo "  md5 $(md5sum "${_stable}" | cut -d' ' -f1)  size $(stat -c%s "${_stable}")"
  echo "  S7c-fixed reference: 93bf7b4a391ac97e2f67cc7141633cb1 40691216"
else
  echo "  [WARN] ${_stable} not found"
fi

echo
echo "===== S8b gate summary ====="
echo "  conventions/kill-list : rc=${RC1}"
echo "  FP8 block-scaled      : rc=${RC2}"
echo "  W4A16 selection test  : rc=${RC3}"
if [ "${RC1}" -eq 0 ] && [ "${RC2}" -eq 0 ] && [ "${RC3}" -eq 0 ]; then
  echo "  RESULT: ALL GREEN"
  exit 0
fi
echo "  RESULT: FAILED"
exit 1
