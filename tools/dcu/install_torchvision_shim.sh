#!/usr/bin/env bash
# Install a minimal `torchvision` import shim into the ROCm 10.1 runtime container.
#
# Why: this tree's vLLM/transformers import chain pulls torchvision at module
# import time even for text-only models -- e.g.
#   vllm/model_executor/models/qwen3_5.py -> .qwen3_vl
#     -> transformers.models.qwen2_vl.image_processing_qwen2_vl
#     -> torchvision.transforms.v2
# A real torchvision cannot be installed here: PyPI's newest release does not
# match the self-built torch 2.16.0.dev20261016 and no ROCm 10.1 build of it
# exists.
#
# Multimodal input is out of scope for this stack (plan Q6), so the shim only
# provides the import surface (InterpolationMode, transforms modules) and makes
# every actual call raise a clear error instead of silently doing nothing.
#
# Usage: bash tools/dcu/install_torchvision_shim.sh [container]
set -euo pipefail
CT=${CT:-z100-vllm-rocm101-rt}
SP=${SP:-/usr/local/lib/python3.12/dist-packages}

docker exec -i -e SP="$SP" "$CT" bash -s <<'INNER'
set -euo pipefail
ROOT="$SP/torchvision"
rm -rf "$ROOT"
mkdir -p "$ROOT/transforms/v2" "$ROOT/ops" "$ROOT/io"

cat > "$ROOT/__init__.py" <<'PY'
"""Minimal torchvision import shim for text-only vLLM deployments.

The real torchvision is not installable for this self-built torch build, but
parts of transformers/vLLM import it at module import time.  Multimodal input
is unsupported on this stack; any actual use raises NotImplementedError.
"""

from enum import IntEnum

__version__ = "0.0.0+shim"
__shim__ = True


class InterpolationMode(IntEnum):
    NEAREST = 0
    NEAREST_EXACT = 1
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5
    LANCZOS = 6


def _unavailable(name, *args, **kwargs):
    raise NotImplementedError(
        f"torchvision.{name} is not available on this stack: torchvision is a "
        "shim (multimodal input is out of scope here). Install a real "
        "torchvision matching torch 2.16 to use it."
    )


def __getattr__(name):
    # Return a placeholder class, not a function: transformers/vLLM use these
    # names in type annotations (e.g. ``EnumType | tvF.SomeName``), which only
    # works with a class object.
    class _Missing:
        def __init__(self, *args, **kwargs):
            _unavailable(name)

    _Missing.__name__ = name
    return _Missing
PY

cat > "$ROOT/transforms/__init__.py" <<'PY'
"""Shim: import surface only (see torchvision/__init__.py)."""

from torchvision import InterpolationMode  # noqa: F401

from . import functional, v2  # noqa: F401


def _unavailable(name, *args, **kwargs):
    raise NotImplementedError(
        f"torchvision.transforms.{name} is not available on this stack "
        "(torchvision is a shim; multimodal is out of scope)."
    )


def __getattr__(name):
    return lambda *a, **k: _unavailable(name, *a, **k)
PY

cat > "$ROOT/transforms/functional.py" <<'PY'
"""Shim: import surface only (see torchvision/__init__.py)."""

from torchvision import InterpolationMode  # noqa: F401


def _unavailable(name, *args, **kwargs):
    raise NotImplementedError(
        f"torchvision.transforms.functional.{name} is not available on this "
        "stack (torchvision is a shim; multimodal is out of scope)."
    )


def __getattr__(name):
    class _Missing:
        def __init__(self, *args, **kwargs):
            _unavailable(name)

    _Missing.__name__ = name
    return _Missing
PY

cat > "$ROOT/transforms/v2/__init__.py" <<'PY'
"""Shim: import surface only (see torchvision/__init__.py)."""

from torchvision import InterpolationMode  # noqa: F401

from . import functional  # noqa: F401


def _unavailable(name, *args, **kwargs):
    raise NotImplementedError(
        f"torchvision.transforms.v2.{name} is not available on this stack "
        "(torchvision is a shim; multimodal is out of scope)."
    )


def __getattr__(name):
    class _Missing:
        def __init__(self, *args, **kwargs):
            _unavailable(name)

    _Missing.__name__ = name
    return _Missing
PY

cat > "$ROOT/transforms/v2/functional.py" <<'PY'
"""Shim: import surface only (see torchvision/__init__.py)."""

from torchvision import InterpolationMode  # noqa: F401


def _unavailable(name, *args, **kwargs):
    raise NotImplementedError(
        f"torchvision.transforms.v2.functional.{name} is not available on "
        "this stack (torchvision is a shim; multimodal is out of scope)."
    )


def __getattr__(name):
    class _Missing:
        def __init__(self, *args, **kwargs):
            _unavailable(name)

    _Missing.__name__ = name
    return _Missing
PY

for sub in ops io; do
  cat > "$ROOT/$sub/__init__.py" <<'PY'
"""Shim: import surface only (see torchvision/__init__.py)."""


def _unavailable(name, *args, **kwargs):
    raise NotImplementedError(
        f"torchvision is shimmed on this stack; {name} is unavailable."
    )


def __getattr__(name):
    class _Missing:
        def __init__(self, *args, **kwargs):
            _unavailable(name)

    _Missing.__name__ = name
    return _Missing
PY
done

echo "torchvision shim installed at $ROOT"
python3 -c "import torchvision; print('shim version', torchvision.__version__)"
python3 -c "from torchvision.transforms import InterpolationMode; print('InterpolationMode', int(InterpolationMode.BICUBIC))"
python3 -c "from torchvision.transforms.v2 import functional as tvF; print('v2.functional ok')"
INNER
