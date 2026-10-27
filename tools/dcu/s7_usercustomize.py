"""S7 runtime shim for the gfx906 / DTK 26.04 environment.

This file is placed on PYTHONPATH by ``tools/dcu/s7_env.sh`` (this fork copy is
listed there), so it runs before *any* vLLM import, including the short-lived
subprocess that ``vllm/model_executor/models/registry.py`` spawns.

It works around two DTK-specific environment problems, both already known from
the project todolist; neither is a vLLM defect:

1. Hard rule 8: on DTK, ``import vllm`` before ``import torch`` aborts with
   ``std::bad_alloc`` (SIGABRT). The registry subprocess does exactly that, so
   torch is imported pre-emptively here.
2. DTK 26.04 publishes no ``torchvision`` wheel for torch 2.11 (only the
   ``torch2100`` build exists, see requirements/dcu.txt). That 2.10 build makes
   ``torch._library.fake_impl.register`` raise
   ``RuntimeError: operator torchvision::nms does not exist`` at import time, and
   vLLM imports ``torchvision.transforms`` from its warmup path
   (``vllm/transformers_utils/processors/minimax_m3.py``). The CPU torchvision
   wheel works with the DTK torch, except for the fake/meta registrations, which
   are irrelevant to inference here.

The shim only installs itself when torchvision is actually broken, so a proper
DTK torchvision build silently takes precedence.
"""

import sys
import types

try:
    import torch  # noqa: F401  (workaround 1, must stay first)
except Exception:  # pragma: no cover - torch is mandatory on this stack
    pass


def _torchvision_meta_registrations_fail() -> bool:
    try:
        import torch  # noqa: F401
        from torch._C import _dispatch_has_kernel_for_dispatch_key as has_kernel

        return not has_kernel("torchvision::nms", "Meta")
    except RuntimeError:
        # "operator torchvision::nms does not exist": the C++ extension did not
        # register its ops against this torch build, which is exactly the case
        # the stub below covers.
        return True
    except Exception:
        return False


try:
    import importlib.util

    if importlib.util.find_spec("torchvision") is not None:
        if _torchvision_meta_registrations_fail():
            # Pretend the meta registrations were already imported and were a
            # no-op, so torchvision's __init__ skips the failing module.
            import torch  # noqa: F401

            _stub = types.ModuleType("torchvision._meta_registrations")
            _stub.defs = {}
            sys.modules["torchvision._meta_registrations"] = _stub
except Exception:  # pragma: no cover - never break interpreter startup
    pass
