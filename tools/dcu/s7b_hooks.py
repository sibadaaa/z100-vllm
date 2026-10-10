# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b layer probe for the gfx906 eager NaN: find the first non-finite tensor.

Imported inside the vLLM worker process through ``LLM.apply_model``
(``vllm/entrypoints/llm.py:602``).  Every module of the loaded model gets a
forward pre/post hook that records the magnitude and the number of non-finite
elements of its inputs and outputs; one JSON line per module per forward pass is
appended to ``$S7B_SCAN`` (default ``/mnt/game/port/s7b/layer_scan.jsonl``).

Why this works in the broken configuration: ``--enforce-eager`` maps to
``CompilationMode.NONE``, so no ``torch.compile`` wrapper sits between the
modules and ``nn.Module.__call__`` still invokes every hook.  The compiled
configuration is deliberately *not* probed this way, because inductor removes
the module boundaries.
"""

import json
import os

import torch

SCAN = os.environ.get("S7B_SCAN", "/mnt/game/port/s7b/layer_scan.jsonl")

# Forward pass bookkeeping.  ``all_passes`` keeps every module line for the
# first N passes (a single prefill + decode pass is enough to see the table),
# after that only rows with a non-finite tensor are emitted.
ST = {"pass": 0, "i": 0, "emit_all_until": 1}


def _tensor_summary(t) -> dict:
    tf = t.detach().float()
    finite = torch.isfinite(tf)
    n = tf.numel()
    nbad = int(n - int(finite.sum().item()))
    amax = float(tf[finite].abs().max().item()) if nbad < n else float("nan")
    return {"n": n, "nbad": nbad, "absmax": amax}


def _tensors(obj, path, out) -> None:
    if isinstance(obj, torch.Tensor):
        out.append((path, obj))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _tensors(v, f"{path}[{i}]", out)


def _summarize(obj, path) -> dict:
    found: list = []
    _tensors(obj, path, found)
    res = {}
    for p, t in found:
        s = _tensor_summary(t)
        res[p] = s
        if s["nbad"] > 0 and "__first_nonfinite__" not in res:
            res["__first_nonfinite__"] = {
                "path": p,
                "nbad": s["nbad"],
                "n": s["n"],
                "dtype": str(t.dtype),
            }
    return res


def _emit(row) -> None:
    os.makedirs(os.path.dirname(SCAN), exist_ok=True)
    with open(SCAN, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def cc_info(model):
    """Return the compilation/custom-op state as seen inside the worker."""
    from vllm.config import get_cached_compilation_config

    cc = get_cached_compilation_config()
    return {
        "mode": str(cc.mode),
        "backend": str(cc.backend),
        "custom_ops": list(cc.custom_ops),
        "enabled_custom_ops": sorted(cc.enabled_custom_ops),
        "disabled_custom_ops": sorted(cc.disabled_custom_ops),
        "cudagraph_mode": str(cc.cudagraph_mode),
    }


def worker_probe(self):
    """``collective_rpc`` entry point: runs with the worker as ``self``.

    Reads the effective compilation/custom-op state from the worker's own
    ``vllm_config`` (``get_cached_compilation_config()`` is not in scope here)
    and installs the layer hooks on the loaded model.
    """
    model = self.get_model()
    cc = self.vllm_config.compilation_config
    info = {
        "mode": str(cc.mode),
        "backend": str(cc.backend),
        "custom_ops": list(cc.custom_ops),
        "enabled_custom_ops": sorted(cc.enabled_custom_ops),
        "disabled_custom_ops": sorted(cc.disabled_custom_ops),
        "cudagraph_mode": str(cc.cudagraph_mode),
    }
    n = _install(model)
    return {"modules": n, "info": info}


def install(model):
    """Install the hooks.  ``model`` is the model inside the vLLM worker."""
    return _install(model)


def _install(model):
    names = {}
    for n, m in model.named_modules():
        names[id(m)] = n or "<root>"

    root = model

    def root_pre(module, args, kwargs):
        ST["pass"] += 1
        ST["i"] = 0
        _emit({"kind": "PASS_BEGIN", "pass": ST["pass"]})

    def pre(module, args, kwargs):
        ST["i"] += 1
        module._s7b_i = ST["i"]
        module._s7b_in = _summarize(list(args) + list(kwargs.values()), "in")

    def post(module, args, output):
        out = _summarize(output, "out")
        row = {
            "kind": "MODULE",
            "pass": ST["pass"],
            "i": module._s7b_i,
            "name": names.get(id(module), type(module).__name__),
            "cls": type(module).__name__,
            "in": module._s7b_in,
            "out": out,
        }
        if "__first_nonfinite__" in out:
            row["first_nonfinite"] = out["__first_nonfinite__"]
        if ST["pass"] <= ST["emit_all_until"] or "first_nonfinite" in row:
            _emit(row)

    root.register_forward_pre_hook(root_pre, with_kwargs=True)
    for m in model.modules():
        m.register_forward_pre_hook(pre, with_kwargs=True)
        m.register_forward_hook(post)
    n = sum(1 for _ in model.named_modules())
    _emit({"kind": "INSTALLED", "modules": n})
    return n


def set_emit_all_until(model, n):
    ST["emit_all_until"] = int(n)
    return ST["emit_all_until"]
