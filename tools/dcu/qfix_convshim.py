"""Diagnostic shim (qfix): dump/compare causal_conv1d_fn calls in situ.

Modes (env VLLM_QFIX_CONVSHIM_MODE):
  dump    - save inputs/outputs of the first N calls per process to disk
  compare - run the gfx906 fallback reference and log the diff
  off     - inactive (default when env unset)
Loaded via a .pth in the shadow container's site-packages.
"""
import os
import sys

_MODE = os.getenv("VLLM_QFIX_CONVSHIM_MODE", "off").strip().lower()
if os.getenv("VLLM_QFIX_CONVSHIM", "0").lower() in ("1", "true", "yes", "on"):
    _MODE = "compare"
_MAX = int(os.getenv("VLLM_QFIX_CONVSHIM_CALLS", "8"))
_DUMP_DIR = os.getenv("VLLM_QFIX_CONVSHIM_DIR", "/mnt/game/qfix/dump")
_TARGETS = (
    "vllm.model_executor.layers.mamba.ops.causal_conv1d",
    "vllm.model_executor.models.qwen3_next",
    "vllm.model_executor.layers.mamba.gdn_linear_attn",
)
_state = {"n": 0}


def _log(msg):
    print(f"[convshim] {msg}", flush=True)


def _capture(p, out_main):
    import torch

    n = _state["n"]
    os.makedirs(_DUMP_DIR, exist_ok=True)
    conv_states = p.get("conv_states")
    cache_indices = p.get("cache_indices")
    cs_sel = None
    if conv_states is not None and cache_indices is not None:
        try:
            cs_sel = conv_states[cache_indices.long()].clone()
        except Exception:
            cs_sel = None
    meta = p.get("metadata")
    meta_dump = {}
    if meta is not None:
        for k in ("batch_ptr", "token_chunk_offset_ptr"):
            v = getattr(meta, k, None)
            if isinstance(v, torch.Tensor):
                meta_dump[k] = v.clone()
        nd = getattr(meta, "nums_dict", None)
        if isinstance(nd, dict):
            meta_dump["nums_dict"] = {
                bm: {
                    kk: (vv.clone() if isinstance(vv, torch.Tensor) else vv)
                    for kk, vv in entry.items()
                }
                for bm, entry in nd.items()
            }
        for k in (
            "block_idx_first_scheduled_token",
            "block_idx_last_scheduled_token",
            "initial_state_idx",
            "num_computed_tokens",
        ):
            v = getattr(meta, k, None)
            if isinstance(v, torch.Tensor):
                meta_dump[k] = v.clone()
            elif v is not None:
                meta_dump[k] = v
    path = os.path.join(_DUMP_DIR, f"call-{os.getpid()}-{n}.pt")
    torch.save(
        {
            "x": p["x"].clone(),
            "weight": p["weight"].clone(),
            "bias": None if p.get("bias") is None else p["bias"].clone(),
            "conv_states_cpu": None,
            "cache_indices": None if cache_indices is None else cache_indices.clone(),
            "cs_sel": None if cs_sel is None else cs_sel.cpu(),
            "query_start_loc": None if p.get("query_start_loc") is None else p["query_start_loc"].clone(),
            "has_initial_state": None
            if p.get("has_initial_state") is None
            else p["has_initial_state"].clone(),
            "activation": p.get("activation"),
            "pad_slot_id": p.get("pad_slot_id"),
            "block_size_to_align": p.get("block_size_to_align", 0),
            "out": None if out_main is None else out_main.clone(),
            "meta": meta_dump,
        },
        path,
    )
    _log(f"dumped call#{n} -> {path}")


def _wrap(orig):
    def shim(*args, **kwargs):
        import inspect

        import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc

        n = _state["n"]
        if n >= _MAX:
            return orig(*args, **kwargs)
        if _MODE == "dump":
            out_main = orig(*args, **kwargs)
            try:
                sig = inspect.signature(orig)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                _capture(bound.arguments, out_main)
            except Exception:
                import traceback

                traceback.print_exc()
            _state["n"] = n + 1
            return out_main

        out_main = None
        try:
            sig = inspect.signature(orig)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            p = bound.arguments
            x = p["x"]
            conv_states = p.get("conv_states")
            cs_ref = conv_states.clone() if conv_states is not None else None
            out_ref = cc._causal_conv1d_gfx906_fallback(
                x=x,
                weight=p["weight"],
                bias=p.get("bias"),
                conv_states=cs_ref,
                query_start_loc=p.get("query_start_loc"),
                cache_indices=p.get("cache_indices"),
                has_initial_state=p.get("has_initial_state"),
                activation=p.get("activation", "silu"),
                pad_slot_id=p.get("pad_slot_id", -1),
            )
            out_main = orig(*args, **kwargs)
            d = (out_main.float() - out_ref.float()).abs()
            msg = (
                f"call#{n} out max={d.max().item():.3e} "
                f"mean={d.mean().item():.3e} bad1e-3={(d > 1e-3).sum().item()}/{d.numel()}"
            )
            if cs_ref is not None and conv_states is not None:
                ds = (conv_states.float() - cs_ref.float()).abs()
                msg += (
                    f" state max={ds.max().item():.3e} "
                    f"bad={(ds > 1e-3).sum().item()}/{ds.numel()}"
                )
            _log(msg)
            _state["n"] = n + 1
            return out_main
        except Exception:
            import traceback

            traceback.print_exc()
            if out_main is None:
                out_main = orig(*args, **kwargs)
            _state["n"] = n + 1
            return out_main

    shim._qfix_wrapped = True
    return shim


def _patch(mod):
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc

        if mod.__name__ == cc.__name__:
            if not getattr(cc.causal_conv1d_fn, "_qfix_wrapped", False):
                cc.causal_conv1d_fn = _wrap(cc.causal_conv1d_fn)
        elif hasattr(mod, "causal_conv1d_fn"):
            mod.causal_conv1d_fn = cc.causal_conv1d_fn
    except Exception:
        import traceback

        traceback.print_exc()
    return mod


if _MODE in ("dump", "compare"):
    import importlib.abc

    class _Loader(importlib.abc.Loader):
        def __init__(self, real):
            self.real = real

        def create_module(self, spec):
            return self.real.create_module(spec)

        def exec_module(self, module):
            self.real.exec_module(module)
            _patch(module)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _TARGETS:
                return None
            for f in list(sys.meta_path):
                if isinstance(f, _Finder):
                    continue
                spec = f.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _Loader(spec.loader)
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())
    _log(f"active mode={_MODE} max_calls={_MAX} pid={os.getpid()}")
