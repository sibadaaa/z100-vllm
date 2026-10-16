#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18: injected into vLLM worker processes (via a site-packages .pth) to log the
# speculative-decoding arguments handed to the gfx906 mamba/GDN update ops.
#
#   installer: /usr/local/lib/python3.10/dist-packages/zz_s18_args.pth
#              -> sys.path.insert(0, "/mnt/game/s18"); import s18_args_patch
#   knobs: S18_ARGS_LOG (default /mnt/game/s18/s18_args.log), S18_ARGS_N (12)
#
# Rationale (S14g): PYTHONPATH / sitecustomize cannot reach VLLM::Worker_TP; the
# only verified channel is a site-packages .pth.  Importing vllm *at .pth time*
# is not possible here (the transformers import chain is not ready yet and the
# engine dies), so a daemon thread patches as soon as the target modules appear
# in sys.modules.
import os
import sys
import threading
import time
import traceback

_LOG = os.environ.get("S18_ARGS_LOG", "/mnt/game/s18/s18_args.log")
_N = int(os.environ.get("S18_ARGS_N", "12"))
_TARGET = "vllm.model_executor.layers.mamba.gdn_linear_attn"
_out = None
_lock = threading.Lock()
_installed = False


def _log(msg: str) -> None:
    global _out
    with _lock:
        try:
            if _out is None:
                _out = open(_LOG, "a", buffering=1)
            _out.write(msg + "\n")
        except Exception:
            pass


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return -1


def _fmt(t) -> str:
    try:
        return f"{tuple(t.shape)}/{str(t.dtype).replace('torch.','')}"
    except Exception:
        return repr(t)


def _patch() -> None:
    global _installed
    cc = sys.modules.get("vllm.model_executor.layers.mamba.ops.causal_conv1d")
    fsg = sys.modules.get("vllm.model_executor.layers.fla.ops.fused_sigmoid_gating")
    gdn = sys.modules.get(_TARGET)
    if cc is None or fsg is None or gdn is None:
        return
    counts = {"conv": 0, "gdn": 0, "state_conv": 0, "state_gdn": 0}
    conv_orig = cc.causal_conv1d_update
    gdn_orig = fsg.fused_sigmoid_gating_delta_rule_update

    def conv_patched(*a, **kw):
        if _STATE_LOG and counts["state_conv"] < _STATE_N:
            _sc = counts["state_conv"]
            counts["state_conv"] = _sc + 1
            cs = _arg(a, kw, 1, "conv_state")
            idx = _arg(a, kw, 5, "conv_state_indices")
            x = _arg(a, kw, 0, "x")
            acc = _arg(a, kw, 6, "num_accepted_tokens")
            qsl = _arg(a, kw, 7, "query_start_loc")
            _log_state(
                f"[state-conv r{_rank()} n={_sc} pass={_sc // 48} layer={_sc % 48} x={_fmt(x)} "
                f"idx={None if idx is None else idx.flatten().tolist()} "
                f"acc={None if acc is None else acc.tolist()} "
                f"qsl={None if qsl is None else qsl.tolist()} "
                f"fp={_slot_fp(cs, idx)} posfp={_slot_pos_fp(cs, idx)} "
            )
        if counts["conv"] < _N:
            counts["conv"] += 1
            qsl, acc = kw.get("query_start_loc"), kw.get("num_accepted_tokens")
            idx = kw.get("conv_state_indices")
            _log(
                f"[conv r{_rank()} #{counts['conv']}] x={_fmt(kw.get('x'))} "
                f"state={_fmt(kw.get('conv_state'))} "
                f"qsl={None if qsl is None else qsl.tolist()} "
                f"acc={None if acc is None else acc.tolist()} "
                f"idx={None if idx is None else idx.tolist()} "
                f"mql={kw.get('max_query_len')}"
            )
        return conv_orig(*a, **kw)

    def gdn_patched(*a, **kw):
        if _STATE_LOG and counts["state_gdn"] < _STATE_N:
            _sg = counts["state_gdn"]
            counts["state_gdn"] = _sg + 1
            st = kw.get("initial_state")
            idx = kw.get("ssm_state_indices")
            cu = kw.get("cu_seqlens")
            acc = kw.get("num_accepted_tokens")
            q = kw.get("q")
            _log_state(
                f"[state-gdn r{_rank()} n={_sg} pass={_sg // 48} layer={_sg % 48} q={_fmt(q)} "
                f"idx={None if idx is None else idx.flatten().tolist()} "
                f"acc={None if acc is None else acc.tolist()} "
                f"cu={None if cu is None else cu.tolist()} "
                f"fp={_slot_fp(st, idx)}"
            )
        if counts["gdn"] < _N:
            counts["gdn"] += 1
            cu, acc = kw.get("cu_seqlens"), kw.get("num_accepted_tokens")
            idx = kw.get("ssm_state_indices")
            _log(
                f"[gdn r{_rank()} #{counts['gdn']}] q={_fmt(kw.get('q'))} "
                f"state={_fmt(kw.get('initial_state'))} "
                f"cu={None if cu is None else cu.tolist()} "
                f"acc={None if acc is None else acc.tolist()} "
                f"idx={None if idx is None else idx.tolist()} "
                f"inplace={kw.get('inplace_final_state')}"
            )
        return gdn_orig(*a, **kw)

    cc.causal_conv1d_update = conv_patched
    fsg.fused_sigmoid_gating_delta_rule_update = gdn_patched
    gdn.causal_conv1d_update = conv_patched
    gdn.fused_sigmoid_gating_delta_rule_update = gdn_patched
    _installed = True
    _log(f"[s18-args] installed r{_rank()} pid={os.getpid()} log={_LOG} n={_N}")


_LAYER_LOG = os.environ.get("S18_LAYER_LOG", "")
_STATE_LOG = os.environ.get("S18_STATE_LOG", "")
_STATE_N = int(os.environ.get("S18_STATE_N", "400"))
_LAYER_N = int(os.environ.get("S18_LAYER_N", "256"))
_layer_state = {"n": 0, "hooked": False}


def _log_state(msg: str) -> None:
    try:
        with open(_STATE_LOG, "a", buffering=1) as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _slot_fp(t, idx) -> str:
    """Fingerprint of the slots named by `idx` inside pool `t` (view-agnostic)."""
    try:
        if t is None or idx is None:
            return "n/a"
        slots = sorted({int(s) for s in idx.flatten().tolist() if int(s) >= 0})
        return "{" + ", ".join(
            f"{s}:{float(t[s].float().abs().sum()):.4f}" for s in slots) + "}"
    except Exception as exc:
        return f"err:{type(exc).__name__}"


def _slot_pos_fp(t, idx) -> str:
    """Per-state-position sums for the conv slots named by `idx`."""
    try:
        if t is None or idx is None:
            return "n/a"
        slots = sorted({int(s) for s in idx.flatten().tolist() if int(s) >= 0})
        out = []
        for s in slots:
            row = t[s].float()
            pos = row.sum(dim=0).tolist() if row.dim() > 1 else [float(row.sum())]
            out.append(f"{s}[" + ",".join(f"{v:.1f}" for v in pos) + "]")
        return " ".join(out)
    except Exception as exc:
        return f"err:{type(exc).__name__}"


def _arg(a, kw, pos, name):
    if name in kw and kw[name] is not None:
        return kw[name]
    return a[pos] if len(a) > pos else None


def _log_layer(msg: str) -> None:
    try:
        with open(_LAYER_LOG, "a", buffering=1) as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _patch_layers() -> None:
    """Per-layer hidden-state fingerprint (S18 layer bisect).

    Enabled only when S18_LAYER_LOG is set.  Records, for the first
    S18_LAYER_N decoder-layer calls: the call index, the `positions` vector
    (which pins the context of every row) and mean/std/max of the whole hidden
    state plus row 0, so two arms can be aligned by position and compared layer
    by layer.  Stats are taken from the layer INPUT (before the call).
    """
    mod = sys.modules.get("vllm.model_executor.models.qwen3_next")
    if mod is None or _layer_state["hooked"]:
        return
    cls = getattr(mod, "Qwen3NextDecoderLayer", None)
    if cls is None:
        return
    orig = cls.forward

    def fwd(self, *a, **kw):
        n = _layer_state["n"]
        if n < _LAYER_N:
            _layer_state["n"] = n + 1
            try:
                hs = kw.get("hidden_states", a[0] if len(a) > 0 else None)
                pos = kw.get("positions", a[2] if len(a) > 2 else None)
                h = hs.detach().float()
                row0 = h[0]
                _log_layer(
                    f"[layer r{_rank()} n={n} pos={None if pos is None else pos.tolist()} "
                    f"shape={tuple(h.shape)} mean={h.mean().item():.6f} "
                    f"std={h.std().item():.6f} max={h.abs().max().item():.6f} "
                    f"r0mean={row0.mean().item():.6f} r0std={row0.std().item():.6f} "
                    f"r0max={row0.abs().max().item():.6f} "
                    f"r0_0={row0[0].item():.6f} r0_1={row0[1].item():.6f}"
                )
            except Exception:
                pass
        return orig(self, *a, **kw)

    cls.forward = fwd
    _layer_state["hooked"] = True
    _log(f"[s18-args] layer hook installed r{_rank()} pid={os.getpid()} "
         f"log={_LAYER_LOG} n={_LAYER_N}")


def _watch() -> None:
    for _ in range(12000):  # ~10 min at 50 ms
        try:
            if _TARGET in sys.modules:
                _patch()
                if _LAYER_LOG:
                    _patch_layers()
                if _installed and (_layer_state["hooked"] or not _LAYER_LOG):
                    return
        except Exception:
            _log("[s18-args] PATCH FAILED:\n" + traceback.format_exc())
            return
        time.sleep(0.05)


try:
    threading.Thread(target=_watch, daemon=True).start()
except Exception:
    _log("[s18-args] THREAD START FAILED:\n" + traceback.format_exc())
