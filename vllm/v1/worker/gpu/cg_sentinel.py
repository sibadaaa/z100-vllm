# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FULL-cudagraph static-input sentinel (WP-B1/B2 VMFault attribution).

Env-gated by ``VLLM_CG_SENTINEL=1``; default off (no behavior change).

Mechanism (post-replay poison + pre-replay check):

1. At FULL graph capture end, register the static input buffers whose
   pointers the graph baked.
2. Only poisonable buffers get poisoned after each replay: buffers whose
   staging rewrites the full padded extent every replay (GDN spec-decode
   staging). Buffers whose padding rows are intentionally stale across
   replays (input_ids/seq_lens/aligned indices — read by no real work
   because padded rows have zero query length) are never poisoned; a magic
   there would fabricate an OOB read of our own.
3. Before the next FULL replay (per-replay staging already refreshed what
   it should), any poisonable row still holding magic proves that row has
   no per-replay writer (H1) -> dump and abort BEFORE replaying.
4. Persistent decision buffers (mamba state_idx/src_col/src_off,
   num_accepted_tokens_gpu, aligned state indices, input block tables) get
   a per-step before/after snapshot diff to attribute in-graph writers (H2).

See docs/design/z100/single-stream/z100-full-decode-vmfault-design.md.
"""

import json
import logging
import os
import threading
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)

_SENTINEL_ON: bool | None = None
_VA_TRACE_STARTED = False


def maybe_start_va_trace() -> None:
    """Start the debug VA tracer (WP-B2 eager-path attribution).

    VLLM_CG_VA_TRACE=1 only: dumps torch.cuda.memory_snapshot() segment
    ranges every 2 s so a fault address from the crash journal can be
    mapped to its owning allocation offline.
    """
    global _VA_TRACE_STARTED
    if _VA_TRACE_STARTED or os.environ.get("VLLM_CG_VA_TRACE", "0") != "1":
        return
    _VA_TRACE_STARTED = True
    path = f"/tmp/cg_vatrace_{os.getpid()}.jsonl"

    def trace_loop() -> None:
        with open(path, "w") as f:
            while True:
                try:
                    segs = [
                        [s["address"], s["total_size"], s.get("segment_type")]
                        for s in torch.cuda.memory_snapshot()
                    ]
                    stats = torch.cuda.memory_stats()
                    f.write(
                        json.dumps(
                            {
                                "t": time.time(),
                                "pid": os.getpid(),
                                "n": len(segs),
                                "allocated": stats.get(
                                    "allocated_bytes.all.current"
                                ),
                                "segs": segs,
                            }
                        )
                        + "\n"
                    )
                    f.flush()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(2.0)

    threading.Thread(target=trace_loop, daemon=True, name="cg-va-trace").start()
    logger.info("[CG_VA_TRACE] dumping allocator segments to %s", path)

_MAGIC_INT32 = -559038737  # 0xDEADBEEF
_MAGIC_INT64 = -2401053088826427905  # 0xDEADBEEFDEADBEEF
_MAGIC_INT16 = -8557  # 0xDEAD
_MAGIC_FLOAT = -12345.0


def sentinel_enabled() -> bool:
    global _SENTINEL_ON
    if _SENTINEL_ON is None:
        _SENTINEL_ON = os.environ.get("VLLM_CG_SENTINEL", "0") == "1"
    return _SENTINEL_ON


def _magic_for(dtype: torch.dtype) -> int | float | None:
    if dtype == torch.int32:
        return _MAGIC_INT32
    if dtype == torch.int64:
        return _MAGIC_INT64
    if dtype == torch.int16:
        return _MAGIC_INT16
    if dtype.is_floating_point:
        return _MAGIC_FLOAT
    # bool / uint8 / other: not poisonable.
    return None


class CGSentinel:
    """Registry + poison/check/diff engine for FULL graph static buffers."""

    def __init__(self) -> None:
        # label -> (tensor, poisonable, rows_kind)  rows_kind: req|tok|sched
        self.entries: dict[str, tuple[torch.Tensor, bool, str]] = {}
        self.decision: dict[str, torch.Tensor] = {}
        self.prev_snapshot: dict[str, torch.Tensor] = {}
        self.step = 0

    # -- registration ----------------------------------------------------

    def register(
        self, label: str, tensor: Any, poison: bool = True, rows: str = "req"
    ) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            return
        if tensor.numel() == 0 or label in self.entries:
            return
        self.entries[label] = (tensor, poison, rows)

    def register_decision(self, label: str, tensor: Any) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            return
        if tensor.numel() == 0 or label in self.decision:
            return
        self.decision[label] = tensor

    # -- poison / check --------------------------------------------------

    def poison_after_replay(self) -> None:
        for tensor, poison, _ in self.entries.values():
            if not poison:
                continue
            magic = _magic_for(tensor.dtype)
            if magic is None:
                continue
            tensor.fill_(magic)

    def check_before_replay(
        self, num_reqs: int, num_tokens: int, num_real: int
    ) -> None:
        self.step += 1
        problems: list[str] = []
        for label, (tensor, poison, rows_kind) in self.entries.items():
            if not poison:
                continue
            magic = _magic_for(tensor.dtype)
            if magic is None:
                continue
            if rows_kind == "sched":
                rows = num_real
            elif rows_kind == "tok":
                rows = num_tokens
            else:
                rows = num_reqs
            flat = tensor.reshape(tensor.shape[0], -1)
            rows = min(rows, flat.shape[0])
            magic_rows = (flat[:rows] == magic).any(dim=-1)
            if magic_rows.any():
                bad = magic_rows.nonzero().flatten().tolist()[:16]
                problems.append(
                    f"  {label}: magic rows {bad} (checked {rows}/"
                    f"{flat.shape[0]}); sample {flat[:rows][magic_rows][:4].tolist()}"
                )
        if problems:
            self._dump(
                "H1 HIT: staged static input rows not refreshed by staging "
                "(still magic) — no per-replay copy for these fields",
                problems,
            )

    # -- decision snapshot diff (H2) --------------------------------------

    def snapshot_before_replay(self) -> None:
        for label, tensor in self.decision.items():
            prev = self.prev_snapshot.get(label)
            if prev is None or prev.shape != tensor.shape:
                prev = torch.empty_like(tensor)
                self.prev_snapshot[label] = prev
            prev.copy_(tensor)

    def diff_after_replay(self, num_reqs: int) -> None:
        problems: list[str] = []
        for label, tensor in self.decision.items():
            prev = self.prev_snapshot.get(label)
            if prev is None:
                continue
            flat = tensor.reshape(tensor.shape[0], -1)
            prev_flat = prev.reshape(prev.shape[0], -1)
            rows = min(num_reqs, flat.shape[0])
            changed = (flat[:rows] != prev_flat[:rows]).any(dim=-1)
            if changed.any():
                bad = changed.nonzero().flatten().tolist()[:16]
                problems.append(
                    f"  {label}: rows changed DURING graph replay: {bad}; "
                    f"before={prev_flat[:rows][changed][:3].tolist()} "
                    f"after={flat[:rows][changed][:3].tolist()}"
                )
        if problems:
            self._dump(
                "H2 HIT: persistent decision buffers written inside the "
                "FULL graph replay (no eager copy corresponds)",
                problems,
            )

    # -- dump --------------------------------------------------------------

    def _dump(self, headline: str, problems: list[str]) -> None:
        lines = [
            f"[CG_SENTINEL] step={self.step} {headline}",
            *problems,
            "  registered buffers:",
            *[
                f"    {lbl}: shape={tuple(t.shape)} dtype={t.dtype} "
                f"ptr={hex(t.data_ptr())} poison={p} rows={r}"
                for lbl, (t, p, r) in self.entries.items()
            ],
            *[
                f"    [decision] {lbl}: shape={tuple(t.shape)} "
                f"dtype={t.dtype} ptr={hex(t.data_ptr())}"
                for lbl, t in self.decision.items()
            ],
        ]
        raise RuntimeError("\n".join(lines))


_SENTINEL: CGSentinel | None = None


def attach_sentinel(owner: Any) -> CGSentinel:
    """Get (or lazily create) the sentinel registry for one graph manager.

    Target and speculator FULL graphs have different static-input sets but
    share some persistent buffers, so each CudaGraphManager instance keeps
    its own registry; the run_fullgraph hook picks it up via the owner.
    """
    global _SENTINEL
    if _SENTINEL is None:
        _SENTINEL = CGSentinel()
    existing = getattr(owner, "_cg_sentinel", None)
    if existing is None:
        existing = CGSentinel()
        owner._cg_sentinel = existing
    return existing


def maybe_register_full_graph_buffers(
    owner: Any,
    input_buffers: Any,
    block_tables: Any,
    model_state: Any,
    attn_groups: list[list[Any]],
    poison_staging: bool = True,
) -> None:
    """Register a graph's static input buffers at capture end.

    Called from ModelCudaGraphManager.capture and the speculator manager's
    capture; no-op unless VLLM_CG_SENTINEL=1. rows kind: ``sched`` buffers
    are refreshed by the scheduler copy (checked over real rows); ``tok``
    buffers are token indexed; ``req`` buffers are request indexed.
    ``poison_staging=False`` (speculator prefill/decode managers) keeps
    those registries check-free: their graphs never stage the GDN spec
    buffers, so leftover magic would be a false H1 hit.
    """
    if not sentinel_enabled():
        return
    s = attach_sentinel(owner)
    ib = input_buffers
    # Shared buffers: refreshed for real rows by the scheduler path
    # (PIECEWISE-safe by construction); their padding rows are intentionally
    # stale across replays and read by no real work (zero query length), so
    # they must NOT be poisoned. Registered for the dump inventory only.
    s.register("sched:input_ids", getattr(ib, "input_ids", None),
               poison=False, rows="sched")
    s.register("sched:positions", getattr(ib, "positions", None),
               poison=False, rows="sched")
    s.register("req:seq_lens", getattr(ib, "seq_lens", None), poison=False)
    s.register("req:idx_mapping", getattr(ib, "idx_mapping", None),
               poison=False)
    s.register("req:query_start_loc", getattr(ib, "query_start_loc", None),
               poison=False)
    for name in ("num_scheduled_tokens", "cu_num_logits", "slot_mappings"):
        t = getattr(ib, name, None)
        if isinstance(t, torch.Tensor):
            s.register(f"req:{name}", t, poison=False)
    input_block_tables = getattr(block_tables, "input_block_tables", None)
    bts = (
        list(input_block_tables.values())
        if isinstance(input_block_tables, dict)
        else list(input_block_tables or [])
    )
    for gid, t in enumerate(bts):
        if isinstance(t, torch.Tensor):
            s.register_decision(f"bt:input_block_table[g{gid}]", t)
    # mamba align decision buffers (snapshot-diffed, not poisoned).
    for name in (
        "num_accepted_tokens_gpu",
        "_mamba_state_idx_gpu",
        "_mamba_src_col_gpu",
        "_mamba_src_off_gpu",
    ):
        t = getattr(model_state, name, None)
        if isinstance(t, torch.Tensor):
            s.register_decision(f"mamba:{name}", t)
    ctx = getattr(model_state, "_mamba_ctx", None)
    aligned = getattr(ctx, "aligned_state_indices", None) if ctx else None
    if isinstance(aligned, torch.Tensor):
        s.register_decision("mamba:aligned_state_indices", aligned)
    # Attention builder staging buffers: the GDN spec-decode path rewrites
    # the full padded extent (values + explicit padding fill) on every
    # staged replay -> the only poisonable surface. The non_spec_* buffers
    # are unused in pure-spec batches (metadata None) -> not poisonable.
    # Speculator managers get poison_staging=False outright: their graphs
    # never stage the GDN spec buffers, so leftover magic there would be a
    # false H1 hit.
    for groups in attn_groups:
        for group in groups:
            builder = group.get_metadata_builder(0)
            poisonable = (
                {
                    "spec_state_indices_tensor",
                    "spec_token_indx",
                    "spec_query_start_loc",
                    "num_accepted_tokens",
                }
                if poison_staging
                else set()
            )
            for name in (
                "spec_state_indices_tensor",
                "non_spec_state_indices_tensor",
                "spec_token_indx",
                "non_spec_token_indx",
                "spec_query_start_loc",
                "non_spec_query_start_loc",
                "num_accepted_tokens",
            ):
                t = getattr(builder, name, None)
                if isinstance(t, torch.Tensor):
                    rows = "tok" if "token_indx" in name else "req"
                    s.register(
                        f"staged:{type(builder).__name__}.{name}",
                        t,
                        poison=name in poisonable,
                        rows=rows,
                    )
    # Forensics: publish every registered VA range so a fault address in the
    # journal can be mapped to its buffer offline.
    for lbl, (t, p, r) in s.entries.items():
        logger.info(
            "[CG_SENTINEL] %s ptr=%#x bytes=%d poison=%s rows=%s",
            lbl,
            t.data_ptr(),
            t.untyped_storage().nbytes(),
            p,
            r,
        )
    for lbl, t in s.decision.items():
        logger.info(
            "[CG_SENTINEL] [decision] %s ptr=%#x bytes=%d",
            lbl,
            t.data_ptr(),
            t.untyped_storage().nbytes(),
        )


# WP-B2 eager-path attribution: start the VA tracer when enabled. Runs at
# import in every process that pulls in this module (the workers, via
# cudagraph_utils); no-op unless VLLM_CG_VA_TRACE=1.
maybe_start_va_trace()
