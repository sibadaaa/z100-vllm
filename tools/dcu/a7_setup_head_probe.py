#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A7 (S11a): ``setup.py::_is_hip()`` under a forced non-DTK condition.

The S4 harness (``check_setup_dtk.py``) injects ``_DTK_PREFIX =
_hygon_dtk_prefix()`` and can therefore only run against the fork's
``setup.py``; on the HEAD worktree that symbol does not exist and the harness
aborts with ``NameError``.  This script extracts ``_is_hip`` from *any* tree
with ``ast`` and evaluates it with ``_is_hygon_dtk() -> False`` (the "DTK is
not installed" condition A7 cares about), comparing it against HEAD's formula
``(target_device in ("cuda", "rocm")) and torch.version.hip is not None``.

Usage:
    python3 tools/dcu/a7_setup_head_probe.py /path/to/tree
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types


def load_is_hip(tree: str):
    path = pathlib.Path(tree) / "setup.py"
    module = ast.parse(path.read_text())
    body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_is_hip"]
    if len(body) != 1:
        raise SystemExit(f"{path}: expected exactly one _is_hip(), got {len(body)}")
    namespace = {
        "torch": types.SimpleNamespace(version=types.SimpleNamespace(hip=None)),
        "VLLM_TARGET_DEVICE": "cpu",
        # The whole point: DTK is absent, so the fork's DTK fallback must not
        # run and must not need its helpers.
        "_is_hygon_dtk": lambda: False,
        "_DTK_PREFIX": None,
        "_DTK_HIP_FALLBACK_WARNED": False,
        "_dtk_version": lambda: None,
        "logger": types.SimpleNamespace(warning=lambda *a, **k: None),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def head_formula(target_device: str, hip: str | None) -> bool:
    return (target_device in ("cuda", "rocm")) and hip is not None


def main() -> int:
    tree = sys.argv[1] if len(sys.argv) > 1 else "."
    namespace = load_is_hip(tree)
    checks = 0
    failures = 0
    for target_device in ("cuda", "rocm", "cpu", "empty", "xpu", "tpu"):
        for hip in (None, "6.3.26113"):
            namespace["VLLM_TARGET_DEVICE"] = target_device
            namespace["torch"].version.hip = hip
            got = namespace["_is_hip"]()
            want = head_formula(target_device, hip)
            checks += 1
            status = "PASS" if got == want else "FAIL"
            if got != want:
                failures += 1
            print(
                f"{status}  _is_hip(target={target_device!r}, hip={hip!r}) = {got}, "
                f"HEAD formula = {want}"
            )
    print(f"\n{checks - failures}/{checks} combos match HEAD's formula ({tree})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
