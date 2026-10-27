#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11m: seconds-long reproduction of the `VLLM_DCU_USE_INT8_SDOT4=1` AOT reload
failure ("Source code has changed since the last compilation").

No GPU and no 27B startup: the check that fails is a pure string comparison
between the source snapshot stored inside the AOT artifact and the file on disk.

    python3 tools/dcu/s11m_min_repro.py                      # synthetic module
    python3 tools/dcu/s11m_min_repro.py --artifact <model>   # real AOT artifact

Run it inside the DTK container (it imports vllm -> torch), e.g.

    docker exec z100-s9 bash -lc '. /opt/dtk/env.sh; cd /src; \
      PYTHONPATH=/src:/mnt/game/triton-gfx906/install \
      python tools/dcu/s11m_min_repro.py --artifact "$AOT"'
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
import tempfile

USAGE = __doc__


def _load_temp_module(tmpdir: str, name: str, body: str):
    path = os.path.join(tmpdir, name + ".py")
    with open(path, "w") as f:
        f.write(body)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, path


def _fake_config():
    from types import SimpleNamespace

    return SimpleNamespace(compilation_config=SimpleNamespace(traced_files=set()))


class _InlinedSource:
    """Stand-in for torch._dynamo.package.InlinedSource (hashable)."""

    def __init__(self, module: str, content: str):
        self.module = module
        self.content = content

    def __hash__(self):
        return hash((self.module, self.content))

    def __eq__(self, other):
        return (self.module, self.content) == (other.module, other.content)


def _source_info(module, name: str):
    from types import SimpleNamespace

    # This is exactly how torch builds it: torch/_dynamo/package.py
    #   SourceInfo.add_code -> InlinedSource(content=_get_module_content(module))
    #   _get_module_content   -> inspect.getsource(module)
    return SimpleNamespace(
        inlined_sources={_InlinedSource(name, inspect.getsource(module))}
    )


def _legacy_check(file_contents: dict[str, str]) -> tuple[bool, str, str]:
    """The pre-S11m comparison: artifact content vs a raw file read."""
    from vllm.compilation.caching import (
        _compute_code_hash,
        _compute_code_hash_with_content,
    )

    expected = _compute_code_hash_with_content(file_contents)
    actual = _compute_code_hash(set(file_contents.keys()))
    return expected == actual, expected, actual


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None,
                    help="real AOT artifact 'model' file to check as well")
    args = ap.parse_args()

    from vllm.compilation.decorators import _verify_source_unchanged

    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        # ---- case 1: file WITHOUT a trailing newline (the =1 situation) ----
        mod, path = _load_temp_module(
            tmp, "s11m_no_newline", "def f():\n    return 42"
        )
        assert not open(path).read().endswith("\n")
        info = _source_info(mod, "s11m_no_newline")
        ok, exp, act = _legacy_check({path: next(iter(info.inlined_sources)).content})
        print(f"[case1] no trailing newline: legacy(old) check equal={ok}")
        print(f"        expected(artifact)={exp[:16]} actual(raw disk)={act[:16]}")
        try:
            _verify_source_unchanged(info, _fake_config())
            print("[case1] new _verify_source_unchanged: PASS (no exception)")
        except RuntimeError as e:
            print(f"[case1] new _verify_source_unchanged: FAIL -> {e}")
            fails += 1

        # ---- case 2: file WITH a trailing newline (all =0 traced files) ----
        mod2, path2 = _load_temp_module(
            tmp, "s11m_newline", "def f():\n    return 42\n"
        )
        info2 = _source_info(mod2, "s11m_newline")
        try:
            _verify_source_unchanged(info2, _fake_config())
            print("[case2] trailing newline  : PASS (both old and new code pass)")
        except RuntimeError as e:
            print(f"[case2] trailing newline  : UNEXPECTED FAIL -> {e}")
            fails += 1

        # ---- case 3: real source change must still be detected ----
        mod3, path3 = _load_temp_module(
            tmp, "s11m_changed", "def f():\n    return 42"
        )
        info3 = _source_info(mod3, "s11m_changed")
        with open(path3, "a") as f:
            f.write("\n\ndef g():\n    return 43\n")
        try:
            _verify_source_unchanged(info3, _fake_config())
            print("[case3] real change       : NOT detected (bad: false negative)")
            fails += 1
        except RuntimeError:
            print("[case3] real change       : detected (RuntimeError) OK")

    if args.artifact:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from s11m_aot_probe import load, walk  # type: ignore

        root = load(args.artifact)
        seen = {}
        for h in walk(root):
            seen[h.module] = h.content
        target = "vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4"
        disk = "/src/vllm/model_executor/kernels/linear/" \
               "scaled_mm/gfx906_sdot4.py"
        if target in seen and os.path.isfile(disk):
            content = seen[target]
            ok, exp, act = _legacy_check({disk: content})
            print(f"\n[artifact] {target}")
            print(f"           artifact bytes={len(content)} "
                  f"disk bytes={os.path.getsize(disk)}")
            print(f"           legacy(old) equal={ok}  "
                  f"expected={exp[:16]} actual={act[:16]}")
            from vllm.compilation.caching import (
                _compute_code_hash_with_content,
                _read_traced_file,
            )

            new = _compute_code_hash_with_content({disk: _read_traced_file(disk)})
            print(f"           new(normalized) equal="
                  f"{_compute_code_hash_with_content({disk: content}) == new}")
        else:
            print(f"\n[artifact] {target} not found in artifact / {disk} missing")

    print(f"\n# cases failed: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
