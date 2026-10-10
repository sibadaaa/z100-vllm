#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11m: inspect a vLLM AOT artifact (``.../torch_aot_compile/<key>/rank_X_0/model``)
without importing torch, and report *exactly* what ``_verify_source_unchanged``
compares: the ``inlined_sources`` embedded in the artifact vs the files on disk.

The artifact is a plain pickle stream (protocol 4) written by
``torch.compiler.load_compiled_function``'s counterpart.  We unpickle it with
stub classes so the probe runs on the host with the system python3.

Usage (host):
    python3 tools/dcu/s11m_aot_probe.py <artifact> [--diff-root DIR] [--json OUT]

``--diff-root DIR`` maps container paths (``/src/...``) to a host directory so
the on-disk content can be compared line by line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys


class _Stub:
    """Generic stand-in for any pickled class."""

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    def __repr__(self):
        cls = type(self).__name__
        keys = sorted(k for k in self.__dict__ if not k.startswith("_"))
        return f"<{cls} {keys}>"


def _make_stub(mod: str, name: str):
    return type(name, (_Stub,), {"__module__": mod, "__qualname__": name})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _make_stub(module, name)


def load(path: str):
    with open(path, "rb") as f:
        return _Unpickler(f).load()


def walk(obj, depth=0, seen=None, hits=None):
    """Find every object that looks like a SourceInfo (has module+content)."""
    if hits is None:
        hits = []
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 40:
        return hits
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        d = obj.__dict__
        if "module" in d and "content" in d:
            hits.append(obj)
        for k, v in d.items():
            if k.startswith("_"):
                continue
            walk(v, depth + 1, seen, hits)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            walk(v, depth + 1, seen, hits)
    elif isinstance(obj, dict):
        for v in obj.values():
            walk(v, depth + 1, seen, hits)
    return hits


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--diff-root", default=None,
                    help="host dir standing in for container /src")
    ap.add_argument("--map", default="/src=", help="container-path prefix mapping")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump-module", default=None,
                    help="write this module's artifact content to --dump-out")
    ap.add_argument("--dump-out", default=None)
    args = ap.parse_args()

    root = load(args.artifact)
    hits = walk(root)
    # de-dup by (module, sha)
    seen = {}
    for h in hits:
        key = (h.module, sha256_text(h.content))
        seen.setdefault(key, h)
    rows = []
    for (module, sha), h in sorted(seen.items()):
        rows.append({
            "module": module,
            "content_len": len(h.content),
            "content_lines": h.content.count("\n"),
            "sha256": sha,
        })

    print(f"# artifact: {args.artifact}")
    print(f"# source_info entries found: {len(rows)}")
    for r in rows:
        print(f"  {r['sha256'][:16]}  {r['content_len']:>8}B "
              f"{r['content_lines']:>6}L  {r['module']}")

    out = {"artifact": args.artifact, "entries": rows}

    if args.dump_module:
        for (module, sha), h in seen.items():
            if module == args.dump_module:
                dest = args.dump_out or "/dev/stdout"
                with open(dest, "w") as f:
                    f.write(h.content)
                print(f"\n# dumped {module} ({len(h.content)}B) -> {dest}")
                break
        else:
            print(f"\n# module not found: {args.dump_module}")

    if args.diff_root:
        cont_prefix = args.map.split("=")[0]
        host_prefix = args.map.split("=", 1)[1] or args.diff_root
        print("\n# on-disk comparison")
        cmp_rows = []
        for (module, sha), h in sorted(seen.items()):
            modfile = module.replace(".", "/") + ".py"
            cands = [
                h.__dict__.get("filename"),
                f"{host_prefix.rstrip('/')}/{modfile}",
            ]
            disk = None
            disk_path = None
            for c in cands:
                if not c:
                    continue
                p = c.replace(cont_prefix, host_prefix) if cont_prefix else c
                try:
                    with open(p) as f:
                        disk = f.read()
                    disk_path = p
                    break
                except OSError:
                    continue
            if disk is None:
                status = "DISK-MISSING"
                disk_sha = "-"
                first_diff = -1
            else:
                disk_sha = sha256_text(disk)
                status = "MATCH" if disk_sha == sha else "DIFF"
                first_diff = -1
                if status == "DIFF":
                    a = h.content.splitlines()
                    b = disk.splitlines()
                    for i in range(max(len(a), len(b))):
                        la = a[i] if i < len(a) else "<EOF>"
                        lb = b[i] if i < len(b) else "<EOF>"
                        if la != lb:
                            first_diff = i + 1
                            break
            print(f"  {status:<12} {module}  artifact={sha[:12]} "
                  f"disk={disk_sha[:12]} first_diff_line={first_diff}")
            cmp_rows.append({
                "module": module,
                "artifact_sha256": sha,
                "disk_path": disk_path,
                "disk_sha256": disk_sha,
                "status": status,
                "artifact_lines": h.content.count("\n"),
                "disk_lines": disk.count("\n") if disk else None,
                "first_diff_line": first_diff,
            })
        out["comparison"] = cmp_rows

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n# json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
