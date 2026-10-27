#!/usr/bin/env python3
"""Extract the *resolved* compilation config from a service startup log slice.

The lead's rule for O7: record the cudagraph mode the engine actually resolved,
not the flag that was passed.  vLLM prints the whole resolved config once:

  ... Initializing a V1 LLM engine (vdev) with config: ... compilation_config=
  {'mode': <CompilationMode.NONE: 0>, ..., 'splitting_ops': [], ...,
   'cudagraph_mode': <CUDAGraphMode.NONE: 0>, ...
   'pass_config': {..., 'fuse_attn_quant': False, ...}}

so this parses that line (plus any cudagraph warnings vLLM emitted) into a small
JSON that the arm descriptor can carry.

  .venv/bin/python tools/dcu/s23_resolved.py --log <arm>/startup.log [--out f]
"""

import argparse
import json
import re

FIELDS = {
    "mode": r"'mode': <CompilationMode\.(\w+)",
    "cudagraph_mode": r"'cudagraph_mode': <CUDAGraphMode\.(\w+)",
    "splitting_ops": r"'splitting_ops': (\[[^\]]*\])",
    "fuse_attn_quant": r"'fuse_attn_quant': (\w+)",
    "max_cudagraph_capture_size": r"'max_cudagraph_capture_size': (\d+)",
    "cudagraph_capture_sizes": r"'cudagraph_capture_sizes': (\[[^\]]*\])",
    "use_inductor_graph_partition": r"'use_inductor_graph_partition': (\w+)",
}
WARN = r"cudagraph|CUDAGraph|splitting_ops|piecewise|PIECEWISE|graph pool|" \
       r"captur|Enforce eager|Inductor compilation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.log, errors="ignore") as f:
        lines = f.read().splitlines()

    # Take the LAST init line: an arm's startup slice must not resolve to an
    # earlier start of the same log file.
    init_line = next(
        (ln for ln in reversed(lines) if "Initializing a V1 LLM engine" in ln), "")
    resolved = {}
    for name, pat in FIELDS.items():
        m = re.search(pat, init_line)
        resolved[name] = m.group(1) if m else None

    rx = re.compile(WARN, re.I)
    notes = [ln.strip()[:300] for ln in lines if rx.search(ln)][:12]
    payload = {
        "log": args.log,
        "has_init_line": bool(init_line),
        "resolved": resolved,
        "cudagraph_notes": notes,
    }
    print(f"[resolved] mode={resolved['mode']} "
          f"cudagraph_mode={resolved['cudagraph_mode']} "
          f"splitting_ops={resolved['splitting_ops']} "
          f"fuse_attn_quant={resolved['fuse_attn_quant']} "
          f"capture_sizes={resolved['cudagraph_capture_sizes']}")
    for note in notes[:6]:
        print(f"[resolved]   {note}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[resolved] saved -> {args.out}")
    return 0 if resolved.get("cudagraph_mode") else 1


if __name__ == "__main__":
    raise SystemExit(main())
