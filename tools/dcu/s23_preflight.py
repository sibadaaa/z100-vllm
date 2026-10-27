#!/usr/bin/env python3
"""Pre-flight gate that must pass before the O6 driver touches the service.

A restart re-imports the working tree, and three agents edit that tree
concurrently.  Two failure modes must never reach production:

* a syntax error or half-written edit in a module the service imports -- caught
  by ``py_compile`` in the driver (host side, same interpreter as the service);
* a broken extension or a missing op registration -- caught here, inside the
  container, where ``_rocm_C`` can actually be loaded.

It also asserts that the three S23 switches exist in ``vllm.envs`` and, when
``--expect NAME=VAL`` is given, that the switch resolves to VAL in this process.

  docker exec z100-regress ... .venv/bin/python tools/dcu/s23_preflight.py \
      --expect VLLM_DCU_FP8_EMU_FASTPATH=1
"""

import argparse
import sys

SWITCHES = (
    "VLLM_DCU_FP8_EMU_FASTPATH",
    "VLLM_DCU_FP8_QUANT_FASTPATH",
    "VLLM_DCU_FP8_FUSE_QUANT",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", action="append", default=[],
                    help="NAME=VAL that must resolve to that value")
    args = ap.parse_args()

    import torch

    from vllm.model_executor.kernels.linear.scaled_mm import gfx906_fp8
    import vllm.envs as envs

    fails = []
    print(f"[preflight] torch={torch.__version__} "
          f"gpus={torch.cuda.device_count()}")

    if not hasattr(torch.ops.vllm, "fp8_emu_mm_block"):
        fails.append("torch.ops.vllm.fp8_emu_mm_block not registered")
    else:
        print("[preflight] vllm.fp8_emu_mm_block registered")
    print(f"[preflight] ops_available_block={gfx906_fp8.ops_available_block()}")
    if not gfx906_fp8.ops_available_block():
        fails.append("gfx906_fp8.ops_available_block() is False")

    for name in SWITCHES:
        if not hasattr(envs, name):
            fails.append(f"vllm.envs.{name} missing (switch not added?)")
        else:
            print(f"[preflight] envs.{name}={getattr(envs, name)}")

    for item in args.expect:
        name, _, want = item.partition("=")
        got = getattr(envs, name, None)
        want_bool = want.lower() in ("1", "true")
        if got != want_bool:
            fails.append(f"{name} resolved to {got}, expected {want_bool}")
        else:
            print(f"[preflight] expect {name}={want_bool} ok")

    if fails:
        for f in fails:
            print(f"[preflight] FAIL {f}", file=sys.stderr)
        return 1
    print("[preflight] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
