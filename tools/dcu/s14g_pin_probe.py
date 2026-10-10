#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14g: 30-second probe that the Triton autotune pin is actually in effect.

Runs one deliberately autotuned kernel and prints the selected config plus the
total autotune bench time.  Unpinned: Triton benchmarks every config (bench time
grows with the number of configs, the winner is the measured-fastest one).
Pinned (sitecustomize from /mnt/game/s14g/pin on PYTHONPATH): only
``prune_configs[0]`` is benchmarked, it always wins, and the ``[s14g-pin]``
marker is printed at exit.

    docker exec z100-s14 bash -lc 'PYTHONPATH=/mnt/game/s14g/pin:/src:/mnt/game/triton-gfx906/install python /src/tools/dcu/s14g_pin_probe.py'
"""

import time

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=w, num_stages=s)
        for w in (1, 4, 8)
        for s in (1, 2)
    ],
    key=[],
)
@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(x_ptr + off, mask=m)
    tl.store(y_ptr + off, v * 2.0 + 1.0, mask=m)


def main() -> None:
    x = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)
    t0 = time.time()
    _k[grid](x, y, x.numel())
    torch.cuda.synchronize()
    print(f"probe: chosen={_k.best_config} bench_time={getattr(_k, 'bench_time', -1):.3f}s "
          f"wall={time.time() - t0:.3f}s all_configs={_k.configs[0]}..{_k.configs[-1]}")


if __name__ == "__main__":
    main()
