#!/usr/bin/env python3
"""Generate the stage-timed variant of the gfx906 int8-KV attention inner loop.

Plan of record: docs/perf/PLAN-20261113-attn-latency-vs-bandwidth.md T1.

The production kernel (``csrc/rocm/attnd_inner_kvint8.hip``) is never edited.
This script copies the kernel + host wrapper text verbatim into a build file and
inserts only ``clock64()`` deltas around five stages.

Edits are anchored on *consecutive line sequences* whose contents are compared
with indentation and the macro continuation mark stripped.  Every anchor must
match exactly once, so a production-source move fails loudly instead of
silently producing a mis-instrumented kernel.

Stages (per CTA, thread 0's timeline):

    load  stage_k x2 + stage_v (+ the half-1 V swap) + LDS writes + the
          barrier that waits for the data to land -> KV bytes / load = GB/s
    qk    the eight v_dot4_i32_i8 chains (two tiles per iteration)
    sm    score finalize + cross-lane max + exp + running-sum + acc rescale
    pv    the 16-accumulator fma loop
    wait  the barriers that wait for slower peers before a staging phase

Numerics are untouched: clock64() has no data dependency on the accumulators
and the arithmetic order is unchanged.  The harness still asserts out_md5
equality against the production op before it trusts any timing.
"""

from __future__ import annotations

import argparse
import os

SRC_DEFAULT = "csrc/rocm/attnd_inner_kvint8.hip"

KERNEL_START = (
    "template <bool kDelayMerge, int kTPS>\n"
    "__global__ void __launch_bounds__(kThreads, 3)\n"
    "attnd_inner_kvsdot4_kernel("
)
KERNEL_END = "\n}\n\n}  // namespace"
WRAPPER_START = "void gfx906_attnd_decode_inner_kvint8("

MACRO_LINES = {
    "const unsigned long long h_sm0 = clock64();",
    "const unsigned long long h_pv0 = clock64();",
    "t_sm += (unsigned int)(h_pv0 - h_sm0);",
    "t_pv += (unsigned int)(clock64() - h_pv0);",
}

# (anchor lines, action, payload)
#   before   -- insert the payload lines before the first anchor line
#   afterN   -- insert after anchor line N (0-based, within the anchor block)
#   replace0 -- replace the first anchor line with the payload
EDITS: list[tuple[list[str], str, list[str]]] = [
    # --- signature: extra counter buffer -----------------------------------
    (["const int segm_fp16,", "const int seg_stride_in) {"], "replaceblk", [
        "const int segm_fp16,",
        "const int seg_stride_in,",
        "unsigned long long* __restrict__ stage_cycles) {",
    ]),
    # --- accumulators (right after the thread index) -----------------------
    (["const int tid = threadIdx.x;"], "after0", [
        "// stage-timed variant: thread 0's per-stage cycles (u32 is enough:",
        "// a single CTA stays under 2^32 cycles even at 131k x 256 segments)",
        "unsigned int t_load = 0, t_qk = 0, t_sm = 0, t_pv = 0, t_wait = 0;",
        "const unsigned long long t_cta0 = clock64();",
    ]),
    # --- QK chain ----------------------------------------------------------
    (["#pragma unroll 4", "for (int c = 0; c < kRowUnits8; ++c) {"], "before", [
        "const unsigned long long c_qk0 = clock64();",
    ]),
    (["S85_HALF(0, (a0 + a1) + (a2 + a3))"], "before", [
        "t_qk += (unsigned int)(clock64() - c_qk0);",
    ]),
    # --- softmax / PV split inside the per-tile macro ----------------------
    (["fmaxf(mt, __shfl_xor(mt, off, 16));"], "after0", [
        "const unsigned long long h_sm0 = clock64();",
    ]),
    (['_Pragma("unroll") for (int i = 0; i < 16; ++i) acc[i] *= alpha;'],
     "after0", [
         "const unsigned long long h_pv0 = clock64();",
         "t_sm += (unsigned int)(h_pv0 - h_sm0);",
     ]),
    (["acc[15] = fmaf(w, f.y, acc[15]);", "}", "}"], "after1", [
        "t_pv += (unsigned int)(clock64() - h_pv0);",
    ]),
    # --- prologue staging --------------------------------------------------
    (["// ---- prologue: K pair (tiles j_lo, j_lo+1) + V of tile j_lo ----"],
     "before", ["const unsigned long long c_ld0 = clock64();"]),
    (["adv();  // cursor -> tile j_lo+1 (consumed by iteration j_lo's V refill)"],
     "before", ["t_load += (unsigned int)(clock64() - c_ld0);"]),
    # --- in-loop V swap: time the existing wait barrier, then the staging --
    (["__syncthreads();", "if (has_h1) {"], "replace0", [
        "{",
        "  const unsigned long long c_w0 = clock64();",
        "  __syncthreads();",
        "  t_wait += (unsigned int)(clock64() - c_w0);",
        "}",
        "const unsigned long long c_vs0 = clock64();",
    ]),
    (["if (has_h1) S85_HALF(1, (b0 + b1) + (b2 + b3))"], "before", [
        "t_load += (unsigned int)(clock64() - c_vs0);",
    ]),
    # --- refill staging at the bottom of the iteration --------------------
    (["__syncthreads();", "if (jn < j_hi) {"], "replace0", [
        "{",
        "  const unsigned long long c_w1 = clock64();",
        "  __syncthreads();",
        "  t_wait += (unsigned int)(clock64() - c_w1);",
        "}",
        "const unsigned long long c_ld1 = clock64();",
    ]),
    (["adv();  // cursor -> tile jn+1 (next iteration's V refill)",
      "}", "__syncthreads();", "}"], "after2", [
        "t_load += (unsigned int)(clock64() - c_ld1);",
    ]),
]


def norm(line: str) -> str:
    """Anchor comparison: ignore indentation and the macro continuation mark."""
    return line.strip().rstrip("\\").strip()


def find_block(lines: list[str], anchor: list[str]) -> int:
    want = [norm(a) for a in anchor]
    hits = []
    for i in range(len(lines) - len(want) + 1):
        if all(norm(lines[i + k]) == want[k] for k in range(len(want))):
            hits.append(i)
    if len(hits) != 1:
        raise SystemExit(
            f"[stage-kernel] anchor {anchor} matched {len(hits)} times "
            "(need 1); the production kernel moved -- re-derive the anchor"
        )
    return hits[0]


def render(payload: list[str], macro: bool, indent: int) -> list[str]:
    out = []
    for p in payload:
        text = " " * indent + p
        if macro and p in MACRO_LINES:
            pad = max(1, 79 - len(text))
            text = text + " " * pad + "\\"
        out.append(text)
    return out


def instrument_kernel(kernel: str) -> str:
    lines = kernel.split("\n")
    m0 = next(i for i, ln in enumerate(lines) if "#define S85_HALF" in ln)
    m1 = next(i for i, ln in enumerate(lines) if "#undef S85_HALF" in ln)

    inserts = []
    for anchor, action, payload in EDITS:
        i = find_block(lines, anchor)
        macro = m0 < i < m1
        indent = len(lines[i]) - len(lines[i].lstrip(" "))
        if action == "before":
            at = i
        elif action.startswith("after"):
            at = i + int(action[5:]) + 1
        elif action in ("replace0", "replaceblk"):
            at = i
        else:  # pragma: no cover
            raise SystemExit(f"unknown action {action}")
        inserts.append(
            (at, action, render(payload, macro, indent), len(anchor)))

    for at, action, payload, alen in sorted(inserts, key=lambda e: e[0],
                                           reverse=True):
        if action == "replace0":
            lines[at:at + 1] = payload
        elif action == "replaceblk":
            lines[at:at + alen] = payload
        else:
            lines[at:at] = payload

    # counters out: just before the kernel's final closing brace
    if lines[-1].strip() != "}":
        raise SystemExit("[stage-kernel] kernel does not end with '}'")
    lines[-1:-1] = [
        "  if (tid == 0) {",
        "    atomicAdd(&stage_cycles[0], (unsigned long long)t_load);",
        "    atomicAdd(&stage_cycles[1], (unsigned long long)t_qk);",
        "    atomicAdd(&stage_cycles[2], (unsigned long long)t_sm);",
        "    atomicAdd(&stage_cycles[3], (unsigned long long)t_pv);",
        "    atomicAdd(&stage_cycles[4], (unsigned long long)t_wait);",
        "    atomicAdd(&stage_cycles[5], 1ull);",
        "    atomicAdd(&stage_cycles[6], clock64() - t_cta0);",
        "  }",
    ]
    return "\n".join(lines)


def stage_wrapper(wrapper: str) -> str:
    new = wrapper.replace(WRAPPER_START,
                          "void gfx906_attnd_decode_inner_kvint8_stage(", 1)
    old = "int64_t seg_stride) {"
    if new.count(old) != 1:
        raise SystemExit("[stage-kernel] wrapper signature anchor moved")
    new = new.replace(old, "int64_t seg_stride, torch::Tensor& stage_cycles) {")

    old = "  const int* cup = cu_query_lens.data_ptr<int>();"
    if new.count(old) != 1:
        raise SystemExit("[stage-kernel] wrapper cup anchor moved")
    new = new.replace(old, old + "\n"
                      "  TORCH_CHECK(stage_cycles.scalar_type() == at::kLong &&\n"
                      "                  stage_cycles.numel() >= 7,\n"
                      "              \"stage: int64 cycle buffer of >= 7 elements required\");\n"
                      "  unsigned long long* stgp = reinterpret_cast<unsigned long long*>(\n"
                      "      stage_cycles.data_ptr<int64_t>());")

    n_launch = new.count("segm_is_fp16, seg_stride);")
    if n_launch != 2:
        raise SystemExit(
            f"[stage-kernel] launch anchor matched {n_launch} times (need 2)")
    new = new.replace("segm_is_fp16, seg_stride);",
                      "segm_is_fp16, seg_stride, stgp);")
    return new.replace("attnd_inner_kvsdot4_kernel<",
                       "attnd_inner_kvsdot4_stage_kernel<")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=os.getcwd())
    a = ap.parse_args()

    src_path = os.path.join(a.repo, a.src) if not os.path.isabs(a.src) else a.src
    text = open(src_path).read()

    kstart = text.index(KERNEL_START)
    kend = text.index(KERNEL_END, kstart)
    kernel = text[kstart:kend] + "\n}"
    wstart = text.index(WRAPPER_START)
    wrapper = text[wstart:]  # source of the *_stage wrapper only

    kernel = instrument_kernel(kernel)
    kernel = kernel.replace("attnd_inner_kvsdot4_kernel(",
                            "attnd_inner_kvsdot4_stage_kernel(", 1)
    stage = stage_wrapper(wrapper)

    header = (
        "// GENERATED by tools/dcu/attn_stage/make_stage_kernel.py -- do not edit.\n"
        f"// Source: {os.path.relpath(src_path, a.repo)}\n"
        "// Production kernel/wrapper are kept verbatim; only *_stage copies are\n"
        "// instrumented with clock64() stage deltas.\n"
    )
    prod_kernel = text[kstart:kend] + "\n}"
    out = (header + text[:kstart] + prod_kernel + "\n\n" + kernel
           + "\n\n}  // namespace\n\n")
    # the production wrapper comes from t1int8.hip.o at link time; keeping a
    # second copy here would be a duplicate symbol
    out += stage
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(out)
    print(f"[stage-kernel] wrote {a.out} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
