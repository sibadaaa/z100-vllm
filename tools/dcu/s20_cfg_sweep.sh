#!/usr/bin/env bash
# S20: sweep the decode tiled-GEMM configurations on the production shape mix.
#
# Runs tools/dcu/s18_fp8_block_mprobe.py (gemv entry == the production decode
# entry) once per VLLM_DCU_FP8_DECODE_CFG value, keeps each run's JSON, and
# prints the mix-weighted ms/decode-step for the 256-call c=1 shape mix:
#   64x 8704:5120 + 64x 5120:4352 + 48x 4096:5120 + 16x 3584:5120 + 64x 5120:1536
#
# Usage (inside the container):
#   bash /src/tools/dcu/s20_cfg_sweep.sh "0 auto 16 17 18 19"
#   bash /src/tools/dcu/s20_cfg_sweep.sh "0 16" 8704:5120,4096:5120 8
set -u
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH="/src:/mnt/game/triton-gfx906/install"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
cd /src || exit 1

CFGS="${1:-0 auto 16 17 18 19}"
SHAPES="${2:-8704:5120,5120:4352,4096:5120,3584:5120,5120:1536}"
MS="${3:-1,8}"
ITERS="${ITERS:-20}"
OUT="${OUT:-/mnt/game/s20}"
mkdir -p "$OUT"

for c in $CFGS; do
  echo "=== VLLM_DCU_FP8_DECODE_CFG=$c"
  VLLM_DCU_FP8_DECODE_CFG="$c" python tools/dcu/s18_fp8_block_mprobe.py \
    --shapes "$SHAPES" --ms "$MS" --modes gemv --threads-list 64 \
    --iters "$ITERS" --outdir "$OUT" 2>&1 | grep -E "gemv "
  cp -f "$OUT/s18_fp8_block_mprobe.json" "$OUT/sweep_cfg_${c}.json" 2>/dev/null
done

python3 - "$OUT" "$CFGS" <<'PY'
import glob, json, os, sys
out, cfgs = sys.argv[1], sys.argv[2].split()
MIX = {8704: 64, 5120: 64, 4096: 48, 3584: 16}   # N -> calls/step (K=5120/4352)
MIX1536 = 64                                     # 5120:1536 -> 64 calls/step
print(f"{'cfg':>6} {'ms/step':>9} {'after-3.1ms-free-win':>20}")
for c in cfgs:
    p = os.path.join(out, f"sweep_cfg_{c}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    tot = 0.0
    have = True
    for r in d["results"]:
        if "ms" not in r or r["m"] != 8:
            continue
        calls = MIX1536 if r["k"] == 1536 else MIX.get(r["n"])
        if calls is None:
            have = False
            continue
        tot += calls * r["ms"]
    print(f"{c:>6} {tot:9.2f} {'(n/a)' if not have else ''}")
PY
