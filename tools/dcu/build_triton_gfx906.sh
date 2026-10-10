#!/usr/bin/env bash
# Reproducible build of a gfx906-capable Triton 3.6.0 for the DTK 26.04 /
# Z100 (gfx906) vLLM port  (Todolist step S6).
#
# Usage (on the host, from the repository root):
#     bash tools/dcu/build_triton_gfx906.sh
#
# What it does
#   1. creates (or reuses) the dedicated container `z100-triton` from the same
#      image as the S4/S4b baselines, with /mnt/game mounted read-write;
#   2. clones triton-lang/triton v3.6.0 into /mnt/game/triton-gfx906/src;
#   3. applies tools/dcu/triton-gfx906-3.6.0.patch (VEGA ISAFamily);
#   4. builds and installs into /mnt/game/triton-gfx906/install (--target), so
#      the DTK image's own triton is never modified;
#   5. prints the gate command (probe) to run afterwards.
#
# Everything (source, build tree, install prefix, JIT cache) lives under
# /mnt/game. The DTK triton in /usr/local/lib/python3.10/dist-packages is left
# untouched; use it by prepending the install prefix to PYTHONPATH:
#
#     PYTHONPATH=/mnt/game/triton-gfx906/install \
#         python tools/dcu/probe_triton.py
set -euo pipefail

IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413
CTR=z100-triton
ROOT=/mnt/game/triton-gfx906
SRC=$ROOT/src
INSTALL=$ROOT/install
LOGDIR=$ROOT/logs
PATCH=/src/tools/dcu/triton-gfx906-3.6.0.patch
TRITON_TAG=${TRITON_TAG:-v3.6.0}
MAX_JOBS=${MAX_JOBS:-4}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

echo "== 1/5 container $CTR =="
if ! docker ps -a --format '{{.Names}}' | grep -qx "$CTR"; then
  # device/group/memory parameters copied from the S4b baseline container.
  docker run -d --name "$CTR" \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri --group-add video \
    --shm-size=64m --memory=10g \
    -v /mnt/game:/mnt/game -v "$REPO":/src:ro -v /opt/hyhal:/opt/hyhal:ro \
    --security-opt label=disable \
    "$IMAGE" sleep infinity
else
  docker start "$CTR" >/dev/null
fi

echo "== 2/5 clone triton $TRITON_TAG =="
docker exec "$CTR" bash -lc "
  set -e
  mkdir -p $LOGDIR $INSTALL
  if [ ! -d $SRC/.git ]; then
    git clone --depth 1 -b $TRITON_TAG https://github.com/triton-lang/triton $SRC
  fi
  cd $SRC && git log --oneline -1
"

echo "== 3/5 apply gfx906 patch (idempotent) =="
docker exec "$CTR" bash -lc "cd $SRC
  if git apply --check $PATCH 2>/dev/null; then
    git apply $PATCH && echo 'patch applied'
  elif git apply --check -R $PATCH 2>/dev/null; then
    echo 'patch already applied - reusing'
  else
    echo 'ERROR: patch neither applies nor reverse-applies cleanly' >&2; exit 1
  fi
  git diff --stat"

echo "== 4/5 build (MAX_JOBS=$MAX_JOBS) =="
docker exec "$CTR" bash -lc "
  set -o pipefail
  . /opt/dtk/env.sh >/dev/null 2>&1
  cd $SRC
  export TRITON_CACHE_DIR=$ROOT/cache
  export MAX_JOBS=$MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  echo \"=== build start \$(date) MAX_JOBS=\$MAX_JOBS ===\"
  pip install . --no-build-isolation --no-deps --target $INSTALL
  echo \"=== build exit=\$? \$(date) ===\"
" 2>&1 | tee "/mnt/game/triton-gfx906/logs/build-$(basename "$TRITON_TAG").log"

echo "== 5/5 gate =="
cat <<EOF
Now run the probe (inside $CTR or any DTK container):

  docker exec $CTR bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; \\
    TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache \\
    PYTHONPATH=$INSTALL python /src/tools/dcu/probe_triton.py'
EOF
