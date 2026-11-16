#!/usr/bin/env bash
# Host relink of the gfx906 _rocm_C extension from the production object set.
#
# Why this exists: the object the CMake build tree hands the linker for
# csrc/rocm/torch_bindings.cpp is from before S81, so every relink that used it
# (S89 included) silently dropped the `gfx906_attnd_decode_inner_kvint8`
# registration and with it the int8-KV HIP inner loop.  `TB_SRC` recompiles the
# bindings from the tree, which is the fix; the rest of the object set is the
# exact one the running .so was built from (.s89/arms/build_ms16.sh).
#
# Knobs (env):
#   OUT=dir            output directory (required)
#   TB_SRC=path        recompile torch_bindings.cpp from this source
#   STAGE_HIP=path     also compile+link the stage-timed attention kernel
#   STAGE_OPS=path     stage op registration TU (required with STAGE_HIP)
#   ATTN_SRC=path      compile this attention inner-loop source and link it in
#                      place of the default t1int8.hip.o (production-arm A/B)
#   LD_DIR=dir         output .so name prefix (default _rocm_C)
#
#   OUT=runs/x/build OUT=... TB_SRC=csrc/rocm/torch_bindings.cpp \
#       bash tools/dcu/attn_stage/build.sh
set -euo pipefail
REPO=/home/lyh/z100-vllm
OUT=$(readlink -m "${OUT:?OUT dir required}")
TB_SRC=${TB_SRC:-}
STAGE_HIP=${STAGE_HIP:-}
STAGE_OPS=${STAGE_OPS:-}
ATTN_SRC=${ATTN_SRC:-}
# resolve before the script cd's into the build tree: the build tree carries an
# OLD copy of csrc/, so a relative path would silently compile that one
[ -n "$TB_SRC" ] && TB_SRC=$(readlink -m "$TB_SRC")
[ -n "$STAGE_HIP" ] && STAGE_HIP=$(readlink -m "$STAGE_HIP")
[ -n "$STAGE_OPS" ] && STAGE_OPS=$(readlink -m "$STAGE_OPS")
[ -n "$ATTN_SRC" ] && ATTN_SRC=$(readlink -m "$ATTN_SRC")
ATTN_OBJ=$REPO/.s89/arms/t1int8.hip.o
[ -n "$STAGE_HIP" ] && [ -z "$STAGE_OPS" ] && {
  echo "STAGE_HIP needs STAGE_OPS" >&2; exit 2; }
S56=$REPO/.s56/rocm_c_rebuild
B=/mnt/game/port/build-sync/temp
DTK=/mnt/game/host-runtime/dtk2604
TORCH=$REPO/.venv/lib/python3.10/site-packages/torch
mkdir -p "$OUT"

get() { grep -m1 "^  $1 = " "$S56/stmt.txt" | sed "s/^  $1 = //"; }
rebase() {
  printf '%s' "$1" | sed 's/\\//g; s/"//g' \
    | sed "s|/usr/local/lib/python3.10/dist-packages/torch|$TORCH|g" \
    | sed "s|/opt/dtk-26.04-DCC2602-0317|$DTK|g" \
    | sed "s|/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26105.2065-3f2faa34|$DTK/hip/lib/libgalaxyhip.so.5|g" \
    | sed "s|/opt/dtk/|$DTK/|g" \
    | sed "s|/src/csrc/dtk_rocprim_compat.h|$REPO/csrc/dtk_rocprim_compat.h|g"
}
DEFINES=$(get DEFINES | sed 's/GOOGLE_GLOG_DLL_DECL=""/GOOGLE_GLOG_DLL_DECL=/g')
FLAGS=$(rebase "$(get FLAGS)")
INCLUDES=$(rebase "$(get INCLUDES)" | sed "s|-I/mnt/game/port/build-sync/temp/csrc|-I$REPO/csrc -I/mnt/game/port/build-sync/temp/csrc|")
INCLUDES="$INCLUDES -isystem $S56/sysroot/usr/include"
LINK_FLAGS=$(rebase "$(get LINK_FLAGS)")
LINK_LIBRARIES=$(rebase "$(get LINK_LIBRARIES)" \
  | sed 's|/usr/lib/x86_64-linux-gnu/libglog.so.0.4.0||' \
  | sed 's|/usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2||' \
  | sed 's|/usr/lib/x86_64-linux-gnu/libunwind.so$|/usr/lib/x86_64-linux-gnu/libunwind.so.8|' \
  | sed 's|/usr/lib/x86_64-linux-gnu/libunwind.so |/usr/lib/x86_64-linux-gnu/libunwind.so.8 |' \
  | sed 's|/usr/lib/x86_64-linux-gnu/librt.so$|/usr/lib/x86_64-linux-gnu/librt.so.1|' \
  | sed 's|/usr/lib/x86_64-linux-gnu/librt.so |/usr/lib/x86_64-linux-gnu/librt.so.1 |')
LINK_PATH=$(rebase "$(get LINK_PATH)")
cd "$B"
# The S91 recovery captured the serving shell's include paths here; the file was
# archived during the 2026-11-13 cleanup, so make it optional and fall back to
# whatever the caller's environment already provides.
SERVE_ENV=$REPO/runs/prefill-lctx/incident/serve.environ.txt
if [ -f "$SERVE_ENV" ]; then
  for _v in C_INCLUDE_PATH CPLUS_INCLUDE_PATH; do
    _val=$(grep -m1 "^$_v=" "$SERVE_ENV" | cut -d= -f2- || true)
    [ -n "$_val" ] && export "$_v=$_val"
  done
fi

# The build tree's copy of these bindings predates S81 and silently drops the
# int8-KV op registration (the S89/S91 incident).  Require the source instead of
# ever linking that object again; the stale object was deleted on 2026-11-13.
if [ -z "$TB_SRC" ] || [ ! -f "$TB_SRC" ]; then
  echo "TB_SRC=<csrc/rocm/torch_bindings.cpp> is required: the build tree's" \
       "bindings object is stale and must not be linked" >&2
  exit 2
fi
echo "[stage-build] compiling bindings $TB_SRC"
$DTK/llvm/bin/clang++ $DEFINES $INCLUDES $FLAGS \
  -o "$OUT/torch_bindings.cpp.o" -x c++ -c "$TB_SRC"
TB_OBJ=$OUT/torch_bindings.cpp.o

EXTRA_OBJS=()
if [ -n "$ATTN_SRC" ]; then
  echo "[stage-build] compiling attn inner loop $ATTN_SRC"
  $DTK/llvm/bin/clang++ $DEFINES $INCLUDES $FLAGS \
    -o "$OUT/attn_inner_variant.hip.o" -x hip -c "$ATTN_SRC"
  ATTN_OBJ="$OUT/attn_inner_variant.hip.o"
fi
if [ -n "$STAGE_HIP" ]; then
  echo "[stage-build] compiling stage kernel $STAGE_HIP"
  $DTK/llvm/bin/clang++ $DEFINES $INCLUDES $FLAGS \
    -o "$OUT/attn_stage.hip.o" -x hip -c "$STAGE_HIP"
  echo "[stage-build] compiling stage ops $STAGE_OPS"
  $DTK/llvm/bin/clang++ $DEFINES $INCLUDES $FLAGS \
    -o "$OUT/attn_stage_ops.cpp.o" -x c++ -c "$STAGE_OPS"
  EXTRA_OBJS=("$OUT/attn_stage.hip.o" "$OUT/attn_stage_ops.cpp.o")
fi

echo "[stage-build] linking"
$DTK/hip/bin/hipcc_cmake_linker_helper $DTK/llvm/bin \
  $LINK_FLAGS \
  "$B/CMakeFiles/_rocm_C.dir/csrc/rocm/w8a8_sdot4.hip.o" \
  "$REPO/.s89/arms/fp8_emu.hip.o" \
  "$B/CMakeFiles/_rocm_C.dir/csrc/rocm/bf16_emu_gemv.hip.o" \
  "$B/CMakeFiles/_rocm_C.dir/csrc/rocm/attention.hip.o" \
  "$TB_OBJ" \
  "$B/CMakeFiles/_rocm_C.dir/csrc/rocm/attnd_inner.hip.o" \
  "$ATTN_OBJ" \
  ${EXTRA_OBJS[@]+"${EXTRA_OBJS[@]}"} \
  -o "$OUT/_rocm_C.abi3.so" $LINK_PATH $LINK_LIBRARIES -shared
ls -la "$OUT/_rocm_C.abi3.so"
echo "[stage-build] done: $OUT/_rocm_C.abi3.so"
