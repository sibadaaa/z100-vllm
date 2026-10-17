#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Rebuild RCCL with gfx906 device code for the Z100 (Hygon DCU) ROCm 10.1 stack.
#
# The ROCm 10.1 SDK ships librccl.so.1 built by TheRock with a GPU_TARGETS list
# that excludes gfx906, so every collective on a Z100 dies with
# "HIP failure: 'invalid device function'". This script rebuilds RCCL from the
# exact rocm-systems commit TheRock used for the SDK (recorded in
# /opt/rocm/core-10.1/share/therock/therock_manifest.json) with
# -DGPU_TARGETS=gfx906 and installs the result to /mnt/game/rccl-gfx906, which
# tools/dcu/env-rocm101.sh already puts first on LD_LIBRARY_PATH.
#
# The source lives inside the ROCm/rocm-systems monorepo (projects/rccl), not in
# ROCm/rccl: the SDK version string "2.30.7-HEAD:7da4302" hashes the
# rocm-systems submodule pin, and that commit is the tip of
# release/therock-10.1.
#
# Usage:
#   tools/dcu/rccl-gfx906-build.sh              # fetch (if needed), build, install, verify
#   REBUILD=1 tools/dcu/rccl-gfx906-build.sh    # wipe the build dir first, full rebuild
#   FORCE_BUILD=1 ...                           # re-run ninja/install with the existing build dir
#   SKIP_FETCH=1 ...                            # reuse an existing source tree
#
# Idempotent: an up-to-date source tree, build dir and installed library are
# reused; re-running only re-runs cmake/ninja, which are incremental.
#
# Everything can be overridden from the environment (see the defaults below).
# The build itself runs in the ROCm 10.1 toolchain container (BUILD_CONTAINER);
# set LOCAL_BUILD=1 to run it in the current shell instead (inside that image).
#
# Evidence lands in ${LOG_DIR} (default /mnt/game/port/rccl-gfx906-build):
# the cmake command line, full configure/ninja logs, and acceptance checks
# (objdump arch counts, md5/size, NEEDED entries, exported symbol diff against
# the SDK lib).

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
GITHUB_REPO="${GITHUB_REPO:-ROCm/rocm-systems}"
ROCM_SYSTEMS_SHA="${ROCM_SYSTEMS_SHA:-7da43026e4f2ef7600433c38c820732f7a205d22}"
ROCM_SYSTEMS_REF="${ROCM_SYSTEMS_REF:-release/therock-10.1}"
GITHUB_API="${GITHUB_API:-https://api.github.com}"
RAW_BASE="${RAW_BASE:-https://raw.githubusercontent.com}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/ROCm/rocm-systems}"
FMT_SHA="${FMT_SHA:-e69e5f977d458f2650bb346dadf2ad30c5320281}" # pinned by rccl cmake
SRC_ROOT="${SRC_ROOT:-/mnt/game/rccl-src}"
SRC_DIR="${SRC_DIR:-${SRC_ROOT}/rocm-systems/projects/rccl}"
FMT_SRC="${FMT_SRC:-${SRC_ROOT}/fmt-${FMT_SHA}}"
BUILD_DIR="${BUILD_DIR:-/mnt/game/rccl-build}"
INSTALL_DIR="${INSTALL_DIR:-/mnt/game/rccl-gfx906}"
LOG_DIR="${LOG_DIR:-/mnt/game/port/rccl-gfx906-build}"
BUILD_CONTAINER="${BUILD_CONTAINER:-z100-rocm10-build}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm/core-10.1}"
GPU_ARCH="${GPU_ARCH:-gfx906}"
JOBS="${JOBS:-6}"
REBUILD="${REBUILD:-0}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SKIP_FETCH="${SKIP_FETCH:-0}"
LOCAL_BUILD="${LOCAL_BUILD:-0}"
SDK_RCCL="${SDK_RCCL:-/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib/librccl.so.1}"
STEP_TIMEOUT="${STEP_TIMEOUT:-0}" # seconds; 0 = no timeout (ninja can be slow)
FETCH_JOBS="${FETCH_JOBS:-32}"
EXPECTED_FILE_COUNT="${EXPECTED_FILE_COUNT:-1905}"
# Extra cmake options appended to the configure command, e.g.
#   EXTRA_CMAKE_ARGS='-DONLY_FUNCS=AllReduce|ReduceScatter'  # subset build
EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS:-}"

mkdir -p "${LOG_DIR}"
log() { printf '[rccl-gfx906] %s\n' "$*" | tee -a "${LOG_DIR}/build-driver.log"; }
die() { log "ERROR: $*"; exit 1; }

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
run_step() {
    # Run a shell script (given on stdin) inside the build container, logging it.
    local name=$1
    local script="${LOG_DIR}/.step-${name}.sh"
    cat >"${script}"
    local wrapper="bash ${script}"
    if [ "${STEP_TIMEOUT}" -gt 0 ]; then
        wrapper="timeout ${STEP_TIMEOUT} ${wrapper}"
    fi
    local rc=0
    if [ "${LOCAL_BUILD}" = "1" ]; then
        bash -lc "${wrapper}" 2>&1 | tee "${LOG_DIR}/${name}.log" || rc=$?
    else
        command -v docker >/dev/null || die "docker not found (set LOCAL_BUILD=1 inside the image)"
        docker inspect "${BUILD_CONTAINER}" >/dev/null 2>&1 ||
            die "container ${BUILD_CONTAINER} not found; start it or set BUILD_CONTAINER/LOCAL_BUILD"
        docker exec "${BUILD_CONTAINER}" bash -lc "${wrapper}" 2>&1 | tee "${LOG_DIR}/${name}.log" || rc=$?
    fi
    return "${rc}"
}

api_get() {
    curl -fsSL --retry 3 --max-time 60 "${GITHUB_API}/$1"
}

# --------------------------------------------------------------------------
# 1. Source: projects/rccl of rocm-systems @ pin
# --------------------------------------------------------------------------
source_version_ok() {
    [ -f "${SRC_DIR}/CMakeLists.txt" ] &&
        grep -q 'NCCL_MINOR *:= *30' "${SRC_DIR}/makefiles/version.mk" &&
        grep -q 'NCCL_PATCH *:= *7' "${SRC_DIR}/makefiles/version.mk"
}

fetch_source() {
    if [ "${SKIP_FETCH}" = "1" ] && source_version_ok; then
        log "SKIP_FETCH=1: reusing ${SRC_DIR}"
        return 0
    fi
    if source_version_ok && [ -f "${SRC_DIR}/.rccl-source-pin" ] &&
        [ "$(cat "${SRC_DIR}/.rccl-source-pin")" = "${ROCM_SYSTEMS_SHA}" ]; then
        log "source OK: ${SRC_DIR} @ ${ROCM_SYSTEMS_SHA} (RCCL 2.30.7)"
        return 0
    fi
    if source_version_ok; then
        log "adopting existing RCCL 2.30.7 source tree at ${SRC_DIR}"
        printf '%s\n' "${ROCM_SYSTEMS_SHA}" >"${SRC_DIR}/.rccl-source-pin"
        return 0
    fi

    log "fetching projects/rccl from ${GITHUB_REPO} @ ${ROCM_SYSTEMS_SHA}"
    # The commit must exist in the monorepo, and should be the tip of the
    # TheRock release branch recorded by the SDK manifest.
    local resolved tip
    resolved=$(api_get "repos/${GITHUB_REPO}/git/commits/${ROCM_SYSTEMS_SHA}" |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])') ||
        die "cannot resolve ${ROCM_SYSTEMS_SHA} in ${GITHUB_REPO}"
    [ "${resolved}" = "${ROCM_SYSTEMS_SHA}" ] ||
        die "commit mismatch: resolved=${resolved} expected=${ROCM_SYSTEMS_SHA}"
    tip=$(git ls-remote "${GIT_REPO_URL}" "refs/heads/${ROCM_SYSTEMS_REF}" 2>/dev/null |
        awk '{print $1}' || true)
    if [ -n "${tip}" ] && [ "${tip}" != "${ROCM_SYSTEMS_SHA}" ]; then
        log "WARNING: tip of ${ROCM_SYSTEMS_REF} is ${tip}, not ${ROCM_SYSTEMS_SHA}"
    fi

    # A git clone of the 8 GB monorepo is unusable on this host: raw-SHA fetches
    # stall server-side and partial-clone lazy blob checkout takes minutes, so
    # fetch only the ~1900 files of projects/rccl through the trees + raw API.
    local work="${LOG_DIR}/.fetch-rccl"
    local commit_json="${LOG_DIR}/source-commit.json"
    rm -rf "${work}" && mkdir -p "${work}"
    api_get "repos/${GITHUB_REPO}/git/commits/${ROCM_SYSTEMS_SHA}" >"${commit_json}"
    local root_tree project_tree rccl_tree
    root_tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"]["sha"])' "${commit_json}")
    project_tree=$(api_get "repos/${GITHUB_REPO}/git/trees/${root_tree}" |
        python3 -c 'import json,sys; print([e["sha"] for e in json.load(sys.stdin)["tree"] if e["path"]=="projects"][0])')
    rccl_tree=$(api_get "repos/${GITHUB_REPO}/git/trees/${project_tree}" |
        python3 -c 'import json,sys; print([e["sha"] for e in json.load(sys.stdin)["tree"] if e["path"]=="rccl"][0])')
    api_get "repos/${GITHUB_REPO}/git/trees/${rccl_tree}?recursive=1" >"${work}/tree.json"
    cp "${work}/tree.json" "${LOG_DIR}/source-tree-rccl.json"

    python3 - "${work}" "${ROCM_SYSTEMS_SHA}" "${RAW_BASE}" "${GITHUB_REPO}" <<'PY'
import json, sys
work, sha, raw_base, repo = sys.argv[1:5]
tree = json.load(open(f"{work}/tree.json"))["tree"]
urls, modes = [], {}
for e in tree:
    if e["type"] != "blob":
        continue
    urls.append(f"{raw_base}/{repo}/{sha}/projects/rccl/{e['path']} {e['path']}")
    modes[e["path"]] = e["mode"]
open(f"{work}/urls.txt", "w").write("\n".join(urls) + "\n")
json.dump(modes, open(f"{work}/modes.json", "w"))
print(f"[rccl-gfx906] {len(urls)} source files to download")
PY

    mkdir -p "${SRC_DIR}"
    (cd "${SRC_DIR}" && xargs -P "${FETCH_JOBS}" -n 2 bash -c \
        'mkdir -p "$(dirname "$2")"; curl -fsSL --retry 3 --max-time 120 -o "$2" "$1"' _ \
        <"${work}/urls.txt") || die "source download failed"
    python3 - "${work}" "${SRC_DIR}" <<'PY'
import json, os, stat, sys
work, dest = sys.argv[1], sys.argv[2]
modes = json.load(open(f"{work}/modes.json"))
for path, mode in modes.items():
    full = os.path.join(dest, path)
    if mode == "120000":  # symlinks are stored as their target path
        target = open(full, encoding="utf-8").read().strip()
        os.remove(full) if os.path.lexists(full) else None
        os.symlink(target, full)
    elif mode == "100755":
        st = os.stat(full)
        os.chmod(full, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY
    local count
    count=$(find "${SRC_DIR}" -type f | wc -l)
    log "source file count: ${count} (expected ${EXPECTED_FILE_COUNT})"
    source_version_ok || die "downloaded tree is not RCCL 2.30.7"
    printf '%s\n' "${ROCM_SYSTEMS_SHA}" >"${SRC_DIR}/.rccl-source-pin"
}

ensure_fmt() {
    # RCCL's cmake/Dependencies.cmake pulls fmt-10.2.1 through FetchContent when
    # no system fmt is found; pre-seed it so the container needs no git clone.
    if [ -d "${FMT_SRC}" ]; then
        log "fmt source OK: ${FMT_SRC}"
        return 0
    fi
    log "fetching fmt ${FMT_SHA} into ${FMT_SRC}"
    mkdir -p "${SRC_ROOT}"
    local tarball="${SRC_ROOT}/fmt-${FMT_SHA}.tar.gz"
    curl -fsSL --retry 3 --max-time 180 -o "${tarball}" \
        "https://codeload.github.com/fmtlib/fmt/tar.gz/${FMT_SHA}"
    tar xzf "${tarball}" -C "${SRC_ROOT}"
    [ -d "${FMT_SRC}" ] || die "fmt extraction did not produce ${FMT_SRC}"
}

# --------------------------------------------------------------------------
# 2. Configure / build / install (ROCm 10.1 toolchain container)
# --------------------------------------------------------------------------
configure() {
    local cmake_cmd="cmake -S ${SRC_DIR} -B ${BUILD_DIR} -G Ninja"
    cmake_cmd+=" -DCMAKE_BUILD_TYPE=Release"
    cmake_cmd+=" -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
    cmake_cmd+=" -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/amdclang++"
    cmake_cmd+=" -DCMAKE_C_COMPILER=${ROCM_PATH}/bin/amdclang"
    cmake_cmd+=" -DROCM_PATH=${ROCM_PATH}"
    cmake_cmd+=" -DGPU_TARGETS=${GPU_ARCH}"
    cmake_cmd+=" -DBUILD_TESTS=OFF"
    cmake_cmd+=" -DFETCHCONTENT_SOURCE_DIR_FMT=${FMT_SRC}"
    if [ -n "${EXTRA_CMAKE_ARGS}" ]; then
        cmake_cmd+=" ${EXTRA_CMAKE_ARGS}"
    fi
    # RCCL's src/device/generate.py deletes and regenerates gensrc/ on every
    # configure, which makes all 1260 device objects look stale to ninja and
    # triggers a full ~40 min rebuild. The stamp below is what keeps re-runs
    # cheap: cmake is only re-run when the options or the source pin changed.
    local stamp="${BUILD_DIR}/.rccl-cmake-command"
    local stamp_content="${ROCM_SYSTEMS_SHA}|${cmake_cmd}"
    if [ "${REBUILD}" != "1" ] && [ -f "${BUILD_DIR}/CMakeCache.txt" ] &&
        [ -f "${stamp}" ] && [ "$(cat "${stamp}")" = "${stamp_content}" ]; then
        log "configure skipped: ${BUILD_DIR} already configured with these options"
        return 0
    fi
    printf '%s\n' "${cmake_cmd}" >"${LOG_DIR}/cmake-command.txt"
    log "cmake: ${cmake_cmd}"
    run_step configure <<EOF
set -eo pipefail
. /mnt/game/port/setenv-rocm10.sh
if [ "${REBUILD}" = "1" ]; then rm -rf "${BUILD_DIR}"; fi
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"
${cmake_cmd}
# The build dir belongs to the container user; record the configure stamp here.
printf '%s\n' "${stamp_content}" >"${stamp}"
EOF
    grep -E 'Compiling for|Device Linker: GPU targets|ROCm version:|Found amd_smi|amd_smi_INCLUDE|hipconfig|HIP compiler' \
        "${LOG_DIR}/configure.log" | head -20 | sed 's/^/[cmake] /' || true
}

build_install() {
    if [ "${REBUILD}" != "1" ] && [ "${FORCE_BUILD}" != "1" ] &&
        [ -f "${INSTALL_DIR}/lib/librccl.so.1" ]; then
        log "build+install skipped: ${INSTALL_DIR}/lib/librccl.so.1 already exists (FORCE_BUILD=1 to rebuild)"
        return 0
    fi
    run_step build <<EOF
set -eo pipefail
. /mnt/game/port/setenv-rocm10.sh
ninja -C "${BUILD_DIR}" -j "${JOBS}"
ninja -C "${BUILD_DIR}" install
EOF
    [ -f "${INSTALL_DIR}/lib/librccl.so.1" ] ||
        die "install did not produce ${INSTALL_DIR}/lib/librccl.so.1"
}

# --------------------------------------------------------------------------
# 3. Acceptance checks: device code, identity, dependencies, symbols
#
# ROCm 10.1 / clang 24 emits RCCL as a split bundle: the host library keeps a
# file-backed .hip_fatbin (PROGBITS, clang offload bundle, compressed device
# code) and llvm-objdump --offloading extracts the per-arch sidecar files next
# to it (librccl.so.1.0.hip-amdgcn-amd-amdhsa--gfx906). The SDK library shipped
# with a NOBITS .hip_fatbin (zero-fill) and no sidecar, i.e. no device code at
# all. Both forms are checked here.
# --------------------------------------------------------------------------
verify_artifact() {
    log "verifying ${INSTALL_DIR}/lib/librccl.so.1"
    run_step acceptance <<EOF
set -eo pipefail
LIB="${INSTALL_DIR}/lib/librccl.so.1"
LIBDIR="${INSTALL_DIR}/lib"
SIDECAR="\${LIBDIR}/librccl.so.1.0.hip-amdgcn-amd-amdhsa--${GPU_ARCH}"
SDK_LIB="${SDK_RCCL}"
OBJDUMP="${ROCM_PATH}/llvm/bin/llvm-objdump"
READELF="${ROCM_PATH}/llvm/bin/llvm-readelf"
NM="${ROCM_PATH}/llvm/bin/llvm-nm"
OUT="${LOG_DIR}/acceptance.txt"
{
  echo "== device code (llvm-objdump --offloading; clang 24 extracts split bundles) =="
  "\${OBJDUMP}" --offloading "\${LIB}" || true
  echo "gfx906_lines=\$("\${OBJDUMP}" --offloading "\${LIB}" | grep -c 'gfx906' || true)"
  echo
  echo "== split-bundle files =="
  ls -l "\${LIBDIR}"
  md5sum "\${LIBDIR}"/librccl.so.1.0*
  echo
  echo "== sidecar device ELF header =="
  if [ -f "\${SIDECAR}" ]; then
    "\${READELF}" -h "\${SIDECAR}" | grep -E 'Class|Machine|Flags'
  else
    echo "no sidecar \${SIDECAR} (device code must then be inside .hip_fatbin)"
  fi
  echo
  echo "== .hip_fatbin (must be PROGBITS, not the SDK NOBITS placeholder) =="
  "\${READELF}" -S "\${LIB}" | grep -i 'hip_fatbin\|hipFatBinSegment' || true
  echo
  echo "== NEEDED / RPATH =="
  "\${READELF}" -d "\${LIB}" | grep -E 'NEEDED|RPATH|RUNPATH' || true
  echo
  echo "== exported symbols vs SDK librccl =="
  "\${NM}" -D --defined-only "\${LIB}" | awk '{print \$NF}' | LC_ALL=C sort -u > "${LOG_DIR}/symbols-new-all.txt"
  "\${NM}" -D --defined-only "\${SDK_LIB}" | awk '{print \$NF}' | LC_ALL=C sort -u > "${LOG_DIR}/symbols-sdk-all.txt"
  grep -v '^__hip_cuid_' "${LOG_DIR}/symbols-new-all.txt" > "${LOG_DIR}/symbols-new.txt" || true
  grep -v '^__hip_cuid_' "${LOG_DIR}/symbols-sdk-all.txt" > "${LOG_DIR}/symbols-sdk.txt" || true
  echo "defined_new=\$(wc -l < "${LOG_DIR}/symbols-new-all.txt") defined_sdk=\$(wc -l < "${LOG_DIR}/symbols-sdk-all.txt")"
  echo "hip_cuid_new=\$(grep -c '^__hip_cuid_' "${LOG_DIR}/symbols-new-all.txt" || true) hip_cuid_sdk=\$(grep -c '^__hip_cuid_' "${LOG_DIR}/symbols-sdk-all.txt" || true) (per-build unique, ignored)"
  echo "non_cuid_new=\$(wc -l < "${LOG_DIR}/symbols-new.txt") non_cuid_sdk=\$(wc -l < "${LOG_DIR}/symbols-sdk.txt")"
  echo "common=\$(comm -12 "${LOG_DIR}/symbols-new.txt" "${LOG_DIR}/symbols-sdk.txt" | wc -l)"
  echo "only_in_new=\$(comm -23 "${LOG_DIR}/symbols-new.txt" "${LOG_DIR}/symbols-sdk.txt" | wc -l)"
  echo "only_in_sdk=\$(comm -13 "${LOG_DIR}/symbols-new.txt" "${LOG_DIR}/symbols-sdk.txt" | wc -l)"
  comm -23 "${LOG_DIR}/symbols-new.txt" "${LOG_DIR}/symbols-sdk.txt" > "${LOG_DIR}/symbols-only-in-new.txt" || true
  comm -13 "${LOG_DIR}/symbols-new.txt" "${LOG_DIR}/symbols-sdk.txt" > "${LOG_DIR}/symbols-only-in-sdk.txt" || true
  echo "-- only_in_new:"; cat "${LOG_DIR}/symbols-only-in-new.txt"
  echo "-- only_in_sdk:"; cat "${LOG_DIR}/symbols-only-in-sdk.txt"
  echo
  echo "== version strings =="
  strings "\${LIB}" | grep -E 'RCCL version|HEAD:' | head -5 || true
} 2>&1 | tee "\${OUT}"
# Hard gates
k=\$("\${OBJDUMP}" --offloading "\${LIB}" | grep -c 'gfx906' || true)
[ "\${k}" -gt 0 ] || { echo "FAIL: llvm-objdump --offloading shows no gfx906"; exit 10; }
if [ -f "\${SIDECAR}" ]; then
  "\${READELF}" -h "\${SIDECAR}" | grep -q 'gfx906' ||
    { echo "FAIL: sidecar \${SIDECAR} is not a gfx906 ELF"; exit 12; }
else
  "\${READELF}" -S "\${LIB}" | grep -q 'hip_fatbin.*PROGBITS' ||
    { echo "FAIL: no file-backed .hip_fatbin and no gfx906 sidecar"; exit 13; }
fi
if "\${READELF}" -d "\${LIB}" | grep -q 'libgalaxyhip'; then
  echo "FAIL: DTK libgalaxyhip dependency leaked in"; exit 11
fi
for sym in ncclAllReduce ncclReduceScatter ncclAllGather ncclBroadcast ncclCommInitRank ncclGetVersion; do
  "\${NM}" -D --defined-only "\${LIB}" | awk '{print \$NF}' | grep -qx "\${sym}" ||
    { echo "FAIL: missing exported API symbol \${sym}"; exit 14; }
done
echo "ACCEPTANCE: PASS (gfx906_lines=\${k}; artifact ${INSTALL_DIR}/lib/librccl.so.1)"
EOF
}

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
main() {
    log "=== rccl gfx906 build start $(date -Is) ==="
    log "pin=${ROCM_SYSTEMS_SHA} ref=${ROCM_SYSTEMS_REF} arch=${GPU_ARCH} jobs=${JOBS}"
    fetch_source
    ensure_fmt
    configure
    build_install
    verify_artifact
    log "=== rccl gfx906 build done $(date -Is) ==="
    log "artifact: ${INSTALL_DIR}/lib/librccl.so.1"
    log "next: four-GPU smoke via tools/dcu/rccl_allreduce_smoke.py under torchrun"
}

main "$@"
