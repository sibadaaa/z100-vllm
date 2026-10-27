#!/usr/bin/env bash
# S5 gate for the gfx906 DTK build. Run inside z100-build:
#     docker exec z100-build bash /mnt/game/port/s5-gate.sh
#
# Same two environment caveats as S4's gate.sh:
#  * DTK 26.04's dccobjdump is a broken symlink and llvm-objdump --offloading
#    prints nothing for DTK's hipfb container, so the arch list is read from the
#    embedded target IDs. That is discriminating: the vendor wheel reports
#    gfx906 gfx926 gfx928 gfx936.
#  * `import vllm` before torch aborts with std::bad_alloc, so torch comes first.
. /mnt/game/port/setenv.sh
cd "${REPO_DIR}" || exit 1

arch_of() {
  strings -a "$1" | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed 's/.*--//' | sort -u
}

_C_SO="${REPO_DIR}/vllm/_C.abi3.so"
_ROCM_SO="${REPO_DIR}/vllm/_rocm_C.abi3.so"
VENDOR=/usr/local/lib/python3.10/dist-packages/vllm/_C.abi3.so

echo "===== S5 gate $(date -Is) ====="
echo "python     : $(python -V 2>&1)"
echo "torch      : $(python -c 'import torch;print(torch.__version__, torch.version.hip)')"
echo "triton     : $(python -c 'import triton;print(triton.__version__)')"
echo
echo "----- 1) import gate, torch imported first -----"
python -c "import torch, vllm, vllm._C, vllm._rocm_C; print('IMPORT_OK', vllm._C.__file__)"; echo "rc=$?"
echo
echo "----- 2) gfx targets -----"
for so in "${_C_SO}" "${_ROCM_SO}"; do
  printf "%-38s archs: " "$(basename "${so}")"
  [ -f "${so}" ] && arch_of "${so}" | tr '\n' ' ' || printf "MISSING "
  echo
done
printf "%-38s archs: " "vendor _C (reference)"
arch_of "${VENDOR}" | tr '\n' ' '; echo
echo
echo "----- 3) _rocm_C registered ops -----"
python -c "
import torch, vllm._rocm_C
import vllm._custom_ops as ops
names = sorted(s.name for s in torch._C._jit_get_all_schemas() if s.name.startswith('_rocm_C::'))
print(names)
print('ROCM_SKINNY_GEMM_ENABLED =', ops.ROCM_SKINNY_GEMM_ENABLED)
print('paged_attention registered =', hasattr(torch.ops._rocm_C, 'paged_attention'))"
echo
echo "===== gate done $(date -Is) ====="
