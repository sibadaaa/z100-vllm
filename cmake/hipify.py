#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#
# A command line tool for running pytorch's hipify preprocessor on CUDA
# source files.
#
# See https://github.com/ROCm/hipify_torch
# and <torch install dir>/utils/hipify/hipify_python.py
#

import argparse
import os
import shutil

from torch.utils.hipify import cuda_to_hip_mappings as _cuda_macros

# DTK's torch 2.11 wheel stores CUDA_MACROS_MAP values as bare strings where
# upstream torch (and DTK's torch 2.10) stores 1-tuples. hipify_python.py does
# `dst = value[0]`, so a bare string degrades to its first character
# (CUDA_VERSION -> "D") and every hipified use of that macro breaks. Normalize
# to the upstream tuple shape before hipify_python is imported, because it
# snapshots the map while it is being imported. The isinstance guard makes this
# a no-op for the already-correct 2.10 tuples.
for _macro, _value in list(_cuda_macros.CUDA_MACROS_MAP.items()):
    if isinstance(_value, str):
        _cuda_macros.CUDA_MACROS_MAP[_macro] = (_value,)

from torch.utils.hipify.hipify_python import get_hip_file_path, hipify


def _expected_hip_build_path(source_abs: str, output_directory: str) -> str:
    """Match torch.utils.hipify.hipify_python.preprocessor fout_path naming."""
    rel = os.path.relpath(source_abs, output_directory)
    return os.path.abspath(
        os.path.join(
            output_directory, get_hip_file_path(rel, is_pytorch_extension=True)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Project directory where all the source + include files live.
    parser.add_argument(
        "-p",
        "--project_dir",
        help="The project directory.",
    )

    # Directory where hipified files are written.
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The output directory.",
    )

    # Source files to convert.
    parser.add_argument(
        "sources", help="Source files to hipify.", nargs="*", default=[]
    )

    args = parser.parse_args()

    # Limit include scope to project_dir only
    includes = [os.path.join(args.project_dir, "*")]

    # Get absolute path for all source files.
    extra_files = [os.path.abspath(s) for s in args.sources]

    # Copy sources from project directory to output directory.
    # The directory might already exist to hold object files so we ignore that.
    shutil.copytree(args.project_dir, args.output_dir, dirs_exist_ok=True)

    def _neutralize_dtk_macros_include() -> None:
        """Replace DTK's injected ``<ATen/dtk_macros.h>`` include.

        DTK's torch patched ``torch.utils.hipify`` to prepend that include to
        every hipified C/C++ file (``HIPIFY_HEADER_FILE`` in
        ``hipify_python.py``). The header itself errors out whenever
        ``TORCH_TARGET_VERSION`` or ``TORCH_STABLE_ONLY`` is defined, which is
        exactly the case for the ``*_stable_libtorch`` extensions (stable ABI),
        so every one of their translation units fails with
        ``"This file should not be included when either TORCH_STABLE_ONLY or
        TORCH_TARGET_VERSION is defined."``

        The only thing the include contributes is ``#define __DTK_ARCH__``
        (DTK's hipify renames ``__CUDA_ARCH__`` to ``__DTK_ARCH__`` in the
        generated code). Expand it in place, without the header's
        ``TORCH_TARGET_VERSION`` guard, so the generated tree builds with both
        stable and regular extensions. Behaviour is otherwise byte-for-byte
        identical.
        """
        marker = "#include <ATen/dtk_macros.h>"
        replacement = (
            "// Replaces the #include <ATen/dtk_macros.h> that DTK's patched\n"
            "// torch.utils.hipify injects; see cmake/hipify.py. The stable-ABI\n"
            "// extensions define TORCH_TARGET_VERSION, which makes that header\n"
            "// fail with #error, yet __DTK_ARCH__ itself is still needed.\n"
            "#if defined(USE_ROCM)\n"
            "#define __DTK_ARCH__ 520\n"
            "#define DTK_VERSION 11800\n"
            "#define DTKRT_VERSION 11080\n"
            "#endif"
        )
        for root, _dirs, files in os.walk(args.output_dir):
            for name in files:
                if not name.endswith((".h", ".hpp", ".cuh", ".c", ".cc", ".cpp", ".cu", ".hip")):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, encoding="utf-8") as fin:
                        text = fin.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if marker not in text:
                    continue
                with open(path, "w", encoding="utf-8") as fout:
                    fout.write(text.replace(marker, replacement))

    hipify_result = hipify(
        project_directory=args.project_dir,
        output_directory=args.output_dir,
        # Hipify resolves quoted includes next to the including file first; vLLM
        # uses paths relative to csrc/ (e.g. "torch_utils.h"
        # from quantization/w8a8/fp8/*.cu). Without an include root here, those
        # headers are never found and are not hipified or rewritten in dependents.
        header_include_dirs=["."],
        includes=includes,
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )

    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        if s_abs in hipify_result and hipify_result[s_abs].hipified_path is not None:
            path = hipify_result[s_abs].hipified_path
            # PyTorch skips writing when is_pytorch_extension and text unchanged;
            # hipified_path then stays *.cu. CMake expects *.hip under output_dir.
            if s_abs.endswith(".cu") and path.endswith(".cu"):
                dest = _expected_hip_build_path(s_abs, args.output_dir)
                if os.path.normpath(path) != os.path.normpath(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(path, dest)
                hipified_s_abs = dest
            else:
                hipified_s_abs = path
        else:
            hipified_s_abs = s_abs
        hipified_sources.append(hipified_s_abs)

    assert len(hipified_sources) == len(args.sources)

    # Must run after the loop above: torch's hipify writes each result lazily,
    # when the caller first touches hipify_result[...].hipified_path.
    _neutralize_dtk_macros_include()

    # Print hipified source files.
    print("\n".join(hipified_sources))
