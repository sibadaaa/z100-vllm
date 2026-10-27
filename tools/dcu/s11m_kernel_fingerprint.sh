#!/usr/bin/env bash
# S11m: compare two AOT artifacts by their *generated-code fingerprint*.
#
#   tools/dcu/s11m_kernel_fingerprint.sh <artifactA> <artifactB>
#
# AOT artifact bytes are NOT comparable: the artifact embeds a random
# `CompileId.id_ = str(uuid.uuid4())` (torch/_dynamo/utils.py:1895-1897), so two
# compiles of the same graph always differ byte-wise.  What *is* comparable is
# the set of generated inductor kernels (`triton_*`, `*_fused_*`), whose names
# hash the generated code.  This script prints both sets and their diff.
set -u
A="${1:?usage: s11m_kernel_fingerprint.sh <artifactA> <artifactB>}"
B="${2:?usage: s11m_kernel_fingerprint.sh <artifactA> <artifactB>}"
TMP=$(mktemp -d)
strings -n 6 "$A" | grep -oE '^(triton|_fused|fused|combo|triton_poi|triton_red)[A-Za-z0-9_]*$' | sort -u > "$TMP/a.txt"
strings -n 6 "$B" | grep -oE '^(triton|_fused|fused|combo|triton_poi|triton_red)[A-Za-z0-9_]*$' | sort -u > "$TMP/b.txt"
echo "[fp] $A -> $(wc -l < "$TMP/a.txt") kernels ($(stat -c%s "$A") B, md5 $(md5sum "$A" | cut -d' ' -f1))"
echo "[fp] $B -> $(wc -l < "$TMP/b.txt") kernels ($(stat -c%s "$B") B, md5 $(md5sum "$B" | cut -d' ' -f1))"
echo "[fp] kernel-name set diff:"
diff "$TMP/a.txt" "$TMP/b.txt" || true
echo "[fp] diff lines = $(diff "$TMP/a.txt" "$TMP/b.txt" | grep -c '^[<>]' || true)"
echo "[fp] files: $TMP/a.txt $TMP/b.txt"
