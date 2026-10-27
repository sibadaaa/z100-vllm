#!/usr/bin/env python3
"""One-shot: derive runs/specbench/launch_ctrim_spec.sh from .s33/launch_ctrim_ar.sh.

The production launcher computes SPEC but never passes --speculative-config (the
s33 no-spec arm).  The copy adds: SPEC_SET wins, SPEC_OFF=1 disables, else the
production default; and passes --speculative-config only when non-empty.
Everything else stays byte-identical.
"""
src = "/home/lyh/z100-vllm/.s33/launch_ctrim_ar.sh"
dst = "/home/lyh/z100-vllm/runs/specbench/launch_ctrim_spec.sh"

text = open(src).read()

anchor = 'SPEC="${SPEC:-${_DEFAULT_SPEC}}"'
assert text.count(anchor) == 1, "SPEC anchor not found"
inject = anchor + '''

# specbench: SPEC_SET wins, SPEC_OFF=1 disables, else the production default.
if [ -n "${SPEC_SET:-}" ]; then SPEC="${SPEC_SET}"; fi
if [ -n "${SPEC_OFF:-}" ]; then SPEC=""; fi
SPEC_ARGS=()
[ -n "${SPEC}" ] && SPEC_ARGS=(--speculative-config "${SPEC}")'''
text = text.replace(anchor, inject)

kv = '  "${KV_ARGS[@]}" \\'
assert text.count(kv) == 1, "KV_ARGS anchor not found"
text = text.replace(kv, kv + '\n  "${SPEC_ARGS[@]}" \\')

open(dst, "w").write(text)
import os
os.chmod(dst, 0o755)
print("wrote", dst)
