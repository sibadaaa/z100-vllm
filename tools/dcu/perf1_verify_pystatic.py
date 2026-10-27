#!/usr/bin/env python3
"""Static guard for the bug class that broke the tree on 2026-10-13 (F1).

A function that ASSIGNS a module-level name without declaring it `global` makes
that name local for the whole function, so an earlier READ raises
`UnboundLocalError` at runtime.  That is exactly what killed
`fused_sigmoid_gating_delta_rule_update` in production while every unit test and
import still passed.

    python3 tools/dcu/perf1_verify_pystatic.py vllm/model_executor/layers/fla/ops/*.py ...

Exit 0 = clean, 1 = offenders found (also reports syntax errors).
"""

import ast
import sys


def module_globals(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def check_file(path):
    try:
        src = open(path).read()
        tree = ast.parse(src, path)
    except SyntaxError as e:
        return [("SYNTAX", e.lineno or 0, str(e))]
    globs = module_globals(tree)
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned, read, declared = set(), set(), set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Global):
                declared |= set(sub.names)
            elif isinstance(sub, ast.Name):
                if isinstance(sub.ctx, ast.Store):
                    assigned.add(sub.id)
                else:
                    read.add(sub.id)
        for name in sorted(((assigned & globs) - declared) & read):
            findings.append((node.name, node.lineno, name))
    return findings


def main():
    bad = 0
    for path in sys.argv[1:]:
        for fn, lineno, name in check_file(path):
            print(f"{path}:{lineno}: {fn}() uses module global '{name}' as a local "
                  f"(missing `global {name}`)")
            bad += 1
    print(f"[pystatic] {len(sys.argv) - 1} file(s), {bad} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
