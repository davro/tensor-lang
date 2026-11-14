#!/usr/bin/env python3
"""Sweep: .tl programs whose result changes with the square-matrix / vector broadcast fix.

Affected: add/minus/mult/div where one operand is a square 2-D tensor (n, n) and the other is a 1-D vector (n,).
Unfused, only `matrix op vector` (matrix FIRST) used the per-row read; inside a fused chain the vector was read per
row in either position. Uses the project's own parser, AST builder and type checker for the shapes.

Run from the repo root:  python3 tools/scan_square_vector_broadcast.py      (exit 1 = hits)
"""
import sys, traceback
from pathlib import Path
sys.path.insert(0, ".")
from lark import Lark
from tensorlang.ast_builder import build_ast
from tensorlang.type_checker import type_checker

OPS = {"add", "minus", "mult", "div"}
parser = Lark(Path("tensorlang.lark").read_text(), start="program", parser="lalr", propagate_positions=True)

def walk(x):
    if isinstance(x, dict):
        yield x
        for k, v in x.items():
            if k != "tree": yield from walk(v)
    elif isinstance(x, (list, tuple)):
        for v in x: yield from walk(v)

hits, skipped, checked = [], [], 0
files = sorted(p for d in ("tests", "apps", "examples", "exploratory") if Path(d).is_dir() for p in Path(d).rglob("*.tl"))
for tl in files:
    code = tl.read_text(errors="replace")
    try:
        ast, _, _ = build_ast(parser.parse(code), False, False, source_code=code)
        ok, env = type_checker(ast, {}, False, False)
    except BaseException as e:
        skipped.append(f"{tl}: {type(e).__name__}"); continue
    if not ok: skipped.append(f"{tl}: does not type-check (or needs generated inputs)"); continue
    checked += 1
    for node in walk(ast):
        e = node.get("expr") if node.get("type") == "let" else None
        if not isinstance(e, dict) or e.get("type") not in OPS: continue
        args = [a if isinstance(a, str) else (a or {}).get("name") for a in e.get("args", [])]
        if len(args) != 2 or any(a not in env for a in args): continue
        s1, s2 = (tuple(env[a]["shape"]) for a in args)
        for (mat, vec, order) in ((s1, s2, "matrix first"), (s2, s1, "vector first")):
            if len(mat) == 2 and mat[0] == mat[1] and len(vec) == 1 and vec[0] == mat[1] and vec[0] != 1:
                hits.append(f"{tl}: let {node['name']} = {e['type']}({', '.join(args)})   {mat} with {vec}, {order}"
                            + ("" if order == "matrix first" else "   [only when fused with a neighbouring op]"))
print(f"checked {len(files) - len(skipped)} of {len(files)} programs")
for s in skipped: print("SKIPPED", s)
for h in hits: print("HIT", h)
print("hits:", len(hits))
sys.exit(1 if hits else 0)
