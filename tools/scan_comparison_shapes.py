#!/usr/bin/env python3
"""
Sweep: find comparisons whose operand shapes the CUDA comparison kernels cannot run correctly.

The six comparison kernels (greater, less, equal, greater_equal, less_equal, not_equal) support:
    * two operands of EQUAL shape (any rank), or
    * a (rows, cols) operand against a (cols,) or (1, cols) row.
Anything else type-checks but either fails in code generation or reads past a buffer.

Run from the repo root (after apply_phase3.py, so literals already take their partner's shape):

    python3 phase3/scan_comparison_shapes.py            # tests/, apps/, examples/
    python3 phase3/scan_comparison_shapes.py a.tl b.tl

Exit status 0 = nothing suspicious, 1 = suspicious comparisons found.
Comparisons on names defined only inside a branch are reported as 'unresolved', not as problems.
"""
import contextlib
import glob
import io
import sys
from lark import Lark, UnexpectedInput
from tensorlang.ast_builder import build_ast
from tensorlang.type_checker import type_checker

COMPARISONS = {"greater", "less", "equal", "greater_equal", "less_equal", "not_equal"}
parser = Lark(open("tensorlang.lark").read(), start="program", parser="lalr", propagate_positions=True)


def supported(s1, s2):
    s1, s2 = tuple(int(d) for d in s1), tuple(int(d) for d in s2)
    if s1 == s2:
        return True
    if len(s1) == 2 and len(s2) in (1, 2):
        cols = s1[1]
        return s2 == (cols,) or s2 == (1, cols)
    return False


def comparisons(nodes):
    """(description, left, right) for every comparison let and every condition, at any depth."""
    if isinstance(nodes, list):
        for n in nodes:
            yield from comparisons(n)
    elif isinstance(nodes, dict):
        e = nodes.get("expr")
        if nodes.get("type") == "let" and isinstance(e, dict) and e.get("type") in COMPARISONS:
            args = e.get("args") or []
            if len(args) == 2 and all(isinstance(a, str) for a in args):
                yield (f"let {nodes['name']} = {e['type']}({args[0]}, {args[1]})", args[0], args[1])
        ce = nodes.get("_condition_expr")
        if isinstance(ce, dict) and ce.get("type") in COMPARISONS and len(ce.get("args") or []) == 2:
            yield (f"condition `{nodes.get('_condition_text', '?')}`", *ce["args"])
        for k, v in nodes.items():
            if k not in ("tree", "expr") and isinstance(v, (list, dict)):
                yield from comparisons(v)


files = sys.argv[1:] or sorted(set(glob.glob("tests/**/*.tl", recursive=True)
                                   + glob.glob("apps/**/*.tl", recursive=True)
                                   + glob.glob("examples/**/*.tl", recursive=True)))
bad = unresolved = checked = skipped = 0
for f in files:
    code = "\n".join("" if l.lstrip().startswith("#") else l for l in open(f).read().split("\n"))
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            ast_, _, _ = build_ast(parser.parse(code), False, False, source_code=code)
            ok, env = type_checker(ast_, {}, False, False)
            found = list(comparisons(ast_)) if ok else None
        except (UnexpectedInput, ValueError):
            found = None
        except Exception:
            found = None
    if found is None:
        skipped += 1          # negative tests, programs that need file-shape inference, ...
        continue
    for desc, l, r in found:
        if l not in env or r not in env:
            unresolved += 1
            continue
        checked += 1
        s1, s2 = env[l]["shape"], env[r]["shape"]
        if not supported(s1, s2):
            bad += 1
            print(f"SUSPICIOUS {f}: {desc}: {tuple(s1)} vs {tuple(s2)}")
print(f"\nscanned {len(files)} files ({skipped} skipped: do not typecheck here), "
      f"comparisons checked: {checked}, unresolved: {unresolved}, suspicious: {bad}")
sys.exit(1 if bad else 0)
