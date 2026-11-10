#!/usr/bin/env python3
"""
Sweep: find comparisons with an f16 operand. The comparison kernels are always generated for f32
(compiler.py and codegen.py pass dtype="f32"), so an f16 operand is read as float. f64 is not reported:
it is stored as f32.

Reports every comparison (let or if/elif condition) where either operand is f16, including an f16 tensor
against a numeric literal (a literal's fill is f32).

Works from any directory (it locates the repo root itself):

    python3 tools/scan_comparison_dtypes.py            # tests/, apps/, examples/
    python3 tools/scan_comparison_dtypes.py a.tl b.tl

Exit status 0 = no f16 comparison, 1 = f16 comparisons found.
Comparisons on names defined only inside a branch are reported as 'unresolved', not as problems.
"""
import contextlib
import glob
import io
import sys
from pathlib import Path

# repo-root-aware (fix_scan_paths)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lark import Lark, UnexpectedInput
from tensorlang.ast_builder import build_ast
from tensorlang.type_checker import type_checker

COMPARISONS = {"greater", "less", "equal", "greater_equal", "less_equal", "not_equal"}
parser = Lark((ROOT / "tensorlang.lark").read_text(), start="program", parser="lalr", propagate_positions=True)


def supported(d1, d2):
    return "f16" not in (d1, d2)


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


files = sys.argv[1:] or sorted(set(glob.glob(str(ROOT / "tests/**/*.tl"), recursive=True)
                                   + glob.glob(str(ROOT / "apps/**/*.tl"), recursive=True)
                                   + glob.glob(str(ROOT / "examples/**/*.tl"), recursive=True)))
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
        d1, d2 = env[l]["dtype"], env[r]["dtype"]
        if not supported(d1, d2):
            bad += 1
            print(f"F16 {f}: {desc}: {d1} vs {d2}")
print(f"\nscanned {len(files)} files ({skipped} skipped: do not typecheck here), "
      f"comparisons checked: {checked}, unresolved: {unresolved}, f16: {bad}")
sys.exit(1 if bad else 0)

