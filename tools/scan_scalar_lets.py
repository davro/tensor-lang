#!/usr/bin/env python3
"""
Sweep: find every .tl program that binds a bare number with `let`.

Phase 2 turns that into [E-SCALAR-LITERAL]; before it was skipped silently.
Run this BEFORE applying apply_phase2.py, from the repo root:

    python3 phase2/scan_scalar_lets.py            # tests/, apps/, examples/
    python3 phase2/scan_scalar_lets.py path/a.tl path/b.tl

Exit status 0 = nothing found (safe to apply), 1 = hits found (send them to me).
Programs that fail to parse (intentional negative tests) are listed, not counted.
"""
import contextlib
import glob
import io
import sys
from lark import Lark, UnexpectedInput
from tensorlang.ast_builder import build_ast

parser = Lark(open("tensorlang.lark").read(), start="program", parser="lalr", propagate_positions=True)


def hits_in(node, path="top-level"):
    """Every `let` whose expression is a bare number, at any nesting depth."""
    if isinstance(node, list):
        for item in node:
            yield from hits_in(item, path)
    elif isinstance(node, dict):
        expr = node.get("expr")
        if node.get("type") == "let" and isinstance(expr, dict) and expr.get("type") == "number":
            yield (node.get("name"), expr.get("value"), path)
        for key, value in node.items():
            if key in ("tree", "expr"):
                continue
            if isinstance(value, (list, dict)):
                yield from hits_in(value, f"{path}/{node.get('type', '?')}.{key}")


files = sys.argv[1:] or sorted(
    set(glob.glob("tests/**/*.tl", recursive=True)
        + glob.glob("apps/**/*.tl", recursive=True)
        + glob.glob("examples/**/*.tl", recursive=True)))
found = skipped = 0
for f in files:
    code = open(f).read()
    # `# loss_scale: ...` pragma lines are consumed by the compiler before parsing;
    # blank them (keeping line numbers) so the grammar does not choke on them.
    code = "\n".join("" if line.lstrip().startswith("#") else line for line in code.split("\n"))
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            ast_, _, _ = build_ast(parser.parse(code), False, False, source_code=code)
            result = list(hits_in(ast_))
        except UnexpectedInput:
            result = None
        except Exception as e:                      # keep sweeping on any builder error
            result = f"builder error: {type(e).__name__}"
    if result is None or isinstance(result, str):
        skipped += 1
        print(f"  skipped {f} ({'does not parse' if result is None else result})")
        continue
    for name, value, where in result:
        found += 1
        print(f"HIT {f}: let {name} = {value}   [{where}]")
print(f"\nscanned {len(files)} files, skipped {skipped}, bare numeric lets found: {found}")
sys.exit(1 if found else 0)
