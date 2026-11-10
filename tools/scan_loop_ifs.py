#!/usr/bin/env python3
"""Sweep: find programs that the v0.5.6 loop fixes (apply_loop_fixes.py) can change.

    python3 tools/scan_loop_ifs.py            # tests/, apps/, examples/
    python3 tools/scan_loop_ifs.py a.tl b.tl

Reports two patterns, using the real parser and AST builder (no GPU needed):
  AFTER-BACKWARD  a `for` body in which an `if` comes after a backward(): it was silently skipped before
                  the fix and runs now, so such a program can change behaviour.
  POST-LOOP       a top-level `for` that contains an `if`, followed by a top-level statement that has a
                  kernel: it ran the wrong kernel before the fix.
Both are listed as hits to look at, not as errors. Exit status 0 = no hits, 1 = hits found.
"""
import contextlib
import glob
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OPTIMIZERS = {"sgd", "adam", "sgd_update", "adam_update"}


def has_if(nodes):
    for n in nodes if isinstance(nodes, list) else []:
        if not isinstance(n, dict):
            continue
        if n.get("type") == "if":
            return True
        if n.get("type") == "for" and has_if(n.get("body", [])):
            return True
    return False


def after_backward(nodes, path="top level"):
    """Loop bodies (at any depth) where an `if` follows a backward()."""
    for n in nodes if isinstance(nodes, list) else []:
        if not isinstance(n, dict):
            continue
        if n.get("type") == "for":
            body = n.get("body", [])
            seen_backward = False
            for b in body:
                if isinstance(b, dict) and b.get("type") == "backward":
                    seen_backward = True
                elif seen_backward and isinstance(b, dict) and b.get("type") == "if":
                    yield f"{path}: for {n.get('loop_var')} has an `if` after backward()"
                    break
            yield from after_backward(body, f"{path} > for {n.get('loop_var')}")
        elif n.get("type") == "if":
            yield from after_backward(n.get("then", []), path)
            for e in n.get("elifs", []):
                yield from after_backward(e.get("body", []), path)
            yield from after_backward(n.get("else") or [], path)


def has_kernel(n):
    if not isinstance(n, dict):
        return False
    if n.get("type") in OPTIMIZERS:
        return True
    e = n.get("expr")
    return n.get("type") == "let" and isinstance(e, dict) and e.get("type") not in ("name", "tensor_literal", "load")


def post_loop(nodes):
    """Top-level statements with a kernel that come after a loop containing an `if`."""
    loop = None
    for n in nodes:
        if isinstance(n, dict) and n.get("type") == "for" and has_if(n.get("body", [])):
            loop = n
        elif loop is not None and has_kernel(n):
            name = n.get("name") or n.get("type")
            yield f"`{name}` follows the loop over `{loop.get('loop_var')}`, which contains an `if`"
            return


def main():
    from lark import Lark, UnexpectedInput
    from tensorlang.ast_builder import build_ast

    parser = Lark((ROOT / "tensorlang.lark").read_text(), start="program", parser="lalr", propagate_positions=True)
    files = sys.argv[1:] or sorted(set(glob.glob(str(ROOT / "tests/**/*.tl"), recursive=True)
                                       + glob.glob(str(ROOT / "apps/**/*.tl"), recursive=True)
                                       + glob.glob(str(ROOT / "examples/**/*.tl"), recursive=True)))
    hits = skipped = 0
    for f in files:
        code = "\n".join("" if l.lstrip().startswith("#") else l for l in open(f).read().split("\n"))
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                ast_, _, _ = build_ast(parser.parse(code), False, False, source_code=code)
            except (UnexpectedInput, ValueError):
                ast_ = None
            except Exception:
                ast_ = None
        if ast_ is None:
            skipped += 1
            continue
        for msg in after_backward(ast_):
            hits += 1
            print(f"AFTER-BACKWARD {f}: {msg}")
        for msg in post_loop(ast_):
            hits += 1
            print(f"POST-LOOP      {f}: {msg}")
    print(f"\nscanned {len(files)} files ({skipped} skipped: do not parse here), hits: {hits}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
