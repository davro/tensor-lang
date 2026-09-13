#!/usr/bin/env python3
"""
apps/games/connect_four/tools/verify_dispatch.py

Replicates compiler.py's exact shape-based branching for every
add/minus/mult/div call in a .tl file (including inside `for` loop
bodies), WITHOUT needing CUDA/pycuda/a GPU (only lark+numpy). Flags:
  - anything routed into KernelGenerator.binary_broadcast whose actual
    output shape isn't genuinely 2D (that function unconditionally does
    `rows, cols = output_shape`, and crashes on anything else — this is
    exactly the bug this script was written after hitting: mult(lr,
    bv_grad) with lr=(1,1) and bv_grad=(1,) is 2D-vs-1D by the letter of
    the dispatch condition, but its real output is 1D (1,))
  - anything with no matching dispatch branch at all

check.py's type checker validates shape COMPATIBILITY but never
simulates compiler.py's own branch-by-branch kernel dispatch, so it will
happily approve code that compiles fine per the type system but crashes
in codegen. Run this in addition to check.py, always, before trusting a
.tl file compiles cleanly on real hardware:

    python3 apps/games/connect_four/tools/verify_dispatch.py apps/games/connect_four/train.tl
    python3 apps/games/connect_four/tools/verify_dispatch.py apps/games/connect_four/infer.tl

This does NOT verify the CUDA kernels themselves execute correctly —
only that the compiler's shape-dispatch logic won't crash before it gets
that far. Run from the tensor-lang repo root.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path.cwd()))
import sys
from lark import Lark
from tensorlang.ast_builder import build_ast
from tensorlang.type_checker import type_checker

path = sys.argv[1]
with open('tensorlang.lark') as f:
    grammar = f.read()
parser = Lark(grammar, start='program', parser='lalr')
with open(path) as f:
    code = f.read()
tree = parser.parse(code)
ast, output_tensor, functions = build_ast(tree, DEBUG_MODE=False, DEBUG_INFO=False)
ok, env = type_checker(ast, {}, DEBUG_INFO=False, DEBUG_MODE=False)
assert ok

def shape_of(name):
    if name in env:
        return tuple(env[name]['shape'])
    if name.endswith('_grad') and name[:-5] in env:
        return tuple(env[name[:-5]]['shape'])
    raise KeyError(name)

def can_broadcast(shape1, shape2):
    ndim = max(len(shape1), len(shape2))
    p1 = (1,)*(ndim-len(shape1)) + tuple(shape1)
    p2 = (1,)*(ndim-len(shape2)) + tuple(shape2)
    return all(d1==d2 or d1==1 or d2==1 for d1,d2 in zip(p1,p2))

def flatten(nodes):
    out = []
    for n in nodes:
        out.append(n)
        if n.get('type') == 'for':
            out.extend(flatten(n['body']))
    return out

def is_scalar_shape(shape):
    # Exactly mirrors type_checker.py's _is_scalar_shape — the earlier
    # version of this script approximated the output-shape rule instead
    # of calling the real one, and that approximation happened to agree
    # with the type checker for train.tl (batch=3000, never (1,1)) but
    # NOT for infer.tl (batch=1, where a width-1 value head genuinely
    # produces (1,1), which this same helper misclassifies as "scalar"
    # and returns the OTHER operand's shape instead) — exactly the bug
    # that slipped through. Always use the REAL computed output shape
    # (env[name]['shape'], from the actual type_checker) as ground
    # truth, never a re-derived approximation.
    return len(shape) == 0 or all(d == 1 for d in shape)

problems = []
for node in flatten(ast):
    if node.get('type') != 'let':
        continue
    expr = node['expr']
    if expr['type'] not in ('add','minus','mult','div'):
        continue
    arg1, arg2 = expr['args']
    shape1, shape2 = shape_of(arg1), shape_of(arg2)
    output_shape = shape_of(node['name'])  # the REAL type-checker output, ground truth
    if shape1 == shape2:
        route = 'elementwise'
    elif len(shape1)==2 and len(shape2)==1:
        route = 'binary_broadcast'
        if len(output_shape) != 2:
            problems.append((node['name'], f'binary_broadcast WOULD CRASH: needs 2D output, '
                              f'got {output_shape} (shape1={shape1} is_scalar={is_scalar_shape(shape1)}, '
                              f'shape2={shape2} is_scalar={is_scalar_shape(shape2)})'))
    elif len(shape1)==1 and len(shape2)==1 and (shape1[0]==1 or shape2[0]==1):
        route = 'binary_1d_broadcast'
    elif can_broadcast(shape1, shape2):
        route = 'binary_general_broadcast'
    else:
        route = 'NO MATCH'
        problems.append((node['name'], 'no dispatch branch matches', shape1, shape2))
    print(f"{node['name']:20s} {expr['type']:6s} {str(shape1):14s} {str(shape2):10s} -> out {str(output_shape):10s} [{route}]")

print()
if problems:
    print(f"=== {len(problems)} PROBLEM(S) in {path} ===")
    for p in problems: print(p)
    sys.exit(1)
else:
    print(f"No dispatch problems detected in {path}.")
