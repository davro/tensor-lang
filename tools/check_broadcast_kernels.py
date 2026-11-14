#!/usr/bin/env python3
"""No-GPU check of the 2-D "matrix op vector" broadcast kernels against NumPy.

Drives the real generators (tensorlang/generators/elementwise.py), pulls the index expression out of
the CUDA text they emit, evaluates it in Python and compares the result with NumPy (which aligns a 1-D
operand with the LAST axis, i.e. one value per column). Covers the unfused kernel (binary_broadcast) and
the fused chain kernel, for square and non-square matrices.

Run from the repo root:  python3 tools/check_broadcast_kernels.py     (exit 1 = a kernel disagrees with NumPy)
"""
import re, sys
import numpy as np
sys.path.insert(0, ".")
from tensorlang.generators.elementwise import ElementwiseMixin

OPS = {"add": np.add, "minus": np.subtract, "mult": np.multiply, "div": np.divide}
gen = ElementwiseMixin(False)

def run_unfused(op, A, w):
    rows, cols = A.shape
    gen._broadcast_helper_emitted = False
    kernel, _ = gen.binary_broadcast(op, op, "t", "A", "B", A.shape, w.shape, A.shape)
    m = re.search(r"B\[([^\]]+)\]", kernel); assert m, "no B[...] read found"
    expr = m.group(1)
    out = np.empty_like(A)
    for i in range(rows):
        for j in range(cols):
            b = eval(expr, {}, dict(i=i, j=j, rows=rows, cols=cols))
            out[i, j] = OPS[op](A[i, j], w[b])
    return out, expr

def run_fused(op, A, w):
    rows, cols = A.shape
    gen._broadcast_helper_emitted = False
    chain = [(op, "c1", ["A", "w"]), ("mult", "c2", ["c1", "w"])]       # a 2-op chain, as fusion.py builds (mult: nothing cancels)
    kernel, _ = gen.fused_elementwise("fused_elementwise", "t", chain, {}, ["A", "w"], ["c1", "c2"], A.size,
                                      broadcast=True, input_shapes=[A.shape, w.shape], output_shape=A.shape)
    idx_expr = {}
    for m in re.finditer(r"int in(\d+)_idx = ([^;]+);", kernel):
        idx_expr[int(m.group(1))] = m.group(2)
    e = idx_expr[1].replace("out_shape[1]", str(cols)).replace("out_shape[0]", str(rows)).replace("/", "//")
    out = np.empty_like(A)
    for idx in range(A.size):
        i, j = divmod(idx, cols)
        a = A[i, j]
        b = w[eval(e, {}, {"idx": idx})]
        c1 = OPS[op](a, b); out[i, j] = c1 * b
    return out, idx_expr[1]

bad = 0
rng = np.random.default_rng(0)
for rows, cols in [(2, 2), (3, 3), (4, 4), (8, 8), (3, 4), (4, 3), (2, 5), (5, 2)]:
    A = np.round(rng.uniform(0.5, 3, (rows, cols)), 2); w = np.round(rng.uniform(0.5, 3, cols), 2)
    for op in OPS:
        ref = OPS[op](A, w)
        for kind, fn in (("unfused", run_unfused), ("fused", run_fused)):
            got, expr = fn(op, A, w)
            want = ref if kind == "unfused" else ref * w
            ok = np.allclose(got, want)
            if not ok:
                bad += 1
                print(f"MISMATCH {kind:7s} {op:5s} ({rows},{cols}) + ({cols},)   index expr: {expr}   max diff {np.abs(got - want).max():.3f}")
print("kernels agree with NumPy" if not bad else f"\n{bad} mismatch(es): a (rows, cols) matrix with a (cols,) vector is read per ROW")
sys.exit(1 if bad else 0)
