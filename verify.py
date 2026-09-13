#!/usr/bin/env python3
"""Generalized version of verify.py: numpy-interpret ANY of the step_*.tl
files and diff against a plain-Python reference for that direction.
Dev/test-only script (not part of the app) used to validate the new
direction files before wiring them into agent.py."""
import sys
import numpy as np
from lark import Lark
from tensorlang.ast_builder import build_ast


def run(path: str, board_4x4: np.ndarray) -> np.ndarray:
    with open('tensorlang.lark') as f:
        grammar = f.read()
    parser = Lark(grammar, start='program', parser='lalr')
    with open(path) as f:
        code = f.read()
    tree = parser.parse(code)
    ast, output_tensor, functions = build_ast(tree, DEBUG_MODE=False, DEBUG_INFO=False)

    env = {}

    def get(name):
        return env[name]

    for node in ast:
        t = node['type']
        if t == 'let':
            name = node['name']
            expr = node['expr']
            et = expr['type']
            if et == 'load':
                env[name] = board_4x4.astype(np.float32)
            elif et == 'fill':
                value = expr['value']
                shape = tuple(expr['shape'])
                env[name] = np.full(shape, value, dtype=np.float32)
            elif et == 'slice':
                src = get(expr['tensor'])
                specs = expr['specs']
                idx = []
                for s in specs:
                    if s['type'] == 'full_slice':
                        idx.append(slice(None))
                    elif s['type'] == 'index':
                        idx.append(s['value'])
                    else:
                        idx.append(slice(s['start'], s['end']))
                env[name] = src[tuple(idx)]
            elif et == 'concat':
                tensors = [get(n) for n in expr['tensors']]
                env[name] = np.concatenate(tensors, axis=expr['axis'])
            elif et == 'transpose':
                src = get(expr['tensor'])
                env[name] = src.T
            elif et == 'equal':
                a, b = [get(n) for n in expr['args']]
                env[name] = (a == b).astype(np.float32)
            elif et == 'greater':
                a, b = [get(n) for n in expr['args']]
                env[name] = (a > b).astype(np.float32)
            elif et == 'mult':
                a, b = [get(n) for n in expr['args']]
                env[name] = a * b
            elif et == 'add':
                a, b = [get(n) for n in expr['args']]
                env[name] = a + b
            elif et == 'minus':
                a, b = [get(n) for n in expr['args']]
                env[name] = a - b
            elif et == 'name':
                env[name] = get(expr['name'])
            else:
                raise NotImplementedError(f"unhandled expr type {et} for {name}: {expr}")
        elif t == 'save':
            pass
        else:
            raise NotImplementedError(f"unhandled node type {t}: {node}")

    return env[output_tensor]


def reference_slide(board_4x4: np.ndarray, direction: str) -> np.ndarray:
    """Ground-truth reference for any of the 4 directions, plain Python."""
    def slide_left_row(vals):
        vals = [v for v in vals if v != 0]
        merged = []
        i = 0
        while i < len(vals):
            if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                merged.append(vals[i] * 2)
                i += 2
            else:
                merged.append(vals[i])
                i += 1
        merged += [0] * (4 - len(merged))
        return merged

    b = board_4x4.copy()
    if direction == 'left':
        pass
    elif direction == 'right':
        b = b[:, ::-1]
    elif direction == 'up':
        b = b.T
    elif direction == 'down':
        b = b.T[:, ::-1]
    else:
        raise ValueError(direction)

    out = np.array([slide_left_row(row) for row in b], dtype=np.float32)

    if direction == 'right':
        out = out[:, ::-1]
    elif direction == 'up':
        out = out.T
    elif direction == 'down':
        out = out[:, ::-1].T
    return out


TEST_BOARDS = [
    ("all 2s row",       [[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    ("444+0 row",        [[4, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    ("4,2,2,0 (tricky)", [[4, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    ("no merge row",     [[4, 2, 8, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    ("mixed full board", [[0, 2, 0, 2], [4, 4, 4, 4], [2, 0, 2, 0], [8, 4, 2, 2]]),
    ("gaps everywhere",  [[0, 0, 2, 0], [0, 4, 0, 4], [2, 0, 0, 2], [0, 0, 0, 8]]),
    ("tricky in a column", [[4, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]]),
]


if __name__ == '__main__':
    path = sys.argv[1]
    direction = sys.argv[2]
    all_ok = True
    for name, board in TEST_BOARDS:
        board = np.array(board, dtype=np.float32)
        got = run(path, board)
        want = reference_slide(board, direction)
        ok = np.array_equal(got, want)
        all_ok &= ok
        status = "OK" if ok else "MISMATCH"
        print(f"[{status}] {name}")
        if not ok:
            print(f"  input : {board.tolist()}")
            print(f"  got     : {got.tolist()}")
            print(f"  expected: {want.tolist()}")

    print()
    print(f"{path} ({direction}): " + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    sys.exit(0 if all_ok else 1)
