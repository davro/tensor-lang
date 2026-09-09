#!/usr/bin/env python3
"""Interpret the flattened+inlined AST with NumPy, to check the actual
arithmetic (not just shapes) without needing CUDA/a GPU."""
import sys
import numpy as np
from lark import Lark
from tensorlang.ast_builder import build_ast

def run(board_4x4: np.ndarray) -> np.ndarray:
    with open('tensorlang.lark') as f:
        grammar = f.read()
    parser = Lark(grammar, start='program', parser='lalr')
    with open('apps/games/2048/step.tl') as f:
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
                # only ever used for `board` in this file
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


def reference_slide_left(board_4x4: np.ndarray) -> np.ndarray:
    """Ground-truth reference implementation (plain Python), to diff against."""
    out = np.zeros_like(board_4x4)
    for r in range(4):
        vals = [v for v in board_4x4[r] if v != 0]
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
        out[r] = merged
    return out


if __name__ == '__main__':
    test_boards = [
        # the hand-checked cases from step.tl's comments, each padded to a 4x4
        ("all 2s row",      [[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("444+0 row",       [[4, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("4,2,2,0 (tricky)",[[4, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("no merge row",    [[4, 2, 8, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("mixed full board",[[0, 2, 0, 2], [4, 4, 4, 4], [2, 0, 2, 0], [8, 4, 2, 2]]),
        ("gaps everywhere", [[0, 0, 2, 0], [0, 4, 0, 4], [2, 0, 0, 2], [0, 0, 0, 8]]),
    ]

    all_ok = True
    for name, board in test_boards:
        board = np.array(board, dtype=np.float32)
        got = run(board)
        want = reference_slide_left(board)
        ok = np.array_equal(got, want)
        all_ok &= ok
        status = "OK" if ok else "MISMATCH"
        print(f"[{status}] {name}")
        print(f"  input : {board.tolist()}")
        print(f"  step.tl : {got.tolist()}")
        print(f"  expected: {want.tolist()}")

    print()
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)
