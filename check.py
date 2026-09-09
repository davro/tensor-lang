#!/usr/bin/env python3
"""Parse + type-check a .tl file without needing pycuda/nvcc/a GPU."""
import sys
from lark import Lark
from tensorlang.ast_builder import build_ast
from tensorlang.type_checker import type_checker

def main(path):
    with open('tensorlang.lark') as f:
        grammar = f.read()
    parser = Lark(grammar, start='program', parser='lalr')

    with open(path) as f:
        code = f.read()

    print(f"--- parsing {path} ---")
    tree = parser.parse(code)
    print("parse OK")

    print("--- building AST ---")
    ast, output_tensor, functions = build_ast(tree, DEBUG_MODE=False, DEBUG_INFO=False)
    print(f"AST OK: {len(ast)} top-level nodes, {len(functions)} function(s) defined, output_tensor={output_tensor}")

    print("--- type checking ---")
    ok, env = type_checker(ast, {}, DEBUG_INFO=True, DEBUG_MODE=False)
    if ok:
        print("TYPE CHECK OK")
        print("--- final tensor shapes ---")
        for name, t in env.items():
            print(f"  {name}: {t}")
    else:
        print("TYPE CHECK FAILED")
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[1])
