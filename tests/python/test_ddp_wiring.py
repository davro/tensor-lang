"""Tests for the data-parallel plumbing that needs no GPU: the --ddp flags, {rank}/{world_size} path placeholders in
load() and save(), and the world_size keyword of the type checker.

Importing these modules does not touch CUDA (tensorlang.compiler is deliberately not imported here: it creates a context).

    python3 -m unittest discover -s tests/python -v
"""
import copy
import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lark import Lark  # noqa: E402

from tensorlang import ast_builder, type_checker  # noqa: E402
from tensorlang.tensor_lang import TensorLang  # noqa: E402

GRAMMAR = (Path(__file__).resolve().parents[2] / "tensorlang.lark").read_text()
PARSER = Lark(GRAMMAR, start="program", parser="lalr", propagate_positions=True)

PLAIN_ENV = {k: v for k, v in os.environ.items() if k not in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}


def parse_args(*argv):
    with mock.patch.object(sys, "argv", ["tensorlang.py", *argv]):
        return TensorLang().parse_arguments()


def build(src):
    ast, _, _ = ast_builder.build_ast(PARSER.parse(src), source_code=src)
    return ast


def file_paths(ast):
    """Every 'file_path' value in an AST, in order (load() and save() nodes)."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "file_path":
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(ast)
    return found


PROGRAM = '''let x: Tensor[f32, (2, 2)] = load("x_{rank}.npy")
let w: Tensor[f32, (2, 2)] = load("w.npy") with grad
let y = matmul(x, w)
save(y, "y_{rank}_of_{world_size}.npy")'''

LOOP_PROGRAM = '''let w: Tensor[f32, (2, 2)] = load("w.npy") with grad
let x: Tensor[f32, (2, 2)] = load("x_{rank}.npy")
for i in range(3) {
  let y = matmul(x, w)
  let loss = mean(y)
  backward(loss)
}'''


class FlagTests(unittest.TestCase):
    def error_for(self, *argv):
        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                parse_args(*argv)
        self.assertEqual(ctx.exception.code, 2)
        return err.getvalue()

    def test_defaults_leave_an_ordinary_run_alone(self):
        a = parse_args("prog.tl")
        self.assertFalse(a.ddp)
        self.assertFalse(a.ddp_same_gpu)
        self.assertIsNone(a.nproc)

    def test_ddp_defaults_to_two_workers(self):
        a = parse_args("--ddp", "prog.tl")
        self.assertTrue(a.ddp)
        self.assertEqual(a.nproc, 2)

    def test_nproc_and_same_gpu(self):
        a = parse_args("--ddp", "--nproc", "4", "--ddp-same-gpu", "prog.tl")
        self.assertEqual((a.ddp, a.nproc, a.ddp_same_gpu), (True, 4, True))

    def test_nproc_needs_ddp(self):
        self.assertIn("--nproc requires --ddp", self.error_for("--nproc", "2", "prog.tl"))

    def test_same_gpu_needs_ddp(self):
        self.assertIn("--ddp-same-gpu requires --ddp", self.error_for("--ddp-same-gpu", "prog.tl"))

    def test_ddp_needs_a_file(self):
        self.assertIn("--ddp requires a .tl file", self.error_for("--ddp"))

    def test_nproc_must_be_positive(self):
        self.assertIn("at least 1", self.error_for("--ddp", "--nproc", "0", "prog.tl"))

    def test_ddp_conflicts_with_modes_that_choose_gpus_themselves(self):
        for extra in (["--test"], ["--app", "some/app"], ["--list-apps"], ["--list-gpus"], ["--gpu", "0", "prog.tl"]):
            with self.subTest(extra=extra):
                self.assertIn("--ddp cannot be combined", self.error_for("--ddp", *extra))


class PlaceholderTests(unittest.TestCase):
    def paths(self, rank, world, src=PROGRAM):
        with mock.patch.dict(os.environ, {**PLAIN_ENV, "RANK": str(rank), "WORLD_SIZE": str(world)}, clear=True):
            return file_paths(build(src))

    def test_a_single_process_is_rank_0_of_1(self):
        with mock.patch.dict(os.environ, PLAIN_ENV, clear=True):
            self.assertEqual(file_paths(build(PROGRAM)), ["x_0.npy", "w.npy", "y_0_of_1.npy"])

    def test_each_rank_gets_its_own_paths(self):
        self.assertEqual(self.paths(1, 2), ["x_1.npy", "w.npy", "y_1_of_2.npy"])
        self.assertEqual(self.paths(3, 4), ["x_3.npy", "w.npy", "y_3_of_4.npy"])

    def test_paths_without_placeholders_are_untouched(self):
        for rank in (0, 1):
            self.assertEqual(self.paths(rank, 2)[1], "w.npy")

    def test_load_inside_a_for_loop_body_is_expanded(self):
        self.assertEqual(self.paths(1, 2, LOOP_PROGRAM), ["w.npy", "x_1.npy"])

    def test_other_braces_are_left_alone(self):
        src = 'let x: Tensor[f32, (2, 2)] = load("data_{name}_{rank}.npy")'
        self.assertEqual(self.paths(1, 2, src), ["data_{name}_1.npy"])

    def test_a_program_without_placeholders_gives_the_same_ast_on_every_rank(self):
        src = 'let x: Tensor[f32, (2, 2)] = load("x.npy")\nsave(x, "out.npy")'
        asts = []
        for rank in (0, 1, 2):
            with mock.patch.dict(os.environ, {**PLAIN_ENV, "RANK": str(rank), "WORLD_SIZE": "3"}, clear=True):
                asts.append(repr(build(src)))
        self.assertEqual(len(set(asts)), 1)


class TypeCheckerWorldSizeTests(unittest.TestCase):
    def check(self, src, **kw):
        with mock.patch.dict(os.environ, PLAIN_ENV, clear=True):
            ast = build(src)
        with redirect_stdout(io.StringIO()):                    # the checker prints its diagnostics
            return type_checker.type_checker(copy.deepcopy(ast), {}, **kw)

    def test_the_keyword_is_accepted_and_defaults_to_one(self):
        self.assertEqual(repr(self.check(PROGRAM)), repr(self.check(PROGRAM, world_size=1)))

    def test_it_is_threaded_through_for_loops_without_changing_the_result(self):
        base = self.check(LOOP_PROGRAM)
        self.assertTrue(base[0])
        for ws in (1, 2, 4):
            self.assertEqual(repr(self.check(LOOP_PROGRAM, world_size=ws)), repr(base))

    def test_type_errors_are_reported_the_same_way(self):
        bad = "let y = matmul(x, w)"
        self.assertEqual(repr(self.check(bad)), repr(self.check(bad, world_size=4)))
        self.assertFalse(self.check(bad, world_size=4)[0])


if __name__ == "__main__":
    unittest.main()
