"""Tests for tools/ddp_check.py, the data-parallel equivalence check.

The tool's own subject (a real GPU run) cannot be tested here, so these tests check everything around it: the generated
programs are valid TensorLang, the float64 reference is right, the data layout is what the programs expect, and the
comparison logic passes a correct simulated run and catches each kind of data-parallel fault. No GPU is used.

    python3 -m unittest discover -s tests/python -v
"""
import copy
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lark import Lark  # noqa: E402

from tensorlang import ast_builder, type_checker  # noqa: E402
from tensorlang.autograd import ComputationGraph  # noqa: E402

_spec = importlib.util.spec_from_file_location("ddp_check", ROOT / "tools" / "ddp_check.py")
ddp_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ddp_check)

PARSER = Lark((ROOT / "tensorlang.lark").read_text(), start="program", parser="lalr", propagate_positions=True)
PLAIN_ENV = {k: v for k, v in os.environ.items() if k not in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}


class ReferenceTests(unittest.TestCase):
    def test_gradients_match_central_finite_differences(self):
        rng = np.random.default_rng(3)
        x, y = rng.standard_normal((5, 4)), rng.standard_normal((5, 1))
        w1, w2 = rng.standard_normal((4, 6)) * 0.5, rng.standard_normal((6, 1)) * 0.5
        _, g1, g2 = ddp_check.loss_and_gradients(x, y, w1, w2)
        eps = 1e-6
        for w, g, which in ((w1, g1, 0), (w2, g2, 1)):
            for idx in [(0, 0), (1, 0), (w.shape[0] - 1, w.shape[1] - 1)]:
                up, down = w.copy(), w.copy()
                up[idx] += eps
                down[idx] -= eps
                args_up = (x, y, up, w2) if which == 0 else (x, y, w1, up)
                args_down = (x, y, down, w2) if which == 0 else (x, y, w1, down)
                numeric = (ddp_check.loss_and_gradients(*args_up)[0] - ddp_check.loss_and_gradients(*args_down)[0]) / (2 * eps)
                self.assertAlmostEqual(g[idx], numeric, places=6)

    def test_gradients_match_the_real_autograd(self):
        x, y, w1, w2 = ddp_check.make_problem()
        g = ComputationGraph()
        g.register_tensor("w1", w1.copy(), requires_grad=True)
        g.register_tensor("w2", w2.copy(), requires_grad=True)
        pre = x @ w1
        act = ddp_check.gelu(pre).astype(np.float32)
        pred = act @ w2
        g.tensors.update({"x": x, "y_true": y, "pre": pre, "act": act, "pred": pred,
                          "loss": np.asarray(((pred - y) ** 2).mean(), dtype=np.float32)})
        g.add_operation("matmul", "pre", ["x", "w1"], {})
        g.add_operation("gelu", "act", ["pre"], {})
        g.add_operation("matmul", "pred", ["act", "w2"], {})
        g.add_operation("mse_loss", "loss", ["pred", "y_true"], {})
        g.backward("loss", scale=1.0)
        _, r1, r2 = ddp_check.loss_and_gradients(x, y, w1, w2)
        self.assertLess(ddp_check.rel_diff(g.gradients["w1"], r1), 1e-4)
        self.assertLess(ddp_check.rel_diff(g.gradients["w2"], r2), 1e-4)

    def test_the_mean_of_equal_shard_gradients_is_the_whole_batch_gradient(self):
        x, y, w1, w2 = ddp_check.make_problem()
        _, full1, full2 = ddp_check.loss_and_gradients(x, y, w1, w2)
        for ranks in (2, 4):
            rows = len(x) // ranks
            parts = [ddp_check.loss_and_gradients(x[r * rows:(r + 1) * rows], y[r * rows:(r + 1) * rows], w1, w2)
                     for r in range(ranks)]
            self.assertLess(ddp_check.rel_diff(np.mean([p[1] for p in parts], axis=0), full1), 1e-12)
            self.assertLess(ddp_check.rel_diff(np.mean([p[2] for p in parts], axis=0), full2), 1e-12)

    def test_training_changes_the_weights(self):
        x, y, w1, w2 = ddp_check.make_problem()
        a, b = ddp_check.reference_train(x, y, w1, w2)
        self.assertGreater(ddp_check.rel_diff(a, w1), 1e-3)
        self.assertGreater(ddp_check.rel_diff(b, w2), 1e-3)


class ProgramTests(unittest.TestCase):
    def test_the_generated_programs_are_valid_tensorlang_for_every_rank(self):
        for rows, world in ((64, 1), (32, 2), (16, 4)):
            for rank in range(world):
                with self.subTest(rows=rows, world=world, rank=rank):
                    src = ddp_check.program_text("build/ddp_check/dir", rows)
                    with mock.patch.dict(os.environ, {**PLAIN_ENV, "RANK": str(rank), "WORLD_SIZE": str(world)}, clear=True):
                        ast, _, _ = ast_builder.build_ast(PARSER.parse(src), source_code=src)
                    with redirect_stdout(io.StringIO()):
                        ok, env = type_checker.type_checker(copy.deepcopy(ast), {}, world_size=world)
                    self.assertTrue(ok)
                    self.assertIn("w1_grad", env)
                    self.assertIn("w2_grad", env)
                    paths = sorted(p for p in repr(ast).split("'file_path': ")[1:])
                    self.assertTrue(any(f"x_{rank}.npy" in p for p in paths))

    def test_every_rank_loads_its_own_shard_and_saves_into_its_own_cache_directory(self):
        src = ddp_check.program_text("build/ddp_check/w2", 32)
        self.assertIn('load("build/ddp_check/w2/x_{rank}.npy")', src)
        self.assertIn('save(w1, "w1_final.npy")', src)                       # relative: lands in the rank's cache directory


class ProblemFilesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_shards_cover_the_whole_batch_and_rank_0_starts_from_the_true_initial_weights(self):
        ddp_check.write_problem(self.root, 4)
        base = self.root / "build" / "ddp_check"
        x_full = np.load(base / "full" / "x_0.npy")
        np.testing.assert_array_equal(np.concatenate([np.load(base / "w4" / f"x_{r}.npy") for r in range(4)]), x_full)
        np.testing.assert_array_equal(np.load(base / "w4" / "w1_0.npy"), np.load(base / "full" / "w1_0.npy"))

    def test_the_other_ranks_start_from_different_weights_so_the_broadcast_is_tested(self):
        ddp_check.write_problem(self.root, 2)
        base = self.root / "build" / "ddp_check" / "w2"
        self.assertFalse(np.array_equal(np.load(base / "w1_0.npy"), np.load(base / "w1_1.npy")))
        self.assertFalse(np.array_equal(np.load(base / "w2_0.npy"), np.load(base / "w2_1.npy")))

    def test_the_program_paths_are_relative_to_the_root(self):
        full, ddp = ddp_check.write_problem(self.root, 2)
        self.assertEqual((str(full), str(ddp)), ("build/ddp_check/full.tl", "build/ddp_check/w2.tl"))
        self.assertTrue((self.root / full).exists() and (self.root / ddp).exists())

    def test_a_rank_count_that_does_not_divide_the_batch_is_refused(self):
        with self.assertRaises(ValueError):
            ddp_check.write_problem(self.root, 3)

    def test_old_files_are_replaced(self):
        stale = self.root / "build" / "ddp_check" / "stale.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("old")
        ddp_check.write_problem(self.root, 2)
        self.assertFalse(stale.exists())


def simulated_run(root, *, average=True, broadcast=True, drift=False, lr_scale=1.0, fail=None, train=True, calls=None):
    """A stand-in for `python3 tensorlang.py ...` that does the training in NumPy, with optional faults."""
    root = Path(root)

    def run(cmd, cwd=None, env=None, capture_output=None, text=None):
        args = cmd[2:]
        if calls is not None:
            calls.append({"args": args, "gpu_backward": (env or {}).get("TENSORLANG_GPU_BACKWARD")})
        done = subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        ddp = "--ddp" in args
        if fail == ("ddp" if ddp else "single"):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom from the simulated run")
        lr = ddp_check.LR * lr_scale if train else 0.0
        steps = ddp_check.STEPS

        def load(d, name, r):
            return np.load(root / "build" / "ddp_check" / d / f"{name}_{r}.npy").astype(np.float64)

        def save(path, w1, w2):
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / "w1_final.npy", w1.astype(np.float32))
            np.save(path / "w2_final.npy", w2.astype(np.float32))

        cache = root / "cache" / "build" / "ddp_check"
        if not ddp:
            x, y, w1, w2 = (load("full", n, 0) for n in ("x", "y", "w1", "w2"))
            for _ in range(steps):
                _, g1, g2 = ddp_check.loss_and_gradients(x, y, w1, w2)
                w1, w2 = w1 - lr * g1, w2 - lr * g2
            save(cache / "full.tl", w1, w2)
            return done
        n = int(args[args.index("--nproc") + 1])
        d = f"w{n}"
        xs, ys = [load(d, "x", r) for r in range(n)], [load(d, "y", r) for r in range(n)]
        w1s, w2s = [load(d, "w1", r) for r in range(n)], [load(d, "w2", r) for r in range(n)]
        if broadcast:
            w1s, w2s = [w1s[0].copy() for _ in range(n)], [w2s[0].copy() for _ in range(n)]
        for _ in range(steps):
            grads = [ddp_check.loss_and_gradients(xs[r], ys[r], w1s[r], w2s[r]) for r in range(n)]
            for r in range(n):
                g1 = np.mean([g[1] for g in grads], axis=0) if average else grads[r][1]
                g2 = np.mean([g[2] for g in grads], axis=0) if average else grads[r][2]
                w1s[r], w2s[r] = w1s[r] - lr * g1, w2s[r] - lr * g2
        if drift:
            w1s[-1] = w1s[-1] + 1e-6
        for r in range(n):
            save(cache / f"{d}.tl" / f"ws{n}" / f"rank{r}", w1s[r], w2s[r])
        return done

    return run


class CheckLogicTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def run_tool(self, argv, run):
        out = io.StringIO()
        with redirect_stdout(out):
            code = ddp_check.main(["--root", str(self.root), *argv], run=run)
        return code, out.getvalue()

    def test_a_correct_data_parallel_run_passes_every_check(self):
        code, out = self.run_tool(["--backward", "numpy"], simulated_run(self.root))
        self.assertEqual(code, 0, out)
        self.assertIn("all checks passed", out)
        for label in ("ranks identical", "ddp vs single", "weights moved"):
            self.assertRegex(out, rf"PASS\s+{label}")

    def test_four_ranks_pass_too(self):
        code, out = self.run_tool(["--ranks", "4", "--backward", "numpy"], simulated_run(self.root))
        self.assertEqual(code, 0, out)

    def test_missing_gradient_averaging_is_caught(self):
        code, out = self.run_tool(["--backward", "numpy"], simulated_run(self.root, average=False))
        self.assertEqual(code, 1)
        self.assertRegex(out, r"FAIL\s+ddp vs single")

    def test_missing_parameter_broadcast_is_caught(self):
        code, out = self.run_tool(["--backward", "numpy"], simulated_run(self.root, broadcast=False))
        self.assertEqual(code, 1)
        self.assertRegex(out, r"FAIL\s+ranks identical")

    def test_a_single_drifting_rank_is_caught(self):
        code, out = self.run_tool(["--backward", "numpy"], simulated_run(self.root, drift=True))
        self.assertEqual(code, 1)
        self.assertRegex(out, r"FAIL\s+ranks identical")
        self.assertIn("rank(s) [1]", out)

    def test_a_run_that_does_not_train_is_caught(self):
        code, out = self.run_tool(["--backward", "numpy"], simulated_run(self.root, train=False))
        self.assertEqual(code, 1)
        self.assertRegex(out, r"FAIL\s+weights moved")

    def test_a_failing_single_process_run_is_reported_with_its_output(self):
        code, out = self.run_tool(["--backward", "numpy"], simulated_run(self.root, fail="single"))
        self.assertEqual(code, 1)
        self.assertIn("single-process run", out)
        self.assertIn("boom from the simulated run", out)

    def test_a_failing_data_parallel_run_is_reported_with_its_output(self):
        code, out = self.run_tool(["--backward", "numpy"], simulated_run(self.root, fail="ddp"))
        self.assertEqual(code, 1)
        self.assertIn("data-parallel run", out)
        self.assertIn("boom from the simulated run", out)

    def test_a_difference_from_the_float64_reference_is_only_a_note(self):
        # both runs use a slightly different learning rate, as a different GELU approximation might: they agree with
        # each other but not exactly with the reference
        code, out = self.run_tool(["--backward", "numpy"], simulated_run(self.root, lr_scale=1.3))
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"note\s+vs float64 reference")

    def test_both_backward_modes_run_by_default_and_only_the_gpu_mode_sets_the_variable(self):
        calls = []
        with mock.patch.dict(os.environ, PLAIN_ENV, clear=True):
            code, out = self.run_tool([], simulated_run(self.root, calls=calls))
        self.assertEqual(code, 0, out)
        self.assertEqual([c["gpu_backward"] for c in calls], [None, None, "1", "1"])    # single, ddp, single, ddp

    def test_the_variable_from_the_callers_environment_does_not_leak_into_the_numpy_mode(self):
        calls = []
        with mock.patch.dict(os.environ, {**PLAIN_ENV, "TENSORLANG_GPU_BACKWARD": "1"}, clear=True):
            self.run_tool(["--backward", "numpy"], simulated_run(self.root, calls=calls))
        self.assertEqual({c["gpu_backward"] for c in calls}, {None})

    def test_the_launch_flags_are_passed_through(self):
        calls = []
        self.run_tool(["--ranks", "4", "--same-gpu", "--backward", "numpy"], simulated_run(self.root, calls=calls))
        ddp_args = [c["args"] for c in calls if "--ddp" in c["args"]][0]
        self.assertEqual(ddp_args[:5], ["--ddp", "--nproc", "4", "--ddp-same-gpu", "build/ddp_check/w4.tl"])
        single_args = [c["args"] for c in calls if "--ddp" not in c["args"]][0]
        self.assertEqual(single_args, ["build/ddp_check/full.tl"])

    def test_files_are_cleaned_up_unless_kept(self):
        self.run_tool(["--backward", "numpy"], simulated_run(self.root))
        self.assertFalse((self.root / "build" / "ddp_check").exists())
        self.assertFalse((self.root / "cache" / "build" / "ddp_check").exists())
        self.run_tool(["--backward", "numpy", "--keep"], simulated_run(self.root))
        self.assertTrue((self.root / "build" / "ddp_check" / "full.tl").exists())

    def test_stale_outputs_from_an_earlier_run_are_removed_before_the_run(self):
        stale = self.root / "cache" / "build" / "ddp_check" / "full.tl"
        stale.mkdir(parents=True)
        np.save(stale / "w1_final.npy", np.zeros((8, 16), np.float32))
        seen = []
        inner = simulated_run(self.root)

        def run(cmd, **kwargs):
            seen.append((stale / "w1_final.npy").exists())
            return inner(cmd, **kwargs)

        code, _ = self.run_tool(["--backward", "numpy"], run)
        self.assertEqual(code, 0)
        self.assertEqual(seen[0], False)                                       # gone before the first program ran

    def test_a_run_that_exits_normally_without_writing_its_outputs_is_a_clean_failure(self):
        def silent(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        code, out = self.run_tool(["--backward", "numpy"], silent)
        self.assertEqual(code, 1)
        self.assertRegex(out, r"FAIL\s+outputs")
        self.assertIn("did not write", out)


if __name__ == "__main__":
    unittest.main()
