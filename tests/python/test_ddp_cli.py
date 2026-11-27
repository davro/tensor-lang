"""Tests for the tensorlang.py entry point (lazy imports, the --ddp hook, --gpu ordering) and for gpu.pin_env().

The compiler, test runner and app runner are replaced by fakes, so nothing here needs a GPU or pycuda. The one thing these
tests exist to protect: tensorlang.compiler imports pycuda.autoinit at module level, which creates a CUDA context, so the
entry point must import it only after --gpu has been applied and never in the --ddp parent.

    python3 -m unittest discover -s tests/python -v
"""
import importlib.abc
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tensorlang import gpu, launcher  # noqa: E402

CLI_PATH = ROOT / "tensorlang.py"
CLEAN_ENV = {k: v for k, v in os.environ.items()
             if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TENSORLANG_DDP_WORKER",
                          "CUDA_VISIBLE_DEVICES")}


def load_cli():
    spec = importlib.util.spec_from_file_location("tensorlang_cli", CLI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_compiler(ok=True):
    calls = []

    class TensorCompiler:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def compile_and_execute(self, path):
            calls.append(("run", path))
            return ok

    mod = types.ModuleType("tensorlang.compiler")
    mod.TensorCompiler = TensorCompiler
    return mod, calls


def run_main(argv, modules=None, env=None, cli=None):
    """Run the CLI's main() with the given argv; return (exit code or None, stdout, stderr)."""
    cli = cli or load_cli()
    out, err = io.StringIO(), io.StringIO()
    code = None
    with mock.patch.object(sys, "argv", ["tensorlang.py", *argv]), \
            mock.patch.dict(sys.modules, modules or {}), \
            mock.patch.dict(os.environ, CLEAN_ENV if env is None else {**CLEAN_ENV, **env}, clear=True), \
            redirect_stdout(out), redirect_stderr(err):
        try:
            cli.main()
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class ImportTests(unittest.TestCase):
    def test_importing_the_entry_point_creates_no_cuda_context(self):
        code = ("import importlib.util, sys\n"
                "spec = importlib.util.spec_from_file_location('tensorlang_cli', 'tensorlang.py')\n"
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                "print('tensorlang.compiler' in sys.modules, "
                "any(n == 'pycuda' or n.startswith('pycuda.') for n in sys.modules))")
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(done.stdout.strip(), "False False", done.stderr)


class NormalModeTests(unittest.TestCase):
    def test_a_file_is_compiled_with_the_flags_it_was_given(self):
        mod, calls = fake_compiler()
        code, _, _ = run_main(["prog.tl", "--debug", "--cache-layers"], {"tensorlang.compiler": mod})
        self.assertIsNone(code)
        self.assertEqual(calls[1], ("run", "prog.tl"))
        kwargs = calls[0][1]
        self.assertTrue(kwargs["debug_mode"])
        self.assertTrue(kwargs["cache_layers"])
        self.assertFalse(kwargs["debug_info"])

    def test_a_failed_run_exits_with_one(self):
        mod, _ = fake_compiler(ok=False)
        code, _, _ = run_main(["prog.tl"], {"tensorlang.compiler": mod})
        self.assertEqual(code, 1)

    def test_no_file_is_an_error(self):
        mod, calls = fake_compiler()
        code, out, _ = run_main([], {"tensorlang.compiler": mod})
        self.assertEqual(code, 1)
        self.assertIn("No file specified", out)
        self.assertEqual(calls, [])

    def test_list_gpus_needs_neither_pycuda_nor_the_compiler(self):
        cli = load_cli()
        gpus = [{"index": 0, "name": "A", "memory_free_mb": 1, "memory_total_mb": 2, "utilization_pct": 3},
                {"index": 1, "name": "B", "memory_free_mb": 4, "memory_total_mb": 5, "utilization_pct": 6}]
        with mock.patch.object(cli, "detect_gpus", return_value=gpus):
            code, out, _ = run_main(["--list-gpus"], cli=cli)
        self.assertEqual(code, 0)
        self.assertIn("Detected 2 GPU(s)", out)


class GpuPinningOrderTests(unittest.TestCase):
    def test_gpu_is_pinned_before_the_compiler_is_imported(self):
        seen = {}
        mod, calls = fake_compiler()

        class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, name, path, target=None):
                if name == "tensorlang.compiler":
                    return importlib.util.spec_from_loader(name, self)
                return None

            def create_module(self, spec):
                return None

            def exec_module(self, module):                     # runs when (and only when) the CLI imports the compiler
                seen["visible_devices_at_import"] = os.environ.get("CUDA_VISIBLE_DEVICES")
                module.TensorCompiler = mod.TensorCompiler

        cli = load_cli()
        finder = Finder()
        with mock.patch.dict(sys.modules):
            sys.modules.pop("tensorlang.compiler", None)
            sys.meta_path.insert(0, finder)
            try:
                with mock.patch.object(cli, "detect_gpus", return_value=[]):
                    run_main(["prog.tl", "--gpu", "1"], cli=cli)
            finally:
                sys.meta_path.remove(finder)
        self.assertEqual(seen.get("visible_devices_at_import"), "1")
        self.assertEqual(calls[0][0], "init")


class OtherModesTests(unittest.TestCase):
    def test_test_mode_builds_the_runner_with_the_same_arguments_as_before(self):
        built = {}

        class TestRunner:
            def __init__(self, **kwargs):
                built.update(kwargs)

            def discover_tests(self):
                return ["a_one.tl", "b_two.tl"]

            def run_test_suite(self, files):
                built["files"] = files

        mod = types.ModuleType("tensorlang.test_runner")
        mod.TestRunner = TestRunner
        code, _, _ = run_main(["--test", "--no-parallel", "--filter", "a_"], {"tensorlang.test_runner": mod})
        self.assertIsNone(code)
        self.assertEqual({k: built[k] for k in ("parallel", "verify_tensors", "debug_mode", "gpu_pinning", "fixed_gpu")},
                         {"parallel": False, "verify_tensors": False, "debug_mode": False, "gpu_pinning": True,
                          "fixed_gpu": None})
        self.assertIsNone(built["jobs"])
        self.assertEqual(built["files"], ["a_one.tl"])

    def test_app_mode_lists_apps(self):
        listed = []

        class AppRunner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def list_apps(self, category=None):
                listed.append(category)

        mod = types.ModuleType("tensorlang.app_runner")
        mod.AppRunner = AppRunner
        code, _, _ = run_main(["--list-apps", "--category", "ml"], {"tensorlang.app_runner": mod})
        self.assertEqual(code, 0)
        self.assertEqual(listed, ["ml"])


class DdpHookTests(unittest.TestCase):
    def test_ddp_hands_over_to_the_launcher_before_anything_heavy_is_imported(self):
        cli = load_cli()
        seen = []

        def fake_maybe_launch(args):
            seen.append((args.ddp, args.nproc, args.ddp_same_gpu, args.file))
            raise SystemExit(7)                                # launch_ddp never returns: it exits with the workers' code

        mod, calls = fake_compiler()
        with mock.patch.object(cli, "maybe_launch", fake_maybe_launch):
            code, _, _ = run_main(["--ddp", "--nproc", "3", "--ddp-same-gpu", "prog.tl"], {"tensorlang.compiler": mod},
                                  cli=cli)
        self.assertEqual(code, 7)
        self.assertEqual(seen, [(True, 3, True, "prog.tl")])
        self.assertEqual(calls, [])

    def test_a_launcher_spawned_worker_runs_the_program_without_launching_again(self):
        mod, calls = fake_compiler()
        env = {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1", "TENSORLANG_DDP_WORKER": "1", "CUDA_VISIBLE_DEVICES": "1"}
        code, _, err = run_main(["--ddp", "--nproc", "2", "prog.tl"], {"tensorlang.compiler": mod}, env=env)
        self.assertIsNone(code)
        self.assertEqual(calls[1], ("run", "prog.tl"))
        self.assertEqual(err, "")                              # no warning for the normal worker case

    def test_a_worker_is_not_pinned_again(self):
        mod, _ = fake_compiler()
        env = {"RANK": "1", "WORLD_SIZE": "2", "TENSORLANG_DDP_WORKER": "1", "CUDA_VISIBLE_DEVICES": "1"}
        cli = load_cli()
        with mock.patch.object(cli, "pin_current_process") as pin:
            run_main(["--ddp", "prog.tl"], {"tensorlang.compiler": mod}, env=env, cli=cli)
        pin.assert_not_called()                                # --ddp and --gpu are mutually exclusive, so no re-pin


class PinEnvTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_dir = os.path.join(tmp.name, "ddp_logs")

    def test_the_gpu_is_pinned_and_the_rest_of_the_environment_is_kept(self):
        env = gpu.pin_env(2, base_env={"PATH": "/bin", "HOME": "/h"})
        self.assertEqual(env, {"PATH": "/bin", "HOME": "/h", "CUDA_VISIBLE_DEVICES": "2"})

    def test_distributed_variables_do_not_leak_into_a_pinned_child(self):
        base = {"PATH": "/bin", "RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2", "MASTER_ADDR": "x",
                "MASTER_PORT": "1", "TENSORLANG_DDP_WORKER": "1"}
        env = gpu.pin_env(0, base_env=base)
        self.assertEqual(env, {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"})

    def test_the_input_mapping_is_not_modified(self):
        base = {"RANK": "1", "PATH": "/bin"}
        gpu.pin_env(0, base_env=base)
        self.assertEqual(base, {"RANK": "1", "PATH": "/bin"})

    def test_it_defaults_to_the_current_environment(self):
        with mock.patch.dict(os.environ, {"RANK": "3", "SOME_OTHER_VAR": "keep"}):
            env = gpu.pin_env(1)
        self.assertNotIn("RANK", env)
        self.assertEqual(env["SOME_OTHER_VAR"], "keep")

    def test_every_variable_the_launcher_sets_is_one_pin_env_removes(self):
        captured = []

        def popen(cmd, env=None, stdout=None):
            captured.append(env)
            return types.SimpleNamespace(poll=lambda: 0, wait=lambda timeout=None: 0, terminate=lambda: None,
                                         kill=lambda: None)

        launcher.run_workers(2, ["x"], environ={}, detect=lambda: [0, 1], pin=gpu.pin_env, popen=popen,
                             sleep=lambda s: None, log_dir=self.log_dir, install_signal_handlers=False)
        # what the launcher adds on top of pin_env's own result, apart from the BLAS thread caps (a tuning choice that is
        # not stripped), must be exactly what pin_env strips
        added = set(captured[0]) - set(gpu.pin_env(0)) - set(launcher.BLAS_THREAD_VARS)
        self.assertEqual(added, set(gpu.DDP_ENV_VARS))

    def test_a_stale_rank_in_the_parent_does_not_reach_the_workers(self):
        captured = []

        def popen(cmd, env=None, stdout=None):
            captured.append(env)
            return types.SimpleNamespace(poll=lambda: 0, wait=lambda timeout=None: 0, terminate=lambda: None,
                                         kill=lambda: None)

        with mock.patch.dict(os.environ, {"RANK": "9", "WORLD_SIZE": "99", "PATH": os.environ.get("PATH", "")}):
            launcher.run_workers(2, ["x"], detect=lambda: [0, 1], pin=gpu.pin_env, popen=popen, sleep=lambda s: None,
                                 log_dir=self.log_dir, install_signal_handlers=False)
        self.assertEqual([(e["RANK"], e["WORLD_SIZE"]) for e in captured], [("0", "2"), ("1", "2")])


if __name__ == "__main__":
    unittest.main()
