"""Tests for tensorlang/kernel_cache.py: cache directories for data-parallel runs and the shared, locked kernel build.

No GPU or nvcc is used; the compile step is a function the tests supply. Several tests start real processes (fork) to race
for the build.

    python3 -m unittest discover -s tests/python -v
"""
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorlang import kernel_cache  # noqa: E402
from tensorlang.kernel_cache import KernelCacheError, build_kernel_shared, cache_dirs  # noqa: E402

CODE = "// kernel A\n__global__ void k() {}\n"
OTHER = "// kernel B\n__global__ void k2() {}\n"

FORK = "fork" in multiprocessing.get_all_start_methods()


def fake_nvcc(log=None, delay=0.0):
    """A compile function that copies the source to the library path and records each call."""

    def compile_fn(cu, so):
        if delay:
            time.sleep(delay)
        if log is not None:
            with open(log, "a") as f:
                f.write(f"{os.getpid()} {Path(cu).name} {Path(so).name}\n")
        Path(so).write_bytes(Path(cu).read_bytes())

    return compile_fn


class CacheDirsTests(unittest.TestCase):
    def test_an_ordinary_run_keeps_one_directory(self):
        kernel_dir, out_dir = cache_dirs("cache", Path("build/prog.tl"))
        self.assertEqual(kernel_dir, Path("cache/build/prog.tl"))
        self.assertEqual(out_dir, kernel_dir)

    def test_world_size_one_is_an_ordinary_run_whatever_the_rank_says(self):
        self.assertEqual(cache_dirs("cache", "p.tl", rank=0, world_size=1), (Path("cache/p.tl"), Path("cache/p.tl")))

    def test_a_data_parallel_run_shares_the_kernel_and_separates_the_outputs(self):
        for rank in (0, 1, 2):
            kernel_dir, out_dir = cache_dirs("cache", Path("build/prog.tl"), rank=rank, world_size=3)
            self.assertEqual(kernel_dir, Path("cache/build/prog.tl/ws3"))
            self.assertEqual(out_dir, Path(f"cache/build/prog.tl/ws3/rank{rank}"))

    def test_different_world_sizes_never_share_a_directory(self):
        a, _ = cache_dirs("cache", "p.tl", rank=0, world_size=2)
        b, _ = cache_dirs("cache", "p.tl", rank=0, world_size=4)
        self.assertNotEqual(a, b)

    def test_the_outputs_of_two_ranks_never_collide(self):
        _, out0 = cache_dirs("cache", "p.tl", rank=0, world_size=2)
        _, out1 = cache_dirs("cache", "p.tl", rank=1, world_size=2)
        self.assertNotEqual(out0, out1)


class BuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "ws2"
        self.log = Path(tmp.name) / "compiles.log"

    def compiles(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def leftovers(self):
        return sorted(p.name for p in self.dir.iterdir() if "tmp" in p.name)

    def test_the_first_call_builds_and_the_second_reuses(self):
        self.assertTrue(build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id=None))
        self.assertEqual((self.dir / "kernel.cu").read_text(), CODE)
        self.assertEqual((self.dir / "kernel.so").read_text(), CODE)
        self.assertFalse(build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id=None))
        self.assertEqual(len(self.compiles()), 1)
        self.assertEqual(self.leftovers(), [])

    def test_changed_code_is_rebuilt(self):
        build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id=None)
        self.assertTrue(build_kernel_shared(self.dir, OTHER, fake_nvcc(self.log), run_id=None))
        self.assertEqual((self.dir / "kernel.so").read_text(), OTHER)
        self.assertEqual(len(self.compiles()), 2)

    def test_the_compiler_is_given_temporary_names_that_nvcc_accepts(self):
        build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id=None)
        _, cu_name, so_name = self.compiles()[0].split()
        self.assertTrue(cu_name.endswith(".cu") and "tmp" in cu_name)
        self.assertTrue(so_name.endswith(".so") and "tmp" in so_name)

    def test_a_failed_build_leaves_nothing_behind_and_does_not_block_the_next(self):
        def failing(cu, so):
            Path(so).write_bytes(b"half a library")
            raise RuntimeError("nvcc failed")

        with self.assertRaises(RuntimeError):
            build_kernel_shared(self.dir, CODE, failing, run_id=None)
        self.assertFalse((self.dir / "kernel.so").exists())
        self.assertFalse((self.dir / "kernel.cu").exists())
        self.assertEqual(self.leftovers(), [])
        self.assertTrue(build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id=None))     # lock was released

    def test_a_failed_rebuild_keeps_the_previous_good_kernel(self):
        build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id=None)

        def failing(cu, so):
            raise RuntimeError("nvcc failed")

        with self.assertRaises(RuntimeError):
            build_kernel_shared(self.dir, OTHER, failing, run_id=None)
        self.assertEqual((self.dir / "kernel.cu").read_text(), CODE)
        self.assertEqual((self.dir / "kernel.so").read_text(), CODE)

    def test_a_source_without_a_matching_library_is_not_trusted(self):
        self.dir.mkdir(parents=True)
        (self.dir / "kernel.cu").write_text(CODE)                                # as a crashed old-style build leaves it
        self.assertTrue(build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id=None))
        self.assertTrue((self.dir / "kernel.so").exists())

    def test_threads_racing_for_the_build_compile_exactly_once(self):
        results = []

        def work():
            results.append(build_kernel_shared(self.dir, CODE, fake_nvcc(self.log, delay=0.3), run_id=None))

        threads = [threading.Thread(target=work) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(sorted(results), [False, False, False, False, True])
        self.assertEqual(len(self.compiles()), 1)

    @unittest.skipUnless(FORK, "needs fork")
    def test_processes_racing_for_the_build_compile_exactly_once(self):
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()

        def work():
            queue.put(build_kernel_shared(self.dir, CODE, fake_nvcc(self.log, delay=0.4), run_id=None))

        procs = [ctx.Process(target=work) for _ in range(4)]
        for p in procs:
            p.start()
        results = [queue.get(timeout=30) for _ in procs]
        for p in procs:
            p.join(30)
        self.assertEqual(sorted(results), [False, False, False, True])
        self.assertEqual(len(self.compiles()), 1)
        self.assertEqual((self.dir / "kernel.so").read_text(), CODE)
        self.assertEqual(self.leftovers(), [])

    @unittest.skipUnless(FORK, "needs fork")
    def test_a_build_that_dies_holding_the_lock_does_not_block_the_others(self):
        ctx = multiprocessing.get_context("fork")
        started = ctx.Event()

        def dying():
            def die(cu, so):
                started.set()
                os._exit(1)                                                      # no cleanup, lock still held

            build_kernel_shared(self.dir, CODE, die, run_id=None)

        p = ctx.Process(target=dying)
        p.start()
        self.assertTrue(started.wait(15))
        p.join(15)
        began = time.monotonic()
        self.assertTrue(build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id=None))
        self.assertLess(time.monotonic() - began, 10)                            # the operating system freed the lock
        self.assertEqual((self.dir / "kernel.so").read_text(), CODE)

    # -- the same-run guard ----------------------------------------------------------------------------------

    def test_ranks_of_one_run_that_generate_different_code_are_stopped(self):
        build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id="run1")
        with self.assertRaises(KernelCacheError) as ctx:
            build_kernel_shared(self.dir, OTHER, fake_nvcc(self.log), run_id="run1")
        self.assertIn("must not depend on the rank", str(ctx.exception))
        self.assertEqual((self.dir / "kernel.cu").read_text(), CODE)             # nothing was overwritten
        self.assertEqual(len(self.compiles()), 1)

    def test_ranks_of_one_run_that_generate_the_same_code_share_the_build(self):
        self.assertTrue(build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id="run1"))
        self.assertFalse(build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id="run1"))

    def test_a_new_run_may_rebuild_with_different_code(self):
        build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id="run1")
        self.assertTrue(build_kernel_shared(self.dir, OTHER, fake_nvcc(self.log), run_id="run2"))
        self.assertEqual((self.dir / "kernel.cu").read_text(), OTHER)

    def test_the_run_id_defaults_to_the_launchers_environment_variable(self):
        with mock.patch.dict(os.environ, {kernel_cache.RUN_ID_ENV: "fromenv"}):
            build_kernel_shared(self.dir, CODE, fake_nvcc(self.log))
            with self.assertRaises(KernelCacheError):
                build_kernel_shared(self.dir, OTHER, fake_nvcc(self.log))

    def test_without_a_run_id_the_guard_is_off(self):
        env = {k: v for k, v in os.environ.items() if k != kernel_cache.RUN_ID_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            build_kernel_shared(self.dir, CODE, fake_nvcc(self.log))
            self.assertTrue(build_kernel_shared(self.dir, OTHER, fake_nvcc(self.log)))

    def test_a_platform_without_file_locking_says_so(self):
        with mock.patch.object(kernel_cache, "fcntl", None):
            with self.assertRaises(KernelCacheError) as ctx:
                build_kernel_shared(self.dir, CODE, fake_nvcc(self.log), run_id=None)
        self.assertIn("file locking", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
