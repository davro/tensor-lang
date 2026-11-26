"""Tests for tensorlang/dist.py: environment helpers, the host-side collectives and gradient averaging.

No GPU is used. Ranks run as threads (or, for the crash test, as a real child process) talking over loopback sockets.

    python3 -m unittest discover -s tests/python -v
"""
import multiprocessing
import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorlang import dist  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def env_for(rank, world, port, timeout="20"):
    return {"RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": str(rank), "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port), "TENSORLANG_DIST_TIMEOUT": timeout}


def run_ranks(world, fn, timeout="20"):
    """Run fn(comm) on every rank in its own thread; return (results, errors) indexed by rank."""
    port = free_port()
    results, errors = [None] * world, [None] * world

    def work(r):
        comm = None
        try:
            comm = dist.init(environ=env_for(r, world, port, timeout))
            results[r] = fn(comm)
        except BaseException as exc:                    # noqa: BLE001 - collected and asserted on by the tests
            errors[r] = exc
        finally:
            if comm is not None:
                comm.close()

    threads = [threading.Thread(target=work, args=(r,), daemon=True) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads), "a rank is still running: the collective hung"
    return results, errors


class EnvTests(unittest.TestCase):
    def test_defaults_are_single_process(self):
        self.assertEqual(dist.env_info({}), (0, 1, 0))

    def test_reads_rank_world_and_local_rank(self):
        self.assertEqual(dist.env_info({"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"}), (1, 4, 1))

    def test_rejects_an_impossible_environment(self):
        for bad in ({"RANK": "2", "WORLD_SIZE": "2"}, {"RANK": "-1", "WORLD_SIZE": "2"}, {"WORLD_SIZE": "0"},
                    {"RANK": "x"}):
            with self.assertRaises(ValueError):
                dist.env_info(bad)

    def test_placeholders_use_the_environment(self):
        e = {"RANK": "1", "WORLD_SIZE": "2"}
        self.assertEqual(dist.expand_placeholders("data/x_{rank}.npy", e), "data/x_1.npy")
        self.assertEqual(dist.expand_placeholders("out_{rank}_of_{world_size}.npy", e), "out_1_of_2.npy")

    def test_placeholders_default_to_rank_0_of_1(self):
        self.assertEqual(dist.expand_placeholders("x_{rank}.npy", {}), "x_0.npy")

    def test_other_braces_and_plain_paths_are_untouched(self):
        e = {"RANK": "1", "WORLD_SIZE": "2"}
        self.assertEqual(dist.expand_placeholders("a_{name}.npy", e), "a_{name}.npy")
        self.assertEqual(dist.expand_placeholders("plain.npy", e), "plain.npy")


class LocalCommTests(unittest.TestCase):
    def test_world_size_one_needs_no_sockets_and_is_the_identity(self):
        comm = dist.init(environ={})
        self.assertIsInstance(comm, dist.LocalComm)
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        for op in ("sum", "mean", "max", "min"):
            np.testing.assert_array_equal(comm.all_reduce(a, op), a)
        np.testing.assert_array_equal(comm.broadcast(a), a)
        np.testing.assert_array_equal(comm.all_gather(a), a)
        comm.barrier()

    def test_returns_copies(self):
        comm = dist.LocalComm()
        a = np.ones(3, np.float32)
        comm.all_reduce(a)[0] = 9
        self.assertEqual(a[0], 1)

    def test_average_gradients_is_a_no_op_for_one_rank(self):
        g = {"w": np.arange(4, dtype=np.float32).reshape(2, 2), "b": np.array([1.5], np.float32)}
        out, flag = dist.average_gradients(dist.LocalComm(), g, overflow=False)
        self.assertFalse(flag)
        for k in g:
            np.testing.assert_array_equal(out[k], g[k])
            self.assertEqual(out[k].dtype, g[k].dtype)
        _, flag = dist.average_gradients(dist.LocalComm(), g, overflow=True)
        self.assertTrue(flag)

    def test_argument_errors(self):
        comm = dist.LocalComm()
        with self.assertRaises(ValueError):
            comm.all_reduce(np.ones(2), "median")
        with self.assertRaises(ValueError):
            comm.all_reduce(np.ones(2, np.int64), "mean")
        with self.assertRaises(ValueError):
            comm.broadcast(np.ones(2), src=1)
        with self.assertRaises(ValueError):
            comm.all_gather(np.ones(2), axis=3)
        with self.assertRaises(ValueError):
            comm.all_reduce(np.array([object()], dtype=object))


class CollectiveTests(unittest.TestCase):
    def test_all_reduce_sum_is_identical_on_every_rank(self):
        def fn(comm):
            a = np.full((3, 4), comm.rank + 1, dtype=np.float32) * np.float32(0.1)
            return comm.all_reduce(a, "sum")
        results, errors = run_ranks(3, fn)
        self.assertEqual(errors, [None] * 3)
        expected = np.float32(0.1) + np.float32(0.2) + np.float32(0.3)      # rank order 0, 1, 2
        for r in results:
            self.assertEqual(r.tobytes(), results[0].tobytes())              # bit-identical across ranks
            np.testing.assert_allclose(r, np.full((3, 4), expected), rtol=1e-6)

    def test_mean_max_min(self):
        def fn(comm):
            a = np.array([comm.rank, 10 - comm.rank], dtype=np.float32)
            return comm.all_reduce(a, "mean"), comm.all_reduce(a, "max"), comm.all_reduce(a, "min")
        results, errors = run_ranks(4, fn)
        self.assertEqual(errors, [None] * 4)
        for mean, mx, mn in results:
            np.testing.assert_allclose(mean, [1.5, 8.5])
            np.testing.assert_array_equal(mx, [3, 10])
            np.testing.assert_array_equal(mn, [0, 7])

    def test_max_propagates_nan(self):
        def fn(comm):
            a = np.array([1.0, np.nan if comm.rank == 1 else 2.0], dtype=np.float32)
            return comm.all_reduce(a, "max")
        results, errors = run_ranks(2, fn)
        self.assertEqual(errors, [None] * 2)
        for r in results:
            self.assertEqual(r[0], 1.0)
            self.assertTrue(np.isnan(r[1]))

    def test_dtypes_are_preserved(self):
        def fn(comm):
            return [comm.all_reduce(np.array([comm.rank + 1], dtype=dt), "sum").dtype
                    for dt in (np.float16, np.float32, np.float64, np.int64)]
        results, errors = run_ranks(2, fn)
        self.assertEqual(errors, [None] * 2)
        self.assertEqual(results[0], [np.float16, np.float32, np.float64, np.int64])

    def test_broadcast_from_each_source(self):
        def fn(comm):
            mine = np.full(3, comm.rank, dtype=np.float32)
            return comm.broadcast(mine, src=0), comm.broadcast(mine, src=2)
        results, errors = run_ranks(3, fn)
        self.assertEqual(errors, [None] * 3)
        for from0, from2 in results:
            np.testing.assert_array_equal(from0, [0, 0, 0])
            np.testing.assert_array_equal(from2, [2, 2, 2])

    def test_all_gather_allows_different_sizes_along_the_axis(self):
        def fn(comm):
            rows = np.full((comm.rank + 1, 2), comm.rank, dtype=np.float32)
            cols = np.full((2, comm.rank + 1), comm.rank, dtype=np.float32)
            return comm.all_gather(rows, axis=0), comm.all_gather(cols, axis=1)
        results, errors = run_ranks(3, fn)
        self.assertEqual(errors, [None] * 3)
        for by_rows, by_cols in results:
            self.assertEqual(by_rows.shape, (6, 2))
            self.assertEqual(by_cols.shape, (2, 6))
            np.testing.assert_array_equal(by_rows[:, 0], [0, 1, 1, 2, 2, 2])
            np.testing.assert_array_equal(by_cols[0], [0, 1, 1, 2, 2, 2])

    def test_barrier_and_many_collectives_in_sequence(self):
        def fn(comm):
            total = 0.0
            for i in range(40):
                comm.barrier()
                total += float(comm.all_reduce(np.array([i + comm.rank], dtype=np.float64), "sum")[0])
            return total
        results, errors = run_ranks(3, fn)
        self.assertEqual(errors, [None] * 3)
        expected = sum(3 * i + 3 for i in range(40))
        self.assertEqual(results, [expected] * 3)

    def test_large_and_empty_arrays(self):
        def fn(comm):
            big = comm.all_reduce(np.full(500_000, comm.rank + 1, dtype=np.float32), "sum")
            empty = comm.all_reduce(np.zeros((0, 4), dtype=np.float32), "sum")
            return big[0], big[-1], big.shape, empty.shape
        results, errors = run_ranks(2, fn)
        self.assertEqual(errors, [None] * 2)
        self.assertEqual(results[0], (3.0, 3.0, (500_000,), (0, 4)))

    def test_zero_dimensional_arrays(self):
        def fn(comm):
            return float(comm.all_reduce(np.float32(comm.rank + 1), "sum")), comm.all_gather(np.float32(comm.rank)).tolist()
        results, errors = run_ranks(2, fn)
        self.assertEqual(errors, [None] * 2)
        self.assertEqual(results[0], (3.0, [0.0, 1.0]))


class FailureTests(unittest.TestCase):
    def test_ranks_that_disagree_on_shape_fail_fast_with_a_message(self):
        def fn(comm):
            return comm.all_reduce(np.ones(4 if comm.rank != 1 else 5, dtype=np.float32), "sum")
        started = time.monotonic()
        results, errors = run_ranks(3, fn)
        self.assertLess(time.monotonic() - started, 15)                       # failed fast, did not wait for a timeout
        for r, err in enumerate(errors):
            self.assertIsInstance(err, dist.DistError, f"rank {r}: {err!r}")
        self.assertIn("disagree", str(errors[0]))
        self.assertIn("rank 1 called", str(errors[0]))
        self.assertIn("[5]", str(errors[0]))
        self.assertIn("disagree", str(errors[2]))                              # the other ranks hear the reason too

    def test_ranks_that_disagree_on_the_operation_fail_fast(self):
        def fn(comm):
            if comm.rank == 2:
                return comm.barrier()
            return comm.all_reduce(np.ones(2, dtype=np.float32), "sum")
        _, errors = run_ranks(3, fn)
        self.assertTrue(all(isinstance(e, dist.DistError) for e in errors), errors)
        self.assertIn("barrier", str(errors[0]))

    def test_ranks_that_disagree_on_the_reduction_fail(self):
        def fn(comm):
            return comm.all_reduce(np.ones(2, dtype=np.float32), "sum" if comm.rank == 0 else "max")
        _, errors = run_ranks(2, fn)
        self.assertTrue(all(isinstance(e, dist.DistError) for e in errors), errors)

    def test_a_failed_communicator_refuses_further_use(self):
        def fn(comm):
            try:
                comm.all_reduce(np.ones(2 + comm.rank, dtype=np.float32), "sum")
            except dist.DistError:
                pass
            return comm.barrier()
        _, errors = run_ranks(2, fn)
        for err in errors:
            self.assertIsInstance(err, dist.DistError)
            self.assertIn("closed", str(err))

    def test_a_rank_that_vanishes_does_not_hang_the_others(self):
        def fn(comm):
            if comm.rank == 1:
                comm.close()                                                   # dies before the collective
                return "gone"
            return comm.all_reduce(np.ones(2, dtype=np.float32), "sum")
        started = time.monotonic()
        results, errors = run_ranks(3, fn)
        self.assertLess(time.monotonic() - started, 15)
        self.assertIsInstance(errors[0], dist.DistError)
        self.assertIn("rank 1", str(errors[0]))
        self.assertIsInstance(errors[2], dist.DistError)
        self.assertIn("rank 1", str(errors[2]))

    def test_a_rank_that_is_too_slow_times_out_with_a_message(self):
        def fn(comm):
            if comm.rank == 1:
                time.sleep(3)
            return comm.all_reduce(np.ones(2, dtype=np.float32), "sum")
        started = time.monotonic()
        _, errors = run_ranks(3, fn, timeout="1")
        self.assertLess(time.monotonic() - started, 20)
        self.assertIsInstance(errors[0], dist.DistError)
        self.assertIn("did not reach", str(errors[0]))
        self.assertIn("rank 1", str(errors[0]))
        self.assertIsInstance(errors[2], dist.DistError)

    def test_missing_ranks_are_named_at_start_up(self):
        port = free_port()
        with self.assertRaises(dist.DistError) as ctx:
            dist.init(environ=env_for(0, 3, port, timeout="1"))
        self.assertIn("only 1 of 3", str(ctx.exception))
        self.assertIn("[1, 2]", str(ctx.exception))

    def test_a_peer_with_no_rank_0_gives_up_with_a_message(self):
        with self.assertRaises(dist.DistError) as ctx:
            dist.init(environ=env_for(1, 2, free_port(), timeout="1"))
        self.assertIn("could not connect to rank 0", str(ctx.exception))

    def test_a_connection_claiming_the_wrong_world_size_is_rejected(self):
        port = free_port()
        outcome = {}

        def root():
            try:
                dist.init(environ=env_for(0, 2, port, timeout="10"))
            except dist.DistError as exc:
                outcome["error"] = exc

        t = threading.Thread(target=root, daemon=True)
        t.start()
        for _ in range(100):
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError:
                time.sleep(0.05)
        dist._send(s, {"hello": 1, "world": 5})
        t.join(15)
        s.close()
        self.assertIn("rejected a connection", str(outcome.get("error")))

    def test_a_port_that_is_taken_is_reported(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            with self.assertRaises(dist.DistError) as ctx:
                dist.init(environ=env_for(0, 2, busy.getsockname()[1], timeout="1"))
        self.assertIn("MASTER_PORT", str(ctx.exception))

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "needs fork")
    def test_a_process_that_crashes_mid_run_is_detected(self):
        port = free_port()

        def child():
            comm = dist.init(environ=env_for(1, 2, port))
            comm.barrier()
            os._exit(3)                                                        # crash: no clean close, no goodbye

        proc = multiprocessing.get_context("fork").Process(target=child)
        proc.start()
        try:
            comm = dist.init(environ=env_for(0, 2, port))
            comm.barrier()
            with self.assertRaises(dist.DistError) as ctx:
                comm.all_reduce(np.ones(2, dtype=np.float32), "sum")
            self.assertIn("rank 1", str(ctx.exception))
        finally:
            proc.join(10)
        self.assertEqual(proc.exitcode, 3)


class AverageGradientsTests(unittest.TestCase):
    def test_averages_with_shapes_and_dtypes_restored(self):
        def fn(comm):
            k = comm.rank
            grads = {"w": np.full((2, 3), 1 + 2 * k, dtype=np.float32),
                     "b": np.array([1.0 + 2 * k, 3.0 + 2 * k], dtype=np.float32),
                     "h": np.full(4, 2.0 * (k + 1), dtype=np.float16)}
            return dist.average_gradients(comm, grads, overflow=False)
        results, errors = run_ranks(2, fn)
        self.assertEqual(errors, [None] * 2)
        for out, flag in results:
            self.assertFalse(flag)
            np.testing.assert_array_equal(out["w"], np.full((2, 3), 2.0))
            np.testing.assert_array_equal(out["b"], [2.0, 4.0])
            np.testing.assert_array_equal(out["h"], np.full(4, 3.0))
            self.assertEqual((out["w"].dtype, out["b"].dtype, out["h"].dtype), (np.float32, np.float32, np.float16))

    def test_the_overflow_flag_is_shared(self):
        def fn(comm):
            return dist.average_gradients(comm, {"w": np.ones(3, dtype=np.float32)}, overflow=(comm.rank == 1))
        results, errors = run_ranks(3, fn)
        self.assertEqual(errors, [None] * 3)
        self.assertEqual([flag for _, flag in results], [True, True, True])      # one rank overflowed: all agree

    def test_a_nan_gradient_on_one_rank_reaches_every_rank(self):
        def fn(comm):
            g = np.ones(4, dtype=np.float32)
            if comm.rank == 0:
                g[1] = np.nan
            return dist.average_gradients(comm, {"w": g}, overflow=bool(np.isnan(g).any()))
        results, errors = run_ranks(2, fn)
        self.assertEqual(errors, [None] * 2)
        for out, flag in results:
            self.assertTrue(flag)
            self.assertTrue(np.isnan(out["w"][1]))
            self.assertEqual(out["w"][0], 1.0)

    def test_float64_gradients_keep_their_precision(self):
        def fn(comm):
            g = {"w": np.array([1.0 + 1e-12 * (comm.rank + 1)], dtype=np.float64)}
            return dist.average_gradients(comm, g)[0]["w"]
        results, errors = run_ranks(2, fn)
        self.assertEqual(errors, [None] * 2)
        self.assertEqual(results[0].dtype, np.float64)
        self.assertAlmostEqual(float(results[0][0]), 1.0 + 1.5e-12, places=15)

    def test_ranks_with_different_parameters_fail_instead_of_mixing(self):
        def fn(comm):
            g = {"w": np.ones(3, dtype=np.float32)}
            if comm.rank == 1:
                g["extra"] = np.ones(2, dtype=np.float32)
            return dist.average_gradients(comm, g)
        _, errors = run_ranks(2, fn)
        self.assertTrue(all(isinstance(e, dist.DistError) for e in errors), errors)

    def test_matches_a_single_process_mean_over_the_full_batch(self):
        rng = np.random.default_rng(0)
        shards = [rng.standard_normal((4, 3)).astype(np.float32) for _ in range(4)]
        def fn(comm):
            return dist.average_gradients(comm, {"g": shards[comm.rank]})[0]["g"]
        results, errors = run_ranks(4, fn)
        self.assertEqual(errors, [None] * 4)
        np.testing.assert_allclose(results[0], np.mean(shards, axis=0), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
