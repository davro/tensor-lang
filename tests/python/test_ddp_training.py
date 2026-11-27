"""Data-parallel training on the CPU: the real autograd graph on simulated ranks (threads over loopback sockets), each
holding a different shard of the data. No GPU is used.

What this proves: gradients averaged across ranks equal the gradients of the whole batch, a few SGD steps stay in lockstep
and match single-process training, the overflow flag is shared, and rank 0's parameters reach every rank. What it cannot
prove is the wiring inside tensorlang/execution.py on a GPU; the last class only checks that the calls are where they must
be in the source.

    python3 -m unittest discover -s tests/python -v
"""
import math
import re
import socket
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tensorlang import dist  # noqa: E402
from tensorlang.autograd import ComputationGraph  # noqa: E402

D, H = 6, 8
PARAMS = ("w1", "b1", "w2")
_erf = np.vectorize(math.erf)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_ranks(world, fn):
    port = free_port()
    results, errors = [None] * world, [None] * world

    def work(r):
        comm = None
        try:
            env = {"RANK": str(r), "WORLD_SIZE": str(world), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                   "TENSORLANG_DIST_TIMEOUT": "20"}
            comm = dist.init(environ=env)
            results[r] = fn(comm)
        except BaseException as exc:                    # noqa: BLE001
            errors[r] = exc
        finally:
            if comm is not None:
                comm.close()

    threads = [threading.Thread(target=work, args=(r,), daemon=True) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads), "a rank is still running"
    return results, errors


def make_data(n=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, D)).astype(np.float32),
            {"w1": (rng.standard_normal((D, H)) * 0.4).astype(np.float32),
             "b1": (rng.standard_normal(H) * 0.1).astype(np.float32),
             "w2": (rng.standard_normal((H, 1)) * 0.4).astype(np.float32)})


def gelu(x):
    return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))).astype(np.float32)


def build_graph(x, params):
    """loss = mean(gelu(x @ w1 + b1) @ w2), built on the real ComputationGraph with the forward values filled in."""
    g = ComputationGraph()
    for name, value in params.items():
        g.register_tensor(name, value.copy(), requires_grad=True)
    h1 = x @ params["w1"]
    h2 = h1 + params["b1"]
    a = gelu(h2)
    y = a @ params["w2"]
    g.tensors.update({"x": x, "h1": h1, "h2": h2, "a": a, "y": y,
                      "loss": np.asarray(y.mean(), dtype=np.float32)})
    g.add_operation("matmul", "h1", ["x", "w1"], {})
    g.add_operation("add", "h2", ["h1", "b1"], {})
    g.add_operation("gelu", "a", ["h2"], {})
    g.add_operation("matmul", "y", ["a", "w2"], {})
    g.add_operation("mean", "loss", ["y"], {})
    return g


def backward(x, params):
    g = build_graph(x, params)
    g.backward("loss", scale=1.0)
    return g, g.reachable_tensors("loss")


def shards(x, world):
    return np.split(x, world)


class SyncGradientsTests(unittest.TestCase):
    def test_gradients_averaged_over_shards_equal_the_full_batch_gradients(self):
        x, params = make_data()
        full, _ = backward(x, params)
        for world in (2, 4):
            with self.subTest(world=world):
                def fn(comm):
                    g, reached = backward(shards(x, world)[comm.rank], params)
                    names = dist.sync_gradients(comm, g, reached)
                    return names, {n: g.gradients[n] for n in PARAMS}, g.overflow_detected
                results, errors = run_ranks(world, fn)
                self.assertEqual(errors, [None] * world)
                for names, grads, overflow in results:
                    self.assertEqual(names, sorted(PARAMS))
                    self.assertFalse(overflow)
                    for n in PARAMS:
                        np.testing.assert_allclose(grads[n], full.gradients[n], rtol=2e-5, atol=1e-7)
                for n in PARAMS:                                    # and identical on every rank, bit for bit
                    self.assertEqual(results[0][1][n].tobytes(), results[-1][1][n].tobytes())

    def test_unsynchronised_gradients_really_do_differ_between_shards(self):
        x, params = make_data()
        a, _ = backward(shards(x, 2)[0], params)
        b, _ = backward(shards(x, 2)[1], params)
        self.assertFalse(np.allclose(a.gradients["w1"], b.gradients["w1"]))      # so the test above proves something

    def test_overflow_on_one_rank_is_seen_by_all(self):
        x, params = make_data(n=18)                                              # divisible by 3 ranks

        def fn(comm):
            shard = shards(x, 3)[comm.rank].copy()
            if comm.rank == 1:
                shard[0, 0] = np.nan
            g, reached = backward(shard, params)
            before = g.overflow_detected
            dist.sync_gradients(comm, g, reached)
            return before, g.overflow_detected
        results, errors = run_ranks(3, fn)
        self.assertEqual(errors, [None] * 3)
        self.assertEqual([r[0] for r in results], [False, True, False])          # only rank 1 saw it locally
        self.assertEqual([r[1] for r in results], [True, True, True])            # afterwards everyone has

    def test_one_rank_changes_nothing(self):
        x, params = make_data()
        g, reached = backward(x, params)
        before = {n: g.gradients[n].copy() for n in PARAMS}
        dist.sync_gradients(dist.LocalComm(), g, reached)
        for n in PARAMS:
            self.assertEqual(g.gradients[n].tobytes(), before[n].tobytes())
            self.assertEqual(g.gradients[n].dtype, before[n].dtype)
        self.assertFalse(g.overflow_detected)

    def test_gradients_that_are_not_exchanged_are_left_alone(self):
        x, params = make_data()
        g, reached = backward(x, params)
        intermediate = g.gradients["h2"].copy()                                  # not a parameter
        dist.sync_gradients(dist.LocalComm(), g, reached)
        np.testing.assert_array_equal(g.gradients["h2"], intermediate)


class BroadcastParametersTests(unittest.TestCase):
    def test_every_rank_receives_rank_0_values(self):
        def fn(comm):
            tensors = {"w": np.full((2, 3), comm.rank + 1, np.float32), "v": np.full(4, 10 * (comm.rank + 1), np.float16),
                       "data": np.full(2, comm.rank, np.float32)}
            dist.broadcast_parameters(comm, tensors, {"w", "v", "not_a_tensor"})
            return tensors
        results, errors = run_ranks(3, fn)
        self.assertEqual(errors, [None] * 3)
        for r, tensors in enumerate(results):
            np.testing.assert_array_equal(tensors["w"], np.full((2, 3), 1))
            np.testing.assert_array_equal(tensors["v"], np.full(4, 10))
            self.assertEqual((tensors["w"].dtype, tensors["v"].dtype), (np.float32, np.float16))
            np.testing.assert_array_equal(tensors["data"], np.full(2, r))        # not a parameter: untouched

    def test_ranks_with_differently_shaped_parameters_fail_with_a_message(self):
        def fn(comm):
            dist.broadcast_parameters(comm, {"w": np.ones((2, 2 + comm.rank), np.float32)}, {"w"})
        _, errors = run_ranks(2, fn)
        self.assertTrue(all(isinstance(e, dist.DistError) for e in errors), errors)
        self.assertIn("disagree", str(errors[0]))


class TrainingEquivalenceTests(unittest.TestCase):
    STEPS, LR = 5, 0.1

    def train(self, comm, x, params, broadcast):
        params = {k: v.copy() for k, v in params.items()}
        if broadcast:
            dist.broadcast_parameters(comm, params, PARAMS)
        shard = shards(x, comm.world_size)[comm.rank]
        for _ in range(self.STEPS):
            g, reached = backward(shard, params)
            dist.sync_gradients(comm, g, reached)
            for n in PARAMS:
                params[n] = (params[n] - np.float32(self.LR) * g.gradients[n]).astype(np.float32)
        return params

    def single_process(self, x, params):
        return self.train(dist.LocalComm(), x, params, broadcast=False)

    def test_sharded_training_matches_single_process_training_and_ranks_stay_identical(self):
        x, params = make_data()
        expected = self.single_process(x, params)
        for world in (2, 4):
            with self.subTest(world=world):
                results, errors = run_ranks(world, lambda comm: self.train(comm, x, params, broadcast=True))
                self.assertEqual(errors, [None] * world)
                for n in PARAMS:
                    for r in range(1, world):
                        self.assertEqual(results[r][n].tobytes(), results[0][n].tobytes(), f"rank {r} drifted on {n}")
                    np.testing.assert_allclose(results[0][n], expected[n], rtol=1e-4, atol=1e-6)
                self.assertFalse(np.allclose(expected["w1"], params["w1"]))      # training really moved the weights

    def test_ranks_that_start_apart_stay_apart_without_the_broadcast_and_agree_with_it(self):
        x, params = make_data()

        def fn(broadcast):
            def run(comm):
                own = {k: v + np.float32(0.05 * comm.rank) for k, v in params.items()}      # each rank starts differently
                return self.train(comm, x, own, broadcast=broadcast)
            return run

        without, errors = run_ranks(2, fn(False))
        self.assertEqual(errors, [None] * 2)
        self.assertFalse(np.allclose(without[0]["w1"], without[1]["w1"]))
        with_broadcast, errors = run_ranks(2, fn(True))
        self.assertEqual(errors, [None] * 2)
        for n in PARAMS:
            self.assertEqual(with_broadcast[0][n].tobytes(), with_broadcast[1][n].tobytes())


class SourceWiringTests(unittest.TestCase):
    """The calls into dist must sit where they have to. This reads the source, which is brittle on purpose: if someone
    moves a call or removes it, a data-parallel run would silently train ranks apart, and this test says so first."""

    def source(self, name):
        return (ROOT / "tensorlang" / name).read_text()

    def test_both_backward_sites_sync_before_the_overflow_flag_is_read(self):
        text = self.source("execution.py")
        sites = [m.start() for m in re.finditer(r"self\.comp_graph\.backward\(", text)]
        self.assertEqual(len(sites), 2, "expected exactly two backward call sites in execution.py")
        for start in sites:
            window = text[start:start + 3000]
            self.assertIn("_ddp_sync_gradients(reached)", window)
            self.assertLess(window.index("_ddp_sync_gradients(reached)"), window.index("comp_graph.overflow_detected"),
                            "the sync must come before the overflow flag is read")
            self.assertLess(window.index("reachable_tensors"), window.index("_ddp_sync_gradients(reached)"))

    def test_the_sync_is_only_for_data_parallel_runs(self):
        text = self.source("execution.py")
        for m in re.finditer(r"self\._ddp_sync_gradients\(reached\)", text):
            if "def _ddp_sync_gradients" in text[max(0, m.start() - 40):m.start()]:
                continue
            self.assertIn("self.world_size > 1", text[max(0, m.start() - 450):m.start()])

    def test_parameters_are_broadcast_after_the_library_loads_and_before_the_upload(self):
        text = self.source("compiler.py")
        load = text.index("cdll.LoadLibrary(os.path.abspath(str(kernel_so_path)))")
        broadcast = text.index("self._ddp_broadcast_parameters(tensors)")
        upload = text.index("gpu_allocs[name] = cuda.mem_alloc(size_bytes)", load)
        self.assertLess(load, broadcast)
        self.assertLess(broadcast, upload)
        self.assertIn("self.world_size > 1", text[broadcast - 400:broadcast])

    def test_data_parallel_errors_are_reported_as_such(self):
        text = self.source("compiler.py")
        self.assertLess(text.index("except DistError as e:"), text.index("except (RuntimeError, OSError, ValueError, TypeError) as e:"))


if __name__ == "__main__":
    unittest.main()
