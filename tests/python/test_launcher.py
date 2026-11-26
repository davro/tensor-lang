"""Tests for tensorlang/launcher.py. No GPU is used: GPU detection and pin_env are injected, and most tests use a fake Popen.
A few tests start real child processes (plain `python -c`) to check termination and exit codes end to end.

    python3 -m unittest discover -s tests/python -v
"""
import io
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorlang import launcher  # noqa: E402


class FakeProc:
    """A stand-in for Popen: poll() walks through `polls`, then keeps returning the last value."""

    def __init__(self, polls=(0,), ignore_terminate=False, on_poll=None):
        self.polls = list(polls)
        self.ignore_terminate = ignore_terminate
        self.on_poll = on_poll
        self.terminated = self.killed = False
        self.returncode = None

    def poll(self):
        if self.on_poll:
            cb, self.on_poll = self.on_poll, None
            cb()
        if self.terminated and not self.ignore_terminate:
            self.returncode = -signal.SIGTERM
        elif self.killed:
            self.returncode = -signal.SIGKILL
        elif len(self.polls) > 1:
            self.returncode = self.polls.pop(0)
        else:
            self.returncode = self.polls[0]
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.terminated and not self.ignore_terminate:
            self.returncode = -signal.SIGTERM
            return self.returncode
        if self.killed:
            self.returncode = -signal.SIGKILL
            return self.returncode
        if self.polls == [None] or (self.ignore_terminate and not self.killed):
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.poll()


class Spawner:
    """A fake Popen factory that records how each rank was started."""

    def __init__(self, procs):
        self.procs = list(procs)
        self.calls = []

    def __call__(self, cmd, env=None, stdout=None):
        self.calls.append({"cmd": cmd, "env": env, "stdout": stdout})
        return self.procs[len(self.calls) - 1]


def fake_pin(gpu_index):
    env = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": str(gpu_index)}
    return env


class RunWorkersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_dir = os.path.join(self.tmp.name, "logs")

    def run_fake(self, procs, nproc=None, same_gpu=False, environ=None, detect=lambda: [0, 1, 2, 3], **kw):
        spawner = Spawner(procs)
        code = launcher.run_workers(
            nproc or len(procs), ["tensorlang.py", "--ddp", "--nproc", "2", "prog.tl"], same_gpu,
            environ={} if environ is None else environ, detect=detect, pin=fake_pin, popen=spawner,
            sleep=lambda s: None, python="PY", log_dir=self.log_dir, install_signal_handlers=False, **kw)
        return code, spawner

    # -- GPU selection ---------------------------------------------------------------------------------------

    def test_each_rank_gets_its_own_detected_gpu(self):
        _, sp = self.run_fake([FakeProc(), FakeProc()])
        self.assertEqual([c["env"]["CUDA_VISIBLE_DEVICES"] for c in sp.calls], ["0", "1"])

    def test_a_parent_cuda_visible_devices_is_respected_in_order(self):
        called = []
        _, sp = self.run_fake([FakeProc(), FakeProc()], environ={"CUDA_VISIBLE_DEVICES": "3,1"},
                              detect=lambda: called.append(1) or [0, 1])
        self.assertEqual([c["env"]["CUDA_VISIBLE_DEVICES"] for c in sp.calls], ["3", "1"])
        self.assertEqual(called, [])                                           # no need to ask nvidia-smi

    def test_non_numeric_cuda_visible_devices_is_an_error(self):
        with self.assertRaises(launcher.LaunchError) as ctx:
            self.run_fake([FakeProc(), FakeProc()], environ={"CUDA_VISIBLE_DEVICES": "GPU-abc,GPU-def"})
        self.assertIn("numeric", str(ctx.exception))

    def test_more_ranks_than_gpus_fails_before_starting_anything(self):
        spawner = Spawner([FakeProc()] * 3)
        with self.assertRaises(launcher.LaunchError) as ctx:
            launcher.run_workers(3, ["tensorlang.py"], environ={}, detect=lambda: [0, 1], pin=fake_pin,
                                 popen=spawner, log_dir=self.log_dir, install_signal_handlers=False)
        self.assertIn("only 2 GPU(s)", str(ctx.exception))
        self.assertIn("--ddp-same-gpu", str(ctx.exception))
        self.assertEqual(spawner.calls, [])

    def test_same_gpu_puts_every_rank_on_the_first_gpu(self):
        _, sp = self.run_fake([FakeProc(), FakeProc(), FakeProc()], same_gpu=True, detect=lambda: [2])
        self.assertEqual([c["env"]["CUDA_VISIBLE_DEVICES"] for c in sp.calls], ["2", "2", "2"])

    def test_same_gpu_with_no_gpu_at_all_is_an_error(self):
        with self.assertRaises(launcher.LaunchError):
            self.run_fake([FakeProc(), FakeProc()], same_gpu=True, detect=lambda: [])

    def test_nproc_must_be_positive(self):
        with self.assertRaises(launcher.LaunchError):
            launcher.run_workers(0, ["x"], environ={}, detect=lambda: [0], pin=fake_pin, popen=Spawner([]),
                                 install_signal_handlers=False)

    # -- what the workers are given ---------------------------------------------------------------------------

    def test_worker_environment(self):
        _, sp = self.run_fake([FakeProc(), FakeProc(), FakeProc()])
        ports = {c["env"]["MASTER_PORT"] for c in sp.calls}
        self.assertEqual(len(ports), 1)
        for r, call in enumerate(sp.calls):
            env = call["env"]
            self.assertEqual((env["RANK"], env["LOCAL_RANK"], env["WORLD_SIZE"]), (str(r), str(r), "3"))
            self.assertEqual(env["MASTER_ADDR"], "127.0.0.1")
            self.assertEqual(env["TENSORLANG_DDP_WORKER"], "1")
            self.assertEqual(env["PATH"], "/bin")                              # pin_env's environment is the base

    def test_the_command_line_is_forwarded_unchanged(self):
        _, sp = self.run_fake([FakeProc(), FakeProc()])
        for call in sp.calls:
            self.assertEqual(call["cmd"], ["PY", "tensorlang.py", "--ddp", "--nproc", "2", "prog.tl"])

    def test_master_port_from_the_parent_wins(self):
        _, sp = self.run_fake([FakeProc(), FakeProc()], environ={"MASTER_PORT": "31337", "MASTER_ADDR": "127.0.0.2"})
        self.assertEqual({c["env"]["MASTER_PORT"] for c in sp.calls}, {"31337"})
        self.assertEqual({c["env"]["MASTER_ADDR"] for c in sp.calls}, {"127.0.0.2"})

    def test_rank_0_keeps_the_terminal_and_others_log_to_files(self):
        _, sp = self.run_fake([FakeProc(), FakeProc(), FakeProc()])
        self.assertIsNone(sp.calls[0]["stdout"])
        for r in (1, 2):
            self.assertTrue(sp.calls[r]["stdout"].closed)                      # closed by the launcher when done
            self.assertTrue(os.path.exists(os.path.join(self.log_dir, f"rank{r}.log")))
        self.assertFalse(os.path.exists(os.path.join(self.log_dir, "rank0.log")))

    # -- outcomes ---------------------------------------------------------------------------------------------

    def test_all_workers_succeeding_returns_zero_and_stops_nothing(self):
        procs = [FakeProc([None, None, 0]), FakeProc([None, 0])]
        code, _ = self.run_fake(procs)
        self.assertEqual(code, 0)
        self.assertFalse(any(p.terminated for p in procs))

    def test_a_failing_worker_stops_the_others_and_sets_the_exit_code(self):
        procs = [FakeProc([None]), FakeProc([None, None, 3]), FakeProc([None])]
        err = io.StringIO()
        with redirect_stderr(err):
            code, _ = self.run_fake(procs)
        self.assertEqual(code, 3)
        self.assertTrue(procs[0].terminated and procs[2].terminated)
        self.assertIn("rank 1 exited with code 3", err.getvalue())
        self.assertIn("rank1.log", err.getvalue())                             # tells you where its output went

    def test_the_exit_code_is_the_largest_failure(self):
        procs = [FakeProc([2]), FakeProc([5]), FakeProc([None])]
        with redirect_stderr(io.StringIO()):
            code, _ = self.run_fake(procs)
        self.assertEqual(code, 5)

    def test_a_worker_killed_by_a_signal_counts_as_128_plus_the_signal(self):
        procs = [FakeProc([None]), FakeProc([-signal.SIGSEGV])]
        with redirect_stderr(io.StringIO()):
            code, _ = self.run_fake(procs)
        self.assertEqual(code, 128 + signal.SIGSEGV)

    def test_a_worker_that_ignores_terminate_is_killed(self):
        stubborn = FakeProc([None], ignore_terminate=True)
        procs = [FakeProc([1]), stubborn]
        with redirect_stderr(io.StringIO()):
            code, _ = self.run_fake(procs, grace=0.01)
        self.assertEqual(code, 1)
        self.assertTrue(stubborn.terminated and stubborn.killed)

    def test_a_worker_that_cannot_be_spawned_stops_the_ones_already_running(self):
        first = FakeProc([None])

        def popen(cmd, env=None, stdout=None):
            if env["RANK"] == "1":
                raise OSError("no such interpreter")
            return first

        with self.assertRaises(launcher.LaunchError) as ctx:
            launcher.run_workers(2, ["x"], environ={}, detect=lambda: [0, 1], pin=fake_pin, popen=popen,
                                 sleep=lambda s: None, log_dir=self.log_dir, install_signal_handlers=False)
        self.assertIn("rank 1", str(ctx.exception))
        self.assertTrue(first.terminated)

    def test_ctrl_c_stops_every_worker(self):
        procs = [FakeProc([None], on_poll=lambda: os.kill(os.getpid(), signal.SIGINT)), FakeProc([None])]
        spawner = Spawner(procs)
        before = signal.getsignal(signal.SIGINT)
        code = launcher.run_workers(2, ["x"], environ={}, detect=lambda: [0, 1], pin=fake_pin, popen=spawner,
                                    sleep=lambda s: None, log_dir=self.log_dir, grace=0.01)
        self.assertEqual(code, 128 + signal.SIGINT)
        self.assertTrue(all(p.terminated for p in procs))
        self.assertIs(signal.getsignal(signal.SIGINT), before)                 # the original handler is back

    def test_launch_ddp_exits_with_the_code_and_reports_launch_errors(self):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
            launcher.launch_ddp(3, ["x"], environ={}, detect=lambda: [0], pin=fake_pin, popen=Spawner([]),
                                log_dir=self.log_dir, install_signal_handlers=False)
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("only 1 GPU(s)", err.getvalue())                          # the reason is printed for the user
        with self.assertRaises(SystemExit) as ctx:
            launcher.launch_ddp(2, ["x"], environ={}, detect=lambda: [0, 1], pin=fake_pin,
                                popen=Spawner([FakeProc([0]), FakeProc([0])]), sleep=lambda s: None,
                                log_dir=self.log_dir, install_signal_handlers=False)
        self.assertEqual(ctx.exception.code, 0)


class PortTests(unittest.TestCase):
    def test_the_default_port_is_used_when_it_is_free(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.assertEqual(launcher.free_port(port), port)

    def test_another_port_is_chosen_when_the_default_is_busy(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            port = busy.getsockname()[1]
            other = launcher.free_port(port)
        self.assertNotEqual(other, port)
        self.assertGreater(other, 0)


class MaybeLaunchTests(unittest.TestCase):
    def args(self, **kw):
        return types.SimpleNamespace(**{"ddp": False, "nproc": 2, "ddp_same_gpu": False, **kw})

    def test_without_ddp_nothing_happens(self):
        calls = []
        self.assertFalse(launcher.maybe_launch(self.args(), environ={}, launch=lambda *a, **k: calls.append(a)))
        self.assertEqual(calls, [])

    def test_the_parent_launches_with_its_argv_and_flags(self):
        calls = []
        done = launcher.maybe_launch(self.args(ddp=True, nproc=3, ddp_same_gpu=True), argv=["tensorlang.py", "p.tl"],
                                     environ={}, launch=lambda n, argv, same_gpu: calls.append((n, argv, same_gpu)))
        self.assertTrue(done)
        self.assertEqual(calls, [(3, ["tensorlang.py", "p.tl"], True)])

    def test_a_launcher_spawned_worker_runs_silently_without_relaunching(self):
        err = io.StringIO()
        with redirect_stderr(err):
            done = launcher.maybe_launch(self.args(ddp=True), environ={"RANK": "1", "TENSORLANG_DDP_WORKER": "1"},
                                         launch=lambda *a, **k: self.fail("a worker must not launch"))
        self.assertFalse(done)
        self.assertEqual(err.getvalue(), "")

    def test_a_rank_exported_by_someone_else_gets_a_note(self):
        err = io.StringIO()
        with redirect_stderr(err):
            done = launcher.maybe_launch(self.args(ddp=True), environ={"RANK": "0"},
                                         launch=lambda *a, **k: self.fail("must not launch"))
        self.assertFalse(done)
        self.assertIn("RANK=0 is already set", err.getvalue())


class RealProcessTests(unittest.TestCase):
    """End to end with real child processes (plain Python, no GPU)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_dir = os.path.join(self.tmp.name, "logs")

    def run_real(self, code, nproc=2, **kw):
        return launcher.run_workers(nproc, ["-c", code], environ={"MASTER_PORT": "29555"}, detect=lambda: [0, 1, 2],
                                    pin=lambda g: {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)},
                                    log_dir=self.log_dir, poll_interval=0.05, grace=3.0,
                                    install_signal_handlers=False, **kw)

    def test_every_rank_sees_its_own_environment(self):
        out = Path(self.tmp.name)
        script = ("import os, pathlib; r = os.environ['RANK']; "
                  f"pathlib.Path(r'{out}', 'rank' + r + '.txt').write_text("
                  "' '.join(os.environ[k] for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES', 'MASTER_PORT')))")
        self.assertEqual(self.run_real(script, nproc=3), 0)
        for r in range(3):
            self.assertEqual((out / f"rank{r}.txt").read_text(), f"{r} {r} 3 {r} 29555")

    def test_rank_1_stdout_goes_to_its_log(self):
        self.assertEqual(self.run_real("import os\nif os.environ['RANK'] == '1':\n    print('hello from rank 1')"), 0)
        self.assertEqual(Path(self.log_dir, "rank1.log").read_text().strip(), "hello from rank 1")

    def test_a_crash_in_one_rank_stops_a_rank_that_would_wait_forever(self):
        script = ("import os, sys, time\n"
                  "if os.environ['RANK'] == '1':\n    sys.exit(3)\n"
                  "time.sleep(120)\n")
        started = time.monotonic()
        with redirect_stderr(io.StringIO()):
            code = self.run_real(script)
        self.assertEqual(code, 3)
        self.assertLess(time.monotonic() - started, 30)                       # did not wait for the 120 s sleep

    def test_a_worker_that_ignores_sigterm_is_killed(self):
        script = ("import os, signal, sys, time\n"
                  "if os.environ['RANK'] == '1':\n    sys.exit(4)\n"
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                  "time.sleep(120)\n")
        started = time.monotonic()
        with redirect_stderr(io.StringIO()):
            code = self.run_real(script)
        self.assertEqual(code, 4)
        self.assertLess(time.monotonic() - started, 30)


if __name__ == "__main__":
    unittest.main()
