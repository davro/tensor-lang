"""Spawn one worker process per GPU for a data-parallel run:  python3 tensorlang.py --ddp --nproc N file.tl

The launcher is the parent. It starts N copies of the same command line, each with its own RANK and its own GPU, and
waits for them. If any worker exits with an error it stops the others (they would otherwise wait forever for a peer
that is gone) and exits with that worker's code. Ctrl-C or SIGTERM in the parent stops every worker.

This module imports no CUDA code: the parent must never create a GPU context of its own, or it would claim GPU 0.
`tensorlang.gpu` (nvidia-smi based) is imported lazily and only when its helpers are not injected.

Worker environment (the torchrun convention):
    RANK, LOCAL_RANK   0..N-1 (one node)
    WORLD_SIZE         N
    MASTER_ADDR/PORT   127.0.0.1 and a free port (MASTER_PORT from the parent's environment wins)
    TENSORLANG_DDP_WORKER  1, so a worker can tell it was started by this launcher
    CUDA_VISIBLE_DEVICES  this rank's GPU, from pin_env()

GPU choice: if the parent's CUDA_VISIBLE_DEVICES is set, rank r takes its r-th entry; otherwise the r-th detected GPU.
Asking for more ranks than GPUs is an error before anything starts, unless same_gpu=True (development and CI only).

Output: rank 0 keeps the terminal; ranks 1..N-1 write stdout to cache/ddp_logs/rank{r}.log; stderr is shared.
"""
import os
import signal
import socket
import subprocess
import sys
import threading
import time

DEFAULT_PORT = 29500
LOG_DIR = os.path.join("cache", "ddp_logs")


class LaunchError(RuntimeError):
    """The run cannot be started (not enough GPUs, bad environment, a worker could not be spawned)."""


def _detected_gpu_indices():
    from tensorlang.gpu import detect_gpus
    return [g["index"] for g in detect_gpus()]


def _default_pin(gpu_index):
    from tensorlang.gpu import pin_env
    return pin_env(gpu_index)


def physical_gpus(environ=None, detect=None):
    """GPU indices this launch may use: the parent's CUDA_VISIBLE_DEVICES if set, otherwise every detected GPU."""
    e = os.environ if environ is None else environ
    cvd = e.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        entries = [x.strip() for x in cvd.split(",") if x.strip()]
        try:
            return [int(x) for x in entries]
        except ValueError:
            raise LaunchError("CUDA_VISIBLE_DEVICES contains entries that are not GPU indices "
                              f"({cvd!r}); --ddp needs numeric indices") from None
    return list((detect or _detected_gpu_indices)())


def free_port(default=DEFAULT_PORT):
    """`default` if nothing is listening on it, otherwise any free port."""
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", default))
            return default
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _exit_code(returncode):
    """A shell-style exit code: a worker killed by signal n counts as 128 + n."""
    return returncode if returncode > 0 else 128 + (-returncode)


def _stop(procs, popen_grace):
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass
    for p in procs:
        try:
            p.wait(timeout=popen_grace)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except OSError:
                pass
            p.wait()


def run_workers(nproc, argv, same_gpu=False, *, environ=None, detect=None, pin=None, popen=subprocess.Popen,
                sleep=time.sleep, python=None, log_dir=LOG_DIR, poll_interval=0.2, grace=5.0,
                install_signal_handlers=True):
    """Run `argv` (the parent's sys.argv, script name first) as nproc ranks and return the exit code."""
    if nproc < 1:
        raise LaunchError("--nproc must be at least 1")
    env_in = dict(os.environ if environ is None else environ)
    gpus = physical_gpus(env_in, detect)
    if same_gpu:
        gpus = [gpus[0]] * nproc if gpus else []
    if len(gpus) < nproc:
        raise LaunchError(f"--nproc {nproc} but only {len(gpus)} GPU(s) available "
                          "(--ddp-same-gpu runs several ranks on one GPU, for development and tests)")

    port = int(env_in["MASTER_PORT"]) if env_in.get("MASTER_PORT") else free_port()
    pin = pin or _default_pin
    python = python or sys.executable
    os.makedirs(log_dir, exist_ok=True)

    procs, logs = [], []
    interrupted = []

    def on_signal(signum, _frame):
        interrupted.append(signum)

    previous = {}
    if install_signal_handlers and threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, on_signal)

    try:
        for r in range(nproc):
            env = dict(pin(gpus[r]))
            env.update(RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE=str(nproc),
                       MASTER_ADDR=env_in.get("MASTER_ADDR", "127.0.0.1"), MASTER_PORT=str(port),
                       TENSORLANG_DDP_WORKER="1")
            out = None
            if r > 0:
                out = open(os.path.join(log_dir, f"rank{r}.log"), "w")
                logs.append(out)
            try:
                procs.append(popen([python, *argv], env=env, stdout=out))
            except OSError as exc:
                _stop(procs, grace)
                raise LaunchError(f"could not start rank {r}: {exc}") from None

        failed = []
        while True:
            if interrupted:
                _stop(procs, grace)
                return 128 + interrupted[0]
            codes = [p.poll() for p in procs]
            failed = [(r, c) for r, c in enumerate(codes) if c not in (None, 0)]
            if failed:
                break
            if all(c == 0 for c in codes):
                return 0
            sleep(poll_interval)

        for r, c in failed:
            where = f" (stdout in {os.path.join(log_dir, f'rank{r}.log')})" if r > 0 else ""
            print(f"[ddp] rank {r} exited with code {c}{where}; stopping the other ranks", file=sys.stderr)
        _stop(procs, grace)
        return max(_exit_code(c) for _, c in failed)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for f in logs:
            f.close()


def launch_ddp(nproc, argv, same_gpu=False, **kwargs):
    """Run the workers and exit with their code. Never returns."""
    try:
        code = run_workers(nproc, argv, same_gpu, **kwargs)
    except LaunchError as exc:
        print(f"[ddp] {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(code)


def maybe_launch(args, argv=None, environ=None, launch=launch_ddp):
    """Call at the top of main(), before any GPU work.

    In the parent of a --ddp run this starts the workers and exits when they finish. In a worker (RANK is already in
    the environment) or without --ddp it returns False, and the caller carries on as a normal single run. Workers get
    the parent's command line unchanged, --ddp included, so a launcher-spawned worker is silent; a RANK that someone
    else exported gets a note, because the process will then run as that rank instead of launching anything."""
    e = os.environ if environ is None else environ
    if not getattr(args, "ddp", False):
        return False
    if "RANK" in e:
        if "TENSORLANG_DDP_WORKER" not in e:
            print(f"[ddp] RANK={e['RANK']} is already set, so --ddp is ignored and this process runs as that rank",
                  file=sys.stderr)
        return False
    launch(args.nproc, sys.argv if argv is None else argv, same_gpu=getattr(args, "ddp_same_gpu", False))
    return True
