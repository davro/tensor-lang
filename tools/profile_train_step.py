#!/usr/bin/env python3
"""Profile where a TensorLang run spends its time: GPU wait, device<->host copies, NumPy backward.

    python3 tools/profile_train_step.py tests/embedding_train_loop.tl
    python3 tools/profile_train_step.py tests/mlp_network.tl --gpu 0

It wraps three things, without changing any behaviour:
  * ComputationGraph.backward      -> the NumPy backward (CPU)
  * pycuda memcpy_dtoh / htod      -> device-to-host and host-to-device copies
  * a Context.synchronize() before every dtoh copy, timed separately as "GPU wait"
    (kernel launches are asynchronous, so without this the first dtoh copy would be
    charged with the forward kernels that are still running).

"rest" is whatever is left of the wall clock: compile, parsing, kernel launches, optimizer
kernels, printing. Run with the debug flags OFF, since printing tensors distorts the numbers.
Use a warm run (run it twice; the first one may compile the kernel).
"""
import argparse
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # the compiler opens tensorlang.lark relative to the repo root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--cprofile", type=int, default=0, metavar="N",
                    help="also print the N most expensive Python functions (by cumulative time)")
    a = ap.parse_args()
    t_proc = time.perf_counter()

    if a.gpu is not None:
        from tensorlang.gpu import pin_current_process
        pin_current_process(a.gpu)  # must happen before pycuda creates its context

    s0 = time.perf_counter()
    import pycuda.autoinit  # noqa: F401  (creates the context)
    import pycuda.driver as cuda
    t_cuda_init = time.perf_counter() - s0
    s0 = time.perf_counter()
    from tensorlang.autograd import ComputationGraph
    from tensorlang.compiler import TensorCompiler
    t_import = time.perf_counter() - s0

    t = defaultdict(float)
    n = defaultdict(int)
    nbytes = defaultdict(int)

    def timed(key, fn, *args, **kw):
        s = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            t[key] += time.perf_counter() - s
            n[key] += 1

    orig_dtoh, orig_htod = cuda.memcpy_dtoh, cuda.memcpy_htod
    orig_backward = ComputationGraph.backward

    inside = {"bw": False}      # copies made INSIDE the backward call are reported under it, not counted twice

    def dtoh(dest, src, *args, **kw):
        pre = "bw_" if inside["bw"] else ""
        timed(pre + "gpu_wait", cuda.Context.synchronize)
        nbytes[pre + "dtoh"] += getattr(dest, "nbytes", 0)
        return timed(pre + "dtoh", orig_dtoh, dest, src, *args, **kw)

    def htod(dest, src, *args, **kw):
        pre = "bw_" if inside["bw"] else ""
        nbytes[pre + "htod"] += getattr(src, "nbytes", 0)
        return timed(pre + "htod", orig_htod, dest, src, *args, **kw)

    def backward(self, *args, **kw):
        inside["bw"] = True
        try:
            return timed("backward", orig_backward, self, *args, **kw)
        finally:
            inside["bw"] = False

    cuda.memcpy_dtoh, cuda.memcpy_htod = dtoh, htod
    ComputationGraph.backward = backward

    compiler = TensorCompiler(debug_mode=False, debug_info=False, debug_ast=False, cache_layers=False)
    prof = None
    if a.cprofile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    start = time.perf_counter()
    ok = compiler.compile_and_execute(a.file)
    total = time.perf_counter() - start
    if prof:
        prof.disable()

    cuda.memcpy_dtoh, cuda.memcpy_htod = orig_dtoh, orig_htod
    ComputationGraph.backward = orig_backward

    outside = ("gpu_wait", "dtoh", "htod", "backward")     # the bw_* keys are already inside "backward"
    accounted = sum(t[k] for k in outside)
    print("\n" + "=" * 64)
    print(f"profile of {a.file}   (ok={ok})")
    print("=" * 64)
    print(f"{'phase':<18}{'seconds':>10}{'share':>9}{'calls':>8}{'MB':>10}")
    for key, label in [("gpu_wait", "GPU wait"), ("dtoh", "copy GPU->host"),
                       ("htod", "copy host->GPU"), ("backward", "backward (total)")]:
        mb = nbytes.get(key, 0) / 1e6 if key in nbytes else 0.0
        print(f"{label:<18}{t[key]:>10.3f}{100 * t[key] / total:>8.1f}%{n[key]:>8}{mb:>10.2f}")
    bw_copy = t["bw_gpu_wait"] + t["bw_dtoh"] + t["bw_htod"]
    print(f"{'  of which copies':<18}{bw_copy:>10.3f}{100 * bw_copy / total:>8.1f}%{n['bw_dtoh'] + n['bw_htod']:>8}"
          f"{(nbytes['bw_dtoh'] + nbytes['bw_htod']) / 1e6:>10.2f}   (inside backward: GPU backward uploads and downloads)")
    print(f"{'  of which compute':<18}{t['backward'] - bw_copy:>10.3f}{100 * (t['backward'] - bw_copy) / total:>8.1f}%"
          f"   (NumPy ops and GPU kernels)")
    rest = total - accounted
    print(f"{'rest':<18}{rest:>10.3f}{100 * rest / total:>8.1f}%")
    print(f"{'TOTAL wall':<18}{total:>10.3f}")
    print(f"\nbefore compile_and_execute: CUDA context + pycuda import {t_cuda_init:.3f} s, "
          f"tensorlang imports {t_import:.3f} s (not in TOTAL wall above)")
    if prof:
        import pstats
        print("\nTop functions by cumulative time:")
        pstats.Stats(prof).sort_stats("cumulative").print_stats(a.cprofile)
    try:
        from tensorlang import gpu_backward as _gb
        if _gb.enabled():
            print(f"\nGPU backward in use: {_gb.STATS['matmul']} matmul and {_gb.STATS['gelu']} gelu backward calls")
    except ImportError:
        pass
    print("\nNotes: 'rest' includes compile, launches, optimizer kernels and printing; "
          "'backward (total)' includes the copies it makes itself, shown on the lines below it.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
