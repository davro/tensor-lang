#!/usr/bin/env python3
"""Where does one GPU matmul-backward call spend its time?

    python3 tools/bench_gpu_backward.py

Needs a GPU and pycuda; run it from anywhere inside the repo. Read-only, takes a few seconds. For each shape it times the whole
tensorlang.gpu_backward.matmul_backward call, then each part on its own: the five device allocations and frees, the three
uploads, the two gemm kernels, the two downloads. "other" is what is left (NumPy conversions, dtype checks, argument boxing).

The shapes are the matmuls of the 4-layer micro-GPT in docs/training.md (sequence 64, width 128, feed-forward 512). Times are
microseconds per call, averaged over many calls after a warm-up. The parts do not add up exactly: allocation and copies overlap
differently when they run together than when they run alone, so read the shares as a guide, not a ledger.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tensorlang import gpu_backward as gbw

REPS = 300
SHAPES = [(64, 128, 128), (64, 128, 512), (64, 512, 128)]   # (M, K, N) for C = a @ b


def per_call_us(fn, cuda):
    for _ in range(10):
        fn()
    cuda.Context.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    cuda.Context.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6


def main():
    dev = gbw._get_device()
    if dev is None:
        print("no usable device (see the message above)")
        return 1
    cuda = dev.cuda
    rng = np.random.default_rng(0)

    print(f"{REPS} calls per measurement; microseconds per matmul-backward call\n")
    header = f"{'shape (M,K,N)':<16}{'whole':>9}{'alloc+free':>12}{'upload':>9}{'gemm x2':>9}{'download':>10}{'other':>8}"
    print(header)
    print("-" * len(header))

    for M, K, N in SHAPES:
        a = rng.random((M, K), dtype=np.float32)
        b = rng.random((K, N), dtype=np.float32)
        og = rng.random((M, N), dtype=np.float32)

        whole = per_call_us(lambda: gbw.matmul_backward(a, b, og), cuda)

        sizes = (a.nbytes, b.nbytes, og.nbytes, a.nbytes, b.nbytes)
        alloc_free = per_call_us(lambda: dev.free(*[dev.alloc(n) for n in sizes]), cuda)

        pa, pb, pg = dev.upload(a), dev.upload(b), dev.upload(og)
        p_ga, p_gb = dev.alloc(a.nbytes), dev.alloc(b.nbytes)
        upload = per_call_us(lambda: (cuda.memcpy_htod(pa, a), cuda.memcpy_htod(pb, b), cuda.memcpy_htod(pg, og)), cuda)
        gemm = per_call_us(lambda: (
            dev.gemm(pg, pb, p_ga, M, K, N, N, N, K, 0, 1, M * N, K * N, M * K, 1),
            dev.gemm(pa, pg, p_gb, K, N, M, K, N, N, 1, 0, M * K, M * N, K * N, 1)), cuda)
        download = per_call_us(lambda: (dev.download(p_ga, a.shape), dev.download(p_gb, b.shape)), cuda)
        dev.free(pa, pb, pg, p_ga, p_gb)

        other = whole - (alloc_free + upload + gemm + download)
        print(f"{str((M, K, N)):<16}{whole:9.0f}{alloc_free:12.0f}{upload:9.0f}{gemm:9.0f}{download:10.0f}{other:8.0f}")

    print("\nIf alloc+free is a large share of 'whole', a buffer pool in gpu_backward._Device removes it without changing any result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
