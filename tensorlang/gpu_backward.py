"""GPU kernels for the backward pass of matmul and gelu (stage 1 of a GPU backward).

Off by default. Switch on with the environment variable  TENSORLANG_GPU_BACKWARD=1  (it is inherited by the test
runner's subprocesses, so  TENSORLANG_GPU_BACKWARD=1 python3 tensorlang.py --test  runs the whole suite through it).
TENSORLANG_GPU_BACKWARD=2 does the same and prints one summary line at exit ("[GPU BACKWARD] used for N matmul and M gelu
backward calls"), which also shows up in the test runner's per-test logs, so a suite run can be checked for really using it.
TENSORLANG_GPU_BACKWARD_MIN=<n> skips the GPU for operands smaller than n elements (default 0: always use it).
TENSORLANG_GPU_BACKWARD_POOL=0 turns off the device-memory pool (default on). Every call used to cudaMalloc and cudaFree its five
buffers, about 260 microseconds per matmul backward; the pool reuses them. Results are identical either way.

Stage 1 keeps autograd's host-side bookkeeping: the operands go up, the gradients come back as NumPy arrays and are
accumulated by the existing code. Every function returns None whenever it cannot handle a case (non-float32 operands,
broadcast batch dimensions, no usable device); the caller then runs the original NumPy code, which stays the reference.

Kernels are generic (shapes are arguments) and compiled once per process by pycuda; pycuda caches the binary on disk.
"""
import atexit
import os
import numpy as np

ENV = "TENSORLANG_GPU_BACKWARD"
MIN_ENV = "TENSORLANG_GPU_BACKWARD_MIN"
POOL_ENV = "TENSORLANG_GPU_BACKWARD_POOL"
TILE = 16

SOURCE = r"""
#define TILE 16
// C[b] (M x N) = op(A[b]) (M x K) @ op(B[b]) (K x N); A and B are row-major with leading dimensions lda and ldb.
// transA: A is stored K x M (op(A)[m][k] = A[k*lda + m]); transB: B is stored N x K (op(B)[k][n] = B[n*ldb + k]).
__global__ void gemm(const float* A, const float* B, float* C, int M, int N, int K,
                     int lda, int ldb, int ldc, int transA, int transB,
                     long long strideA, long long strideB, long long strideC) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int batch = blockIdx.z;
    A += (size_t)batch * strideA;
    B += (size_t)batch * strideB;
    C += (size_t)batch * strideC;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    int tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int ka = t * TILE + threadIdx.x;      // k index this thread loads for A
        int kb = t * TILE + threadIdx.y;      // k index this thread loads for B
        float av = 0.0f, bv = 0.0f;
        if (row < M && ka < K) av = transA ? A[(size_t)ka * lda + row] : A[(size_t)row * lda + ka];
        if (kb < K && col < N) bv = transB ? B[(size_t)col * ldb + kb] : B[(size_t)kb * ldb + col];
        As[threadIdx.y][threadIdx.x] = av;
        Bs[threadIdx.y][threadIdx.x] = bv;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * ldc + col] = acc;
}

// d gelu(x) / dx = Phi(x) + x * phi(x)   (exact erf form, as the forward kernel), times the incoming gradient
__global__ void gelu_backward(const float* x, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float cdf = 0.5f * (1.0f + erff(v * 0.70710678118654752f));
        float pdf = expf(-0.5f * v * v) * 0.39894228040143268f;
        dx[i] = dy[i] * (cdf + v * pdf);
    }
}
"""


STATS = {"matmul": 0, "gelu": 0}
_atexit_registered = False


def enabled():
    return os.environ.get(ENV, "") not in ("", "0")


def _count(kind):
    global _atexit_registered
    STATS[kind] += 1
    if os.environ.get(ENV) == "2" and not _atexit_registered:
        _atexit_registered = True
        atexit.register(lambda: print(f"[GPU BACKWARD] used for {STATS['matmul']} matmul and {STATS['gelu']} gelu backward calls"))


class _Device:
    """The only code that touches pycuda. Tests replace it with a NumPy stand-in."""

    def __init__(self):
        import pycuda.autoinit  # noqa: F401  (the process's context; the same one the program's kernels use)
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        self.cuda = cuda
        self.pool = None
        if os.environ.get(POOL_ENV, "1") != "0":
            from pycuda.tools import DeviceMemoryPool
            self.pool = DeviceMemoryPool()
            atexit.register(self.pool.stop_holding)   # runs before pycuda.autoinit destroys the context
        mod = SourceModule(SOURCE)
        self._gemm = mod.get_function("gemm")
        self._gelu_bw = mod.get_function("gelu_backward")

    def upload(self, arr):
        ptr = self.alloc(arr.nbytes)
        self.cuda.memcpy_htod(ptr, arr)
        return ptr

    def alloc(self, nbytes):
        # pooled: free() hands the block back to the pool instead of calling cudaFree
        return self.pool.allocate(nbytes) if self.pool else self.cuda.mem_alloc(nbytes)

    def download(self, ptr, shape):
        out = np.empty(shape, dtype=np.float32)
        self.cuda.memcpy_dtoh(out, ptr)
        return out

    def free(self, *ptrs):
        for p in ptrs:
            p.free()

    def gemm(self, a, b, c, M, N, K, lda, ldb, ldc, ta, tb, sa, sb, sc, batch):
        i32, i64 = np.int32, np.int64
        self._gemm(a, b, c, i32(M), i32(N), i32(K), i32(lda), i32(ldb), i32(ldc), i32(ta), i32(tb),
                   i64(sa), i64(sb), i64(sc),
                   block=(TILE, TILE, 1), grid=((N + TILE - 1) // TILE, (M + TILE - 1) // TILE, batch))

    def gelu_backward(self, x, dy, dx, n):
        self._gelu_bw(x, dy, dx, np.int32(n), block=(256, 1, 1), grid=((n + 255) // 256, 1))


_device = None
_device_failed = False


def _get_device():
    global _device, _device_failed
    if _device is None and not _device_failed:
        try:
            _device = _Device()
        except Exception as exc:  # no pycuda, no context, nvcc missing...: fall back to NumPy, say so once
            _device_failed = True
            print(f"[GPU BACKWARD] unavailable ({exc}); using the NumPy backward")
    return _device


def _too_small(*arrays):
    return max(a.size for a in arrays) < int(os.environ.get(MIN_ENV, "0") or 0)


def matmul_backward(a, b, og, device=None):
    """(grad_a, grad_b) for C = a @ b, or None if this case is not handled. a (..,M,K), b (..,K,N), og (..,M,N)."""
    a, b, og = np.asarray(a), np.asarray(b), np.asarray(og)
    if a.dtype != np.float32 or b.dtype != np.float32 or og.dtype not in (np.float32, np.float64):
        return None
    if a.ndim != b.ndim or a.ndim != og.ndim or a.ndim not in (2, 3):
        return None
    if a.shape[:-2] != b.shape[:-2] or a.shape[:-2] != og.shape[:-2]:
        return None                                     # broadcast batch dimensions: NumPy path
    M, K = a.shape[-2:]
    K2, N = b.shape[-2:]
    if K2 != K or og.shape[-2:] != (M, N) or min(M, N, K) == 0:
        return None
    if _too_small(a, b, og):
        return None
    dev = device or _get_device()
    if dev is None:
        return None
    batch = int(np.prod(a.shape[:-2])) if a.ndim == 3 else 1
    a, b, og = (np.ascontiguousarray(x, dtype=np.float32) for x in (a, b, og))
    pa, pb, pg = dev.upload(a), dev.upload(b), dev.upload(og)
    ga = dev.alloc(a.nbytes)
    gb = dev.alloc(b.nbytes)
    try:
        # grad_a (M x K) = og (M x N) @ b^T : op(B)[n][k] = b[k][n] with b stored K x N  -> transB
        dev.gemm(pg, pb, ga, M, K, N, N, N, K, 0, 1, M * N, K * N, M * K, batch)
        # grad_b (K x N) = a^T @ og            : op(A)[k][m] = a[m][k] with a stored M x K  -> transA
        dev.gemm(pa, pg, gb, K, N, M, K, N, N, 1, 0, M * K, M * N, K * N, batch)
        result = dev.download(ga, a.shape), dev.download(gb, b.shape)
        _count("matmul")
        return result
    finally:
        dev.free(pa, pb, pg, ga, gb)


def gelu_backward(x, og, device=None):
    """og * gelu'(x) as float32, or None if not handled."""
    x, og = np.asarray(x), np.asarray(og)
    if x.dtype != np.float32 or og.dtype not in (np.float32, np.float64) or x.shape != og.shape or x.size == 0:
        return None
    if _too_small(x):
        return None
    dev = device or _get_device()
    if dev is None:
        return None
    x, og = np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(og, dtype=np.float32)
    px, pg = dev.upload(x), dev.upload(og)
    po = dev.alloc(x.nbytes)
    try:
        dev.gelu_backward(px, pg, po, x.size)
        result = dev.download(po, x.shape)
        _count("gelu")
        return result
    finally:
        dev.free(px, pg, po)
