"""NumPy zero-copy interop for TensorCompiler."""

import numpy as np
import pycuda.driver as cuda


class InteropMixin:
    """Mixin providing zero-copy NumPy interop.

    CPU-side tensors (self.tensors) are numpy arrays.  This mixin lets
    callers share storage with external numpy arrays instead of copying.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep references to externally-owned arrays to prevent GC
        self._interop_refs = {}

    # ------------------------------------------------------------------
    # load_numpy: zero-copy CPU buffer → TensorLang
    # ------------------------------------------------------------------
    def load_numpy(self, name, arr):
        """Register a numpy array as a TensorLang tensor (zero-copy CPU side).

        The array's buffer is used directly as ``self.tensors[name]``.
        A GPU allocation is created and data is copied H→D once.
        """
        self.env = getattr(self, 'env', {})
        self.tensors = getattr(self, 'tensors', {})
        self.gpu_allocs = getattr(self, 'gpu_allocs', {})
        self.env = getattr(self, 'env', {})
        self.tensors = getattr(self, 'tensors', {})
        self.gpu_allocs = getattr(self, 'gpu_allocs', {})
        self._interop_refs = getattr(self, '_interop_refs', {})
        if not isinstance(arr, np.ndarray):
            raise TypeError("load_numpy requires a numpy.ndarray")
        if arr.dtype != np.float32:
            raise TypeError(f"load_numpy requires float32, got {arr.dtype}")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1)

        shape = tuple(int(d) for d in arr.shape)
        self.env[name] = {"dtype": "f32", "shape": shape}
        self.tensors[name] = arr  # zero-copy: same buffer
        self._interop_refs[name] = arr

        # GPU allocation + H→D copy
        size_bytes = int(arr.size * np.float32().nbytes)
        self.gpu_allocs[name] = cuda.mem_alloc(size_bytes)
        cuda.memcpy_htod(self.gpu_allocs[name], arr)

        if self.debug_mode:
            self.tensorlang.print(
                message=f"[INTEROP] load_numpy: {name} shape={shape} "
                        f"ptr={arr.ctypes.data} (zero-copy)"
            )
        return name

    # ------------------------------------------------------------------
    # to_numpy: zero-copy view of a TensorLang tensor
    # ------------------------------------------------------------------
    def to_numpy(self, name):
        """Return a numpy array sharing storage with ``self.tensors[name]``.

        If the tensor is not yet in ``self.tensors`` (e.g. GPU-only),
        a D→H copy is performed first into a new buffer.
        """
        self.env = getattr(self, 'env', {})
        self.tensors = getattr(self, 'tensors', {})
        self.gpu_allocs = getattr(self, 'gpu_allocs', {})
        self.env = getattr(self, 'env', {})
        self.tensors = getattr(self, 'tensors', {})
        self.gpu_allocs = getattr(self, 'gpu_allocs', {})
        if name in self.tensors:
            arr = self.tensors[name]
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=np.float32)
                self.tensors[name] = arr
            return arr

        # GPU-only: copy D→H
        if name in self.gpu_allocs and name in self.env:
            shape = tuple(int(d) for d in self.env[name]["shape"])
            arr = np.zeros(shape, dtype=np.float32)
            cuda.memcpy_dtoh(arr, self.gpu_allocs[name])
            self.tensors[name] = arr
            return arr

        raise KeyError(f"to_numpy: tensor '{name}' not found")

    # ------------------------------------------------------------------
    # sync_to_gpu: push CPU buffer → GPU
    # ------------------------------------------------------------------
    def sync_to_gpu(self, name):
        """Copy ``self.tensors[name]`` to ``self.gpu_allocs[name]``."""
        self.env = getattr(self, 'env', {})
        self.tensors = getattr(self, 'tensors', {})
        self.gpu_allocs = getattr(self, 'gpu_allocs', {})
        self.env = getattr(self, 'env', {})
        self.tensors = getattr(self, 'tensors', {})
        self.gpu_allocs = getattr(self, 'gpu_allocs', {})
        if name not in self.tensors:
            raise KeyError(f"sync_to_gpu: '{name}' not in tensors")
        if name not in self.gpu_allocs:
            shape = tuple(int(d) for d in self.env[name]["shape"])
            size_bytes = int(np.prod(shape) * np.float32().nbytes)
            self.gpu_allocs[name] = cuda.mem_alloc(size_bytes)
        arr = np.ascontiguousarray(self.tensors[name], dtype=np.float32)
        cuda.memcpy_htod(self.gpu_allocs[name], arr)
        if self.debug_mode:
            self.tensorlang.print(message=f"[INTEROP] sync_to_gpu: {name}")

    # ------------------------------------------------------------------
    # sync_to_cpu: pull GPU → CPU buffer (in-place when possible)
    # ------------------------------------------------------------------
    def sync_to_cpu(self, name):
        """Copy ``self.gpu_allocs[name]`` into ``self.tensors[name]`` in place."""
        self.env = getattr(self, 'env', {})
        self.tensors = getattr(self, 'tensors', {})
        self.gpu_allocs = getattr(self, 'gpu_allocs', {})
        self.env = getattr(self, 'env', {})
        self.tensors = getattr(self, 'tensors', {})
        self.gpu_allocs = getattr(self, 'gpu_allocs', {})
        if name not in self.gpu_allocs:
            raise KeyError(f"sync_to_cpu: '{name}' not in gpu_allocs")
        if name not in self.env:
            raise KeyError(f"sync_to_cpu: '{name}' not in env")
        shape = tuple(int(d) for d in self.env[name]["shape"])
        if name in self.tensors and isinstance(self.tensors[name], np.ndarray):
            arr = self.tensors[name]
            if arr.shape != shape or arr.dtype != np.float32:
                arr = np.zeros(shape, dtype=np.float32)
                self.tensors[name] = arr
        else:
            arr = np.zeros(shape, dtype=np.float32)
            self.tensors[name] = arr
        cuda.memcpy_dtoh(arr, self.gpu_allocs[name])
        if self.debug_mode:
            self.tensorlang.print(message=f"[INTEROP] sync_to_cpu: {name}")
        return arr

    # ------------------------------------------------------------------
    # mmap-based zero-copy load for .npy files
    # ------------------------------------------------------------------
    def load_numpy_mmap(self, file_path):
        """Memory-map a .npy file (zero-copy, read-only)."""
        loaded = np.load(file_path, mmap_mode="r")
        if loaded.dtype != np.float32:
            loaded = loaded.astype(np.float32)
        return np.ascontiguousarray(loaded)
