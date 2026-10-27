import numpy as np
import os
import tempfile
from tensorlang.compiler import TensorCompiler


def test_load_numpy_zero_copy():
    """load_numpy shares the CPU buffer (no copy)."""
    compiler = TensorCompiler()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    compiler.load_numpy("x", arr)

    # Same data pointer
    assert compiler.tensors["x"].ctypes.data == arr.ctypes.data

    # Shape/dtype registered
    assert compiler.env["x"]["shape"] == (2, 2)
    assert compiler.env["x"]["dtype"] == "f32"


def test_to_numpy_zero_copy():
    """to_numpy returns a view, not a copy."""
    compiler = TensorCompiler()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    compiler.load_numpy("x", arr)

    view = compiler.to_numpy("x")
    assert view.ctypes.data == arr.ctypes.data
    assert view.ctypes.data == compiler.tensors["x"].ctypes.data


def test_sync_roundtrip():
    """sync_to_gpu then sync_to_cpu round-trips correctly."""
    compiler = TensorCompiler()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    compiler.load_numpy("x", arr)

    # Mutate via numpy
    arr[0, 0] = 99.0
    compiler.sync_to_gpu("x")

    # Verify GPU has the new value
    gpu_val = np.zeros((2, 2), dtype=np.float32)
    import pycuda.driver as cuda
    cuda.memcpy_dtoh(gpu_val, compiler.gpu_allocs["x"])
    assert gpu_val[0, 0] == 99.0

    # Sync back to CPU
    compiler.sync_to_cpu("x")
    assert compiler.tensors["x"][0, 0] == 99.0


def test_mutation_visibility():
    """Mutations through the numpy view are visible in tensors."""
    compiler = TensorCompiler()
    arr = np.zeros((2, 2), dtype=np.float32)
    compiler.load_numpy("buf", arr)

    view = compiler.to_numpy("buf")
    view[:] = 7.0

    # The underlying tensors dict should see the change
    # (same buffer, so yes)
    assert np.allclose(compiler.tensors["buf"], 7.0)


def test_load_numpy_rejects_non_float32():
    """load_numpy rejects non-f32 arrays."""
    compiler = TensorCompiler()
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    try:
        compiler.load_numpy("x", arr)
        assert False, "should have raised TypeError"
    except TypeError:
        pass


def test_load_numpy_rejects_non_contiguous():
    """load_numpy handles non-contiguous arrays by making them contiguous."""
    compiler = TensorCompiler()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).T  # non-contiguous
    compiler.load_numpy("x", arr)
    assert compiler.tensors["x"].flags["C_CONTIGUOUS"]

