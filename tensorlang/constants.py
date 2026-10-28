# File: tensorlang/constants.py

import numpy as np

OPTIMIZER_NAMES = {"sgd", "adam", "sgd_update", "adam_update"}

# ---------------------------------------------------------------------------
# Dtype registry
# ---------------------------------------------------------------------------
# Maps dtype name -> (numpy dtype, C/CUDA type string, byte size)
DTYPE_INFO = {
    "f16": (np.dtype(np.float16), "__half", 2),
    "f32": (np.dtype(np.float32), "float",  4),
    "f64": (np.dtype(np.float64), "double", 8),
}

VALID_DTYPES = set(DTYPE_INFO.keys())

# Dtype promotion table (NumPy-style)
_PROMOTION = {
    ("f16", "f16"): "f16",
    ("f16", "f32"): "f32",
    ("f32", "f16"): "f32",
    ("f32", "f32"): "f32",
    ("f16", "f64"): "f64",
    ("f64", "f16"): "f64",
    ("f32", "f64"): "f64",
    ("f64", "f32"): "f64",
    ("f64", "f64"): "f64",
}


def dtype_numpy(dtype_name: str):
    """Return the numpy dtype for a TensorLang dtype name."""
    return DTYPE_INFO[dtype_name][0]


def dtype_ctype(dtype_name: str):
    """Return the C/CUDA type string for a TensorLang dtype name."""
    return DTYPE_INFO[dtype_name][1]


def dtype_size(dtype_name: str):
    """Return the byte size of a TensorLang dtype."""
    return DTYPE_INFO[dtype_name][2]


def promote_dtype(a: str, b: str) -> str:
    """Promote two dtypes following NumPy rules."""
    if a == b:
        return a
    key = (a, b)
    if key in _PROMOTION:
        return _PROMOTION[key]
    # Fallback: promote to the wider of the two
    order = {"f16": 0, "f32": 1, "f64": 2}
    return a if order.get(a, 1) >= order.get(b, 1) else b


# ---------------------------------------------------------------------------
# CUDA includes
# ---------------------------------------------------------------------------
F16_INCLUDE = "#include <cuda_fp16.h>\n"


def dtype_include(dtype_name: str) -> str:
    """Return the CUDA include directive needed for a dtype (f16 needs cuda_fp16.h)."""
    if dtype_name == "f16":
        return F16_INCLUDE
    return ""


# ---------------------------------------------------------------------------
# Kernel launch defaults
# ---------------------------------------------------------------------------
DEFAULT_BLOCK_SIZE = 256
