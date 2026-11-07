# File: tensorlang/io.py

"""I/O and memory helpers for TensorCompiler."""


import numpy as np


class IOMixin:
    def prod(self, lst):
        from functools import reduce
        return reduce(lambda x, y: x * y, lst, 1)

    def _eval_scalar(self, arg, env, tensors, default=None):
        if arg is None:
            return default
        if isinstance(arg, bool):
            return float(arg)
        if isinstance(arg, (int, float)):
            return float(arg)
        if isinstance(arg, str):
            if arg in tensors:
                arr = np.asarray(tensors[arg])
                if arr.size == 0:
                    return default
                return float(arr.reshape(-1)[0])
            return default
        if isinstance(arg, dict):
            if arg.get("type") in ("number", "float", "int", "const", "scalar"):
                value = arg.get("value", arg.get("val"))
                if value is not None:
                    return float(value)
            if arg.get("type") == "name":
                name = arg.get("name")
                if name in tensors:
                    arr = np.asarray(tensors[name])
                    if arr.size:
                        return float(arr.reshape(-1)[0])
            if "value" in arg:
                try:
                    return float(arg["value"])
                except (TypeError, ValueError):
                    pass
        return default
