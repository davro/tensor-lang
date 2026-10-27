"""I/O and memory helpers for TensorCompiler."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pycuda.driver as cuda

from tensorlang.tensor_lang import TensorLang


class IOMixin:
    def prod(self, lst):
        from functools import reduce
        return reduce(lambda x, y: x * y, lst, 1)

    def _ensure_gpu_alloc(self, name, env, tensors, gpu_allocs, cuda):
        if name in gpu_allocs:
            return gpu_allocs[name]

        shape = tuple(int(dim) for dim in env[name]["shape"])
        numel = int(np.prod(shape)) if shape else 1
        alloc = cuda.mem_alloc(numel * np.dtype(np.float32).itemsize)

        if name in tensors:
            arr = np.asarray(tensors[name], dtype=np.float32).reshape(shape if shape else (1,))
            if not shape:
                arr = arr.reshape(())
            cuda.memcpy_htod(alloc, arr)
        else:
            cuda.memset_d32(alloc, 0, numel)

        gpu_allocs[name] = alloc
        return alloc

    def _allocate_zeros(self, shape, cuda):
        shape = tuple(int(dim) for dim in shape)
        numel = int(np.prod(shape)) if shape else 1
        alloc = cuda.mem_alloc(numel * np.dtype(np.float32).itemsize)
        arr = np.zeros(shape if shape else (1,), dtype=np.float32)
        if not shape:
            arr = arr.reshape(())
        cuda.memcpy_htod(alloc, arr)
        return alloc

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

    def _load_data_file(self, file_path, array_name=None):
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        suffix = path.suffix.lower()
        metadata = {"format": suffix[1:], "path": str(path)}

        if suffix == ".npy":
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Loading .npy file: {path.name}")
            data = np.load(path).astype(np.float32)
            metadata["shape"] = data.shape
            metadata["dtype"] = "f32"
            return data, metadata

        if suffix == ".npz":
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Loading .npz file: {path.name}")
            npz_file = np.load(path)
            available_arrays = list(npz_file.files)
            metadata["available_arrays"] = available_arrays
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER]   Available arrays: {available_arrays}")

            if array_name:
                if array_name not in available_arrays:
                    raise KeyError(f"Array '{array_name}' not found in {path.name}. Available: {available_arrays}")
                data = npz_file[array_name].astype(np.float32)
                metadata["array_name"] = array_name
            else:
                if "data" in available_arrays:
                    data = npz_file["data"].astype(np.float32)
                    metadata["array_name"] = "data"
                else:
                    data = npz_file[available_arrays[0]].astype(np.float32)
                    metadata["array_name"] = available_arrays[0]
                    if self.debug_mode:
                        self.tensorlang.print(message=f"[COMPILER]   No 'data' array found, using '{available_arrays[0]}'")

            if "columns" in available_arrays:
                metadata["columns"] = npz_file["columns"].tolist()
                if self.debug_mode:
                    self.tensorlang.print(message=f"[COMPILER]   Columns: {metadata['columns']}")
            if "index" in available_arrays:
                index_data = npz_file["index"]
                if index_data.dtype.kind == "S":
                    metadata["index"] = [x.decode("utf-8") if isinstance(x, bytes) else x for x in index_data]
                else:
                    metadata["index"] = index_data.tolist()
                if self.debug_mode:
                    self.tensorlang.print(
                        message=f"[COMPILER]   Index: {len(metadata['index'])} entries "
                                f"({metadata['index'][0]} to {metadata['index'][-1]})"
                    )

            metadata["shape"] = data.shape
            metadata["dtype"] = "f32"
            return data, metadata

        if suffix == ".csv":
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Loading .csv file: {path.name}")
            try:
                df = pd.read_csv(path)
            except (OSError, ValueError, UnicodeError) as e:
                raise ValueError(f"Failed to parse CSV file {path.name}: {e}")

            metadata["columns"] = df.columns.tolist()
            metadata["rows"] = len(df)
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER]   Columns: {metadata['columns']}")
                self.tensorlang.print(message=f"[COMPILER]   Rows: {metadata['rows']}")

            first_col = df.columns[0]
            if first_col.lower() in ["date", "time", "datetime", "index", "id"]:
                metadata["index"] = df[first_col].tolist()
                df = df.drop(columns=[first_col])
                if self.debug_mode:
                    self.tensorlang.print(message=f"[COMPILER]   Index column detected: {first_col}")

            try:
                data = df.to_numpy().astype(np.float32)
            except ValueError as e:
                raise ValueError(
                    f"CSV contains non-numeric data that cannot be converted to float32. "
                    f"Columns: {df.columns.tolist()}. Error: {e}"
                )

            metadata["shape"] = data.shape
            metadata["dtype"] = "f32"
            metadata["columns"] = df.columns.tolist()
            return data, metadata

        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .npy, .npz, .csv"
        )

