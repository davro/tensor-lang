# File: tensorlang/compiler.py

import errno
import os
import subprocess
import sys
import time
import traceback
from ctypes import c_float, c_int, c_void_p, cdll
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lark import Lark, UnexpectedInput

import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

from tensorlang.ast_builder import build_ast
from tensorlang.autograd import AutogradContext, ComputationGraph
from tensorlang.fusion import fuse_elementwise_chains
from tensorlang.kernel_generator import KernelGenerator
from tensorlang.tensor_lang import TensorLang
from tensorlang.type_checker import _is_scalar_shape, type_checker

OPTIMIZER_NAMES = {"sgd", "adam", "sgd_update", "adam_update"}


class TensorCompiler:

    def __init__(self, debug_mode=False, debug_info=False, debug_ast=False, cache_layers=False, transpile=False):
        self.debug_mode = debug_mode
        self.debug_info = debug_info
        self.debug_ast = debug_ast
        self.cache_layers = cache_layers
        self.transpile = transpile
        self.tensorlang = TensorLang()
        self.version = self.tensorlang.version

        self.comp_graph = ComputationGraph(debug_mode=debug_mode)
        self.requires_grad_tensors = set()
        self._loop_scope_counter = 0
        self._kernel_name_map = {}
        self.optimizer_state = {}

    def _should_track_gradients(self, tensor_name):
        return tensor_name in self.requires_grad_tensors

    def _record_operation(self, op_type, name, inputs, metadata=None):
        if not AutogradContext.is_enabled():
            return
        if any(self._should_track_gradients(inp) for inp in inputs):
            self.comp_graph.add_operation(op_type, name, inputs, metadata)
            self.requires_grad_tensors.add(name)
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] [Autograd] Recorded {op_type}: {inputs} -> {name}")

    def prod(self, lst):
        return reduce(lambda x, y: x * y, lst, 1)

    def can_broadcast(self, shape1, shape2):
        ndim = max(len(shape1), len(shape2))
        s1 = (1,) * (ndim - len(shape1)) + shape1
        s2 = (1,) * (ndim - len(shape2)) + shape2
        for d1, d2 in zip(s1, s2):
            if d1 != d2 and d1 != 1 and d2 != 1:
                return False
        return True

    def _optimizer_arg_name(self, arg):
        if isinstance(arg, str):
            return arg
        if isinstance(arg, dict):
            if "name" in arg:
                return arg["name"]
            if arg.get("type") == "name":
                return arg["name"]
        return None

    def _optimizer_parts(self, node):
        if node.get("type") == "call":
            opt_name = node.get("name")
        else:
            opt_name = node.get("type")

        args = node.get("args", node.get("arguments", []))
        param_name = self._optimizer_arg_name(args[0]) if len(args) > 0 else None
        grad_name = self._optimizer_arg_name(args[1]) if len(args) > 1 else None
        lr_arg = args[2] if len(args) > 2 else None
        beta1_arg = args[3] if len(args) > 3 else 0.9
        beta2_arg = args[4] if len(args) > 4 else 0.999
        eps_arg = args[5] if len(args) > 5 else 1e-8
        return opt_name, param_name, grad_name, lr_arg, beta1_arg, beta2_arg, eps_arg

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

    def _load_data_file(self, file_path: str, array_name: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
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
                        self.tensorlang.print(
                            message=f"[COMPILER]   No 'data' array found, using '{available_arrays[0]}'"
                        )

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

    # =========================================================================
    # FOR LOOP EXECUTION
    # =========================================================================

    def _execute_for_loop(self, for_node, kernels, ast_to_kernel_map,
                          lib, gpu_allocs, env, tensors, cache_file_dir, cuda):
        iterations = for_node["iterations"]
        body = for_node["body"]

        rebinds = [n for n in body if n["type"] == "rebind"]
        exec_nodes = [n for n in body if n["type"] != "rebind"]
        backward_node = next((n for n in exec_nodes if n["type"] == "backward"), None)

        bw_list_pos = None
        for pos, (gi, n) in enumerate(for_node["_flat_items"]):
            if n["type"] == "backward":
                bw_list_pos = pos
                break

        if self.debug_mode:
            self.tensorlang.print(
                message=f"[COMPILER] for loop: {iterations} iterations, "
                        f"{len(exec_nodes)} exec nodes, "
                        f"{len(rebinds)} rebinds: {[r['name'] for r in rebinds]}, "
                        f"backward={'yes' if backward_node else 'no'}"
            )

        for i in range(iterations):
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] for loop: iteration {i + 1}/{iterations}")

            for pos, (global_idx, node) in enumerate(for_node["_flat_items"]):
                if node["type"] in ("backward", "save", "rebind"):
                    continue
                if global_idx is None:
                    continue
                if bw_list_pos is not None and pos >= bw_list_pos:
                    continue
                self._execute_single_kernel(
                    kernels[global_idx], lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                )

            if backward_node is not None:
                loss_name = backward_node["loss_tensor"]

                for global_idx, body_node in for_node["_flat_items"]:
                    if body_node["type"] != "let":
                        continue
                    bname = body_node["name"]
                    if bname in gpu_allocs and bname in env:
                        bshape = tuple(int(d) for d in env[bname]["shape"])
                        barr = np.empty(bshape, dtype=np.float32)
                        cuda.memcpy_dtoh(barr, gpu_allocs[bname])
                        tensors[bname] = barr

                for tname in list(tensors.keys()):
                    if tname in gpu_allocs and tname in env:
                        tshape = tuple(int(d) for d in env[tname]["shape"])
                        tarr = np.empty(tshape, dtype=np.float32)
                        cuda.memcpy_dtoh(tarr, gpu_allocs[tname])
                        tensors[tname] = tarr

                for tname, tval in tensors.items():
                    requires_grad = tname in self.requires_grad_tensors
                    self.comp_graph.register_tensor(tname, tval, requires_grad)

                self.comp_graph.gradients = {}
                self.comp_graph.backward(loss_name)

                if self.debug_mode:
                    loss_val = tensors.get(loss_name)
                    if loss_val is not None:
                        self.tensorlang.print(
                            message=f"[COMPILER] for loop iter {i + 1}: "
                                    f"loss={float(np.asarray(loss_val).reshape(-1)[0]):.6f}"
                        )

                for grad_name in self.comp_graph.requires_grad:
                    if grad_name in self.comp_graph.gradients:
                        grad_tensor = self.comp_graph.gradients[grad_name]
                        grad_tensor_name = f"{grad_name}_grad"
                        tensors[grad_tensor_name] = grad_tensor

                        if grad_tensor_name in gpu_allocs:
                            cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)
                        else:
                            gpu_allocs[grad_tensor_name] = cuda.mem_alloc(grad_tensor.nbytes)
                            cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)

                        env[grad_tensor_name] = {
                            "dtype": "f32",
                            "shape": grad_tensor.shape,
                        }

                if bw_list_pos is not None:
                    for pos, (global_idx, node) in enumerate(for_node["_flat_items"]):
                        if node["type"] in ("backward", "save", "rebind"):
                            continue
                        if global_idx is None:
                            continue
                        if pos <= bw_list_pos:
                            continue
                        self._execute_single_kernel(
                            kernels[global_idx], lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                        )

            pending_gpu = {}
            pending_tensors = {}

            for rebind in rebinds:
                target = rebind["name"]
                src_expr = rebind["expr"]

                if src_expr.get("type") != "name":
                    self.tensorlang.print(
                        message=f"[COMPILER] for loop rebind: only NAME expressions supported, "
                                f"got {src_expr.get('type')} for '{target}'"
                    )
                    continue

                src_name = src_expr["name"]

                if src_name not in gpu_allocs:
                    self.tensorlang.print(
                        message=f"[COMPILER] for loop rebind: '{src_name}' not in gpu_allocs"
                    )
                    continue

                pending_gpu[target] = gpu_allocs[src_name]
                if src_name in tensors:
                    pending_tensors[target] = tensors[src_name]

                if self.debug_mode:
                    self.tensorlang.print(
                        message=f"[COMPILER] for loop iter {i + 1}: rebind {target} <- {src_name}"
                    )

            gpu_allocs.update(pending_gpu)
            tensors.update(pending_tensors)

            for tname, tval in tensors.items():
                requires_grad = tname in self.requires_grad_tensors
                self.comp_graph.register_tensor(tname, tval, requires_grad)

        if self.debug_mode:
            self.tensorlang.print(
                message=f"[COMPILER] for loop: completed {iterations} iterations"
            )

    # =========================================================================
    # KERNEL EXECUTION
    # =========================================================================

    def _inline_functions(self, ast, functions):
        return ast

    def _execute_single_kernel(self, kernel_info, lib, gpu_allocs, env, tensors, cache_file_dir, cuda):
        if not kernel_info:
            return

        op_type = kernel_info[0]

        # if op_type == "fused_elementwise":
            # _, kernel_func_name, chain, external_inputs, outputs, size = kernel_info
            # launch_args = [c_void_p(int(gpu_allocs[n])) for n in external_inputs] + \
                          # [c_void_p(int(gpu_allocs[n])) for n in outputs] + [c_int(size)]
            # getattr(lib, f"launch_fused_elementwise_{kernel_func_name}")(*launch_args)
            # for out_name in outputs:
                # self._save_kernel_result(out_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            # return

        if op_type == "fused_elementwise":
            (_, kernel_func_name, chain, external_inputs, outputs, size,
             broadcast, input_shapes, output_shape) = kernel_info
            launch_args = [c_void_p(int(gpu_allocs[n])) for n in external_inputs] + \
                          [c_void_p(int(gpu_allocs[n])) for n in outputs]
            shape_bufs = []
            if broadcast:
                ndim = len(output_shape)
                padded_in = [(1,) * (ndim - len(s)) + tuple(s) for s in input_shapes]
                for s in padded_in:
                    arr = np.array(s, dtype=np.int32)
                    buf = cuda.mem_alloc(ndim * np.int32().nbytes)
                    cuda.memcpy_htod(buf, arr)
                    shape_bufs.append(buf)
                    launch_args.append(c_void_p(int(buf)))
                out_arr = np.array(output_shape, dtype=np.int32)
                out_buf = cuda.mem_alloc(ndim * np.int32().nbytes)
                cuda.memcpy_htod(out_buf, out_arr)
                shape_bufs.append(out_buf)
                launch_args.append(c_void_p(int(out_buf)))
                launch_args.append(c_int(ndim))
            launch_args.append(c_int(size))
            getattr(lib, f"launch_fused_elementwise_{kernel_func_name}")(*launch_args)
            for buf in shape_bufs:
                buf.free()
            for out_name in outputs:
                self._save_kernel_result(out_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        if op_type in ("sgd_update", "adam_update"):
            kernel_func_name = kernel_info[1]
            param_name = kernel_info[2]
            grad_name = kernel_info[3]
            size = int(kernel_info[4])

            param_ptr = int(self._ensure_gpu_alloc(param_name, env, tensors, gpu_allocs, cuda))
            grad_ptr = int(self._ensure_gpu_alloc(grad_name, env, tensors, gpu_allocs, cuda))

            if op_type == "sgd_update":
                lr_value = self._eval_scalar(kernel_info[5], env, tensors, 0.0)
                getattr(lib, f"launch_sgd_update_{kernel_func_name}")(
                    c_void_p(param_ptr),
                    c_void_p(grad_ptr),
                    c_float(lr_value),
                    c_int(size),
                )
            else:
                lr_value = self._eval_scalar(kernel_info[5], env, tensors, 0.01)
                beta1 = self._eval_scalar(kernel_info[6], env, tensors, 0.9)
                beta2 = self._eval_scalar(kernel_info[7], env, tensors, 0.999)
                eps = self._eval_scalar(kernel_info[8], env, tensors, 1e-8)

                state = self.optimizer_state.setdefault(param_name, {
                    "m": self._allocate_zeros(env[param_name]["shape"], cuda),
                    "v": self._allocate_zeros(env[param_name]["shape"], cuda),
                    "t": 0,
                })
                state["t"] += 1

                getattr(lib, f"launch_adam_update_{kernel_func_name}")(
                    c_void_p(param_ptr),
                    c_void_p(grad_ptr),
                    c_void_p(int(state["m"])),
                    c_void_p(int(state["v"])),
                    c_float(lr_value),
                    c_float(beta1),
                    c_float(beta2),
                    c_float(eps),
                    c_int(state["t"]),
                    c_int(size),
                )

            tensor_name = self._kernel_name_map.get(kernel_func_name, param_name)
            self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        if op_type == "general_broadcast":
            _, actual_op, kernel_func_name, arg1, arg2, padded_shape1, padded_shape2, output_shape_tuple, total_elements = kernel_info
            tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)
            shape = tuple(int(dim) for dim in env[tensor_name]["shape"])
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Executing {op_type} ({actual_op}) for {tensor_name}, shape: {shape}")
            ndim = len(output_shape_tuple)
            shape1_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            shape2_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            out_shape_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            shape1_array = np.array(padded_shape1, dtype=np.int32)
            shape2_array = np.array(padded_shape2, dtype=np.int32)
            out_shape_array = np.array(output_shape_tuple, dtype=np.int32)
            cuda.memcpy_htod(shape1_gpu, shape1_array)
            cuda.memcpy_htod(shape2_gpu, shape2_array)
            cuda.memcpy_htod(out_shape_gpu, out_shape_array)
            getattr(lib, f"launch_{actual_op}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_void_p(int(shape1_gpu)),
                c_void_p(int(shape2_gpu)),
                c_void_p(int(out_shape_gpu)),
                c_int(ndim),
                c_int(total_elements),
            )
            self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        op_type, kernel_func_name, arg1, arg2, *dims = kernel_info
        tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)
        shape = tuple(int(dim) for dim in env[tensor_name]["shape"])

        if op_type == "matmul":
            m, n, p = dims
            getattr(lib, f"launch_matmul_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(m), c_int(n), c_int(p),
            )

        elif op_type in ("add", "minus", "mult", "div"):
            size = dims[0]
            getattr(lib, f"launch_{op_type}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
            )

        elif op_type in ("add_scalar", "minus_scalar", "mult_scalar", "div_scalar"):
            size = dims[0]
            op_base = op_type.split("_")[0]
            getattr(lib, f"launch_{op_base}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
            )

        elif op_type in (
            "add_broadcast", "minus_broadcast", "mult_broadcast", "div_broadcast",
            "greater_broadcast", "less_broadcast", "equal_broadcast",
        ):
            op_name = op_type.split("_")[0]
            rows, cols = dims
            getattr(lib, f"launch_{op_name}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols),
            )

        elif op_type in ("relu", "sigmoid", "tanh"):
            op_name = op_type.split("_")[0]
            size = dims[0]
            getattr(lib, f"launch_{op_name}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
            )

        elif op_type in ("greater", "less", "equal"):
            size = dims[0]
            getattr(lib, f"launch_{op_type}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
            )

        elif op_type == "softmax_1d":
            size = dims[0]
            getattr(lib, f"launch_softmax_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
            )

        elif op_type == "softmax":
            rows, cols, _axis = dims
            getattr(lib, f"launch_softmax_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols),
            )

        elif op_type in ("sum_axis", "sum_axis0"):
            rows, cols, _axis = dims
            getattr(lib, f"launch_sum_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols),
            )

        elif op_type == "slice_2d":
            rows, cols, row_start, row_end, col_start, col_end, out_rows, out_cols = dims
            getattr(lib, f"launch_slice_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols),
                c_int(row_start), c_int(row_end), c_int(col_start), c_int(col_end),
                c_int(out_rows), c_int(out_cols),
            )

        elif op_type == "slice_1d":
            start, out_size = dims
            getattr(lib, f"launch_slice_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(start), c_int(out_size),
            )

        elif op_type == "sum_full":
            size = dims[0]
            getattr(lib, f"launch_sum_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
            )

        elif op_type == "mean_full":
            size = dims[0]
            getattr(lib, f"launch_mean_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
            )

        elif op_type in ("mean_axis", "mean_axis0"):
            rows, cols, _axis = dims
            getattr(lib, f"launch_mean_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols),
            )

        elif op_type in ("max_full", "min_full"):
            size = dims[0]
            op_name = op_type.split("_")[0]
            getattr(lib, f"launch_{op_name}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
            )

        elif op_type in ("max_axis", "max_axis0", "min_axis", "min_axis0"):
            rows, cols, _axis = dims
            op_name = op_type.split("_")[0]
            getattr(lib, f"launch_{op_name}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols),
            )

        elif op_type in ("argmax_full", "argmax_axis"):
            if len(dims) == 1:
                size = dims[0]
                getattr(lib, f"launch_argmax_{kernel_func_name}")(
                    c_void_p(int(gpu_allocs[arg1])),
                    c_void_p(int(gpu_allocs[tensor_name])),
                    c_int(size),
                )
            else:
                rows, cols, _axis = dims
                getattr(lib, f"launch_argmax_{kernel_func_name}")(
                    c_void_p(int(gpu_allocs[arg1])),
                    c_void_p(int(gpu_allocs[tensor_name])),
                    c_int(rows), c_int(cols),
                )

        elif op_type in ("argmin_full", "argmin_axis"):
            if len(dims) == 1:
                size = dims[0]
                getattr(lib, f"launch_argmin_{kernel_func_name}")(
                    c_void_p(int(gpu_allocs[arg1])),
                    c_void_p(int(gpu_allocs[tensor_name])),
                    c_int(size),
                )
            else:
                rows, cols, _axis = dims
                getattr(lib, f"launch_argmin_{kernel_func_name}")(
                    c_void_p(int(gpu_allocs[arg1])),
                    c_void_p(int(gpu_allocs[tensor_name])),
                    c_int(rows), c_int(cols),
                )

        elif op_type == "fill":
            size, value = dims
            getattr(lib, f"launch_fill_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[tensor_name])),
                c_float(value), c_int(size),
            )

        elif op_type == "linear_1d":
            input_name, weight_name, bias_name, in_features, out_features = arg1, arg2, dims[0], dims[1], dims[2]
            getattr(lib, f"launch_linear_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[input_name])),
                c_void_p(int(gpu_allocs[weight_name])),
                c_void_p(int(gpu_allocs[bias_name])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(in_features), c_int(out_features),
            )

        elif op_type == "linear_2d":
            input_name, weight_name, bias_name, batch_size, in_features, out_features = arg1, arg2, dims[0], dims[1], dims[2], dims[3]
            getattr(lib, f"launch_linear_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[input_name])),
                c_void_p(int(gpu_allocs[weight_name])),
                c_void_p(int(gpu_allocs[bias_name])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(batch_size), c_int(in_features), c_int(out_features),
            )

        elif op_type in ("layer_norm_2d", "layer_norm_axis0"):
            rows, cols, eps = dims
            getattr(lib, f"launch_layer_norm_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols), c_float(eps),
            )

        elif op_type == "layer_norm_1d":
            size, eps = dims
            getattr(lib, f"launch_layer_norm_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size), c_float(eps),
            )

        elif op_type == "cross_entropy":
            batch_size, num_classes = dims
            getattr(lib, f"launch_cross_entropy_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(batch_size), c_int(num_classes),
            )

        elif op_type == "mse_loss":
            total_elements = dims[0]
            getattr(lib, f"launch_mse_loss_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(total_elements),
            )

        elif op_type == "transpose":
            total_elements = dims[0]
            getattr(lib, f"launch_transpose_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(total_elements),
            )

        elif op_type == "reshape":
            total_elements = dims[0]
            getattr(lib, f"launch_reshape_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(total_elements),
            )

        elif op_type == "concat_axis0":
            rows1, rows2, cols = dims
            getattr(lib, f"launch_concat_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows1), c_int(rows2), c_int(cols),
            )

        elif op_type == "concat_axis1":
            rows, cols1, cols2 = dims
            getattr(lib, f"launch_concat_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols1), c_int(cols2),
            )

        elif op_type == "batch_norm_2d":
            running_mean_name, batch_size, num_features, eps, running_var_name = arg2, dims[0], dims[1], dims[2], dims[3]
            getattr(lib, f"launch_batch_norm_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[running_mean_name])),
                c_void_p(int(gpu_allocs[running_var_name])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(batch_size), c_int(num_features), c_float(eps),
            )

        elif op_type == "instance_norm_2d":
            batch_size, num_features, eps = dims[-3], dims[-2], dims[-1]
            getattr(lib, f"launch_instance_norm_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(batch_size), c_int(num_features), c_float(eps),
            )

        elif op_type in ("minus_broadcast_rows", "add_broadcast_rows"):
            rows, cols = dims
            op_name = op_type.split("_")[0]
            getattr(lib, f"launch_{op_name}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols),
            )

        self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)

    def _save_kernel_result(self, name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda):
        if name in env:
            shape = tuple(int(dim) for dim in env[name]["shape"])
            output = np.zeros(shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, gpu_allocs[name])
            tensors[name] = output
            if self.cache_layers:
                cache_npy_path = cache_file_dir / f"{name}.npy"
                np.save(cache_npy_path, output)
            if self.debug_mode:
                self.tensorlang.print(type="[COMPILER]", message=f"Result {name} ({op_type}):\n{output}")

    def _resolve_alias_inline(self, node, gpu_allocs, tensors):
        alias_name = node["name"]
        source_name = node["expr"]["name"]
        if source_name in tensors and alias_name not in tensors:
            tensors[alias_name] = tensors[source_name].copy()
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] CUDA Execute: created alias: {alias_name} -> {source_name}")
        if source_name in gpu_allocs and alias_name in gpu_allocs:
            gpu_allocs[alias_name] = gpu_allocs[source_name]
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] CUDA Execute: created GPU alias: {alias_name} -> {source_name}")

    def _execute_save_statement(self, save_node, tensors, gpu_allocs, env, cache_file_dir, cuda):
        tensor_name = save_node["tensor"]
        file_path = save_node["file_path"]
        if self.debug_mode:
            self.tensorlang.print(message=f"[COMPILER] Executing save: {tensor_name} -> {file_path}")
        if tensor_name in tensors:
            tensor_data = tensors[tensor_name]
        elif tensor_name in gpu_allocs and tensor_name in env:
            shape = tuple(int(dim) for dim in env[tensor_name]["shape"])
            tensor_data = np.zeros(shape, dtype=np.float32)
            cuda.memcpy_dtoh(tensor_data, gpu_allocs[tensor_name])
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Copied {tensor_name} from GPU for save")
        else:
            self.tensorlang.print(message=f"[COMPILER] Error: Tensor '{tensor_name}' not found for save")
            return

        save_path = Path(file_path)
        if not save_path.is_absolute():
            save_path = cache_file_dir / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(save_path, tensor_data)
            if self.debug_mode:
                self.tensorlang.print(
                    message=f"[COMPILER] Saved {tensor_name}\n"
                            f"  Path:  {save_path}\n"
                            f"  Shape: {tensor_data.shape}\n"
                            f"  Size:  {save_path.stat().st_size} bytes"
                )
            else:
                self.tensorlang.print(message=f"[COMPILER] Saved {tensor_name} to {save_path}")
        except OSError as e:
            self.tensorlang.print(message=f"[COMPILER] Error saving {tensor_name} to {file_path}: {e}")

    # =========================================================================
    # KERNEL GENERATION HELPERS
    # =========================================================================

    def _generate_kernels_for_nodes(self, nodes, env, generator, scope_prefix=None):
        nodes = fuse_elementwise_chains(nodes, env)
        kernels = []
        cuda_code = ""
        flat_items = []

        for node in nodes:
            if node.get("type") in ("backward", "save", "rebind", "for"):
                flat_items.append((None, node))
                continue

            if node.get("type") in OPTIMIZER_NAMES or (
                node.get("type") == "call" and node.get("name") in OPTIMIZER_NAMES
            ):
                opt_name, param_name, grad_name, lr_arg, beta1_arg, beta2_arg, eps_arg = self._optimizer_parts(node)
                if param_name is None or grad_name is None or param_name not in env or grad_name not in env:
                    flat_items.append((None, node))
                    continue

                kernel_name = f"opt{len(kernels)}"
                if scope_prefix:
                    kernel_name = f"{scope_prefix}_{kernel_name}"
                self._kernel_name_map[kernel_name] = param_name

                kernel_str, kernel_info = self._kernel_for_optimizer(
                    node, env, generator, kernel_name=kernel_name
                )
                if kernel_str is not None:
                    flat_idx = len(kernels)
                    kernels.append(kernel_info)
                    cuda_code += kernel_str
                    flat_items.append((flat_idx, node))
                else:
                    flat_items.append((None, node))
                continue

            if node.get("type") != "let":
                flat_items.append((None, node))
                continue

            name = node["name"]
            expr = node.get("expr")

            if not isinstance(expr, dict):
                flat_items.append((None, node))
                continue

            if expr.get("type") in ("name", "tensor_literal", "load"):
                flat_items.append((None, node))
                continue

            kernel_name = name
            if scope_prefix:
                kernel_name = f"{scope_prefix}_{name}"
                self._kernel_name_map[kernel_name] = name

            kernel_str, kernel_info = self._kernel_for_expr(
                name, expr, env, generator, kernel_name=kernel_name
            )
            if kernel_str is not None:
                flat_idx = len(kernels)
                kernels.append(kernel_info)
                cuda_code += kernel_str
                flat_items.append((flat_idx, node))
            else:
                flat_items.append((None, node))

        return kernels, cuda_code, flat_items

    def _kernel_for_optimizer(self, node, env, generator, kernel_name=None):
        kernel_name = kernel_name or f"opt{len(self._kernel_name_map)}"

        opt_name, param_name, grad_name, lr_arg, beta1_arg, beta2_arg, eps_arg = self._optimizer_parts(node)
        if opt_name not in OPTIMIZER_NAMES or param_name is None or grad_name is None:
            return None, None
        if param_name not in env or grad_name not in env:
            return None, None

        size = int(np.prod([int(d) for d in env[param_name]["shape"]]))

        if opt_name in ("sgd", "sgd_update"):
            kernel_str, kernel_info = generator.sgd_update(
                "sgd_update", kernel_name, param_name, grad_name, size
            )
            kernel_info = kernel_info + (lr_arg,)
        else:
            kernel_str, kernel_info = generator.adam_update(
                "adam_update", kernel_name, param_name, grad_name, size
            )
            kernel_info = kernel_info + (lr_arg, beta1_arg, beta2_arg, eps_arg)

        self._kernel_name_map[kernel_name] = param_name
        return kernel_str, kernel_info

    # def _kernel_for_fused_expr(self, name, expr, env, generator, kernel_name=None):
        # kernel_name = kernel_name or name
        # chain = expr["chain"]
        # for op_type, out_name, args in chain:
            # self._record_operation(op_type, out_name, list(args))
        # final_name = chain[-1][1]
        # shape = env[final_name]["shape"]
        # size = int(np.prod([int(d) for d in shape])) if shape else 1
        # outputs = [op[1] for op in chain]
        # output_set = set(outputs)
        # external_inputs = []
        # seen = set()
        # for op in chain:
            # for a in op[2]:
                # if a not in output_set and a not in seen:
                    # seen.add(a)
                    # external_inputs.append(a)
        # kernel_str, kernel_info = generator.fused_elementwise(
            # "fused_elementwise", kernel_name, chain, env, external_inputs, outputs, size
        # )
        # self._kernel_name_map[kernel_name] = final_name
        # return kernel_str, kernel_info

    def _kernel_for_fused_expr(self, name, expr, env, generator, kernel_name=None):
        kernel_name = kernel_name or name
        chain = expr["chain"]
        for op_type, out_name, args in chain:
            self._record_operation(op_type, out_name, list(args))
        final_name = chain[-1][1]
        shape = env[final_name]["shape"]
        size = int(np.prod([int(d) for d in shape])) if shape else 1
        outputs = [op[1] for op in chain]
        output_set = set(outputs)
        external_inputs = []
        seen = set()
        for op in chain:
            for a in op[2]:
                if a not in output_set and a not in seen:
                    seen.add(a)
                    external_inputs.append(a)
        broadcast = bool(expr.get("broadcast", False))
        input_shapes = [list(env[a]["shape"]) for a in external_inputs] if broadcast else None
        output_shape = list(shape) if broadcast else None
        kernel_str, kernel_info = generator.fused_elementwise(
            "fused_elementwise", kernel_name, chain, env, external_inputs, outputs, size,
            broadcast=broadcast, input_shapes=input_shapes, output_shape=output_shape,
        )
        self._kernel_name_map[kernel_name] = final_name
        return kernel_str, kernel_info

    def _kernel_for_expr(self, name, expr, env, generator, kernel_name=None):
        kernel_name = kernel_name or name

        if expr.get("type") == "fused_elementwise":
            return self._kernel_for_fused_expr(name, expr, env, generator, kernel_name=kernel_name)

        if expr.get("type") in OPTIMIZER_NAMES:
            node_like = {"type": expr.get("type"), "args": expr.get("args", expr.get("arguments", []))}
            return self._kernel_for_optimizer(node_like, env, generator, kernel_name=kernel_name)

        if expr.get("type") in ("add", "minus", "mult", "div"):
            arg1, arg2 = expr["args"]
            shape1, shape2 = env[arg1]["shape"], env[arg2]["shape"]
            output_shape = env[name]["shape"]
            self._record_operation(expr["type"], name, [arg1, arg2])
            if shape1 == shape2:
                size = int(np.prod([int(d) for d in output_shape]))
                return generator.elementwise(expr["type"], kernel_name, arg1, arg2, size)
            if _is_scalar_shape(shape1) or _is_scalar_shape(shape2):
                scalar_on_left = _is_scalar_shape(shape1) and not _is_scalar_shape(shape2)
                size = int(np.prod([int(d) for d in output_shape]))
                return generator.binary_1d_broadcast(
                    expr["type"], kernel_name, arg1, arg2, size, scalar_on_left=scalar_on_left
                )
            if len(shape1) == 2 and len(shape2) == 1:
                return generator.binary_broadcast(
                    expr["type"], expr["type"], kernel_name, arg1, arg2, shape1, shape2, output_shape
                )
            if len(shape1) == 1 and len(shape2) == 1 and (shape1[0] == 1 or shape2[0] == 1):
                size = int(np.prod([int(d) for d in output_shape]))
                return generator.binary_1d_broadcast(expr["type"], kernel_name, arg1, arg2, size)
            if self.can_broadcast(shape1, shape2):
                return generator.binary_general_broadcast(
                    expr["type"], kernel_name, arg1, arg2, shape1, shape2, output_shape
                )

        if expr.get("type") == "matmul":
            arg1, arg2 = expr["args"]
            a_shape = env[arg1]["shape"]
            b_shape = env[arg2]["shape"]
            if len(a_shape) == 2 and len(b_shape) == 2:
                m, n, p = a_shape[0], a_shape[1], b_shape[1]
            elif len(a_shape) == 2 and len(b_shape) == 1:
                m, n, p = a_shape[0], a_shape[1], 1
            else:
                raise TypeError(f"Unsupported matmul shapes: {a_shape} @ {b_shape}")
            self._record_operation("matmul", name, [arg1, arg2])
            return generator.matmul(expr["type"], kernel_name, arg1, arg2, int(m), int(n), int(p))

        if expr.get("type") in ("relu", "sigmoid", "tanh", "gelu", "swish"):
            arg1 = expr["args"][0]
            size = int(np.prod([int(d) for d in env[arg1]["shape"]]))
            self._record_operation(expr["type"], name, [arg1])
            method = getattr(generator, expr["type"], None)
            if method:
                return method(expr["type"], kernel_name, arg1, size)

        if expr.get("type") == "softmax":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            self._record_operation("softmax", name, [tensor_name], metadata={"axis": axis})
            return generator.softmax(expr["type"], kernel_name, tensor_name, input_shape, axis)

        if expr.get("type") == "mse_loss":
            pred_name, target_name = expr["args"]
            pred_shape = env[pred_name]["shape"]
            total_elements = int(np.prod([int(d) for d in pred_shape]))
            self._record_operation("mse_loss", name, [pred_name, target_name])
            return generator.mse_loss(expr["type"], kernel_name, pred_name, target_name, total_elements)

        if expr.get("type") == "linear":
            input_name, weight_name, bias_name = expr["args"]
            input_shape = env[input_name]["shape"]
            weight_shape = env[weight_name]["shape"]
            output_shape = env[name]["shape"]
            self._record_operation("linear", name, [input_name, weight_name, bias_name])
            return generator.linear(
                expr["type"], kernel_name, input_name, weight_name, bias_name,
                input_shape, weight_shape, output_shape,
            )

        if expr.get("type") == "sum":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            self._record_operation("sum", name, [tensor_name], metadata={"axis": axis})
            return generator.sum(expr["type"], kernel_name, tensor_name, axis, input_shape)

        if expr.get("type") == "mean":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            self._record_operation("mean", name, [tensor_name], metadata={"axis": axis})
            return generator.mean(expr["type"], kernel_name, tensor_name, axis, input_shape)

        return None, None

    # =========================================================================
    # COMPILE AND EXECUTE
    # =========================================================================

    def compile_and_execute(self, tensorlang_file):
        if self.debug_mode:
            self.tensorlang.print_header(f"[COMPILER] TensorLang {self.version}")

        if tensorlang_file:
            file_path = Path(tensorlang_file)
        else:
            self.tensorlang.print(message="[COMPILER] Error missing file")
            return False

        if not file_path.exists():
            self.tensorlang.print(message=f"[COMPILER] Error: {tensorlang_file} not found at {file_path}")
            sys.exit(1)

        if file_path.suffix == '.tl':
            if self.debug_mode:
                file_details = {
                    "// Path     " : str(file_path),
                    "// Name     " : file_path.name,
                    "// Suffix   " : file_path.suffix or "None",
                    "// Size     " : f"{file_path.stat().st_size} bytes",
                    "// Modified " : time.ctime(file_path.stat().st_mtime),
                }
                details_str = "\n".join(f"{key} {value}" for key, value in file_details.items())
                self.tensorlang.print(type=details_str)
                self.tensorlang.print(type="// ============================================================================")
                self.tensorlang.print(type="// ============================================================================\n")
            else:
                print(f"[COMPILER] TensorLang {self.version} — compiling {file_path.name}")
        else:
            self.tensorlang.print(message="[COMPILER] file not found, suffix is not .tl")
            self.tensorlang.separator()
            sys.exit(1)

        compile_start_time = time.time()

        try:
            if self.debug_mode:
                self.tensorlang.separator()
            grammar_file = "tensorlang.lark"
            with open(grammar_file, "r") as f:
                grammar = f.read()
            parser = Lark(grammar, start="program", parser="lalr")
            if self.debug_mode:
                self.tensorlang.print(type="[COMPILER]", message=f"Loaded Lark Grammar file: {grammar_file}")
        except FileNotFoundError:
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Missing Lark Grammar file: {grammar_file}")
            sys.exit(1)

        try:
            with open(file_path, "r") as f:
                code = f.read()
                if self.debug_mode:
                    self.tensorlang.print(type="[COMPILER]", message=f"Loaded TensorLang file: {file_path}")
        except FileNotFoundError:
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Missing TensorLang file: {file_path}")
            sys.exit(1)

        try:
            if self.debug_mode:
                self.tensorlang.separator()
                self.tensorlang.print(type="")
                self.tensorlang.print(type=f"{code}")
                self.tensorlang.separator()
                self.tensorlang.print(type="")
                self.tensorlang.separator()
                self.tensorlang.print(type="[COMPILER] TensorLang > Lark > Parser > Compiler > CUDA Kernel -> Results")
                self.tensorlang.separator()

            cache_base = Path("cache")
            cache_file_dir = cache_base / file_path
            cache_file_dir.mkdir(parents=True, exist_ok=True)

            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] CACHE: {cache_base}")
                self.tensorlang.print(message=f"[COMPILER] CACHE FILE_PATH: {file_path}")
                self.tensorlang.print(message=f"[COMPILER] CACHE FILE_PATH STEM: {file_path.stem}")
                self.tensorlang.print(message=f"[COMPILER] CACHE TENSORLANG_FILE: {tensorlang_file}")

            if self.debug_info:
                self.tensorlang.print(type="[INFO]", message="Created cache for tensor outputs directory")

            parse_tree = parser.parse(code)
            if self.debug_ast:
                self.tensorlang.print(message=f"Parsed AST:\n{parse_tree.pretty()}")

            ast, output_tensor, functions = build_ast(parse_tree, self.debug_mode, self.debug_info)

            functions_called = list(functions.keys())
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] AST:\n{ast}")
                if functions_called:
                    self.tensorlang.print(message=f"[COMPILER] Functions: {functions_called}")
                self.tensorlang.print(message=f"[COMPILER] Output Tensor: {output_tensor}")

            success, env = type_checker(ast, {}, self.debug_info, self.debug_mode)

            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] TYPE CHECKER ENV:\n{env}")

            if success:
                ast = fuse_elementwise_chains(ast, env)

                tensors = {}
                gpu_allocs = {}

                for node in ast:
                    if node.get("type") == "let" and node.get("requires_grad", False):
                        grad_name = f"{node['name']}_grad"
                        source_shape = env[node["name"]]["shape"]
                        env[grad_name] = {"dtype": "f32", "shape": source_shape}
                        if self.debug_mode:
                            self.tensorlang.print(
                                message=f"[COMPILER] Pre-registered gradient shape: "
                                        f"{grad_name} -> {source_shape}"
                            )
                    if node.get("type") == "for":
                        for body_node in node.get("body", []):
                            if body_node.get("type") == "let" and body_node.get("requires_grad", False):
                                grad_name = f"{body_node['name']}_grad"
                                if body_node["name"] in env:
                                    source_shape = env[body_node["name"]]["shape"]
                                    env[grad_name] = {"dtype": "f32", "shape": source_shape}

                generator = KernelGenerator(self.debug_mode)
                kernels = []
                cuda_code = generator.cuda_header()

                for node in ast:
                    if node.get("type") == "save":
                        tensor_name = node["tensor"]
                        if tensor_name not in env:
                            self.tensorlang.print(message=f"[COMPILER] Error: Cannot save undefined tensor '{tensor_name}'")
                            return False
                        if self.debug_mode:
                            self.tensorlang.print(message=f"[COMPILER] Queued save operation: {tensor_name} -> {node['file_path']}")
                        continue

                    if node.get("type") == "backward":
                        continue

                    if node.get("type") == "rebind":
                        continue

                    if node.get("type") == "for":
                        scope_prefix = f"loop{self._loop_scope_counter}"
                        self._loop_scope_counter += 1
                        body_kernels, body_cuda, flat_items = self._generate_kernels_for_nodes(
                            node["body"], env, generator, scope_prefix=scope_prefix
                        )
                        global_flat_items = []
                        for local_idx, body_node in flat_items:
                            if local_idx is not None:
                                global_idx = len(kernels) + local_idx
                                global_flat_items.append((global_idx, body_node))
                            else:
                                global_flat_items.append((None, body_node))
                        kernels.extend(body_kernels)
                        cuda_code += body_cuda
                        node["_flat_items"] = global_flat_items
                        continue

                    if node.get("type") in OPTIMIZER_NAMES or (
                        node.get("type") == "call" and node.get("name") in OPTIMIZER_NAMES
                    ):
                        kernel_name = f"opt{len(kernels)}"
                        kernel_str, kernel_info = self._kernel_for_optimizer(node, env, generator, kernel_name=kernel_name)
                        if kernel_str is not None:
                            kernels.append(kernel_info)
                            cuda_code += kernel_str
                        continue

                    if node.get("type") == "let" and node.get("requires_grad", False):
                        name = node["name"]
                        self.requires_grad_tensors.add(name)
                        if self.debug_mode:
                            self.tensorlang.print(message=f"[COMPILER] [Autograd] Tensor '{name}' marked for gradient tracking")

                    if node.get("type") == "let" and isinstance(node.get("expr"), dict):
                        name = node["name"]
                        expr = node["expr"]

                        if expr.get("type") == "name":
                            continue

                        if self.debug_mode:
                            self.tensorlang.print(message=f"[COMPILER] Processing {name} ({expr.get('type')})")

                        if expr.get("type") == "tensor_literal":
                            shape = tuple(int(dim) for dim in env[name]["shape"])
                            tensors[name] = np.array(expr["data"], dtype=np.float32).reshape(shape)
                            if self.debug_info:
                                self.tensorlang.print(type="[INFO]", message=f"Kernel Tensor Initialized {name} with shape {shape}")

                        elif expr.get("type") == "load":
                            file_path_load = expr["file_path"]
                            array_name = expr.get("array_name")
                            try:
                                loaded_data, metadata = self._load_data_file(file_path_load, array_name)
                                if self.debug_mode:
                                    self.tensorlang.print(
                                        message=f"[COMPILER] Loaded {metadata['format'].upper()}: "
                                                f"shape={loaded_data.shape}, "
                                                f"size={loaded_data.size:,} elements"
                                    )
                                if name in env:
                                    expected_shape = tuple(int(dim) for dim in env[name]["shape"])
                                    if loaded_data.shape != expected_shape:
                                        self.tensorlang.print(
                                            message=f"[COMPILER] Shape mismatch: {file_path_load}\n"
                                                    f"  File shape:     {loaded_data.shape}\n"
                                                    f"  Declared shape: {expected_shape}"
                                        )
                                        return False
                                else:
                                    env[name] = {
                                        "dtype": "f32",
                                        "shape": loaded_data.shape,
                                        "metadata": metadata,
                                    }
                                tensors[name] = loaded_data
                                if self.cache_layers:
                                    cache_npy_path = cache_file_dir / f"{name}.npy"
                                    np.save(cache_npy_path, loaded_data)
                                if self.debug_info:
                                    if "columns" in metadata:
                                        self.tensorlang.print(type="[INFO]", message=f"Columns: {metadata['columns']}")
                                    preview = loaded_data.flatten()[:5]
                                    self.tensorlang.print(type="[INFO]", message=f"First values: {preview}")
                            except FileNotFoundError as e:
                                self.tensorlang.print(message=f"[COMPILER] Error: {e}")
                                return False
                            except (OSError, ValueError, KeyError) as e:
                                self.tensorlang.print(message=f"[COMPILER] Error loading {file_path_load}: {e}")
                                if self.debug_mode:
                                    traceback.print_exc()
                                return False

                        elif expr.get("type") in ("add", "minus", "mult", "div"):
                            arg1, arg2 = expr["args"]
                            shape1, shape2 = env[arg1]["shape"], env[arg2]["shape"]
                            output_shape = env[name]["shape"]
                            self._record_operation(expr["type"], name, [arg1, arg2])
                            if self.debug_mode:
                                self.tensorlang.print(message=f"Tensor {expr['type']}")
                                self.tensorlang.print(message=f"Tensor Shape1: {len(shape1)} Shape2: {len(shape2)}")
                            if shape1 == shape2:
                                size = int(np.prod([int(d) for d in output_shape]))
                                kernel, kernel_info = generator.elementwise(expr["type"], name, arg1, arg2, size)
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif _is_scalar_shape(shape1) or _is_scalar_shape(shape2):
                                scalar_on_left = _is_scalar_shape(shape1) and not _is_scalar_shape(shape2)
                                size = int(np.prod([int(d) for d in output_shape]))
                                kernel, kernel_info = generator.binary_1d_broadcast(
                                    expr["type"], name, arg1, arg2, size, scalar_on_left=scalar_on_left
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif len(shape1) == 2 and len(shape2) == 1:
                                kernel, kernel_info = generator.binary_broadcast(
                                    expr["type"], expr["type"], name, arg1, arg2, shape1, shape2, output_shape
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif len(shape1) == 1 and len(shape2) == 1 and (shape1[0] == 1 or shape2[0] == 1):
                                size = int(np.prod([int(d) for d in output_shape]))
                                kernel, kernel_info = generator.binary_1d_broadcast(expr["type"], name, arg1, arg2, size)
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif self.can_broadcast(shape1, shape2):
                                kernel, kernel_info = generator.binary_general_broadcast(
                                    expr["type"], name, arg1, arg2, shape1, shape2, output_shape
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            else:
                                self.tensorlang.print(
                                    message=(
                                        f"[COMPILER] Error: {expr['type']}: shape mismatch computing '{name}': "
                                        f"'{arg1}' {tuple(shape1)} vs '{arg2}' {tuple(shape2)}. "
                                        f"Shapes must match, one operand must be a scalar, or they must "
                                        f"broadcast (trailing dimensions equal or 1)"
                                    )
                                )
                                return False

                        elif expr.get("type") == "fused_elementwise":
                            kernel, kernel_info = self._kernel_for_fused_expr(name, expr, env, generator, kernel_name=name)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "fill":
                            size = int(np.prod([int(dim) for dim in expr["shape"]]))
                            kernel, kernel_info = generator.fill(expr["type"], name, None, None, size, expr["value"])
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "matmul":
                            arg1, arg2 = expr["args"]
                            a_shape = env[arg1]["shape"]
                            b_shape = env[arg2]["shape"]
                            if len(a_shape) == 2 and len(b_shape) == 2:
                                m, n, p = a_shape[0], a_shape[1], b_shape[1]
                            elif len(a_shape) == 2 and len(b_shape) == 1:
                                m, n, p = a_shape[0], a_shape[1], 1
                            else:
                                raise TypeError(f"Unsupported matmul shapes: {a_shape} @ {b_shape}")
                            kernel, kernel_info = generator.matmul(
                                expr["type"], name, arg1, arg2, int(m), int(n), int(p)
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            if len(a_shape) == 2 and len(b_shape) == 2:
                                out_shape = (a_shape[0], b_shape[1])
                            elif len(a_shape) == 2 and len(b_shape) == 1:
                                out_shape = (a_shape[0],)
                            elif len(a_shape) == 1 and len(b_shape) == 2:
                                out_shape = (b_shape[1],)
                            else:
                                raise TypeError(f"Unsupported matmul shapes: {a_shape} @ {b_shape}")
                            env[name] = {"dtype": env[arg1]["dtype"], "shape": out_shape}
                            self._record_operation("matmul", name, [arg1, arg2])

                        elif expr.get("type") in ("relu", "sigmoid", "tanh", "gelu", "swish"):
                            arg1 = expr["args"][0]
                            size = int(np.prod([int(dim) for dim in env[arg1]["shape"]]))
                            method = getattr(generator, expr["type"], None)
                            if method is None:
                                raise ValueError(f"No method found for operation: {expr['type']}")
                            kernel, kernel_info = method(expr["type"], name, arg1, size)
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation(expr["type"], name, [arg1])

                        elif expr.get("type") == "softmax":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.softmax(expr["type"], name, tensor_name, input_shape, axis)
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("softmax", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "greater":
                            arg1, arg2 = expr["args"]
                            shape1, shape2 = env[arg1]["shape"], env[arg2]["shape"]
                            output_shape = env[name]["shape"]
                            size = int(np.prod([int(dim) for dim in output_shape]))
                            kernel, kernel_info = generator.greater_broadcast(
                                expr["type"], name, arg1, arg2, shape1, shape2, output_shape, size
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "less":
                            arg1, arg2 = expr["args"]
                            shape1, shape2 = env[arg1]["shape"], env[arg2]["shape"]
                            output_shape = env[name]["shape"]
                            kernel, kernel_info = generator.less_broadcast(
                                expr["type"], name, arg1, arg2, shape1, shape2, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "equal":
                            arg1, arg2 = expr["args"]
                            shape1, shape2 = env[arg1]["shape"], env[arg2]["shape"]
                            output_shape = env[name]["shape"]
                            kernel, kernel_info = generator.equal(
                                expr["type"], name, arg1, arg2, shape1, shape2, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "sum":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.sum(expr["type"], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("sum", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "mean":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.mean(expr["type"], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("mean", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "slice":
                            tensor_name = expr["tensor"]
                            slice_specs = expr["specs"]
                            input_shape = env[tensor_name]["shape"]
                            output_shape = env[name]["shape"]
                            kernel, kernel_info = generator.slice(
                                expr["type"], name, tensor_name, slice_specs, input_shape, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "max":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.max(expr["type"], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "min":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.min(expr["type"], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "argmax":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.argmax(expr["type"], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "argmin":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.argmin(expr["type"], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "linear":
                            input_name, weight_name, bias_name = expr["args"]
                            input_shape = env[input_name]["shape"]
                            weight_shape = env[weight_name]["shape"]
                            output_shape = env[name]["shape"]
                            self._record_operation("linear", name, [input_name, weight_name, bias_name])
                            kernel, kernel_info = generator.linear(
                                expr["type"], name, input_name, weight_name, bias_name,
                                input_shape, weight_shape, output_shape,
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "layer_norm":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            eps = expr.get("eps", 1e-5)
                            input_shape = env[tensor_name]["shape"]
                            if axis is None:
                                axis = len(input_shape) - 1
                            self._record_operation("layer_norm", name, [tensor_name], metadata={"axis": axis, "eps": eps})
                            kernel, kernel_info = generator.layer_norm(
                                expr["type"], name, tensor_name, axis, eps, input_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "cross_entropy":
                            pred_name, target_name = expr["args"]
                            pred_shape = env[pred_name]["shape"]
                            target_shape = env[target_name]["shape"]
                            self._record_operation("cross_entropy", name, [pred_name, target_name])
                            kernel, kernel_info = generator.cross_entropy(
                                expr["type"], name, pred_name, pred_shape, target_shape, target_name
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "mse_loss":
                            pred_name, target_name = expr["args"]
                            pred_shape = env[pred_name]["shape"]
                            total_elements = int(np.prod([int(dim) for dim in pred_shape]))
                            kernel, kernel_info = generator.mse_loss(
                                expr["type"], name, pred_name, target_name, total_elements
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("mse_loss", name, [pred_name, target_name])

                        elif expr.get("type") == "transpose":
                            tensor_name = expr["tensor"]
                            input_shape = env[tensor_name]["shape"]
                            axes = expr.get("axes")
                            kernel, kernel_info = generator.transpose(expr["type"], name, input_shape, tensor_name, axes)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "reshape":
                            tensor_name = expr["tensor"]
                            input_shape = env[tensor_name]["shape"]
                            total_elements = int(np.prod([int(dim) for dim in input_shape]))
                            kernel, kernel_info = generator.reshape(expr["type"], name, tensor_name, total_elements)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "concat":
                            tensor_names = expr["tensors"]
                            axis = expr["axis"]
                            kernel, kernel_info = generator.concat(expr["type"], name, tensor_names, axis, env)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "batch_norm":
                            tensor_name = expr["tensor"]
                            running_mean_name = expr["running_mean"]
                            running_var_name = expr["running_var"]
                            eps = expr.get("eps", 1e-5)
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.batch_norm(
                                expr["type"], name, input_shape, tensor_name, running_mean_name, eps, running_var_name
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "instance_norm":
                            tensor_name = expr["tensor"]
                            eps = expr.get("eps", 1e-5)
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.instance_norm(
                                expr["type"], name, tensor_name, input_shape, eps
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                if kernels:
                    if self.debug_mode:
                        self.tensorlang.print(message="[COMPILER] KERNEL CUDA!")

                    kernel_cu_path = cache_file_dir / "kernel.cu"
                    kernel_so_path = cache_file_dir / "kernel.so"

                    already_compiled = (
                        kernel_so_path.exists()
                        and kernel_cu_path.exists()
                        and kernel_cu_path.read_text() == cuda_code
                    )

                    if already_compiled:
                        if self.debug_mode:
                            self.tensorlang.print(
                                message=f"[COMPILER] CUDA Compile: kernel unchanged, reusing cached {kernel_so_path}"
                            )
                    else:
                        with open(kernel_cu_path, "w") as f:
                            f.write(cuda_code)
                            if self.debug_mode:
                                self.tensorlang.print(message=f"[COMPILER] CUDA Compile: {kernel_cu_path} written!")

                        try:
                            subprocess.run(
                                [
                                    "nvcc", "-o", str(kernel_so_path),
                                    "--shared", "-Xcompiler", "-fPIC", "-lcudart", "-lcublas",
                                    str(kernel_cu_path),
                                ],
                                check=True,
                            )
                            if self.debug_mode:
                                self.tensorlang.print(message=f"[COMPILER] CUDA Compile: {kernel_so_path} compiled!")
                        except subprocess.CalledProcessError as e:
                            self.tensorlang.print(message=f"[COMPILER] CUDA Compile: error {e}")
                            sys.exit(1)

                    if self.transpile:
                        try:
                            from tensorlang.transpiler.wgsl import WGSLTranspiler

                            if self.debug_mode:
                                self.tensorlang.print(message=f"[TRANSPILE] Reading {kernel_cu_path} for WGSL...")
                            transpiler = WGSLTranspiler(wgsl_workgroup_size=64)
                            wgsl = transpiler.transpile(cuda_code)
                            transpiler.save(wgsl, cache_file_dir)
                            if self.debug_mode:
                                self.tensorlang.print(f"[WGSL] Saved to {cache_file_dir}")
                        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as e:
                            self.tensorlang.print(message=f"[TRANSPILE] ERROR: {e}")
                            traceback.print_exc()
                            sys.exit(1)

                    try:
                        lib = cdll.LoadLibrary(os.path.abspath(str(kernel_so_path)))

                        backward_index = None
                        for i, node in enumerate(ast):
                            if node.get("type") == "backward":
                                backward_index = i
                                if self.debug_mode:
                                    self.tensorlang.print(message=f"[COMPILER] CUDA Execute: found backward() at AST index {i}")
                                break

                        for name in env:
                            shape = tuple(int(dim) for dim in env[name]["shape"])
                            size_bytes = int(np.prod(shape) * np.float32().nbytes) if shape else np.float32().nbytes
                            gpu_allocs[name] = cuda.mem_alloc(size_bytes)
                            if name in tensors:
                                cuda.memcpy_htod(gpu_allocs[name], tensors[name])
                                if self.debug_mode:
                                    self.tensorlang.print(
                                        message=f"[COMPILER] CUDA Execute: copied \"{name}\" to GPU\n"
                                                f"shape: {tensors[name].shape} \n"
                                                f"sample: {tensors[name][:2] if getattr(tensors[name], 'ndim', 0) > 1 else tensors[name]}"
                                    )

                        ast_to_kernel_map = {}
                        kernel_idx = 0
                        for ast_idx, node in enumerate(ast):
                            if node.get("type") == "for":
                                kernel_idx += len([
                                    fi for fi, _ in node.get("_flat_items", [])
                                    if fi is not None
                                ])
                            elif node.get("type") in OPTIMIZER_NAMES or (
                                node.get("type") == "call" and node.get("name") in OPTIMIZER_NAMES
                            ):
                                ast_to_kernel_map[ast_idx] = kernel_idx
                                kernel_idx += 1
                            elif node.get("type") == "let" and isinstance(node.get("expr"), dict):
                                expr = node["expr"]
                                if expr.get("type") not in ("name", "tensor_literal", "load"):
                                    ast_to_kernel_map[ast_idx] = kernel_idx
                                    kernel_idx += 1

                        if backward_index is not None:
                            if self.debug_mode:
                                self.tensorlang.print(
                                    message=f"[COMPILER] CUDA Execute: {len([k for a, k in ast_to_kernel_map.items() if a < backward_index])} kernels before backward"
                                )
                            for ast_idx in range(backward_index):
                                if ast_idx in ast_to_kernel_map:
                                    k_idx = ast_to_kernel_map[ast_idx]
                                    if k_idx < len(kernels):
                                        self._execute_single_kernel(
                                            kernels[k_idx], lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                                        )
                                elif ast[ast_idx].get("type") == "for":
                                    self._execute_for_loop(
                                        ast[ast_idx], kernels, ast_to_kernel_map,
                                        lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                                    )
                                elif (
                                    ast[ast_idx].get("type") == "let"
                                    and isinstance(ast[ast_idx].get("expr"), dict)
                                    and ast[ast_idx]["expr"].get("type") == "name"
                                ):
                                    self._resolve_alias_inline(ast[ast_idx], gpu_allocs, tensors)
                        else:
                            for ast_idx, node in enumerate(ast):
                                if node.get("type") == "for":
                                    self._execute_for_loop(
                                        node, kernels, ast_to_kernel_map,
                                        lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                                    )
                                elif ast_idx in ast_to_kernel_map:
                                    k_idx = ast_to_kernel_map[ast_idx]
                                    if k_idx < len(kernels):
                                        self._execute_single_kernel(
                                            kernels[k_idx], lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                                        )
                                elif (
                                    node.get("type") == "let"
                                    and isinstance(node.get("expr"), dict)
                                    and node["expr"].get("type") == "name"
                                ):
                                    self._resolve_alias_inline(node, gpu_allocs, tensors)

                        for node in ast:
                            if node.get("type") == "let" and isinstance(node.get("expr"), dict) and node["expr"].get("type") == "name":
                                self._resolve_alias_inline(node, gpu_allocs, tensors)

                        if self.cache_layers:
                            for name, value in tensors.items():
                                cache_npy_path = cache_file_dir / f"{name}.npy"
                                if not cache_npy_path.exists():
                                    np.save(cache_npy_path, value)
                                    if self.debug_mode:
                                        self.tensorlang.print(message=f"[COMPILER] CUDA Execute: tensor literal: {name}")

                        for name, value in tensors.items():
                            requires_grad = name in self.requires_grad_tensors
                            self.comp_graph.register_tensor(name, value, requires_grad)
                            if self.debug_mode and requires_grad:
                                self.tensorlang.print(message=f"[COMPILER] [Autograd] CUDA Execute: registered '{name}' with gradient tracking")

                        for node in ast:
                            if node.get("type") == "backward":
                                loss_name = node["loss_tensor"]
                                print(f"\n{'=' * 80}")
                                print(f"BACKWARD PASS from '{loss_name}'")
                                print("=" * 80)
                                try:
                                    self.comp_graph.backward(loss_name)
                                    for grad_name in self.comp_graph.requires_grad:
                                        if grad_name not in self.comp_graph.gradients:
                                            continue
                                        grad_tensor = self.comp_graph.gradients[grad_name]
                                        print(f"\nGradient {grad_name}.grad:\n{grad_tensor}")
                                        if self.cache_layers:
                                            grad_cache_path = cache_file_dir / f"{grad_name}.grad.npy"
                                            np.save(grad_cache_path, grad_tensor)
                                    print("=" * 80)
                                    for grad_name in self.comp_graph.requires_grad:
                                        if grad_name in self.comp_graph.gradients:
                                            grad_tensor = self.comp_graph.gradients[grad_name]
                                            grad_tensor_name = f"{grad_name}_grad"
                                            tensors[grad_tensor_name] = grad_tensor
                                            if grad_tensor_name in gpu_allocs:
                                                cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)
                                                if self.debug_mode:
                                                    self.tensorlang.print(message=f"[COMPILER] [Autograd] Copied gradient {grad_tensor_name} to GPU")
                                            else:
                                                if self.debug_mode:
                                                    self.tensorlang.print(message=f"[COMPILER] [Autograd] Warning: {grad_tensor_name} not in GPU allocations")
                                            env[grad_tensor_name] = {
                                                "dtype": "f32",
                                                "shape": grad_tensor.shape,
                                            }
                                            if self.debug_mode:
                                                self.tensorlang.print(message=f"[COMPILER] [Autograd] Made gradient accessible: {grad_tensor_name}")
                                            if self.cache_layers:
                                                grad_var_cache_path = cache_file_dir / f"{grad_tensor_name}.npy"
                                                np.save(grad_var_cache_path, grad_tensor)
                                except (RuntimeError, ValueError, KeyError, OSError, TypeError) as e:
                                    self.tensorlang.print(message=f"[COMPILER] [Autograd] Error during backward pass: {e}")
                                    if self.debug_mode:
                                        traceback.print_exc()

                        if backward_index is not None:
                            for ast_idx in range(backward_index + 1, len(ast)):
                                if ast_idx in ast_to_kernel_map:
                                    k_idx = ast_to_kernel_map[ast_idx]
                                    if k_idx < len(kernels):
                                        self._execute_single_kernel(
                                            kernels[k_idx], lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                                        )
                                elif (
                                    ast[ast_idx].get("type") == "let"
                                    and isinstance(ast[ast_idx].get("expr"), dict)
                                    and ast[ast_idx]["expr"].get("type") == "name"
                                ):
                                    self._resolve_alias_inline(ast[ast_idx], gpu_allocs, tensors)

                        save_count = sum(1 for n in ast if n.get("type") == "save")
                        if save_count > 0:
                            if self.debug_mode:
                                self.tensorlang.print(message=f"[COMPILER] Executing {save_count} save statement(s)")
                            for node in ast:
                                if node.get("type") == "save":
                                    self._execute_save_statement(node, tensors, gpu_allocs, env, cache_file_dir, cuda)

                        freed_ptrs = set()
                        for name, alloc in gpu_allocs.items():
                            ptr = int(alloc)
                            if ptr not in freed_ptrs:
                                alloc.free()
                                freed_ptrs.add(ptr)
                                if self.debug_mode:
                                    self.tensorlang.print(message=f"[COMPILER] Freed GPU memory for {name}")

                        if output_tensor and output_tensor in tensors:
                            print(f"{output_tensor} = {tensors[output_tensor]}")
                        elif output_tensor and self.debug_mode:
                            self.tensorlang.print(message=f"[COMPILER] Warning: output tensor '{output_tensor}' not found in computed tensors")

                        print(f"Done in {time.time() - compile_start_time:.2f}s")
                        return True

                    except ImportError as e:
                        self.tensorlang.print(message=f"[COMPILER] PyCUDA error: {e}. Run 'pip install pycuda' and ensure CUDA toolkit is installed.")
                        sys.exit(1)
                    except (RuntimeError, OSError, ValueError, TypeError) as e:
                        self.tensorlang.print(message=f"[COMPILER] Error executing CUDA kernel: {e}")
                        traceback.print_exc()
                        sys.exit(1)

            else:
                self.tensorlang.print(
                    message="[COMPILER] Compilation aborted: type checking failed "
                            "(see [TYPE CHECKER] error above)."
                )
                return False

        except ValueError as e:
            self.tensorlang.print(message=f"[COMPILER] Value Error: failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        except UnexpectedInput as e:
            self.tensorlang.print(message=f"[COMPILER] Parse error: {e}")
            traceback.print_exc()
            sys.exit(1)

        except Exception as e:
            self.tensorlang.print(message=f"[COMPILER] Unexpected error: {e}")
            traceback.print_exc()
            sys.exit(1)
