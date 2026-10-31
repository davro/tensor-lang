"""
TensorLang TensorCompiler

File: tensorlang/compiler.py
"""

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
from tensorlang.codegen import CodegenMixin
from tensorlang.constants import (OPTIMIZER_NAMES, dtype_numpy, promote_dtype,
                                 DEFAULT_LOSS_SCALE_FP32, DEFAULT_LOSS_SCALE_FP16)

from tensorlang.execution import ExecutionMixin
from tensorlang.fusion import fuse_elementwise_chains
from tensorlang.io import IOMixin
from tensorlang.interop import InteropMixin
from tensorlang.kernel_generator import KernelGenerator
from tensorlang.tensor_lang import TensorLang
from tensorlang.type_checker import _is_scalar_shape, type_checker

class TensorCompiler(CodegenMixin, ExecutionMixin, IOMixin, InteropMixin):
    def __init__(self, debug_mode=False, debug_info=False, debug_ast=False, cache_layers=False, transpile=False, loss_scale=1.0, dynamic_loss_scale=False, loss_scale_patience=2000):

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
        self.loss_scale = loss_scale
        self._loss_scale = None  # resolved during compile_and_execute
        self._loss_scale_auto = False
        self.dynamic_loss_scale = dynamic_loss_scale
        self.loss_scale_patience = loss_scale_patience
        self._current_scale = None
        self._scale_growth_counter = 0
        self._overflow_detected = False
        self._skip_optimizer = False



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
                    "// Path     ": str(file_path),
                    "// Name     ": file_path.name,
                    "// Suffix   ": file_path.suffix or "None",
                    "// Size     ": f"{file_path.stat().st_size} bytes",
                    "// Modified ": time.ctime(file_path.stat().st_mtime),
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
            # parser = Lark(grammar, start="program", parser="lalr")
            # parser = Lark(grammar, start="program", parser="lalr", keep_all_tokens=True)
            # parser = Lark(grammar, start="program", parser="earley")
            parser = Lark(grammar, start="program", parser="lalr", propagate_positions=True)

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

            # Parse loss scale pragma: "# loss_scale: 65536", "# loss_scale: auto",
            # or "# loss_scale: dynamic [init_scale] [patience]"
            import re
            scale_match = re.search(r'#\s*loss_scale:\s*(.+)', code)
            if scale_match:
                scale_parts = scale_match.group(1).strip().split()
                scale_keyword = scale_parts[0].lower()
                if scale_keyword == "auto":
                    self._loss_scale_auto = True
                elif scale_keyword == "dynamic":
                    self.dynamic_loss_scale = True
                    if len(scale_parts) > 1:
                        try:
                            self.loss_scale = float(scale_parts[1])
                        except ValueError:
                            pass
                    if len(scale_parts) > 2:
                        try:
                            self.loss_scale_patience = int(scale_parts[2])
                        except ValueError:
                            pass
                else:
                    try:
                        self.loss_scale = float(scale_parts[0])
                    except ValueError:
                        self.tensorlang.print(message=f"[COMPILER] Warning: invalid loss_scale pragma '{scale_parts[0]}', using default")
            # Strip pragma lines from code before parsing
            code = re.sub(r'#\s*loss_scale:\s*.+\n?', '', code)


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

            # ast, output_tensor, functions = build_ast(parse_tree, self.debug_mode, self.debug_info)
            ast, output_tensor, functions = build_ast(parse_tree, self.debug_mode, self.debug_info, source_code=code)

            functions_called = list(functions.keys())
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] AST:\n{ast}")
                if functions_called:
                    self.tensorlang.print(message=f"[COMPILER] Functions: {functions_called}")
                self.tensorlang.print(message=f"[COMPILER] Output Tensor: {output_tensor}")

            success, env = type_checker(ast, {}, self.debug_info, self.debug_mode)
            self.env = env
            self.env = env

            # Resolve loss scale: pragma "auto", explicit value, or default 1.0
            if getattr(self, "_loss_scale_auto", False):
                has_fp16 = any(
                    isinstance(v, dict) and v.get("dtype") == "f16"
                    for v in env.values()
                )
                resolved = (DEFAULT_LOSS_SCALE_FP16 if has_fp16
                            else DEFAULT_LOSS_SCALE_FP32)
            else:
                resolved = float(self.loss_scale)

            self._loss_scale = resolved
            self._current_scale = resolved
            if self.debug_mode:
                mode = "dynamic" if self.dynamic_loss_scale else "static"
                self.tensorlang.print(
                    message=f"[COMPILER] Loss scale resolved to {self._current_scale} ({mode})"
                )


            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] TYPE CHECKER ENV:\n{env}")

            if success:
                ast = fuse_elementwise_chains(ast, env)

                tensors = {}
                gpu_allocs = {}
                self.tensors = tensors
                self.gpu_allocs = gpu_allocs
                self.tensors = tensors
                self.gpu_allocs = gpu_allocs

                for node in ast:
                    if node.get("type") == "if":
                        raise NotImplementedError("Control flow (if/elif/else) is not implemented yet")

                    if node.get("type") == "let" and node.get("requires_grad", False):
                        grad_name = f"{node['name']}_grad"
                        source_shape = env[node["name"]]["shape"]
                        source_dtype = env[node["name"]].get("dtype", "f32")
                        if source_dtype == "f16" and self._loss_scale > 65504.0:
                            grad_dtype = "f32"
                        else:
                            grad_dtype = source_dtype
                        env[grad_name] = {"dtype": grad_dtype, "shape": source_shape}
                        # Register FP32 master weight for FP16 params
                        if source_dtype == "f16":
                            master_name = f"{node['name']}_master"
                            env[master_name] = {"dtype": "f32", "shape": source_shape}

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
                                    source_dtype = env[body_node["name"]].get("dtype", "f32")
                                    if source_dtype == "f16" and self._loss_scale > 65504.0:
                                        grad_dtype = "f32"
                                    else:
                                        grad_dtype = source_dtype
                                    env[grad_name] = {"dtype": grad_dtype, "shape": source_shape}
                                    if source_dtype == "f16":
                                        master_name = f"{body_node['name']}_master"
                                        env[master_name] = {"dtype": "f32", "shape": source_shape}

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
                        kernels.extend(body_kernels)
                        cuda_code += body_cuda
                        node["_kernels"] = body_kernels
                        node["_flat_items"] = flat_items
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
                            np_dtype = dtype_numpy(env[name].get("dtype", "f32"))
                            tensors[name] = np.array(expr["data"], dtype=np_dtype).reshape(shape)
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
                                    if loaded_data.dtype == np.float16:
                                        inferred_dtype = "f16"
                                    elif loaded_data.dtype == np.float64:
                                        inferred_dtype = "f64"
                                    else:
                                        inferred_dtype = "f32"
                                    env[name] = {
                                        "dtype": inferred_dtype,
                                        "shape": loaded_data.shape,
                                        "metadata": metadata,
                                    }
                                target_dtype = dtype_numpy(env[name].get("dtype", "f32"))
                                tensors[name] = loaded_data.astype(target_dtype)
                                if self.cache_layers:
                                    cache_npy_path = cache_file_dir / f"{name}.npy"
                                    np.save(cache_npy_path, tensors[name])
                                if self.debug_info:
                                    if "columns" in metadata:
                                        self.tensorlang.print(type="[INFO]", message=f"Columns: {metadata['columns']}")
                                    preview = tensors[name].flatten()[:5]
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
                            result_dtype = promote_dtype(env[arg1].get("dtype", "f32"), env[arg2].get("dtype", "f32"))
                            if self.debug_mode:
                                self.tensorlang.print(message=f"Tensor {expr['type']}")
                                self.tensorlang.print(message=f"Tensor Shape1: {len(shape1)} Shape2: {len(shape2)}")
                            if shape1 == shape2:
                                size = int(np.prod([int(d) for d in output_shape]))
                                kernel, kernel_info = generator.elementwise(expr["type"], name, arg1, arg2, size, dtype=result_dtype)
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif _is_scalar_shape(shape1) or _is_scalar_shape(shape2):
                                scalar_on_left = _is_scalar_shape(shape1) and not _is_scalar_shape(shape2)
                                size = int(np.prod([int(d) for d in output_shape]))
                                kernel, kernel_info = generator.binary_1d_broadcast(
                                    expr["type"], name, arg1, arg2, size, scalar_on_left=scalar_on_left, dtype=result_dtype
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif len(shape1) == 2 and len(shape2) == 1:
                                kernel, kernel_info = generator.binary_broadcast(
                                    expr["type"], expr["type"], name, arg1, arg2, shape1, shape2, output_shape, dtype=result_dtype
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif len(shape1) == 1 and len(shape2) == 1 and (shape1[0] == 1 or shape2[0] == 1):
                                size = int(np.prod([int(d) for d in output_shape]))
                                kernel, kernel_info = generator.binary_1d_broadcast(expr["type"], name, arg1, arg2, size, dtype=result_dtype)
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif self.can_broadcast(shape1, shape2):
                                kernel, kernel_info = generator.binary_general_broadcast(
                                    expr["type"], expr["type"], name, arg1, arg2, shape1, shape2, output_shape, dtype=result_dtype
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
                            fill_dtype = env[name].get("dtype", "f32") if name in env else "f32"
                            kernel, kernel_info = generator.fill(expr["type"], name, None, None, size, expr["value"], dtype=fill_dtype)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "matmul":
                            arg1, arg2 = expr["args"]
                            a_shape = [int(d) for d in env[arg1]["shape"]]
                            b_shape = [int(d) for d in env[arg2]["shape"]]

                            if len(a_shape) >= 2 and len(b_shape) >= 2:
                                m, n = a_shape[-2], a_shape[-1]
                                p = b_shape[-1]
                                if n != b_shape[-2]:
                                    self.tensorlang.print(
                                        message=f"[COMPILER] Error: matmul inner dimension mismatch: "
                                                f"{a_shape} @ {b_shape}"
                                    )
                                    return False

                                batch_a = a_shape[:-2]
                                batch_b = b_shape[:-2]
                                ndim_batch = max(len(batch_a), len(batch_b))
                                padded_a = [1] * (ndim_batch - len(batch_a)) + batch_a
                                padded_b = [1] * (ndim_batch - len(batch_b)) + batch_b

                                out_batch = []
                                for d1, d2 in zip(padded_a, padded_b):
                                    if d1 != d2 and d1 != 1 and d2 != 1:
                                        self.tensorlang.print(
                                            message=f"[COMPILER] Error: matmul batch dims must broadcast: "
                                                    f"{batch_a} vs {batch_b}"
                                        )
                                        return False
                                    out_batch.append(max(d1, d2))

                                batch_count = 1
                                for d in out_batch:
                                    batch_count *= d
                                if not out_batch:
                                    batch_count = 1

                                out_shape = tuple(out_batch + [m, p])

                                if batch_count <= 1 and len(a_shape) == 2 and len(b_shape) == 2:
                                    kernel, kernel_info = generator.matmul(
                                        expr["type"], name, arg1, arg2, int(m), int(n), int(p),
                                        dtype=promote_dtype(env[arg1].get("dtype", "f32"), env[arg2].get("dtype", "f32"))
                                    )
                                else:
                                    kernel, kernel_info = generator.matmul(
                                        expr["type"], name, arg1, arg2, int(m), int(n), int(p),
                                        batch_count=batch_count,
                                        dtype=promote_dtype(env[arg1].get("dtype", "f32"), env[arg2].get("dtype", "f32"))
                                    )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                                env[name] = {"dtype": promote_dtype(env[arg1].get("dtype", "f32"), env[arg2].get("dtype", "f32")), "shape": out_shape}
                                self._record_operation("matmul", name, [arg1, arg2])

                            elif len(a_shape) == 2 and len(b_shape) == 1:
                                m, n = a_shape[0], a_shape[1]
                                if n != b_shape[0]:
                                    self.tensorlang.print(
                                        message=f"[COMPILER] Error: matmul inner dimension mismatch: "
                                                f"{a_shape} @ {b_shape}"
                                    )
                                    return False
                                kernel, kernel_info = generator.matmul(
                                    expr["type"], name, arg1, arg2, int(m), int(n), 1,
                                    dtype=promote_dtype(env[arg1].get("dtype", "f32"), env[arg2].get("dtype", "f32"))
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                                env[name] = {"dtype": promote_dtype(env[arg1].get("dtype", "f32"), env[arg2].get("dtype", "f32")), "shape": (m,)}
                                self._record_operation("matmul", name, [arg1, arg2])
                            else:
                                self.tensorlang.print(
                                    message=f"[COMPILER] Error: unsupported matmul shapes: {a_shape} @ {b_shape}"
                                )
                                return False



                        elif expr.get("type") in ("relu", "sigmoid", "tanh", "gelu", "swish"):
                            arg1 = expr["args"][0]
                            size = int(np.prod([int(dim) for dim in env[arg1]["shape"]]))
                            method = getattr(generator, expr["type"], None)
                            if method is None:
                                raise ValueError(f"No method found for operation: {expr['type']}")
                            kernel, kernel_info = method(expr["type"], name, arg1, size, dtype=env[arg1].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation(expr["type"], name, [arg1])

                        elif expr.get("type") == "softmax":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.softmax(expr["type"], name, tensor_name, input_shape, axis, dtype=env[tensor_name].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("softmax", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "greater":
                            arg1, arg2 = expr["args"]
                            shape1, shape2 = env[arg1]["shape"], env[arg2]["shape"]
                            output_shape = env[name]["shape"]
                            size = int(np.prod([int(dim) for dim in output_shape]))
                            kernel, kernel_info = generator.greater_broadcast(
                                expr["type"], name, arg1, arg2, shape1, shape2, output_shape, size, dtype="f32"
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("greater", name, [arg1, arg2])

                        elif expr.get("type") == "less":
                            arg1, arg2 = expr["args"]
                            shape1, shape2 = env[arg1]["shape"], env[arg2]["shape"]
                            output_shape = env[name]["shape"]
                            kernel, kernel_info = generator.less_broadcast(
                                expr["type"], name, arg1, arg2, shape1, shape2, output_shape, dtype="f32"
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("less", name, [arg1, arg2])

                        elif expr.get("type") == "equal":
                            arg1, arg2 = expr["args"]
                            shape1, shape2 = env[arg1]["shape"], env[arg2]["shape"]
                            output_shape = env[name]["shape"]
                            kernel, kernel_info = generator.equal(
                                expr["type"], name, arg1, arg2, shape1, shape2, output_shape, dtype="f32"
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("equal", name, [arg1, arg2])

                        elif expr.get("type") == "sum":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.sum(expr["type"], name, tensor_name, axis, input_shape, dtype=env[tensor_name].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("sum", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "mean":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.mean(expr["type"], name, tensor_name, axis, input_shape, dtype=env[tensor_name].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("mean", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "slice":
                            tensor_name = expr["tensor"]
                            slice_specs = expr["specs"]
                            input_shape = env[tensor_name]["shape"]
                            output_shape = env[name]["shape"]
                            kernel, kernel_info = generator.slice(
                                expr["type"], name, tensor_name, slice_specs, input_shape, output_shape, dtype=env[tensor_name].get("dtype", "f32")
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("slice", name, [tensor_name], metadata={"specs": slice_specs})

                        elif expr.get("type") == "max":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.max(expr["type"], name, tensor_name, axis, input_shape, dtype=env[tensor_name].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("max", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "min":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.min(expr["type"], name, tensor_name, axis, input_shape, dtype=env[tensor_name].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("min", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "argmax":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.argmax(expr["type"], name, tensor_name, axis, input_shape, dtype=env[tensor_name].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("argmax", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "argmin":
                            tensor_name = expr["tensor"]
                            axis = expr.get("axis")
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.argmin(expr["type"], name, tensor_name, axis, input_shape, dtype=env[tensor_name].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("argmin", name, [tensor_name], metadata={"axis": axis})

                        elif expr.get("type") == "linear":
                            input_name, weight_name, bias_name = expr["args"]
                            input_shape = env[input_name]["shape"]
                            weight_shape = env[weight_name]["shape"]
                            output_shape = env[name]["shape"]
                            self._record_operation("linear", name, [input_name, weight_name, bias_name])
                            kernel, kernel_info = generator.linear(
                                expr["type"], name, input_name, weight_name, bias_name,
                                input_shape, weight_shape, output_shape,
                                dtype=promote_dtype(promote_dtype(env[input_name].get("dtype", "f32"), env[weight_name].get("dtype", "f32")), env[bias_name].get("dtype", "f32"))
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
                                expr["type"], name, tensor_name, axis, eps, input_shape, dtype=env[tensor_name].get("dtype", "f32")
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "cross_entropy":
                            pred_name, target_name = expr["args"]
                            pred_shape = env[pred_name]["shape"]
                            target_shape = env[target_name]["shape"]
                            self._record_operation("cross_entropy", name, [pred_name, target_name])
                            kernel, kernel_info = generator.cross_entropy(
                                expr["type"], name, pred_name, pred_shape, target_shape, target_name,
                                dtype=promote_dtype(env[pred_name].get("dtype", "f32"), env[target_name].get("dtype", "f32"))
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr.get("type") == "mse_loss":
                            pred_name, target_name = expr["args"]
                            pred_shape = env[pred_name]["shape"]
                            total_elements = int(np.prod([int(dim) for dim in pred_shape]))
                            kernel, kernel_info = generator.mse_loss(
                                expr["type"], name, pred_name, target_name, total_elements,
                                dtype=promote_dtype(env[pred_name].get("dtype", "f32"), env[target_name].get("dtype", "f32"))
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("mse_loss", name, [pred_name, target_name])

                        elif expr.get("type") == "transpose":
                            tensor_name = expr["tensor"]
                            input_shape = env[tensor_name]["shape"]
                            axes = expr.get("axes")
                            kernel, kernel_info = generator.transpose(expr["type"], name, input_shape, tensor_name, axes, dtype=env[tensor_name].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("transpose", name, [tensor_name], metadata={"axes": axes})

                        elif expr.get("type") == "reshape":
                            tensor_name = expr["tensor"]
                            input_shape = env[tensor_name]["shape"]
                            total_elements = int(np.prod([int(dim) for dim in input_shape]))
                            kernel, kernel_info = generator.reshape(expr["type"], name, tensor_name, total_elements, dtype=env[tensor_name].get("dtype", "f32"))
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("reshape", name, [tensor_name])

                        elif expr.get("type") == "concat":
                            tensor_names = expr["tensors"]
                            axis = expr["axis"]
                            concat_dtype = env[tensor_names[0]].get("dtype", "f32")
                            for tn in tensor_names[1:]:
                                concat_dtype = promote_dtype(concat_dtype, env[tn].get("dtype", "f32"))
                            kernel, kernel_info = generator.concat(expr["type"], name, tensor_names, axis, env, dtype=concat_dtype)
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("concat", name, tensor_names, metadata={"axis": axis})

                        elif expr.get("type") == "batch_norm":
                            tensor_name = expr["tensor"]
                            running_mean_name = expr["running_mean"]
                            running_var_name = expr["running_var"]
                            eps = expr.get("eps", 1e-5)
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.batch_norm(
                                expr["type"], name, input_shape, tensor_name, running_mean_name, eps, running_var_name,
                                dtype=env[tensor_name].get("dtype", "f32")
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("batch_norm", name, [tensor_name, running_mean_name, running_var_name], metadata={"eps": eps})

                        elif expr.get("type") == "instance_norm":
                            tensor_name = expr["tensor"]
                            eps = expr.get("eps", 1e-5)
                            input_shape = env[tensor_name]["shape"]
                            kernel, kernel_info = generator.instance_norm(
                                expr["type"], name, tensor_name, input_shape, eps, dtype=env[tensor_name].get("dtype", "f32")
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation("instance_norm", name, [tensor_name], metadata={"eps": eps})

                if kernels or cuda_code.strip():
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

                        for name, info in env.items():
                            if name.endswith("_grad"):
                                continue
                            shape = tuple(int(dim) for dim in info["shape"])
                            np_dtype = dtype_numpy(info.get("dtype", "f32"))
                            size_bytes = int(np.prod(shape) * np_dtype.itemsize) if shape else np_dtype.itemsize
                            if size_bytes == 0:
                                size_bytes = np_dtype.itemsize
                            gpu_allocs[name] = cuda.mem_alloc(size_bytes)
                            if name in tensors:
                                arr = np.ascontiguousarray(tensors[name], dtype=np_dtype)
                                cuda.memcpy_htod(gpu_allocs[name], arr)
                                if self.debug_mode:
                                    self.tensorlang.print(
                                        message=f"[COMPILER] CUDA Execute: copied \"{name}\" to GPU\n"
                                                f"shape: {tensors[name].shape} \n"
                                                f"sample: {tensors[name][:2] if getattr(tensors[name], 'ndim', 0) > 1 else tensors[name]}"
                                    )

                        # Pre-allocate gradient tensors
                        for name, info in env.items():
                            if name.endswith("_grad"):
                                shape = tuple(int(dim) for dim in info["shape"])
                                np_dtype = dtype_numpy(info.get("dtype", "f32"))
                                size_bytes = int(np.prod(shape) * np_dtype.itemsize) if shape else np_dtype.itemsize
                                if size_bytes == 0:
                                    size_bytes = np_dtype.itemsize
                                gpu_allocs[name] = cuda.mem_alloc(size_bytes)
                                zeros = np.zeros(shape if shape else (1,), dtype=np_dtype)
                                cuda.memcpy_htod(gpu_allocs[name], zeros)

                        # Pre-allocate master weight tensors (FP32 copies of FP16 params)
                        for name, info in env.items():
                            if name.endswith("_master"):
                                shape = tuple(int(dim) for dim in info["shape"])
                                np_dtype = dtype_numpy("f32")
                                size_bytes = int(np.prod(shape) * np_dtype.itemsize) if shape else np_dtype.itemsize
                                if size_bytes == 0:
                                    size_bytes = np_dtype.itemsize
                                gpu_allocs[name] = cuda.mem_alloc(size_bytes)
                                zeros = np.zeros(shape if shape else (1,), dtype=np_dtype)
                                cuda.memcpy_htod(gpu_allocs[name], zeros)

                        # Initialize master weights from FP16 params (before forward pass)
                        for name, info in env.items():
                            if name.endswith("_master"):
                                param_name = name[:-7]
                                if param_name in tensors:
                                    param_arr = np.ascontiguousarray(tensors[param_name], dtype=dtype_numpy(env[param_name].get("dtype", "f16")))
                                    master_arr = param_arr.astype(np.float32)
                                    cuda.memcpy_htod(gpu_allocs[name], master_arr)
                                    tensors[name] = master_arr
                                    if self.debug_mode:
                                        self.tensorlang.print(message=f"[COMPILER] [Autograd] Initialized master weight '{name}' from '{param_name}'")

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
                                scale = self._current_scale if getattr(self, "_current_scale", None) is not None else 1.0
                                self._skip_optimizer = False
                                print(f"\n{'=' * 80}")
                                print(f"BACKWARD PASS from '{loss_name}' (loss_scale={scale})")
                                print("=" * 80)
                                try:
                                    self.comp_graph.backward(loss_name, scale=scale)
                                    for grad_name in self.comp_graph.requires_grad:
                                        if grad_name not in self.comp_graph.gradients:
                                            continue
                                        grad_tensor = self.comp_graph.gradients[grad_name]
                                        print(f"\nGradient {grad_name}.grad (scaled):\n{grad_tensor}")
                                        if self.cache_layers:
                                            grad_cache_path = cache_file_dir / f"{grad_name}.grad.npy"
                                            np.save(grad_cache_path, grad_tensor)
                                    print("=" * 80)
                                    # Overflow detection
                                    self._overflow_detected = self.comp_graph.overflow_detected

                                    if self._overflow_detected:
                                        # Reduce scale, skip optimizer step
                                        old_scale = self._current_scale
                                        self._current_scale = max(self._current_scale / 2.0, 1.0)
                                        self._scale_growth_counter = 0
                                        self._skip_optimizer = True
                                        print(f"[COMPILER] Overflow detected — scale {old_scale} -> {self._current_scale}, skipping optimizer step")
                                    else:
                                        self._skip_optimizer = False
                                        for grad_name in self.comp_graph.requires_grad:
                                            if grad_name in self.comp_graph.gradients:
                                                grad_tensor = self.comp_graph.gradients[grad_name]
                                                # Unscale
                                                if scale != 1.0 and scale != 0.0:
                                                    grad_tensor = grad_tensor / scale
                                                self.comp_graph.gradients[grad_name] = grad_tensor
                                                grad_tensor_name = f"{grad_name}_grad"
                                                tensors[grad_tensor_name] = grad_tensor
                                                if grad_tensor_name in gpu_allocs:
                                                    cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)
                                                    if self.debug_mode:
                                                        self.tensorlang.print(message=f"[COMPILER] [Autograd] Copied gradient {grad_tensor_name} to GPU")
                                                else:
                                                    if self.debug_mode:
                                                        self.tensorlang.print(message=f"[COMPILER] [Autograd] Warning: {grad_tensor_name} not in GPU allocations")
                                                # Use actual gradient dtype, not forward tensor dtype
                                                if grad_tensor.dtype == np.float16:
                                                    grad_dtype_name = "f16"
                                                elif grad_tensor.dtype == np.float64:
                                                    grad_dtype_name = "f64"
                                                else:
                                                    grad_dtype_name = "f32"
                                                env[grad_tensor_name] = {
                                                    "dtype": grad_dtype_name,
                                                    "shape": tuple(int(d) for d in grad_tensor.shape),
                                                }
                                                if self.debug_mode:
                                                    self.tensorlang.print(message=f"[COMPILER] [Autograd] Made gradient accessible: {grad_tensor_name}")
                                                if self.cache_layers:
                                                    grad_var_cache_path = cache_file_dir / f"{grad_tensor_name}.npy"
                                                    np.save(grad_var_cache_path, grad_tensor)

                                        # Dynamic scale growth
                                        if self.dynamic_loss_scale and not self._overflow_detected:
                                            self._scale_growth_counter += 1
                                            if self._scale_growth_counter >= self.loss_scale_patience:
                                                old_scale = self._current_scale
                                                self._current_scale *= 2.0
                                                self._scale_growth_counter = 0
                                                if self.debug_mode:
                                                    self.tensorlang.print(
                                                        message=f"[COMPILER] Scale grown {old_scale} -> {self._current_scale} after {self.loss_scale_patience} successful steps"
                                                    )

                                except (RuntimeError, ValueError, KeyError, OSError, TypeError) as e:
                                    self.tensorlang.print(message=f"[COMPILER] [Autograd] Error during backward pass: {e}")
                                    if self.debug_mode:
                                        traceback.print_exc()


                        if backward_index is not None:
                            for ast_idx in range(backward_index + 1, len(ast)):
                                # Skip optimizer steps on overflow
                                node = ast[ast_idx]
                                is_optimizer = (node.get("type") in OPTIMIZER_NAMES or
                                                (node.get("type") == "call" and node.get("name") in OPTIMIZER_NAMES))
                                if is_optimizer and getattr(self, "_skip_optimizer", False):
                                    if self.debug_mode:
                                        self.tensorlang.print(
                                            message=f"[COMPILER] Skipping optimizer step (overflow)"
                                        )
                                    continue

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

    # def can_broadcast(self, shape1, shape2):
        # if len(shape1) != len(shape2):
            # return False
        # for d1, d2 in zip(shape1, shape2):
            # if d1 != d2 and d1 != 1 and d2 != 1:
                # return False
        # return True

    def can_broadcast(self, shape1, shape2):
        ndim = max(len(shape1), len(shape2))
        s1 = (1,) * (ndim - len(shape1)) + tuple(int(d) for d in shape1)
        s2 = (1,) * (ndim - len(shape2)) + tuple(int(d) for d in shape2)
        for d1, d2 in zip(s1, s2):
            if d1 != d2 and d1 != 1 and d2 != 1:
                return False
        return True



    def _load_data_file(self, file_path, array_name=None):
        import numpy as np
        import pandas as pd
        from pathlib import Path

        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            data = np.load(file_path)
            return data, {"format": "npy"}

        elif suffix == ".npz":
            npz = np.load(file_path)
            if "data" in npz.files:
                data = npz["data"]
            elif len(npz.files) > 0:
                data = npz[npz.files[0]]
            else:
                npz.close()
                raise ValueError("NPZ file has no arrays")
            result = data
            npz.close()
            return result, {"format": "npz"}

        elif suffix == ".csv":
            df = pd.read_csv(file_path)
            first_col = df.columns[0]
            if first_col.lower() in ["date", "time", "datetime", "index", "id"]:
                df = df.drop(columns=[first_col])
            return df.to_numpy(dtype=np.float32), {"format": "csv", "columns": list(df.columns)}

        else:
            raise ValueError(f"Unsupported file format: {suffix}")
