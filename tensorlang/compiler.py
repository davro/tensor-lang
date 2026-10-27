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
from tensorlang.constants import OPTIMIZER_NAMES
from tensorlang.execution import ExecutionMixin
from tensorlang.fusion import fuse_elementwise_chains
from tensorlang.io import IOMixin
from tensorlang.kernel_generator import KernelGenerator
from tensorlang.tensor_lang import TensorLang
from tensorlang.type_checker import _is_scalar_shape, type_checker

OPTIMIZER_NAMES = {"sgd", "adam", "sgd_update", "adam_update"}


class TensorCompiler(CodegenMixin, ExecutionMixin, IOMixin):
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
