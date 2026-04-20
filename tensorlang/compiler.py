import sys
import os
import time
import errno
import textwrap  
import traceback 
import subprocess

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from pathlib import Path
import numpy as np
from lark import Lark, Tree, Token, UnexpectedInput
from datetime import datetime
from ctypes import c_void_p, c_int, c_float
from functools import reduce
from typing import Optional
from pathlib import Path

from tensorlang.tensor_lang import TensorLang
from tensorlang.kernel_generator import KernelGenerator
from tensorlang.type_checker import type_checker
from tensorlang.ast_builder import build_ast

from tensorlang.autograd import ComputationGraph, AutogradContext


class TensorCompiler:

    def __init__(self, debug_mode=False, debug_info=False, debug_ast=False, cache_layers=False, transpile=False):
        self.debug_mode   = debug_mode
        self.debug_info   = debug_info
        self.debug_ast    = debug_ast
        self.cache_layers = cache_layers
        self.transpile    = transpile
        self.tensorlang   = TensorLang()
        self.version      = self.tensorlang.version
        
        self.comp_graph = ComputationGraph(debug_mode=debug_mode)
        self.requires_grad_tensors = set()

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

    def _load_data_file(self, file_path: str, array_name: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        suffix = path.suffix.lower()
        metadata = {'format': suffix[1:], 'path': str(path)}

        if suffix == '.npy':
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Loading .npy file: {path.name}")
            data = np.load(path).astype(np.float32)
            metadata['shape'] = data.shape
            metadata['dtype'] = 'f32'
            return data, metadata

        elif suffix == '.npz':
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Loading .npz file: {path.name}")
            npz_file = np.load(path)
            available_arrays = list(npz_file.files)
            metadata['available_arrays'] = available_arrays
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER]   Available arrays: {available_arrays}")
            if array_name:
                if array_name not in available_arrays:
                    raise KeyError(f"Array '{array_name}' not found in {path.name}. Available: {available_arrays}")
                data = npz_file[array_name].astype(np.float32)
                metadata['array_name'] = array_name
            else:
                if 'data' in available_arrays:
                    data = npz_file['data'].astype(np.float32)
                    metadata['array_name'] = 'data'
                else:
                    data = npz_file[available_arrays[0]].astype(np.float32)
                    metadata['array_name'] = available_arrays[0]
                    if self.debug_mode:
                        self.tensorlang.print(
                            message=f"[COMPILER]   No 'data' array found, using '{available_arrays[0]}'"
                        )
            if 'columns' in available_arrays:
                metadata['columns'] = npz_file['columns'].tolist()
                if self.debug_mode:
                    self.tensorlang.print(message=f"[COMPILER]   Columns: {metadata['columns']}")
            if 'index' in available_arrays:
                index_data = npz_file['index']
                if index_data.dtype.kind == 'S':
                    metadata['index'] = [x.decode('utf-8') if isinstance(x, bytes) else x for x in index_data]
                else:
                    metadata['index'] = index_data.tolist()
                if self.debug_mode:
                    self.tensorlang.print(
                        message=f"[COMPILER]   Index: {len(metadata['index'])} entries "
                                f"({metadata['index'][0]} to {metadata['index'][-1]})"
                    )
            metadata['shape'] = data.shape
            metadata['dtype'] = 'f32'
            return data, metadata

        elif suffix == '.csv':
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Loading .csv file: {path.name}")
            try:
                df = pd.read_csv(path)
            except Exception as e:
                raise ValueError(f"Failed to parse CSV file {path.name}: {e}")
            metadata['columns'] = df.columns.tolist()
            metadata['rows'] = len(df)
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER]   Columns: {metadata['columns']}")
                self.tensorlang.print(message=f"[COMPILER]   Rows: {metadata['rows']}")
            first_col = df.columns[0]
            if first_col.lower() in ['date', 'time', 'datetime', 'index', 'id']:
                metadata['index'] = df[first_col].tolist()
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
            metadata['shape'] = data.shape
            metadata['dtype'] = 'f32'
            metadata['columns'] = df.columns.tolist()
            return data, metadata

        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .npy, .npz, .csv"
            )

    # =========================================================================
    # FOR LOOP EXECUTION
    # =========================================================================

    def _execute_for_loop(self, for_node, kernels, ast_to_kernel_map,
                          lib, gpu_allocs, env, tensors, cache_file_dir, cuda):
        """
        Execute the body of a for_statement N times.
        After each iteration apply rebind_statements (pointer swaps, no GPU copy).

        Key design:
          _flat_items stores (global_kernel_idx, body_node) where global_kernel_idx
          is a direct index into the `kernels` list — NOT a key into ast_to_kernel_map
          (which only covers top-level AST nodes). We execute body kernels via
          kernels[global_kernel_idx] directly.
        """
        iterations = for_node['iterations']
        body       = for_node['body']

        rebinds       = [n for n in body if n['type'] == 'rebind']
        exec_nodes    = [n for n in body if n['type'] != 'rebind']
        backward_node = next((n for n in exec_nodes if n['type'] == 'backward'), None)

        # Pre-compute the LIST POSITION of the backward node in _flat_items.
        # We use list position (not global_idx) because backward nodes have
        # global_idx=None (they generate no kernel). We split pre/post-backward
        # body kernels by whether their list position is before or after the
        # backward entry.
        bw_list_pos = None
        for pos, (gi, n) in enumerate(for_node['_flat_items']):
            if n['type'] == 'backward':
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

            # ------------------------------------------------------------------
            # 1. Execute pre-backward body kernels.
            #    global_idx is a direct index into kernels[] — not ast_to_kernel_map.
            # ------------------------------------------------------------------
            for pos, (global_idx, node) in enumerate(for_node['_flat_items']):
                if node['type'] in ('backward', 'save', 'rebind'):
                    continue
                if global_idx is None:
                    continue
                # If there's a backward in the body, only run nodes before it
                if bw_list_pos is not None and pos >= bw_list_pos:
                    continue
                self._execute_single_kernel(
                    kernels[global_idx],
                    lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                )

            # ------------------------------------------------------------------
            # 2. Backward pass (if present in loop body)
            # ------------------------------------------------------------------
            if backward_node is not None:
                loss_name = backward_node['loss_tensor']

                # Pull ALL body tensors from GPU into tensors dict.
                # _execute_single_kernel already copies results via _save_kernel_result,
                # but we do a full sweep here to ensure the comp graph sees current values
                # for ALL tensors (inputs like x, w, y_pred, loss) before backward().
                for global_idx, body_node in for_node['_flat_items']:
                    if body_node['type'] != 'let':
                        continue
                    bname = body_node['name']
                    if bname in gpu_allocs and bname in env:
                        bshape = tuple(int(d) for d in env[bname]['shape'])
                        barr   = np.empty(bshape, dtype=np.float32)
                        cuda.memcpy_dtoh(barr, gpu_allocs[bname])
                        tensors[bname] = barr

                # Also pull top-level tensors that the loop reads (x, y_true, w, lr)
                for tname in list(tensors.keys()):
                    if tname in gpu_allocs and tname in env:
                        tshape = tuple(int(d) for d in env[tname]['shape'])
                        tarr   = np.empty(tshape, dtype=np.float32)
                        cuda.memcpy_dtoh(tarr, gpu_allocs[tname])
                        tensors[tname] = tarr

                # Re-register all tensors in comp graph with current iteration values
                for tname, tval in tensors.items():
                    requires_grad = tname in self.requires_grad_tensors
                    self.comp_graph.register_tensor(tname, tval, requires_grad)

                # Reset gradients then run backward
                self.comp_graph.gradients = {}
                self.comp_graph.backward(loss_name)

                if self.debug_mode:
                    loss_val = tensors.get(loss_name)
                    if loss_val is not None:
                        self.tensorlang.print(
                            message=f"[COMPILER] for loop iter {i + 1}: "
                                    f"loss={float(loss_val.flat[0]):.6f}"
                        )

                # Copy computed gradients back to GPU
                for grad_name in self.comp_graph.requires_grad:
                    if grad_name in self.comp_graph.gradients:
                        grad_tensor      = self.comp_graph.gradients[grad_name]
                        grad_tensor_name = f"{grad_name}_grad"
                        tensors[grad_tensor_name] = grad_tensor

                        if grad_tensor_name in gpu_allocs:
                            cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)
                        else:
                            # First iteration — allocate gradient buffer on GPU
                            gpu_allocs[grad_tensor_name] = cuda.mem_alloc(grad_tensor.nbytes)
                            cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)

                        env[grad_tensor_name] = {
                            'dtype': 'f32',
                            'shape': grad_tensor.shape
                        }

                # Execute post-backward body kernels (e.g. grad_step, w_updated)
                if bw_list_pos is not None:
                    for pos, (global_idx, node) in enumerate(for_node['_flat_items']):
                        if node['type'] in ('backward', 'save', 'rebind'):
                            continue
                        if global_idx is None:
                            continue
                        if pos <= bw_list_pos:
                            continue
                        self._execute_single_kernel(
                            kernels[global_idx],
                            lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                        )

            # ------------------------------------------------------------------
            # 3. Apply rebinds — swap GPU buffer pointers atomically, no memcpy.
            #    Collect all pending swaps first to handle "a=b; b=a" correctly.
            # ------------------------------------------------------------------
            pending_gpu     = {}
            pending_tensors = {}

            for rebind in rebinds:
                target   = rebind['name']
                src_expr = rebind['expr']

                if src_expr['type'] != 'name':
                    self.tensorlang.print(
                        message=f"[COMPILER] for loop rebind: only NAME expressions supported, "
                                f"got {src_expr['type']} for '{target}'"
                    )
                    continue

                src_name = src_expr['name']

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

            # Apply swaps
            gpu_allocs.update(pending_gpu)
            tensors.update(pending_tensors)

            # Re-register after rebind so next iteration's backward sees updated w
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
        """Basic function inlining - replaces fn calls with their body."""
        # This is a stub. If build_ast already returns expanded AST, this can be empty.
        # For full inlining you would walk the AST and substitute calls.
        # For now, return as-is and let the existing kernel gen handle synthetic return nodes.
        return ast

    def _execute_single_kernel(self, kernel_info, lib, gpu_allocs, env, tensors, cache_file_dir, cuda):
        """Execute a single CUDA kernel and save results."""
        op_type = kernel_info[0]

        if op_type == 'general_broadcast':
            _, actual_op, name, arg1, arg2, padded_shape1, padded_shape2, output_shape_tuple, total_elements = kernel_info
            shape = tuple(int(dim) for dim in env[name]['shape'])
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Executing {op_type} ({actual_op}) for {name}, shape: {shape}")
            ndim = len(output_shape_tuple)
            shape1_gpu    = cuda.mem_alloc(ndim * np.int32().nbytes)
            shape2_gpu    = cuda.mem_alloc(ndim * np.int32().nbytes)
            out_shape_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            shape1_array    = np.array(padded_shape1,    dtype=np.int32)
            shape2_array    = np.array(padded_shape2,    dtype=np.int32)
            out_shape_array = np.array(output_shape_tuple, dtype=np.int32)
            cuda.memcpy_htod(shape1_gpu,    shape1_array)
            cuda.memcpy_htod(shape2_gpu,    shape2_array)
            cuda.memcpy_htod(out_shape_gpu, out_shape_array)
            getattr(lib, f'launch_{actual_op}_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[name])),
                c_void_p(int(shape1_gpu)),
                c_void_p(int(shape2_gpu)),
                c_void_p(int(out_shape_gpu)),
                c_int(ndim),
                c_int(total_elements)
            )
            self._save_kernel_result(name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        op_type, name, arg1, arg2, *dims = kernel_info
        shape = tuple(int(dim) for dim in env[name]['shape'])

        if op_type == 'matmul':
            m, n, p = dims
            getattr(lib, f'launch_matmul_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[name])),
                c_int(m), c_int(n), c_int(p)
            )

        elif op_type in ['add', 'minus', 'mult', 'div']:
            size = dims[0]
            getattr(lib, f'launch_{op_type}_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[name])),
                c_int(size)
            )

        elif op_type in ['add_scalar', 'minus_scalar', 'mult_scalar', 'div_scalar']:
            size = dims[0]
            op_base = op_type.split('_')[0]
            getattr(lib, f'launch_{op_base}_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[name])),
                c_int(size)
            )

        elif op_type in [
                'add_broadcast', 'minus_broadcast', 'mult_broadcast', 'div_broadcast',
                'greater_broadcast', 'less_broadcast', 'equal_broadcast'
            ]:
            op_name = op_type.split("_")[0]
            rows, cols = dims
            getattr(lib, f'launch_{op_name}_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols)
            )

        elif op_type in ['relu', 'sigmoid', 'tanh']:
            op_name = op_type.split("_")[0]
            size = dims[0]
            getattr(lib, f'launch_{op_name}_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(size)
            )

        elif op_type in ['greater', 'less', 'equal']:
            size = dims[0]
            getattr(lib, f'launch_{op_type}_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[name])),
                c_int(size)
            )

        elif op_type == 'softmax_1d':
            size = dims[0]
            getattr(lib, f'launch_softmax_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(size)
            )

        elif op_type == 'softmax':
            rows, cols, axis = dims
            getattr(lib, f'launch_softmax_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols)
            )

        elif op_type in ['sum_axis', 'sum_axis0']:
            rows, cols, axis = dims
            getattr(lib, f'launch_sum_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols)
            )

        elif op_type == 'slice_2d':
            rows, cols, row_start, row_end, col_start, col_end, out_rows, out_cols = dims
            getattr(lib, f'launch_slice_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols),
                c_int(row_start), c_int(row_end), c_int(col_start), c_int(col_end),
                c_int(out_rows), c_int(out_cols)
            )

        elif op_type == 'slice_1d':
            start, out_size = dims
            getattr(lib, f'launch_slice_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(start), c_int(out_size)
            )

        elif op_type == 'sum_full':
            size = dims[0]
            getattr(lib, f'launch_sum_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(size)
            )

        elif op_type == 'mean_full':
            size = dims[0]
            getattr(lib, f'launch_mean_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(size)
            )

        elif op_type in ['mean_axis', 'mean_axis0']:
            rows, cols, axis = dims
            getattr(lib, f'launch_mean_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols)
            )

        elif op_type in ['max_full', 'min_full']:
            size = dims[0]
            op_name = op_type.split('_')[0]
            getattr(lib, f'launch_{op_name}_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(size)
            )

        elif op_type in ['max_axis', 'max_axis0', 'min_axis', 'min_axis0']:
            rows, cols, axis = dims
            op_name = op_type.split('_')[0]
            getattr(lib, f'launch_{op_name}_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols)
            )

        elif op_type in ['argmax_full', 'argmax_axis']:
            if len(dims) == 1:
                size = dims[0]
                getattr(lib, f'launch_argmax_{name}')(
                    c_void_p(int(gpu_allocs[arg1])),
                    c_void_p(int(gpu_allocs[name])),
                    c_int(size)
                )
            else:
                rows, cols, axis = dims
                getattr(lib, f'launch_argmax_{name}')(
                    c_void_p(int(gpu_allocs[arg1])),
                    c_void_p(int(gpu_allocs[name])),
                    c_int(rows), c_int(cols)
                )

        elif op_type == 'argmin_axis':
            rows, cols, axis = dims
            getattr(lib, f'launch_argmin_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols)
            )

        elif op_type == 'fill':
            size, value = dims
            getattr(lib, f'launch_fill_{name}')(
                c_void_p(int(gpu_allocs[name])),
                c_float(value), c_int(size)
            )

        elif op_type == 'linear_1d':
            input_name, weight_name, bias_name, in_features, out_features = arg1, arg2, dims[0], dims[1], dims[2]
            getattr(lib, f'launch_linear_{name}')(
                c_void_p(int(gpu_allocs[input_name])),
                c_void_p(int(gpu_allocs[weight_name])),
                c_void_p(int(gpu_allocs[bias_name])),
                c_void_p(int(gpu_allocs[name])),
                c_int(in_features), c_int(out_features)
            )

        elif op_type == 'linear_2d':
            input_name, weight_name, bias_name, batch_size, in_features, out_features = arg1, arg2, dims[0], dims[1], dims[2], dims[3]
            getattr(lib, f'launch_linear_{name}')(
                c_void_p(int(gpu_allocs[input_name])),
                c_void_p(int(gpu_allocs[weight_name])),
                c_void_p(int(gpu_allocs[bias_name])),
                c_void_p(int(gpu_allocs[name])),
                c_int(batch_size), c_int(in_features), c_int(out_features)
            )

        elif op_type in ['layer_norm_2d', 'layer_norm_axis0']:
            rows, cols, eps = dims
            getattr(lib, f'launch_layer_norm_{name}')(
                c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols), c_float(eps)
            )

        elif op_type == 'layer_norm_1d':
            size, eps = dims
            getattr(lib, f'launch_layer_norm_{name}')(
                c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])),
                c_int(size), c_float(eps)
            )

        elif op_type == 'cross_entropy':
            batch_size, num_classes = dims
            getattr(lib, f'launch_cross_entropy_{name}')(
                c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[name])),
                c_int(batch_size), c_int(num_classes)
            )

        elif op_type == 'mse_loss':
            total_elements = dims[0]
            getattr(lib, f'launch_mse_loss_{name}')(
                c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[name])),
                c_int(total_elements)
            )

        elif op_type == 'transpose_2d':
            rows, cols = dims
            getattr(lib, f'launch_transpose_{name}')(
                c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols)
            )

        elif op_type == 'reshape':
            total_elements = dims[0]
            getattr(lib, f'launch_reshape_{name}')(
                c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])),
                c_int(total_elements)
            )

        elif op_type == 'concat_axis0':
            rows1, rows2, cols = dims
            getattr(lib, f'launch_concat_{name}')(
                c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])),
                c_int(rows1), c_int(rows2), c_int(cols)
            )

        elif op_type == 'batch_norm_2d':
            running_mean_name, batch_size, num_features, eps, running_var_name = arg2, dims[0], dims[1], dims[2], dims[3]
            getattr(lib, f'launch_batch_norm_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[running_mean_name])),
                c_void_p(int(gpu_allocs[running_var_name])),
                c_void_p(int(gpu_allocs[name])),
                c_int(batch_size), c_int(num_features), c_float(eps)
            )

        elif op_type == 'instance_norm_2d':
            batch_size, num_features, eps = dims[-3], dims[-2], dims[-1]
            getattr(lib, f'launch_instance_norm_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[name])),
                c_int(batch_size), c_int(num_features), c_float(eps)
            )

        elif op_type in ['minus_broadcast_rows', 'add_broadcast_rows']:
            rows, cols = dims
            op_name = op_type.split('_')[0]
            getattr(lib, f'launch_{op_name}_{name}')(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[name])),
                c_int(rows), c_int(cols)
            )

        self._save_kernel_result(name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)

    def _save_kernel_result(self, name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda):
        """Save kernel result from GPU to CPU."""
        if name in env:
            shape = tuple(int(dim) for dim in env[name]['shape'])
            output = np.zeros(shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, gpu_allocs[name])
            tensors[name] = output
            if self.cache_layers:
                cache_npy_path = cache_file_dir / f"{name}.npy"
                np.save(cache_npy_path, output)
            self.tensorlang.print(type=f"[COMPILER] Result {name} ({op_type}):\n{output}")

    def _execute_save_statement(self, save_node, tensors, gpu_allocs, env, cache_file_dir, cuda):
        tensor_name = save_node['tensor']
        file_path   = save_node['file_path']
        if self.debug_mode:
            self.tensorlang.print(message=f"[COMPILER] Executing save: {tensor_name} -> {file_path}")
        if tensor_name in tensors:
            tensor_data = tensors[tensor_name]
        elif tensor_name in gpu_allocs and tensor_name in env:
            shape = tuple(int(dim) for dim in env[tensor_name]['shape'])
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
        except IOError as e:
            self.tensorlang.print(message=f"[COMPILER] Error saving {tensor_name} to {file_path}: {e}")

    # =========================================================================
    # KERNEL GENERATION HELPER — used by for loop body compilation
    # =========================================================================

    def _generate_kernels_for_nodes(self, nodes, env, generator):
        """
        Generate CUDA kernels for a list of AST nodes (let_bindings only).
        Returns (kernels_list, cuda_code_string, flat_items).
        flat_items is a list of (flat_idx, node) so the executor can map
        kernel indices back to body nodes.
        """
        kernels   = []
        cuda_code = ""
        flat_items = []   # (flat_idx, node) — index into the returned kernels list

        for node in nodes:
            if node['type'] in ('backward', 'save', 'rebind', 'for'):
                flat_items.append((None, node))
                continue

            if node['type'] != 'let':
                flat_items.append((None, node))
                continue

            name = node['name']
            expr = node.get('expr')

            if not isinstance(expr, dict):
                flat_items.append((None, node))
                continue

            if expr['type'] in ('name', 'tensor_literal'):
                flat_items.append((None, node))
                continue

            kernel_str, kernel_info = self._kernel_for_expr(name, expr, env, generator)
            if kernel_str is not None:
                flat_idx = len(kernels)
                kernels.append(kernel_info)
                cuda_code += kernel_str
                flat_items.append((flat_idx, node))
            else:
                flat_items.append((None, node))

        return kernels, cuda_code, flat_items

    def _kernel_for_expr(self, name, expr, env, generator):
        """
        Generate (cuda_code_str, kernel_info) for a single expression.
        Returns (None, None) if not applicable.
        """
        if expr['type'] in ['add', 'minus', 'mult', 'div']:
            arg1, arg2     = expr['args']
            shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
            output_shape   = env[name]['shape']
            self._record_operation(expr['type'], name, [arg1, arg2])
            if shape1 == shape2:
                size = int(np.prod([int(d) for d in output_shape]))
                return generator.elementwise(expr['type'], name, arg1, arg2, size)
            elif len(shape1) == 2 and len(shape2) == 1:
                return generator.binary_broadcast(
                    expr['type'], expr['type'], name, arg1, arg2, shape1, shape2, output_shape
                )
            elif len(shape1) == 1 and len(shape2) == 1 and (shape1[0] == 1 or shape2[0] == 1):
                size = int(np.prod([int(d) for d in output_shape]))
                return generator.binary_1d_broadcast(expr['type'], name, arg1, arg2, size)
            elif self.can_broadcast(shape1, shape2):
                return generator.binary_general_broadcast(
                    expr['type'], name, arg1, arg2, shape1, shape2, output_shape
                )

        elif expr['type'] == 'matmul':
            arg1, arg2 = expr['args']
            a_shape = env[arg1]['shape']
            b_shape = env[arg2]['shape']
            if len(a_shape) == 2 and len(b_shape) == 2:
                m, n, p = a_shape[0], a_shape[1], b_shape[1]
            elif len(a_shape) == 2 and len(b_shape) == 1:
                m, n, p = a_shape[0], a_shape[1], b_shape[0]
            else:
                raise TypeError(f"Unsupported matmul shapes: {a_shape} @ {b_shape}")
            self._record_operation('matmul', name, [arg1, arg2])
            return generator.matmul(expr['type'], name, arg1, arg2, int(m), int(n), int(p))

        elif expr['type'] in ['relu', 'sigmoid', 'tanh']:
            arg1 = expr['args'][0]
            size = int(np.prod([int(d) for d in env[arg1]['shape']]))
            self._record_operation(expr['type'], name, [arg1])
            method = getattr(generator, expr['type'], None)
            if method:
                return method(expr['type'], name, arg1, size)

        elif expr['type'] == 'softmax':
            tensor_name = expr['tensor']
            axis        = expr.get('axis')
            input_shape = env[tensor_name]['shape']
            self._record_operation('softmax', name, [tensor_name], metadata={'axis': axis})
            return generator.softmax(expr['type'], name, tensor_name, input_shape, axis)

        elif expr['type'] == 'mse_loss':
            pred_name, target_name = expr['args']
            pred_shape     = env[pred_name]['shape']
            total_elements = int(np.prod([int(d) for d in pred_shape]))
            self._record_operation('mse_loss', name, [pred_name, target_name])
            return generator.mse_loss(expr['type'], name, pred_name, target_name, total_elements)

        elif expr['type'] == 'linear':
            input_name, weight_name, bias_name = expr['args']
            input_shape  = env[input_name]['shape']
            weight_shape = env[weight_name]['shape']
            output_shape = env[name]['shape']
            return generator.linear(
                expr['type'], name, input_name, weight_name, bias_name,
                input_shape, weight_shape, output_shape
            )

        elif expr['type'] == 'sum':
            tensor_name = expr['tensor']
            axis        = expr.get('axis')
            input_shape = env[tensor_name]['shape']
            self._record_operation('sum', name, [tensor_name], metadata={'axis': axis})
            return generator.sum(expr['type'], name, tensor_name, axis, input_shape)

        elif expr['type'] == 'mean':
            tensor_name = expr['tensor']
            axis        = expr.get('axis')
            input_shape = env[tensor_name]['shape']
            self._record_operation('mean', name, [tensor_name], metadata={'axis': axis})
            return generator.mean(expr['type'], name, tensor_name, axis, input_shape)

        return None, None

    # =========================================================================
    # COMPILE AND EXECUTE
    # =========================================================================

    # WORKING WITHOUT GPU → CPU copy for all tensors after execution
    def compile_and_execute(self, tensorlang_file):
        """Compile and execute a single .tl file."""

        self.tensorlang.print_header(f"[COMPILER] TensorLang {self.version}")

        if tensorlang_file:
            file_path = Path(tensorlang_file)
        else:
            self.tensorlang.print(message=f"[COMPILER] Error missing file")
            return

        if not file_path.exists():
            self.tensorlang.print(message=f"[COMPILER] Error: {tensorlang_file} not found at {file_path}")
            sys.exit(1)

        if file_path.suffix == '.tl':
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
            self.tensorlang.print(message=f"[COMPILER] file not found, suffix is not .tl")
            self.tensorlang.separator()
            sys.exit(1)

        try:
            self.tensorlang.separator()
            grammar_file = 'tensorlang.lark'
            with open(grammar_file, 'r') as f:
                grammar = f.read()
            parser = Lark(grammar, start='program', parser='lalr')
            self.tensorlang.print(type="[COMPILER]", message=f"Loaded Lark Grammer file: {grammar_file}")
        except FileNotFoundError:
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Missing Lark Grammer file: {grammar_file}")
            sys.exit(1)

        try:
            with open(file_path, 'r') as f:
                code = f.read()
                self.tensorlang.print(type="[COMPILER]", message=f"Loaded TensorLang   file: {file_path}")
        except FileNotFoundError:
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Missing TensorLang file: {file_path}")
            sys.exit(1)

        try:
            self.tensorlang.separator()
            self.tensorlang.print(type=f"")
            self.tensorlang.print(type=f"{code}")
            self.tensorlang.separator()
            self.tensorlang.print(type=f"")
            self.tensorlang.separator()
            self.tensorlang.print(type=f"[COMPILER] TensorLang > Lark > Parser > Compiler > CUDA Kernel -> Results")
            self.tensorlang.separator()

            cache_base     = Path("cache")
            cache_file_dir = cache_base / file_path
            cache_file_dir.mkdir(parents=True, exist_ok=True)

            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] CACHE: {cache_base}")
                self.tensorlang.print(message=f"[COMPILER] CACHE FILE_PATH: {file_path}")
                self.tensorlang.print(message=f"[COMPILER] CACHE FILE_PATH STEM: {file_path.stem}")
                self.tensorlang.print(message=f"[COMPILER] CACHE TENSORLANG_FILE: {tensorlang_file}")

            if self.debug_info:
                self.tensorlang.print(type="[INFO]", message=f"Created cache for tensor outputs directory")

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
                tensors    = {}
                gpu_allocs = {}

                # ============================================================
                # Pre-register gradient tensor shapes in env.
                # w_grad doesn't exist at type-check time — it only appears
                # after backward() runs. But the for loop body compiler
                # (_generate_kernels_for_nodes / _kernel_for_expr) needs to
                # look up shapes in env at compile time. We pre-populate env
                # now: for every tensor tagged 'with grad', its gradient has
                # the same shape, and will be accessible as '{name}_grad'.
                # ============================================================
                for node in ast:
                    if node['type'] == 'let' and node.get('requires_grad', False):
                        grad_name = f"{node['name']}_grad"
                        source_shape = env[node['name']]['shape']
                        env[grad_name] = {'dtype': 'f32', 'shape': source_shape}
                        if self.debug_mode:
                            self.tensorlang.print(
                                message=f"[COMPILER] Pre-registered gradient shape: "
                                        f"{grad_name} -> {source_shape}"
                            )
                    # Also walk for loop bodies — a tensor declared with grad
                    # inside a for loop would need the same treatment.
                    if node['type'] == 'for':
                        for body_node in node['body']:
                            if body_node['type'] == 'let' and body_node.get('requires_grad', False):
                                grad_name = f"{body_node['name']}_grad"
                                if body_node['name'] in env:
                                    source_shape = env[body_node['name']]['shape']
                                    env[grad_name] = {'dtype': 'f32', 'shape': source_shape}


                generator = KernelGenerator(self.debug_mode)
                kernels   = []
                cuda_code = generator.cuda_header()

                # ============================================================
                # KERNEL GENERATION LOOP
                # For top-level nodes: generate kernels as before.
                # For for_statement nodes: compile the body once via helper,
                #   storing flat_items on the node for use during execution.
                # ============================================================
                for node in ast:

                    if node['type'] == 'save':
                        tensor_name = node['tensor']
                        if tensor_name not in env:
                            self.tensorlang.print(message=f"[COMPILER] Error: Cannot save undefined tensor '{tensor_name}'")
                            return False, env
                        if self.debug_mode:
                            self.tensorlang.print(message=f"[COMPILER] Queued save operation: {tensor_name} -> {node['file_path']}")
                        continue

                    if node['type'] == 'backward':
                        continue

                    if node['type'] == 'rebind':
                        continue

                    # --------------------------------------------------------
                    # FOR LOOP — compile body once, tag each body node with
                    # its kernel index so _execute_for_loop can find it.
                    # --------------------------------------------------------
                    if node['type'] == 'for':
                        body_kernels, body_cuda, flat_items = self._generate_kernels_for_nodes(
                            node['body'], env, generator
                        )
                        # Remap flat_items indices to the global kernels list
                        global_flat_items = []
                        for local_idx, body_node in flat_items:
                            if local_idx is not None:
                                global_idx = len(kernels) + local_idx
                                global_flat_items.append((global_idx, body_node))
                            else:
                                global_flat_items.append((None, body_node))
                        kernels.extend(body_kernels)
                        cuda_code += body_cuda
                        # Store on the node for use during execution
                        node['_flat_items'] = global_flat_items
                        continue

                    if node['type'] == 'let' and node.get('requires_grad', False):
                        name = node['name']
                        self.requires_grad_tensors.add(name)
                        if self.debug_mode:
                            self.tensorlang.print(message=f"[COMPILER] [Autograd] Tensor '{name}' marked for gradient tracking")

                    if node['type'] == 'let' and isinstance(node['expr'], dict):
                        name = node['name']
                        expr = node['expr']

                        if expr['type'] == 'name':
                            continue

                        if self.debug_mode:
                            self.tensorlang.print(message=f"[COMPILER] Processing {name} ({expr['type']})")

                        # ====================================================
                        # Tensor Literal
                        # ====================================================
                        if expr['type'] == 'tensor_literal':
                            shape = tuple(int(dim) for dim in env[name]['shape'])
                            tensors[name] = np.array(expr['data'], dtype=np.float32).reshape(shape)
                            if self.debug_info:
                                self.tensorlang.print(type="[INFO]", message=f"Kernel Tensor Initialized {name} with shape {shape}")

                        # ====================================================
                        # LOAD
                        # ====================================================
                        elif expr['type'] == 'load':
                            file_path_load = expr['file_path']
                            array_name     = expr.get('array_name')
                            try:
                                loaded_data, metadata = self._load_data_file(file_path_load, array_name)
                                if self.debug_mode:
                                    self.tensorlang.print(
                                        message=f"[COMPILER] Loaded {metadata['format'].upper()}: "
                                                f"shape={loaded_data.shape}, "
                                                f"size={loaded_data.size:,} elements"
                                    )
                                if name in env:
                                    expected_shape = tuple(int(dim) for dim in env[name]['shape'])
                                    if loaded_data.shape != expected_shape:
                                        self.tensorlang.print(
                                            message=f"[COMPILER] Shape mismatch: {file_path_load}\n"
                                                    f"  File shape:     {loaded_data.shape}\n"
                                                    f"  Declared shape: {expected_shape}"
                                        )
                                        return False, env
                                else:
                                    env[name] = {
                                        'dtype': 'f32',
                                        'shape': loaded_data.shape,
                                        'metadata': metadata
                                    }
                                tensors[name] = loaded_data
                                if self.cache_layers:
                                    cache_npy_path = cache_file_dir / f"{name}.npy"
                                    np.save(cache_npy_path, loaded_data)
                                if self.debug_info:
                                    if 'columns' in metadata:
                                        self.tensorlang.print(type="[INFO]", message=f"Columns: {metadata['columns']}")
                                    preview = loaded_data.flatten()[:5]
                                    self.tensorlang.print(type="[INFO]", message=f"First values: {preview}")
                            except FileNotFoundError as e:
                                self.tensorlang.print(message=f"[COMPILER] Error: {e}")
                                return False, env
                            except Exception as e:
                                self.tensorlang.print(message=f"[COMPILER] Error loading {file_path_load}: {e}")
                                if self.debug_mode:
                                    traceback.print_exc()
                                return False, env

                        # ====================================================
                        # ADD / MINUS / MULT / DIV
                        # ====================================================
                        elif expr['type'] in ['add', 'minus', 'mult', 'div']:
                            arg1, arg2     = expr['args']
                            shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                            output_shape   = env[name]['shape']
                            self._record_operation(expr['type'], name, [arg1, arg2])
                            if self.debug_mode:
                                self.tensorlang.print(message=f"Tensor {expr['type']}")
                                self.tensorlang.print(message=f"Tensor Shape1: {len(shape1)} Shape2: {len(shape2)}")
                            if shape1 == shape2:
                                size = int(np.prod([int(dim) for dim in output_shape]))
                                kernel, kernel_info = generator.elementwise(expr['type'], name, arg1, arg2, size)
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif len(shape1) == 2 and len(shape2) == 1:
                                kernel, kernel_info = generator.binary_broadcast(
                                    expr['type'], expr['type'], name, arg1, arg2, shape1, shape2, output_shape
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif len(shape1) == 1 and len(shape2) == 1 and (shape1[0] == 1 or shape2[0] == 1):
                                size = int(np.prod([int(dim) for dim in output_shape]))
                                kernel, kernel_info = generator.binary_1d_broadcast(expr['type'], name, arg1, arg2, size)
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            elif self.can_broadcast(shape1, shape2):
                                kernel, kernel_info = generator.binary_general_broadcast(
                                    expr['type'], name, arg1, arg2, shape1, shape2, output_shape
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            else:
                                self.tensorlang.print(
                                    message=f"[COMPILER] Error: Cannot {expr['type']} tensors with incompatible shapes {shape1} and {shape2}."
                                )
                                return False, env

                        # ====================================================
                        # FILL
                        # ====================================================
                        elif expr['type'] == 'fill':
                            size = int(np.prod([int(dim) for dim in expr['shape']]))
                            kernel, kernel_info = generator.fill(expr['type'], name, None, None, size, expr['value'])
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ====================================================
                        # MATMUL
                        # ====================================================
                        elif expr['type'] == 'matmul':
                            arg1, arg2 = expr['args']
                            a_shape = env[arg1]['shape']
                            b_shape = env[arg2]['shape']
                            if len(a_shape) == 2 and len(b_shape) == 2:
                                m, n, p = a_shape[0], a_shape[1], b_shape[1]
                            elif len(a_shape) == 2 and len(b_shape) == 1:
                                m, n, p = a_shape[0], a_shape[1], b_shape[0]
                            else:
                                raise TypeError(f"Unsupported matmul shapes: {a_shape} @ {b_shape}")
                            kernel, kernel_info = generator.matmul(
                                expr['type'], name, arg1, arg2, int(m), int(n), int(p)
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
                            self._record_operation('matmul', name, [arg1, arg2])

                        # ====================================================
                        # ReLU | SIGMOID | TANH
                        # ====================================================
                        elif expr['type'] in ['relu', 'sigmoid', 'tanh']:
                            arg1 = expr['args'][0]
                            size = int(np.prod([int(dim) for dim in env[arg1]['shape']]))
                            method = getattr(generator, f"{expr['type']}", None)
                            if method is None:
                                raise ValueError(f"No method found for operation: {expr['type']}")
                            kernel, kernel_info = method(expr['type'], name, arg1, size)
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation(expr['type'], name, [arg1])

                        # ====================================================
                        # SOFTMAX
                        # ====================================================
                        elif expr['type'] == 'softmax':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            kernel, kernel_info = generator.softmax(expr['type'], name, tensor_name, input_shape, axis)
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation('softmax', name, [tensor_name], metadata={'axis': axis})

                        # ====================================================
                        # GREATER / LESS / EQUAL
                        # ====================================================
                        elif expr['type'] == 'greater':
                            arg1, arg2   = expr['args']
                            shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                            output_shape = env[name]['shape']
                            size         = int(np.prod([int(dim) for dim in output_shape]))
                            kernel, kernel_info = generator.greater_broadcast(
                                expr['type'], name, arg1, arg2, shape1, shape2, output_shape, size
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr['type'] == 'less':
                            arg1, arg2   = expr['args']
                            shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                            output_shape = env[name]['shape']
                            kernel, kernel_info = generator.less_broadcast(
                                expr['type'], name, arg1, arg2, shape1, shape2, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr['type'] == 'equal':
                            arg1, arg2   = expr['args']
                            shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                            output_shape = env[name]['shape']
                            kernel, kernel_info = generator.equal(
                                expr['type'], name, arg1, arg2, shape1, shape2, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ====================================================
                        # SUM
                        # ====================================================
                        elif expr['type'] == 'sum':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            kernel, kernel_info = generator.sum(expr['type'], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation('sum', name, [tensor_name], metadata={'axis': axis})

                        # ====================================================
                        # MEAN
                        # ====================================================
                        elif expr['type'] == 'mean':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            kernel, kernel_info = generator.mean(expr['type'], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation('mean', name, [tensor_name], metadata={'axis': axis})

                        # ====================================================
                        # SLICE
                        # ====================================================
                        elif expr['type'] == 'slice':
                            tensor_name  = expr['tensor']
                            slice_specs  = expr['specs']
                            input_shape  = env[tensor_name]['shape']
                            output_shape = env[name]['shape']
                            kernel, kernel_info = generator.slice(
                                expr['type'], name, tensor_name, slice_specs, input_shape, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ====================================================
                        # MAX / MIN
                        # ====================================================
                        elif expr['type'] == 'max':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            kernel, kernel_info = generator.max(expr['type'], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr['type'] == 'min':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            kernel, kernel_info = generator.min(expr['type'], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ====================================================
                        # ARGMAX / ARGMIN
                        # ====================================================
                        elif expr['type'] == 'argmax':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            kernel, kernel_info = generator.argmax(expr['type'], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr['type'] == 'argmin':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            kernel, kernel_info = generator.argmin(expr['type'], name, tensor_name, axis, input_shape)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ====================================================
                        # LINEAR LAYER
                        # ====================================================
                        elif expr['type'] == 'linear':
                            input_name, weight_name, bias_name = expr['args']
                            input_shape  = env[input_name]['shape']
                            weight_shape = env[weight_name]['shape']
                            output_shape = env[name]['shape']
                            kernel, kernel_info = generator.linear(
                                expr['type'], name, input_name, weight_name, bias_name,
                                input_shape, weight_shape, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ====================================================
                        # LAYER NORM
                        # ====================================================
                        elif expr['type'] == 'layer_norm':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            eps         = expr.get('eps', 1e-5)
                            input_shape = env[tensor_name]['shape']
                            kernel, kernel_info = generator.layer_norm(
                                expr['type'], name, tensor_name, axis, eps, input_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ====================================================
                        # CROSS ENTROPY
                        # ====================================================
                        elif expr['type'] == 'cross_entropy':
                            pred_name, target_name = expr['args']
                            pred_shape   = env[pred_name]['shape']
                            target_shape = env[target_name]['shape']
                            kernel, kernel_info = generator.cross_entropy(
                                expr['type'], name, pred_name, pred_shape, target_shape, target_name
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ====================================================
                        # MSE LOSS
                        # ====================================================
                        elif expr['type'] == 'mse_loss':
                            pred_name, target_name = expr['args']
                            pred_shape     = env[pred_name]['shape']
                            total_elements = int(np.prod([int(dim) for dim in pred_shape]))
                            kernel, kernel_info = generator.mse_loss(
                                expr['type'], name, pred_name, target_name, total_elements
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            self._record_operation('mse_loss', name, [pred_name, target_name])

                        # ====================================================
                        # TRANSPOSE / RESHAPE / CONCAT
                        # ====================================================
                        elif expr['type'] == 'transpose':
                            tensor_name  = expr['tensor']
                            input_shape  = env[tensor_name]['shape']
                            kernel, kernel_info = generator.transpose(expr['type'], name, input_shape, tensor_name)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr['type'] == 'reshape':
                            tensor_name    = expr['tensor']
                            input_shape    = env[tensor_name]['shape']
                            total_elements = int(np.prod([int(dim) for dim in input_shape]))
                            kernel, kernel_info = generator.reshape(expr['type'], name, tensor_name, total_elements)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr['type'] == 'concat':
                            tensor_names = expr['tensors']
                            axis         = expr['axis']
                            kernel, kernel_info = generator.concat(expr['type'], name, tensor_names, axis, env)
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ====================================================
                        # BATCH NORM / INSTANCE NORM
                        # ====================================================
                        elif expr['type'] == 'batch_norm':
                            tensor_name       = expr['tensor']
                            running_mean_name = expr['running_mean']
                            running_var_name  = expr['running_var']
                            eps               = expr.get('eps', 1e-5)
                            input_shape       = env[tensor_name]['shape']
                            kernel, kernel_info = generator.batch_norm(
                                expr['type'], name, input_shape, tensor_name, running_mean_name, eps, running_var_name
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        elif expr['type'] == 'instance_norm':
                            tensor_name = expr['tensor']
                            eps         = expr.get('eps', 1e-5)
                            input_shape = env[tensor_name]['shape']
                            kernel, kernel_info = generator.instance_norm(
                                expr['type'], name, tensor_name, input_shape, eps
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                # ============================================================
                # KERNEL COMPILATION
                # ============================================================
                if kernels:
                    self.tensorlang.print(message=f"[COMPILER] KERNEL CUDA!")

                    kernel_cu_path = cache_file_dir / "kernel.cu"
                    kernel_so_path = cache_file_dir / "kernel.so"

                    with open(kernel_cu_path, 'w') as f:
                        f.write(cuda_code)
                        if self.debug_mode:
                            self.tensorlang.print(message=f"[COMPILER] CUDA Compile: {kernel_cu_path} written!")

                    try:
                        subprocess.run([
                            'nvcc', '-o', str(kernel_so_path),
                            '--shared', '-Xcompiler', '-fPIC', '-lcudart',
                            str(kernel_cu_path)
                        ], check=True)
                        if self.debug_mode:
                            self.tensorlang.print(message=f"[COMPILER] CUDA Compile: {kernel_so_path} compiled!")
                    except subprocess.CalledProcessError as e:
                        self.tensorlang.print(message=f"[COMPILER] CUDA Compile: error {e}")
                        sys.exit(1)

                    if self.transpile:
                        try:
                            import json
                            from tensorlang.transpiler.wgsl import WGSLTranspiler
                            if self.debug_mode:
                                self.tensorlang.print(message=f"[TRANSPILE] Reading {kernel_cu_path} for WGSL...")
                            transpiler = WGSLTranspiler(wgsl_workgroup_size=64)
                            wgsl = transpiler.transpile(cuda_code)
                            transpiler.save(wgsl, cache_file_dir)
                            if self.debug_mode:
                                self.tensorlang.print(f"[WGSL] Saved to {cache_file_dir}")
                        except Exception as e:
                            self.tensorlang.print(message=f"[TRANSPILE] ERROR: {e}")
                            traceback.print_exc()
                            sys.exit(1)

                    # =========================================================
                    # EXECUTION
                    # =========================================================
                    try:
                        import pycuda.driver as cuda
                        import pycuda.autoinit
                        from ctypes import cdll

                        lib = cdll.LoadLibrary(str(kernel_so_path))

                        # =====================================================
                        # PHASE 0: Find top-level backward index
                        # =====================================================
                        backward_index = None
                        for i, node in enumerate(ast):
                            if node['type'] == 'backward':
                                backward_index = i
                                if self.debug_mode:
                                    self.tensorlang.print(message=f"[COMPILER] CUDA Execute: found backward() at AST index {i}")
                                break

                        # =====================================================
                        # PHASE 1: Allocate ALL GPU memory upfront
                        # =====================================================
                        for name in env:
                            shape      = tuple(int(dim) for dim in env[name]['shape'])
                            size_bytes = int(np.prod(shape) * np.float32().nbytes)
                            gpu_allocs[name] = cuda.mem_alloc(size_bytes)
                            if name in tensors:
                                cuda.memcpy_htod(gpu_allocs[name], tensors[name])
                                if self.debug_mode:
                                    self.tensorlang.print(
                                        message=f"[COMPILER] CUDA Execute: copied \"{name}\" to GPU\n"
                                                f"shape: {tensors[name].shape} \n"
                                                f"sample: {tensors[name][:2] if tensors[name].ndim > 1 else tensors[name]}"
                                    )

                        # =====================================================
                        # PHASE 2: Build kernel map and execute top-level kernels
                        # =====================================================
                        # Build mapping: AST index -> kernel index
                        # (for top-level nodes only — for loop bodies are
                        #  tracked via _flat_items on each for_node)
                        ast_to_kernel_map = {}
                        kernel_idx        = 0

                        for ast_idx, node in enumerate(ast):
                            if node['type'] == 'for':
                                # Count kernels consumed by this loop body
                                kernel_idx += len([
                                    fi for fi, _ in node.get('_flat_items', [])
                                    if fi is not None
                                ])
                            elif node['type'] == 'let' and isinstance(node.get('expr'), dict):
                                expr = node['expr']
                                if expr['type'] not in ('name', 'tensor_literal', 'load'):
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
                                            kernels[k_idx], lib, gpu_allocs, env,
                                            tensors, cache_file_dir, cuda
                                        )
                                elif ast[ast_idx]['type'] == 'for':
                                    self._execute_for_loop(
                                        ast[ast_idx], kernels, ast_to_kernel_map,
                                        lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                                    )
                        else:
                            # No top-level backward — execute all top-level kernels
                            # and run any for loops
                            for ast_idx, node in enumerate(ast):
                                if node['type'] == 'for':
                                    self._execute_for_loop(
                                        node, kernels, ast_to_kernel_map,
                                        lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                                    )
                                elif ast_idx in ast_to_kernel_map:
                                    k_idx = ast_to_kernel_map[ast_idx]
                                    if k_idx < len(kernels):
                                        self._execute_single_kernel(
                                            kernels[k_idx], lib, gpu_allocs, env,
                                            tensors, cache_file_dir, cuda
                                        )

                        # =====================================================
                        # Handle alias assignments
                        # =====================================================
                        for node in ast:
                            if node['type'] == 'let' and isinstance(node.get('expr'), dict) and node['expr']['type'] == 'name':
                                alias_name  = node['name']
                                source_name = node['expr']['name']
                                if source_name in tensors and alias_name not in tensors:
                                    tensors[alias_name] = tensors[source_name].copy()
                                    if self.debug_mode:
                                        self.tensorlang.print(message=f"[COMPILER] CUDA Execute: created alias: {alias_name} -> {source_name}")

                        # =====================================================
                        # Cache tensor literals
                        # =====================================================
                        if self.cache_layers:
                            for name in tensors:
                                cache_npy_path = cache_file_dir / f"{name}.npy"
                                if not cache_npy_path.exists():
                                    np.save(cache_npy_path, tensors[name])
                                    if self.debug_mode:
                                        self.tensorlang.print(message=f"[COMPILER] CUDA Execute: tensor literal: {name}")

                        # =====================================================
                        # Register tensors in autograd graph
                        # =====================================================
                        for name in tensors:
                            requires_grad = name in self.requires_grad_tensors
                            self.comp_graph.register_tensor(name, tensors[name], requires_grad)
                            if self.debug_mode and requires_grad:
                                self.tensorlang.print(message=f"[COMPILER] [Autograd] CUDA Execute: registered '{name}' with gradient tracking")

                        # =====================================================
                        # PHASE 3: Top-level backward()
                        # =====================================================
                        for node in ast:
                            if node['type'] == 'backward':
                                loss_name = node['loss_tensor']
                                print(f"\n{'='*80}")
                                print(f"BACKWARD PASS from '{loss_name}'")
                                print('='*80)
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
                                    print('='*80)
                                    for grad_name in self.comp_graph.requires_grad:
                                        if grad_name in self.comp_graph.gradients:
                                            grad_tensor      = self.comp_graph.gradients[grad_name]
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
                                                'dtype': 'f32',
                                                'shape': grad_tensor.shape
                                            }
                                            if self.debug_mode:
                                                self.tensorlang.print(message=f"[COMPILER] [Autograd] Made gradient accessible: {grad_tensor_name}")
                                except Exception as e:
                                    self.tensorlang.print(message=f"[COMPILER] [Autograd] Error during backward pass: {e}")
                                    if self.debug_mode:
                                        traceback.print_exc()

                        # =====================================================
                        # PHASE 4: Top-level post-backward kernels
                        # =====================================================
                        if backward_index is not None:
                            for ast_idx in range(backward_index + 1, len(ast)):
                                if ast_idx in ast_to_kernel_map:
                                    k_idx = ast_to_kernel_map[ast_idx]
                                    if k_idx < len(kernels):
                                        self._execute_single_kernel(
                                            kernels[k_idx], lib, gpu_allocs, env,
                                            tensors, cache_file_dir, cuda
                                        )

                        # =====================================================
                        # PHASE 5: Save statements
                        # =====================================================
                        save_count = sum(1 for n in ast if n['type'] == 'save')
                        if save_count > 0:
                            if self.debug_mode:
                                self.tensorlang.print(message=f"[COMPILER] Executing {save_count} save statement(s)")
                            for node in ast:
                                if node['type'] == 'save':
                                    self._execute_save_statement(
                                        node, tensors, gpu_allocs, env, cache_file_dir, cuda
                                    )

                        # Free GPU memory.
                        # Rebinds cause multiple names to share one allocation,
                        # so deduplicate by integer pointer address before freeing.
                        freed_ptrs = set()
                        for name, alloc in gpu_allocs.items():
                            ptr = int(alloc)
                            if ptr not in freed_ptrs:
                                alloc.free()
                                freed_ptrs.add(ptr)
                                if self.debug_mode:
                                    self.tensorlang.print(message=f"[COMPILER] Freed GPU memory for {name}")

                    except ImportError as e:
                        self.tensorlang.print(message=f"[COMPILER] PyCUDA error: {e}. Run 'pip install pycuda' and ensure CUDA toolkit is installed.")
                        sys.exit(1)
                    except Exception as e:
                        self.tensorlang.print(message=f"[COMPILER] Error executing CUDA kernel: {e}")
                        traceback.print_exc()
                        sys.exit(1)

        except ValueError as e:
            self.tensorlang.print(message=f"[COMPILER] Value Error: failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        except UnexpectedInput as e:
            self.tensorlang.print(message=f"[COMPILER] Parse error: {e}")
            traceback.print_exc()
            sys.exit(1)

        except Exception as e:
            self.tensorlang.print(message=f"[COMPILER] Exception at: {e}")
            traceback.print_exc()
            raise

        except (SyntaxError, RuntimeError) as e:
            self.tensorlang.print(message=f"[COMPILER] Error during parsing or execution: {e}")
            sys.exit(errno.EINVAL)

        self.tensorlang.separator()