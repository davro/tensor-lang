"""
TensorLang ExecutionMixin

Execution helpers for TensorCompiler.

File: tensorlang/execution.py
"""

import numpy as np
import pycuda.driver as cuda
from ctypes import c_float, c_int, c_void_p
from pathlib import Path

from tensorlang.tensor_lang import TensorLang
from tensorlang.constants import dtype_numpy, OPTIMIZER_NAMES


class TensorLangRuntimeError(RuntimeError):
    """A run-time failure with a stable diagnostic code; the message is already formatted."""


class ExecutionMixin:
    def _inline_functions(self, ast, functions):
        return ast

    def _tl_dtype(self, env, name):
        """Get the TensorLang dtype name for a tensor, defaulting to f32."""
        if name in env and isinstance(env[name], dict):
            return env[name].get("dtype", "f32")
        return "f32"

    def _np_dtype(self, env, name):
        """Get the numpy dtype for a tensor, defaulting to float32."""
        return dtype_numpy(self._tl_dtype(env, name))

    def _ensure_gpu_alloc(self, name, env, tensors, gpu_allocs, cuda):
        """Ensure a GPU allocation exists for *name*, creating one if needed."""
        if name in gpu_allocs:
            return gpu_allocs[name]
        if name not in env:
            raise KeyError(f"_ensure_gpu_alloc: '{name}' not in env")
        shape = tuple(int(d) for d in env[name]["shape"])
        np_dtype = self._np_dtype(env, name)
        size_bytes = int(np.prod(shape) * np_dtype.itemsize) if shape else np_dtype.itemsize
        if size_bytes == 0:
            size_bytes = np_dtype.itemsize
        alloc = cuda.mem_alloc(size_bytes)
        gpu_allocs[name] = alloc
        if name in tensors and isinstance(tensors[name], np.ndarray):
            arr = np.ascontiguousarray(tensors[name], dtype=np_dtype)
            cuda.memcpy_htod(alloc, arr)
        return alloc

    def _allocate_zeros(self, shape, cuda, dtype_name="f32"):
        """Allocate a GPU buffer of zeros with the given dtype."""
        np_dtype = dtype_numpy(dtype_name)
        size_bytes = int(np.prod(shape) * np_dtype.itemsize) if shape else np_dtype.itemsize
        if size_bytes == 0:
            size_bytes = np_dtype.itemsize
        alloc = cuda.mem_alloc(size_bytes)
        zeros = np.zeros(shape if shape else (1,), dtype=np_dtype)
        cuda.memcpy_htod(alloc, zeros)
        return alloc

    def _execute_single_kernel(self, kernel_info, lib, gpu_allocs, env, tensors, cache_file_dir, cuda):
        if not kernel_info:
            return

        op_type = kernel_info[0]

        # ================================================================
        # Special cases with non-standard kernel_info layouts
        # ================================================================

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
            master_name = kernel_info[5]  # None or string

            param_ptr = int(self._ensure_gpu_alloc(param_name, env, tensors, gpu_allocs, cuda))
            grad_ptr = int(self._ensure_gpu_alloc(grad_name, env, tensors, gpu_allocs, cuda))

            if master_name is not None:
                master_ptr = int(self._ensure_gpu_alloc(master_name, env, tensors, gpu_allocs, cuda))
            else:
                master_ptr = 0

            if op_type == "sgd_update":
                lr_value = self._eval_scalar(kernel_info[6], env, tensors, 0.0)
                if master_name is not None:
                    getattr(lib, f"launch_sgd_update_{kernel_func_name}")(
                        c_void_p(param_ptr),
                        c_void_p(grad_ptr),
                        c_void_p(master_ptr),
                        c_float(lr_value),
                        c_int(size),
                    )
                else:
                    getattr(lib, f"launch_sgd_update_{kernel_func_name}")(
                        c_void_p(param_ptr),
                        c_void_p(grad_ptr),
                        c_float(lr_value),
                        c_int(size),
                    )
            else:
                lr_value = self._eval_scalar(kernel_info[6], env, tensors, 0.01)
                beta1 = self._eval_scalar(kernel_info[7], env, tensors, 0.9)
                beta2 = self._eval_scalar(kernel_info[8], env, tensors, 0.999)
                eps = self._eval_scalar(kernel_info[9], env, tensors, 1e-8)

                state_dtype = "f32" if master_name is not None else self._tl_dtype(env, param_name)
                state = self.optimizer_state.setdefault(param_name, {
                    "m": self._allocate_zeros(env[param_name]["shape"], cuda,
                                             dtype_name=state_dtype),
                    "v": self._allocate_zeros(env[param_name]["shape"], cuda,
                                             dtype_name=state_dtype),
                    "t": 0,
                })
                state["t"] += 1

                if master_name is not None:
                    getattr(lib, f"launch_adam_update_{kernel_func_name}")(
                        c_void_p(param_ptr),
                        c_void_p(grad_ptr),
                        c_void_p(master_ptr),
                        c_void_p(int(state["m"])),
                        c_void_p(int(state["v"])),
                        c_float(lr_value),
                        c_float(beta1),
                        c_float(beta2),
                        c_float(eps),
                        c_int(state["t"]),
                        c_int(size),
                    )
                else:
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
            (_, actual_op, kernel_func_name, arg1, arg2, padded_shape1, padded_shape2, output_shape_tuple, total_elements) = kernel_info
            tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] Executing {op_type} ({actual_op}) for {tensor_name}")
            ndim = len(output_shape_tuple)
            shape1_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            shape2_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            out_shape_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            cuda.memcpy_htod(shape1_gpu, np.array(padded_shape1, dtype=np.int32))
            cuda.memcpy_htod(shape2_gpu, np.array(padded_shape2, dtype=np.int32))
            cuda.memcpy_htod(out_shape_gpu, np.array(output_shape_tuple, dtype=np.int32))
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
            shape1_gpu.free()
            shape2_gpu.free()
            out_shape_gpu.free()
            self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        # ================================================================
        # N-D special cases with shape arrays
        # ================================================================

        if op_type == "matmul_batched":
            (_, kernel_func_name, arg1, arg2, m, n, p, batch_count) = kernel_info
            tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)
            getattr(lib, f"launch_matmul_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(m), c_int(n), c_int(p), c_int(batch_count),
            )
            self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        if op_type == "softmax_nd":
            (_, kernel_func_name, tensor_name_src, _, total_elements, ndim, axis, input_shape) = kernel_info
            tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)
            in_shape_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            cuda.memcpy_htod(in_shape_gpu, np.array(input_shape, dtype=np.int32))
            getattr(lib, f"launch_softmax_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[tensor_name_src])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_void_p(int(in_shape_gpu)),
                c_int(ndim), c_int(axis), c_int(total_elements),
            )
            in_shape_gpu.free()
            self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        if op_type == "slice_nd":
            (_, kernel_func_name, tensor_name_src, _, in_ndim, out_ndim, total_output,
             input_shape, output_shape, offsets, is_index, fixed_vals) = kernel_info
            tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)
            in_shape_gpu = cuda.mem_alloc(in_ndim * np.int32().nbytes)
            out_shape_gpu = cuda.mem_alloc(max(out_ndim, 1) * np.int32().nbytes)
            offsets_gpu = cuda.mem_alloc(in_ndim * np.int32().nbytes)
            is_index_gpu = cuda.mem_alloc(in_ndim * np.int32().nbytes)
            fixed_vals_gpu = cuda.mem_alloc(in_ndim * np.int32().nbytes)
            cuda.memcpy_htod(in_shape_gpu, np.array(input_shape, dtype=np.int32))
            cuda.memcpy_htod(out_shape_gpu, np.array(output_shape if output_shape else [1], dtype=np.int32))
            cuda.memcpy_htod(offsets_gpu, np.array(offsets, dtype=np.int32))
            cuda.memcpy_htod(is_index_gpu, np.array(is_index, dtype=np.int32))
            cuda.memcpy_htod(fixed_vals_gpu, np.array(fixed_vals, dtype=np.int32))
            getattr(lib, f"launch_slice_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[tensor_name_src])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_void_p(int(in_shape_gpu)), c_void_p(int(out_shape_gpu)),
                c_void_p(int(offsets_gpu)), c_void_p(int(is_index_gpu)), c_void_p(int(fixed_vals_gpu)),
                c_int(in_ndim), c_int(out_ndim), c_int(total_output),
            )
            in_shape_gpu.free(); out_shape_gpu.free(); offsets_gpu.free()
            is_index_gpu.free(); fixed_vals_gpu.free()
            self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        if op_type == "concat_nd":
            (_, kernel_func_name, t1, t2, ndim, total_output,
             shape1, shape2, output_shape, axis, size1_axis) = kernel_info
            tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)
            shape1_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            shape2_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            out_shape_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            cuda.memcpy_htod(shape1_gpu, np.array(shape1, dtype=np.int32))
            cuda.memcpy_htod(shape2_gpu, np.array(shape2, dtype=np.int32))
            cuda.memcpy_htod(out_shape_gpu, np.array(output_shape, dtype=np.int32))
            getattr(lib, f"launch_concat_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[t1])), c_void_p(int(gpu_allocs[t2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_void_p(int(shape1_gpu)), c_void_p(int(shape2_gpu)), c_void_p(int(out_shape_gpu)),
                c_int(ndim), c_int(axis), c_int(size1_axis), c_int(total_output),
            )
            shape1_gpu.free(); shape2_gpu.free(); out_shape_gpu.free()
            self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        if op_type.endswith("_axis_nd"):
            (_, kernel_func_name, tensor_name_src, _, num_outputs, axis_size, axis, ndim, input_shape) = kernel_info
            tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)
            in_shape_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            cuda.memcpy_htod(in_shape_gpu, np.array(input_shape, dtype=np.int32))
            op_base = op_type.split("_")[0]
            getattr(lib, f"launch_{op_base}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[tensor_name_src])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_void_p(int(in_shape_gpu)),
                c_int(ndim), c_int(axis), c_int(num_outputs), c_int(axis_size),
            )
            in_shape_gpu.free()
            self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        if op_type == "layer_norm_nd":
            (_, kernel_func_name, tensor_name_src, _, total_elements, ndim, axis, axis_size, num_lines, eps, input_shape) = kernel_info
            tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)
            in_shape_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
            cuda.memcpy_htod(in_shape_gpu, np.array(input_shape, dtype=np.int32))
            getattr(lib, f"launch_layer_norm_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[tensor_name_src])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_void_p(int(in_shape_gpu)),
                c_int(ndim), c_int(axis), c_int(axis_size), c_int(num_lines), c_float(eps),
            )
            in_shape_gpu.free()
            self._save_kernel_result(tensor_name, op_type, env, gpu_allocs, tensors, cache_file_dir, cuda)
            return

        # ================================================================
        # Generic unpacking for standard kernel_info layouts
        # ================================================================

        op_type, kernel_func_name, arg1, arg2, *dims = kernel_info
        tensor_name = self._kernel_name_map.get(kernel_func_name, kernel_func_name)

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

        elif op_type in ("add_broadcast", "minus_broadcast", "mult_broadcast", "div_broadcast",
                        "greater_broadcast", "less_broadcast", "equal_broadcast"):
            op_name = op_type.split("_")[0]
            rows, cols = dims
            getattr(lib, f"launch_{op_name}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols),
            )

        elif op_type in ("greater_equal_broadcast", "less_equal_broadcast", "not_equal_broadcast"):
            op_name = op_type[:-len("_broadcast")]
            rows, cols = dims
            getattr(lib, f"launch_{op_name}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(rows), c_int(cols),
            )

        elif op_type in ("relu", "sigmoid", "tanh", "gelu", "swish"):
            op_name = op_type.split("_")[0]
            size = dims[0]
            getattr(lib, f"launch_{op_name}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
            )

        elif op_type in ("greater", "less", "equal", "greater_equal", "less_equal", "not_equal"):
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

        elif op_type in ("any_full", "all_full"):
            size = dims[0]
            op_name = op_type.split("_")[0]
            getattr(lib, f"launch_{op_name}_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size),
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

        elif op_type == "linear_nd":
            input_name = arg1
            weight_name = arg2
            bias_name = dims[0]
            batch_size = dims[1]
            in_features = dims[2]
            out_features = dims[3]
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

        elif op_type == "concat_1d":
            size1, size2 = dims
            getattr(lib, f"launch_concat_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(size1), c_int(size2),
            )

        elif op_type == "batch_norm":
            # v0.5.6: batch_norm gamma/beta (one kernel for every rank >= 2, channel at dim 1)
            total, num_features, inner, eps, running_var_name, gamma_name, beta_name = dims
            bn_args = [
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[arg2])),
                c_void_p(int(gpu_allocs[running_var_name])),
            ]
            if gamma_name is not None:
                bn_args += [c_void_p(int(gpu_allocs[gamma_name])), c_void_p(int(gpu_allocs[beta_name]))]
            bn_args += [
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(total), c_int(num_features), c_int(inner), c_float(eps),
            ]
            getattr(lib, f"launch_batch_norm_{kernel_func_name}")(*bn_args)

        elif op_type == "batch_norm_2d":
            running_mean_name, batch_size, num_features, eps, running_var_name = arg2, dims[0], dims[1], dims[2], dims[3]
            getattr(lib, f"launch_batch_norm_{kernel_func_name}")(
                c_void_p(int(gpu_allocs[arg1])),
                c_void_p(int(gpu_allocs[running_mean_name])),
                c_void_p(int(gpu_allocs[running_var_name])),
                c_void_p(int(gpu_allocs[tensor_name])),
                c_int(batch_size), c_int(num_features), c_float(eps),
            )

        elif op_type == "batch_norm_nd":
            running_mean_name = arg2
            batch_size = dims[0]
            num_features = dims[1]
            eps = dims[2]
            running_var_name = dims[3]
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

        elif op_type == "instance_norm_nd":
            batch_size = dims[1]
            num_features = dims[2]
            eps = dims[3]
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
            np_dtype = self._np_dtype(env, name)
            output = np.zeros(shape, dtype=np_dtype)
            cuda.memcpy_dtoh(output, gpu_allocs[name])
            tensors[name] = output
            if self.cache_layers:
                cache_npy_path = cache_file_dir / f"{name}.npy"
                np.save(cache_npy_path, output)
            if self.debug_mode:
                self.tensorlang.print(type="[COMPILER]", message=f"Result {name} ({op_type}):\n{output}")

    def _materialize_let_tensor(self, node, env, tensors, gpu_allocs, cache_file_dir, cuda):
        name = node["name"]
        expr = node.get("expr", {})
        if expr.get("type") == "tensor_literal":
            shape = tuple(int(d) for d in env[name]["shape"])
            np_dtype = self._np_dtype(env, name)
            tensors[name] = np.array(expr["data"], dtype=np_dtype).reshape(shape)
        elif expr.get("type") == "load":
            loaded_data, metadata = self._load_data_file(expr["file_path"], expr.get("array_name"))
            target_dtype = self._np_dtype(env, name)
            tensors[name] = loaded_data.astype(target_dtype)
        else:
            return
        # Validate loaded/literal data against the declared shape (loop bodies skip
        # the top-level shape check done in compiler.py).
        declared = tuple(int(d) for d in env[name]["shape"])
        if tuple(tensors[name].shape) != declared:
            raise ValueError(
                f"[COMPILER] Shape mismatch for '{name}': data has shape "
                f"{tuple(tensors[name].shape)}, declared {declared}"
            )
        alloc = self._ensure_gpu_alloc(name, env, tensors, gpu_allocs, cuda)
        gpu_allocs[name] = alloc
        # _ensure_gpu_alloc returns an existing buffer without uploading, and the
        # compiler pre-allocates every env name (including loop-local ones) before
        # execution.  Always push the freshly materialised host data to the GPU.
        np_dtype = self._np_dtype(env, name)
        cuda.memcpy_htod(alloc, np.ascontiguousarray(tensors[name], dtype=np_dtype))
        if node.get("requires_grad"):
            self.requires_grad_tensors.add(name)
        if self.cache_layers:
            np.save(cache_file_dir / f"{name}.npy", tensors[name])

    def _resolve_alias_inline(self, node, gpu_allocs, tensors):
        alias_name = node["name"]
        source_name = node["expr"]["name"]
        if source_name in tensors:
            src_dtype = tensors[source_name].dtype
            tensors[alias_name] = tensors[source_name].copy()
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] CUDA Execute: alias {alias_name} -> {source_name}")
        if source_name in gpu_allocs:
            gpu_allocs[alias_name] = gpu_allocs[source_name]
            if self.debug_mode:
                self.tensorlang.print(message=f"[COMPILER] CUDA Execute: GPU alias {alias_name} -> {source_name}")

    def _execute_save_statement(self, save_node, tensors, gpu_allocs, env, cache_file_dir, cuda):
        tensor_name = save_node["tensor"]
        file_path = save_node["file_path"]
        if self.debug_mode:
            self.tensorlang.print(message=f"[COMPILER] Executing save: {tensor_name} -> {file_path}")
        if tensor_name in tensors:
            tensor_data = tensors[tensor_name]
        elif tensor_name in gpu_allocs and tensor_name in env:
            shape = tuple(int(dim) for dim in env[tensor_name]["shape"])
            np_dtype = self._np_dtype(env, tensor_name)
            tensor_data = np.zeros(shape, dtype=np_dtype)
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

    def _execute_for_loop(self, for_node, kernels, ast_to_kernel_map,
                          lib, gpu_allocs, env, tensors, cache_file_dir, cuda):
        iterations = for_node["iterations"]
        body = for_node["body"]

        loop_kernels = for_node.get("_kernels", kernels)

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

            for pos, (local_idx, node) in enumerate(for_node["_flat_items"]):
                if node["type"] in ("backward", "save", "rebind"):
                    continue
                if bw_list_pos is not None and pos >= bw_list_pos:
                    continue

                if node.get("type") == "if":
                    # Shared with top level and nested ifs: also resolves which branch ran, so a
                    # backward pass inside it ignores the nodes of the other branches.
                    self._execute_if_node(node, kernels, lib, gpu_allocs, env, tensors, cache_file_dir, cuda)
                    continue

                if node.get("type") == "for":
                    self._execute_for_loop(node, kernels, ast_to_kernel_map,
                                           lib, gpu_allocs, env, tensors, cache_file_dir, cuda)
                    continue

                if (node.get("type") == "let"
                        and isinstance(node.get("expr"), dict)
                        and node["expr"].get("type") == "name"):
                    self._resolve_alias_inline(node, gpu_allocs, tensors)
                    continue
                if (node.get("type") == "let"
                        and isinstance(node.get("expr"), dict)
                        and node["expr"].get("type") in ("tensor_literal", "load")):
                    self._materialize_let_tensor(node, env, tensors, gpu_allocs, cache_file_dir, cuda)
                    continue

                if local_idx is None:
                    continue
                self._execute_single_kernel(
                    loop_kernels[local_idx], lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                )

            if backward_node is not None:
                loss_name = backward_node["loss_tensor"]

                for idx, body_node in for_node["_flat_items"]:
                    if body_node["type"] != "let":
                        continue
                    bname = body_node["name"]
                    if bname in gpu_allocs and bname in env:
                        bshape = tuple(int(d) for d in env[bname]["shape"])
                        bnp_dtype = self._np_dtype(env, bname)
                        barr = np.empty(bshape, dtype=bnp_dtype)
                        cuda.memcpy_dtoh(barr, gpu_allocs[bname])
                        tensors[bname] = barr

                for tname in list(tensors.keys()):
                    if tname in gpu_allocs and tname in env:
                        tshape = tuple(int(d) for d in env[tname]["shape"])
                        tnp_dtype = self._np_dtype(env, tname)
                        tarr = np.empty(tshape, dtype=tnp_dtype)
                        cuda.memcpy_dtoh(tarr, gpu_allocs[tname])
                        tensors[tname] = tarr

                for tname, tval in tensors.items():
                    requires_grad = tname in self.requires_grad_tensors
                    self.comp_graph.register_tensor(tname, tval, requires_grad)

                scale = self._current_scale if getattr(self, "_current_scale", None) is not None else (self._loss_scale if getattr(self, "_loss_scale", None) is not None else 1.0)

                self.comp_graph.gradients = {}
                self.comp_graph.backward(loss_name, scale=scale)
                reached = self.comp_graph.reachable_tensors(loss_name)

                # Overflow detection
                overflow = self.comp_graph.overflow_detected
                if overflow:
                    old_scale = self._current_scale
                    self._current_scale = max(self._current_scale / 2.0, 1.0)
                    self._scale_growth_counter = 0
                    self._skip_optimizer = True
                    if self.debug_mode:
                        self.tensorlang.print(
                            message=f"[COMPILER] for loop iter {i + 1}: overflow detected — "
                                    f"scale {old_scale} -> {self._current_scale}, skipping optimizer step"
                        )
                else:
                    self._skip_optimizer = False

                if self.debug_mode:
                    loss_val = tensors.get(loss_name)
                    if loss_val is not None:
                        self.tensorlang.print(
                            message=f"[COMPILER] for loop iter {i + 1}: "
                                    f"loss={float(np.asarray(loss_val).reshape(-1)[0]):.6f} "
                                    f"(scale={scale})"
                        )

                if not overflow:
                    for grad_name in self.comp_graph.requires_grad:
                        if (grad_name in self.comp_graph.gradients
                                and not self.comp_graph.keeps_previous_gradient(grad_name, reached)):
                            self.comp_graph.grads_written.add(grad_name)
                            grad_tensor = self.comp_graph.gradients[grad_name]
                            # Unscale gradients before optimizer step
                            if scale != 1.0 and scale != 0.0:
                                grad_tensor = grad_tensor / scale
                            self.comp_graph.gradients[grad_name] = grad_tensor
                            grad_tensor_name = f"{grad_name}_grad"
                            tensors[grad_tensor_name] = grad_tensor

                            if grad_tensor_name in gpu_allocs:
                                cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)
                            else:
                                gpu_allocs[grad_tensor_name] = cuda.mem_alloc(grad_tensor.nbytes)
                                cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)

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

                    # Dynamic scale growth
                    if getattr(self, "dynamic_loss_scale", False) and not overflow:
                        self._scale_growth_counter = getattr(self, "_scale_growth_counter", 0) + 1
                        patience = getattr(self, "loss_scale_patience", 2000)
                        if self._scale_growth_counter >= patience:
                            old = self._current_scale
                            self._current_scale *= 2.0
                            self._scale_growth_counter = 0
                            if self.debug_mode:
                                self.tensorlang.print(
                                    message=f"[COMPILER] for loop iter {i + 1}: scale grown {old} -> {self._current_scale}"
                                )

                if bw_list_pos is not None:
                    for pos, (local_idx, node) in enumerate(for_node["_flat_items"]):
                        if pos <= bw_list_pos:
                            continue
                        if node["type"] == "backward":
                            # a further backward pass in the same loop body: run it here, in order
                            # (the first one was handled above, with the loss-scaling bookkeeping)
                            self._execute_backward_pass(
                                node["loss_tensor"], tensors, gpu_allocs, env, cache_file_dir, cuda
                            )
                            continue
                        if node["type"] in ("save", "rebind"):
                            continue

                        if node.get("type") == "if":
                            # v0.5.6: an `if` after the first backward() runs here, like any other `if`: it
                            # sees the fresh loss, and an optimiser step in its branch sees the fresh gradient.
                            self._execute_if_node(node, kernels, lib, gpu_allocs, env, tensors, cache_file_dir, cuda)
                            continue

                        if node.get("type") == "for":
                            self._execute_for_loop(node, kernels, ast_to_kernel_map,
                                                   lib, gpu_allocs, env, tensors, cache_file_dir, cuda)
                            continue

                        if (node.get("type") == "let"
                                and isinstance(node.get("expr"), dict)
                                and node["expr"].get("type") == "name"):
                            self._resolve_alias_inline(node, gpu_allocs, tensors)
                            continue
                        if (node.get("type") == "let"
                                and isinstance(node.get("expr"), dict)
                                and node["expr"].get("type") in ("tensor_literal", "load")):
                            self._materialize_let_tensor(node, env, tensors, gpu_allocs, cache_file_dir, cuda)
                            continue

                        if local_idx is None:
                            continue
                        _is_opt = (node.get("type", "") in OPTIMIZER_NAMES or
                                   (node.get("type", "") == "call" and node.get("name", "") in OPTIMIZER_NAMES))
                        if _is_opt and getattr(self, "_skip_optimizer", False):
                            if self.debug_mode:
                                self.tensorlang.print(message="[COMPILER] for loop: skipping optimizer step (overflow)")
                            continue

                        self._execute_single_kernel(
                            loop_kernels[local_idx], lib, gpu_allocs, env, tensors, cache_file_dir, cuda
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

                if target in env and src_name in env:
                    if tuple(env[target]["shape"]) != tuple(env[src_name]["shape"]):
                        self.tensorlang.print(
                            message=f"[COMPILER] for loop rebind: shape mismatch for '{target}': "
                                    f"{env[target]['shape']} vs {env[src_name]['shape']}"
                        )
                        continue

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

            for node in body:
                if (node.get("type") == "let"
                        and isinstance(node.get("expr"), dict)
                        and node["expr"].get("type") == "name"):
                    self._resolve_alias_inline(node, gpu_allocs, tensors)

            for node in body:
                if node.get("type") == "save":
                    self._execute_save_statement(node, tensors, gpu_allocs, env, cache_file_dir, cuda)

        if self.debug_mode:
            self.tensorlang.print(
                message=f"[COMPILER] for loop: completed {iterations} iterations"
            )

    def _eval_if_condition(self, node, tensors, gpu_allocs, env, cuda):
        if "_condition_constant" in node:
            return bool(node["_condition_constant"])

        name = node.get("_condition_name")
        if name is None:
            return False

        self._reject_nan_condition(node, tensors, gpu_allocs, env, cuda)

        if name not in tensors and name in gpu_allocs and name in env:
            shape = tuple(int(d) for d in env[name]["shape"])
            np_dtype = self._np_dtype(env, name)
            arr = np.empty(shape if shape else (1,), dtype=np_dtype)
            cuda.memcpy_dtoh(arr, gpu_allocs[name])
            tensors[name] = arr

        if name not in tensors:
            raise KeyError(f"_eval_if_condition: missing tensor '{name}'")

        arr = np.asarray(tensors[name]).reshape(-1)
        if arr.size == 0:
            return False
        return bool(np.all(arr != 0))

    def _reject_nan_condition(self, node, tensors, gpu_allocs, env, cuda):
        """A NaN operand makes a condition meaningless: every comparison with NaN is false
        (except !=), so the program would silently take a branch. Stop instead.
        The operands are checked, not the 0/1 result, which has already lost the NaN."""
        expr = node.get("_condition_expr")
        if isinstance(expr, dict) and isinstance(expr.get("args"), list):
            operands = list(expr["args"])
        elif isinstance(expr, dict) and "tensor" in expr:
            operands = [expr["tensor"]]
        else:
            operands = [node.get("_condition_name")]
        for op_name in operands:
            if not isinstance(op_name, str) or op_name not in env:
                continue
            if op_name in gpu_allocs:
                shape = tuple(int(d) for d in env[op_name]["shape"])
                arr = np.empty(shape if shape else (1,), dtype=self._np_dtype(env, op_name))
                cuda.memcpy_dtoh(arr, gpu_allocs[op_name])
            elif op_name in tensors:
                arr = np.asarray(tensors[op_name])
            else:
                continue
            if arr.dtype.kind != "f":
                continue
            nan_count = int(np.isnan(arr).sum())
            if nan_count:
                text = node.get("_condition_text") or op_name
                label = "a nested expression" if op_name.startswith("__tmp_") else f"'{op_name}'"
                raise TensorLangRuntimeError(
                    f"[RUNTIME] error: [E-NAN-IN-CONDITION] {label} contains NaN ({nan_count} of "
                    f"{arr.size} elements), so the condition `{text}` cannot be evaluated: a NaN "
                    f"makes every comparison false (except !=) and would silently pick a branch. "
                    f"Find the computation that produced it.")

    def _select_if_branch(self, node, tensors, gpu_allocs, env, cuda, lib, cache_file_dir):
        if self._eval_if_condition(node, tensors, gpu_allocs, env, cuda):
            return node.get("then", []), node.get("_then_kernels", []), node.get("_then_flat_items", [])

        for elif_node in node.get("elifs", []):
            for k in elif_node.get("_condition_kernels", []):
                self._execute_single_kernel(k, lib, gpu_allocs, env, tensors, cache_file_dir, cuda)
            if self._eval_if_condition(elif_node, tensors, gpu_allocs, env, cuda):
                return elif_node.get("body", []), elif_node.get("_body_kernels", []), elif_node.get("_body_flat_items", [])

        if node.get("else") is not None:
            return node["else"], node.get("_else_kernels", []), node.get("_else_flat_items", [])

        return None, None, None

    def _execute_if_branch(self, branch_nodes, branch_flat, branch_kernels, lib, gpu_allocs, env, tensors, cache_file_dir, cuda, branch_path=None):
        rebinds = [n for n in branch_nodes if n.get("type") == "rebind"]
        exec_nodes = [n for n in branch_nodes if n.get("type") != "rebind"]

        for node in exec_nodes:
            if (
                node.get("type") == "let"
                and isinstance(node.get("expr"), dict)
                and node["expr"].get("type") == "name"
            ):
                self._resolve_alias_inline(node, gpu_allocs, tensors)
            if (
                node.get("type") == "let"
                and isinstance(node.get("expr"), dict)
                and node["expr"].get("type") in ("tensor_literal", "load")
            ):
                self._materialize_let_tensor(node, env, tensors, gpu_allocs, cache_file_dir, cuda)

        for local_idx, branch_node in branch_flat:
            if branch_node.get("type") == "backward":
                # Run each backward pass where it appears, in order: a later sgd/let sees its
                # gradients and several passes in one branch all run. Sync every device tensor,
                # as at top level: the loss may depend on values computed before this branch or
                # in an enclosing branch, which this branch's own statements do not mention.
                self._execute_backward_pass(
                    branch_node["loss_tensor"], tensors, gpu_allocs, env, cache_file_dir, cuda,
                    active_branch_path=branch_path or (),
                )
                continue
            if branch_node.get("type") in ("save", "rebind"):
                continue
            if branch_node.get("type") == "for":
                self._execute_for_loop(branch_node, branch_kernels, None, lib, gpu_allocs, env, tensors, cache_file_dir, cuda)
                continue
            if branch_node.get("type") == "if":
                self._execute_if_node(branch_node, branch_kernels, lib, gpu_allocs, env, tensors, cache_file_dir, cuda)
                continue
            if (
                branch_node.get("type") == "let"
                and isinstance(branch_node.get("expr"), dict)
                and branch_node["expr"].get("type") == "name"
            ):
                self._resolve_alias_inline(branch_node, gpu_allocs, tensors)
            if (
                branch_node.get("type") == "let"
                and isinstance(branch_node.get("expr"), dict)
                and branch_node["expr"].get("type") in ("tensor_literal", "load")
            ):
                self._materialize_let_tensor(branch_node, env, tensors, gpu_allocs, cache_file_dir, cuda)
                continue
            if local_idx is None:
                continue
            self._execute_single_kernel(branch_kernels[local_idx], lib, gpu_allocs, env, tensors, cache_file_dir, cuda)

        pending_gpu = {}
        pending_tensors = {}
        for rebind in rebinds:
            target = rebind["name"]
            src_expr = rebind["expr"]
            if src_expr.get("type") != "name":
                continue
            src_name = src_expr["name"]
            if src_name not in gpu_allocs:
                continue
            pending_gpu[target] = gpu_allocs[src_name]
            if src_name in tensors:
                pending_tensors[target] = tensors[src_name]
        gpu_allocs.update(pending_gpu)
        tensors.update(pending_tensors)

        for node in branch_nodes:
            if (
                node.get("type") == "let"
                and isinstance(node.get("expr"), dict)
                and node["expr"].get("type") == "name"
            ):
                self._resolve_alias_inline(node, gpu_allocs, tensors)

        for node in branch_nodes:
            if node.get("type") == "save":
                self._execute_save_statement(node, tensors, gpu_allocs, env, cache_file_dir, cuda)

    def _execute_if_node(self, node, kernels, lib, gpu_allocs, env, tensors, cache_file_dir, cuda):
        for k in node.get("_condition_kernels", []):
            self._execute_single_kernel(k, lib, gpu_allocs, env, tensors, cache_file_dir, cuda)
        selected_nodes, selected_kernels, selected_flat = self._select_if_branch(
            node, tensors, gpu_allocs, env, cuda, lib, cache_file_dir
        )

        branch_path = None
        if selected_nodes is node.get("then", []):
            branch_path = node.get("_then_branch_path")
        elif selected_nodes is node.get("else", []):
            branch_path = node.get("_else_branch_path")
        else:
            for i, elif_node in enumerate(node.get("elifs", [])):
                if selected_nodes is elif_node.get("body", []):
                    paths = node.get("_elif_branch_paths", [])
                    branch_path = paths[i] if i < len(paths) else None
                    break

        node["_selected_branch_path"] = branch_path
        if selected_nodes is not None:
            self._execute_if_branch(
                selected_nodes, selected_flat, selected_kernels,
                lib, gpu_allocs, env, tensors, cache_file_dir, cuda,
                branch_path=branch_path,
            )
        node["_selected_branch_flat_items"] = selected_flat
        node["_selected_branch_kernels"] = selected_kernels

    def _execute_if_branch_post(self, branch_flat, branch_kernels, lib, gpu_allocs, env, tensors, cache_file_dir, cuda):
        for local_idx, branch_node in branch_flat:
            if branch_node.get("type") == "for":
                continue
            if branch_node.get("type") == "if":
                selected_flat = branch_node.get("_selected_branch_flat_items")
                selected_kernels = branch_node.get("_selected_branch_kernels")
                if selected_flat is not None:
                    self._execute_if_branch_post(
                        selected_flat, selected_kernels, lib, gpu_allocs, env, tensors, cache_file_dir, cuda
                    )
                continue
            is_optimizer = (
                branch_node.get("type") in OPTIMIZER_NAMES
                or (branch_node.get("type") == "call" and branch_node.get("name") in OPTIMIZER_NAMES)
            )
            if not is_optimizer:
                continue
            if getattr(self, "_skip_optimizer", False):
                continue
            if local_idx is None:
                continue
            self._execute_single_kernel(branch_kernels[local_idx], lib, gpu_allocs, env, tensors, cache_file_dir, cuda)

    def _collect_executed_output_names(self, flat_items):
        names = set()
        for _, node in flat_items or []:
            if node.get("type") == "let":
                names.add(node["name"])
            elif node.get("type") == "rebind":
                names.add(node["name"])
            elif node.get("type") == "for":
                names |= self._collect_executed_output_names(node.get("_flat_items", []))
            elif node.get("type") == "if":
                names |= self._collect_executed_output_names(node.get("_selected_branch_flat_items", []))
        return names

    def _execute_backward_pass(self, loss_name, tensors, gpu_allocs, env, cache_file_dir, cuda, sync_names=None, active_branch_path=None):
        if sync_names is None:
            names_to_sync = set(gpu_allocs.keys()) & set(env.keys())
        else:
            names_to_sync = set(sync_names)
        names_to_sync |= set(tensors.keys())

        for name in names_to_sync:
            if name in gpu_allocs and name in env:
                shape = tuple(int(d) for d in env[name]["shape"])
                np_dtype = self._np_dtype(env, name)
                arr = np.empty(shape if shape else (1,), dtype=np_dtype)
                cuda.memcpy_dtoh(arr, gpu_allocs[name])
                tensors[name] = arr

        for name, value in list(tensors.items()):
            requires_grad = name in self.requires_grad_tensors
            self.comp_graph.register_tensor(name, value, requires_grad)

        scale = self._current_scale if getattr(self, "_current_scale", None) is not None else 1.0
        self.comp_graph.active_branch_path = tuple(active_branch_path or ())
        self._skip_optimizer = False

        try:
            self.comp_graph.gradients = {}
            self.comp_graph.backward(loss_name, scale=scale, active_branch_path=self.comp_graph.active_branch_path)
            reached = self.comp_graph.reachable_tensors(loss_name, self.comp_graph.active_branch_path)

            for grad_name in self.comp_graph.requires_grad:
                if grad_name not in self.comp_graph.gradients:
                    continue
                if self.comp_graph.keeps_previous_gradient(grad_name, reached):
                    continue
                grad_tensor = self.comp_graph.gradients[grad_name]
                if self.debug_mode:
                    self.tensorlang.print(message=f"\nGradient {grad_name}.grad (scaled):\n{grad_tensor}")
                if self.cache_layers:
                    grad_cache_path = cache_file_dir / f"{grad_name}.grad.npy"
                    np.save(grad_cache_path, grad_tensor)

            self._overflow_detected = self.comp_graph.overflow_detected

            if self._overflow_detected:
                old_scale = self._current_scale
                self._current_scale = max(self._current_scale / 2.0, 1.0)
                self._scale_growth_counter = 0
                self._skip_optimizer = True
                if self.debug_mode:
                    self.tensorlang.print(
                        message=f"[COMPILER] Overflow detected — scale {old_scale} -> {self._current_scale}, skipping optimizer step"
                    )
            else:
                self._skip_optimizer = False
                for grad_name in self.comp_graph.requires_grad:
                    if grad_name not in self.comp_graph.gradients:
                        continue
                    if self.comp_graph.keeps_previous_gradient(grad_name, reached):
                        continue
                    self.comp_graph.grads_written.add(grad_name)
                    grad_tensor = self.comp_graph.gradients[grad_name]
                    if scale != 1.0 and scale != 0.0:
                        grad_tensor = grad_tensor / scale
                    self.comp_graph.gradients[grad_name] = grad_tensor
                    grad_tensor_name = f"{grad_name}_grad"
                    tensors[grad_tensor_name] = grad_tensor
                    if grad_tensor_name in gpu_allocs:
                        cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)
                    else:
                        gpu_allocs[grad_tensor_name] = cuda.mem_alloc(grad_tensor.nbytes)
                        cuda.memcpy_htod(gpu_allocs[grad_tensor_name], grad_tensor)
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
                    if self.cache_layers:
                        grad_var_cache_path = cache_file_dir / f"{grad_tensor_name}.npy"
                        np.save(grad_var_cache_path, grad_tensor)

                if getattr(self, "dynamic_loss_scale", False) and not self._overflow_detected:
                    self._scale_growth_counter = getattr(self, "_scale_growth_counter", 0) + 1
                    patience = getattr(self, "loss_scale_patience", 2000)
                    if self._scale_growth_counter >= patience:
                        old_scale = self._current_scale
                        self._current_scale *= 2.0
                        self._scale_growth_counter = 0
                        if self.debug_mode:
                            self.tensorlang.print(
                                message=f"[COMPILER] Scale grown {old_scale} -> {self._current_scale} after {patience} successful steps"
                            )

        except (RuntimeError, ValueError, KeyError, OSError, TypeError) as e:
            self.tensorlang.print(message=f"[COMPILER] [Autograd] Error during backward pass: {e}")
            import traceback
            if self.debug_mode:
                traceback.print_exc()

