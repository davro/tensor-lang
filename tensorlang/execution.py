"""Execution helpers for TensorCompiler."""

import numpy as np
import pycuda.driver as cuda
from ctypes import c_float, c_int, c_void_p
from pathlib import Path

from tensorlang.tensor_lang import TensorLang


class ExecutionMixin:
    def _inline_functions(self, ast, functions):
        return ast

    def _execute_single_kernel(self, kernel_info, lib, gpu_allocs, env, tensors, cache_file_dir, cuda):
        if not kernel_info:
            return

        op_type = kernel_info[0]

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
        if source_name in tensors:
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

                if node.get("type") == "for":
                    self._execute_for_loop(node, kernels, ast_to_kernel_map,
                                           lib, gpu_allocs, env, tensors, cache_file_dir, cuda)
                    continue

                if (node.get("type") == "let"
                        and isinstance(node.get("expr"), dict)
                        and node["expr"].get("type") == "name"):
                    self._resolve_alias_inline(node, gpu_allocs, tensors)
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
                    for pos, (local_idx, node) in enumerate(for_node["_flat_items"]):
                        if node["type"] in ("backward", "save", "rebind"):
                            continue
                        if pos <= bw_list_pos:
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

                        if local_idx is None:
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

                # Guard 1: shape stability
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

            # Re-resolve alias lets now that rebinds have been applied,
            # so `let y = w` tracks the rebound storage within the same iteration.
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
