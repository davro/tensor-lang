"""
TensorLang CodegenMixin
Code generation helpers for TensorCompiler.

File: tensorlang/codegen.py
"""

import inspect

import numpy as np

from tensorlang.autograd import AutogradContext, ComputationGraph
from tensorlang.constants import OPTIMIZER_NAMES, promote_dtype
from tensorlang.fusion import fuse_elementwise_chains
from tensorlang.kernel_generator import KernelGenerator
from tensorlang.tensor_lang import TensorLang
from tensorlang.type_checker import _is_scalar_shape


class CodegenMixin:
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

    def can_broadcast(self, shape1, shape2):
        ndim = max(len(shape1), len(shape2))
        s1 = (1,) * (ndim - len(shape1)) + shape1
        s2 = (1,) * (ndim - len(shape2)) + shape2
        for d1, d2 in zip(s1, s2):
            if d1 != d2 and d1 != 1 and d2 != 1:
                return False
        return True

    def _kernel_for_optimizer(self, node, env, generator, kernel_name=None):
        kernel_name = kernel_name or f"opt{len(self._kernel_name_map)}"

        opt_name, param_name, grad_name, lr_arg, beta1_arg, beta2_arg, eps_arg = self._optimizer_parts(node)
        if opt_name not in OPTIMIZER_NAMES or param_name is None or grad_name is None:
            return None, None
        if param_name not in env or grad_name not in env:
            return None, None

        size = int(np.prod([int(d) for d in env[param_name]["shape"]]))
        opt_dtype = env[param_name].get("dtype", "f32")

        if opt_name in ("sgd", "sgd_update"):
            kernel_str, kernel_info = generator.sgd_update(
                "sgd_update", kernel_name, param_name, grad_name, size, dtype=opt_dtype
            )
            kernel_info = kernel_info + (lr_arg,)
        else:
            kernel_str, kernel_info = generator.adam_update(
                "adam_update", kernel_name, param_name, grad_name, size, dtype=opt_dtype
            )
            kernel_info = kernel_info + (lr_arg, beta1_arg, beta2_arg, eps_arg)

        self._kernel_name_map[kernel_name] = param_name
        return kernel_str, kernel_info

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
        fused_dtype = env[name].get("dtype", "f32") if name in env else env[final_name].get("dtype", "f32")
        kernel_str, kernel_info = generator.fused_elementwise(
            "fused_elementwise", kernel_name, chain, env, external_inputs, outputs, size,
            broadcast=broadcast, input_shapes=input_shapes, output_shape=output_shape,
            dtype=fused_dtype,
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
            result_dtype = promote_dtype(env[arg1].get("dtype", "f32"), env[arg2].get("dtype", "f32"))
            if shape1 == shape2:
                size = int(np.prod([int(d) for d in output_shape]))
                return generator.elementwise(expr["type"], kernel_name, arg1, arg2, size, dtype=result_dtype)
            if _is_scalar_shape(shape1) or _is_scalar_shape(shape2):
                scalar_on_left = _is_scalar_shape(shape1) and not _is_scalar_shape(shape2)
                size = int(np.prod([int(d) for d in output_shape]))
                return generator.binary_1d_broadcast(
                    expr["type"], kernel_name, arg1, arg2, size, scalar_on_left=scalar_on_left, dtype=result_dtype
                )
            if len(shape1) == 2 and len(shape2) == 1:
                return generator.binary_broadcast(
                    expr["type"], expr["type"], kernel_name, arg1, arg2, shape1, shape2, output_shape, dtype=result_dtype
                )
            if len(shape1) == 1 and len(shape2) == 1 and (shape1[0] == 1 or shape2[0] == 1):
                size = int(np.prod([int(d) for d in output_shape]))
                return generator.binary_1d_broadcast(expr["type"], kernel_name, arg1, arg2, size, dtype=result_dtype)
            if self.can_broadcast(shape1, shape2):
                return generator.binary_general_broadcast(
                    expr["type"], kernel_name, arg1, arg2, shape1, shape2, output_shape, dtype=result_dtype
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
            result_dtype = promote_dtype(env[arg1].get("dtype", "f32"), env[arg2].get("dtype", "f32"))
            return generator.matmul(expr["type"], kernel_name, arg1, arg2, int(m), int(n), int(p), dtype=result_dtype)

        if expr.get("type") in ("relu", "sigmoid", "tanh", "gelu", "swish"):
            arg1 = expr["args"][0]
            size = int(np.prod([int(d) for d in env[arg1]["shape"]]))
            self._record_operation(expr["type"], name, [arg1])
            result_dtype = env[arg1].get("dtype", "f32")
            method = getattr(generator, expr["type"], None)
            if method:
                return method(expr["type"], kernel_name, arg1, size, dtype=result_dtype)

        if expr.get("type") == "softmax":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            self._record_operation("softmax", name, [tensor_name], metadata={"axis": axis})
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.softmax(expr["type"], kernel_name, tensor_name, input_shape, axis, dtype=result_dtype)

        if expr.get("type") == "mse_loss":
            pred_name, target_name = expr["args"]
            pred_shape = env[pred_name]["shape"]
            total_elements = int(np.prod([int(d) for d in pred_shape]))
            self._record_operation("mse_loss", name, [pred_name, target_name])
            result_dtype = promote_dtype(env[pred_name].get("dtype", "f32"), env[target_name].get("dtype", "f32"))
            return generator.mse_loss(expr["type"], kernel_name, pred_name, target_name, total_elements, dtype=result_dtype)

        if expr.get("type") == "linear":
            input_name, weight_name, bias_name = expr["args"]
            input_shape = env[input_name]["shape"]
            weight_shape = env[weight_name]["shape"]
            output_shape = env[name]["shape"]
            self._record_operation("linear", name, [input_name, weight_name, bias_name])
            result_dtype = promote_dtype(promote_dtype(env[input_name].get("dtype", "f32"), env[weight_name].get("dtype", "f32")), env[bias_name].get("dtype", "f32"))
            return generator.linear(
                expr["type"], kernel_name, input_name, weight_name, bias_name,
                input_shape, weight_shape, output_shape, dtype=result_dtype,
            )

        if expr.get("type") == "sum":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            self._record_operation("sum", name, [tensor_name], metadata={"axis": axis})
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.sum(expr["type"], kernel_name, tensor_name, axis, input_shape, dtype=result_dtype)

        if expr.get("type") == "mean":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            self._record_operation("mean", name, [tensor_name], metadata={"axis": axis})
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.mean(expr["type"], kernel_name, tensor_name, axis, input_shape, dtype=result_dtype)

        if expr.get("type") in ("greater", "less", "equal"):
            arg1, arg2 = expr["args"]
            shape1, shape2 = env[arg1]["shape"], env[arg2]["shape"]
            output_shape = env[name]["shape"]
            if expr.get("type") == "greater":
                size = int(np.prod([int(d) for d in output_shape]))
                return generator.greater_broadcast(expr["type"], name, arg1, arg2, shape1, shape2, output_shape, size, dtype="f32")
            if expr.get("type") == "less":
                return generator.less_broadcast(expr["type"], name, arg1, arg2, shape1, shape2, output_shape, dtype="f32")
            if expr.get("type") == "equal":
                return generator.equal(expr["type"], name, arg1, arg2, shape1, shape2, output_shape, dtype="f32")

        if expr.get("type") == "softmax":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            self._record_operation("softmax", name, [tensor_name], metadata={"axis": axis})
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.softmax(expr["type"], kernel_name, tensor_name, input_shape, axis, dtype=result_dtype)

        if expr.get("type") == "slice":
            tensor_name = expr["tensor"]
            slice_specs = expr["specs"]
            input_shape = env[tensor_name]["shape"]
            output_shape = env[name]["shape"]
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.slice(expr["type"], kernel_name, tensor_name, slice_specs, input_shape, output_shape, dtype=result_dtype)

        if expr.get("type") == "max":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.max(expr["type"], kernel_name, tensor_name, axis, input_shape, dtype=result_dtype)

        if expr.get("type") == "min":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.min(expr["type"], kernel_name, tensor_name, axis, input_shape, dtype=result_dtype)

        if expr.get("type") == "argmax":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.argmax(expr["type"], kernel_name, tensor_name, axis, input_shape, dtype=result_dtype)

        if expr.get("type") == "argmin":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            input_shape = env[tensor_name]["shape"]
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.argmin(expr["type"], kernel_name, tensor_name, axis, input_shape, dtype=result_dtype)

        if expr.get("type") == "layer_norm":
            tensor_name = expr["tensor"]
            axis = expr.get("axis")
            eps = expr.get("eps", 1e-5)
            input_shape = env[tensor_name]["shape"]
            if axis is None:
                axis = len(input_shape) - 1
            self._record_operation("layer_norm", name, [tensor_name], metadata={"axis": axis, "eps": eps})
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.layer_norm(expr["type"], kernel_name, tensor_name, axis, eps, input_shape, dtype=result_dtype)

        if expr.get("type") == "cross_entropy":
            pred_name, target_name = expr["args"]
            pred_shape = env[pred_name]["shape"]
            target_shape = env[target_name]["shape"]
            self._record_operation("cross_entropy", name, [pred_name, target_name])
            result_dtype = promote_dtype(env[pred_name].get("dtype", "f32"), env[target_name].get("dtype", "f32"))
            return generator.cross_entropy(expr["type"], kernel_name, pred_name, pred_shape, target_shape, target_name, dtype=result_dtype)

        if expr.get("type") == "transpose":
            tensor_name = expr["tensor"]
            input_shape = env[tensor_name]["shape"]
            axes = expr.get("axes")
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.transpose(expr["type"], kernel_name, input_shape, tensor_name, axes, dtype=result_dtype)

        if expr.get("type") == "reshape":
            tensor_name = expr["tensor"]
            input_shape = env[tensor_name]["shape"]
            total_elements = int(np.prod([int(d) for d in input_shape]))
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.reshape(expr["type"], kernel_name, tensor_name, total_elements, dtype=result_dtype)

        if expr.get("type") == "concat":
            tensor_names = expr["tensors"]
            axis = expr["axis"]
            # Promote across all concat inputs
            concat_dtype = env[tensor_names[0]].get("dtype", "f32")
            for tn in tensor_names[1:]:
                concat_dtype = promote_dtype(concat_dtype, env[tn].get("dtype", "f32"))
            return generator.concat(expr["type"], kernel_name, tensor_names, axis, env, dtype=concat_dtype)

        if expr.get("type") == "batch_norm":
            tensor_name = expr["tensor"]
            running_mean_name = expr["running_mean"]
            running_var_name = expr["running_var"]
            eps = expr.get("eps", 1e-5)
            input_shape = env[tensor_name]["shape"]
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.batch_norm(expr["type"], kernel_name, input_shape, tensor_name, running_mean_name, eps, running_var_name, dtype=result_dtype)

        if expr.get("type") == "instance_norm":
            tensor_name = expr["tensor"]
            eps = expr.get("eps", 1e-5)
            input_shape = env[tensor_name]["shape"]
            result_dtype = env[tensor_name].get("dtype", "f32")
            return generator.instance_norm(expr["type"], kernel_name, tensor_name, input_shape, eps, dtype=result_dtype)

        if expr.get("type") == "fill":
            size = int(np.prod([int(dim) for dim in expr["shape"]]))
            result_dtype = env[name].get("dtype", "f32") if name in env else "f32"
            return generator.fill(expr["type"], kernel_name, None, None, size, expr["value"], dtype=result_dtype)

        return None, None

    def _generate_kernels_for_nodes(self, nodes, env, generator, scope_prefix=None):
        nodes = fuse_elementwise_chains(nodes, env)
        kernels = []
        cuda_code = ""
        flat_items = []

        for node in nodes:
            if node.get("type") in ("backward", "save", "rebind"):
                flat_items.append((None, node))
                continue

            if node.get("type") == "for":
                if scope_prefix:
                    nested_scope = f"{scope_prefix}_loop{self._loop_scope_counter}"
                else:
                    nested_scope = f"loop{self._loop_scope_counter}"
                self._loop_scope_counter += 1

                nested_kernels, nested_cuda, nested_flat = self._generate_kernels_for_nodes(
                    node["body"], env, generator, scope_prefix=nested_scope
                )

                node["_kernels"] = nested_kernels
                node["_flat_items"] = nested_flat
                cuda_code += nested_cuda
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
