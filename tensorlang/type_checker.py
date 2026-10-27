# File: tensorlang/type_checker.py

import numpy as np

from tensorlang.tensor_lang import TensorLang
from tensorlang.constants import OPTIMIZER_NAMES

tensorlang = TensorLang()

def _is_scalar_shape(shape):
    return len(shape) == 0 or all(d == 1 for d in shape)

def _shape_error(op, name, shape1, shape2, arg_names=None, hint=None):
    if arg_names and len(arg_names) == 2:
        operands = f"'{arg_names[0]}' {tuple(shape1)} vs '{arg_names[1]}' {tuple(shape2)}"
    else:
        operands = f"left {tuple(shape1)} vs right {tuple(shape2)}"
    msg = f"[TYPE CHECKER] error: {op}: shape mismatch computing '{name}': {operands}"
    if hint:
        msg += f". {hint}"
    return msg


_ELEMENTWISE_HINT = (
    "Shapes must match, one operand must be a scalar, or they must broadcast "
    "(trailing dimensions equal or 1)"
)
_MATRIX_VECTOR_HINT = (
    "A 1D operand combined with a 2D operand must have length equal to "
    "one of the 2D operand's dimensions"
)

def _resolve_tensor_type(env, tensor_name):
    t = env.get(tensor_name)
    if t is not None:
        return t
    if tensor_name.endswith("_grad"):
        base_name = tensor_name[:-5]
        base_t = env.get(base_name)
        if base_t is not None:
            return base_t.copy()
    return None

def _optimizer_arg_name(arg):
    if isinstance(arg, str):
        return arg
    if isinstance(arg, dict):
        if "name" in arg:
            return arg["name"]
        if arg.get("type") == "name":
            return arg["name"]
    return None

def _validate_optimizer_call(node_or_expr, env):
    args = node_or_expr.get("args", node_or_expr.get("arguments", []))
    label = node_or_expr.get("name", node_or_expr.get("type", "optimizer"))

    if len(args) < 3:
        tensorlang.print(message=f"[TYPE CHECKER] error: {label} requires param, grad, and lr")
        return False

    param_name = _optimizer_arg_name(args[0])
    grad_name = _optimizer_arg_name(args[1])

    if param_name is None or grad_name is None:
        tensorlang.print(message="[TYPE CHECKER] error: optimizer arguments must be tensor names")
        return False

    if param_name not in env:
        tensorlang.print(message=f"[TYPE CHECKER] error: undefined param '{param_name}'")
        return False

    if grad_name not in env and grad_name.endswith("_grad"):
        base_name = grad_name[:-5]
        if base_name in env:
            env[grad_name] = {
                "dtype": env[base_name]["dtype"],
                "shape": env[base_name]["shape"],
            }

    if grad_name not in env:
        tensorlang.print(message=f"[TYPE CHECKER] error: undefined grad '{grad_name}'")
        return False

    if env[param_name]["shape"] != env[grad_name]["shape"]:
        tensorlang.print(message=(
            f"[TYPE CHECKER] error: optimizer shape mismatch: "
            f"'{param_name}' {tuple(env[param_name]['shape'])} vs "
            f"'{grad_name}' {tuple(env[grad_name]['shape'])}"
        ))
        return False

    return True

def type_checker(ast, env, DEBUG_INFO=False, DEBUG_MODE=False):
    # ================================================================
    # PASS 1: Type check all statements
    # ================================================================
    for node in ast:
        if node.get("type") == "rebind":
            continue

        if node.get("type") == "for":
            success, env = type_checker(node["body"], env, DEBUG_INFO, DEBUG_MODE)
            if not success:
                return False, env
            continue

        if node.get("type") in OPTIMIZER_NAMES or (
            node.get("type") == "call" and node.get("name") in OPTIMIZER_NAMES
        ):
            if not _validate_optimizer_call(node, env):
                return False, env
            continue

        if node.get("type") == "save":
            tensor_name = node.get("tensor")
            if tensor_name not in env:
                tensorlang.print(message=f"[TYPE CHECKER] Error: Cannot save undefined tensor '{tensor_name}'")
                return False, env
            if DEBUG_INFO:
                print(f"[INFO] Save statement validated for {tensor_name}")
            continue

        if node.get("type") == "backward":
            continue

        if node.get("type") != "let":
            if DEBUG_MODE:
                tensorlang.print(message=f"[TYPE CHECKER] Skipping node type: {node.get('type')}")
            continue

        name = node["name"]
        expr = node["expr"]
        ty = node.get("ty")

        arg_names = expr.get("args", []) if isinstance(expr, dict) else []
        args = [env.get(a) for a in arg_names if a in env]

        if DEBUG_MODE:
            tensorlang.print(message=f"[TYPE CHECKER] Arg Names: {arg_names}")
            tensorlang.print(message=f"[TYPE CHECKER] Args: {args}")

        # Annotated type: trust the declared type.
        if ty and isinstance(ty, dict) and "dtype" in ty:
            shape = ty["shape"]
            if any(isinstance(dim, str) for dim in shape):
                if DEBUG_INFO:
                    tensorlang.print(message=f"[TYPE CHECKER] Generic type for {name}: {ty}")
            else:
                env[name] = {"dtype": ty["dtype"], "shape": ty["shape"]}
                if DEBUG_INFO:
                    tensorlang.print(message=f"[TYPE CHECKER] Tensor {name} {env[name]}")
            continue

        # Tensor literal.
        if isinstance(expr, dict) and expr.get("type") == "tensor_literal":
            data = expr["data"]
            num_elements = len(data)
            if expr.get("is_1d"):
                shape = (num_elements,)
            else:
                rows = sum(1 for child in expr["tree"].children if child.data == "inner_array")
                cols = num_elements // rows if rows > 0 else num_elements
                shape = (rows, cols)
            env[name] = {"dtype": "f32", "shape": shape}
            if DEBUG_MODE:
                tensorlang.print(message=f"[TYPE CHECKER] Inferred shape for {name}: {env[name]['shape']}")
            continue

        # Name alias.
        if isinstance(expr, dict) and expr.get("type") == "name":
            ref_name = expr["name"]
            if ref_name not in env:
                tensorlang.print(message=f"[TYPE CHECKER] Type error: Undefined tensor {ref_name} referenced by {name}")
                return False, env
            env[name] = {"dtype": env[ref_name]["dtype"], "shape": env[ref_name]["shape"]}
            if DEBUG_INFO:
                print(f"[INFO] Type {name} assigned from {ref_name}: {env[name]}")
            continue

        # Optimizer call used as a let-binding.
        if isinstance(expr, dict) and expr.get("type") in OPTIMIZER_NAMES:
            if not _validate_optimizer_call(expr, env):
                return False, env
            continue

        # Operation expressions.
        if isinstance(expr, dict) and expr.get("type") in [
            "matmul", "add", "minus", "mult", "div",
            "relu", "sigmoid", "tanh", "gelu", "swish", "softmax",
            "fill", "load", "save",
            "sum", "mean", "max", "min", "argmax", "argmin",
            "greater", "less", "equal",
            "linear", "layer_norm", "batch_norm", "instance_norm",
            "cross_entropy", "mse_loss",
            "transpose", "reshape", "concat", "slice",
            "sgd", "adam", "sgd_update", "adam_update",
        ]:
            if expr["type"] == "fill":
                env[name] = {"dtype": "f32", "shape": expr["shape"]}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned from fill: {env[name]}")
                continue

            if expr["type"] == "load":
                file_path = expr["file_path"]

                if ty:
                    env[name] = {"dtype": ty["dtype"], "shape": ty["shape"]}
                    if DEBUG_MODE:
                        tensorlang.print(message=f"[TYPE CHECKER] {name}: explicit shape from declaration: {env[name]}")
                    continue

                try:
                    from pathlib import Path

                    path = Path(file_path)
                    suffix = path.suffix.lower()

                    if suffix == ".npy":
                        loaded_data = np.load(file_path)
                        inferred_shape = tuple(loaded_data.shape)

                    elif suffix == ".npz":
                        npz_file = np.load(file_path)
                        if "data" in npz_file.files:
                            data_array = npz_file["data"]
                        elif len(npz_file.files) > 0:
                            data_array = npz_file[npz_file.files[0]]
                            if DEBUG_MODE:
                                tensorlang.print(message=f"[TYPE CHECKER] NPZ: Using array '{npz_file.files[0]}' (no 'data' array found)")
                        else:
                            npz_file.close()
                            tensorlang.print(message=f"[TYPE CHECKER] error: NPZ file has no arrays: {file_path}")
                            return False, env

                        inferred_shape = tuple(data_array.shape)
                        npz_file.close()

                    elif suffix == ".csv":
                        try:
                            import pandas as pd
                        except ImportError:
                            tensorlang.print(message="[TYPE CHECKER] error: pandas required for CSV loading")
                            return False, env

                        df = pd.read_csv(file_path)
                        first_col = df.columns[0]
                        if first_col.lower() in ["date", "time", "datetime", "index", "id"]:
                            df = df.drop(columns=[first_col])
                        inferred_shape = tuple(df.shape)

                    else:
                        tensorlang.print(message=f"[TYPE CHECKER] error: Unsupported file format: {suffix}")
                        return False, env

                    env[name] = {"dtype": "f32", "shape": inferred_shape}
                    if DEBUG_MODE:
                        tensorlang.print(message=f"[TYPE CHECKER] {name}: inferred shape {inferred_shape} from {suffix.upper()} file")

                except FileNotFoundError:
                    tensorlang.print(message=f"[TYPE CHECKER] error: File not found during type inference: {file_path}")
                    return False, env
                except (OSError, UnicodeError, ValueError) as e:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Could not infer shape from {file_path}: {e}")
                    return False, env
                continue

            if expr["type"] in ("sum", "mean"):
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    print(f"Type error: Undefined tensor {tensor_name} for {expr['type']}")
                    return False, env

                input_shape = tensor_type["shape"]
                axis = expr.get("axis")
                if axis is None:
                    output_shape = ()
                else:
                    if axis < 0 or axis >= len(input_shape):
                        print(f"Type error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                        return False, env
                    output_shape = tuple(dim for i, dim in enumerate(input_shape) if i != axis)
                    if not output_shape:
                        output_shape = ()

                env[name] = {"dtype": "f32", "shape": output_shape}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "softmax":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined tensor {tensor_name} for {expr['type']}")
                    return False, env
                env[name] = {"dtype": "f32", "shape": tensor_type["shape"]}
                if DEBUG_INFO:
                    print(f"[INFO] Tensor {tensor_name} {name}:({expr['type']}):{env[name]}")
                continue

            if expr["type"] in ("relu", "sigmoid", "tanh", "gelu", "swish"):
                arg_name = expr["args"][0]
                arg_type = _resolve_tensor_type(env, arg_name)
                if arg_type is None:
                    print(f"Type error: Undefined tensor {arg_name} for {expr['type']}")
                    return False, env
                env[name] = {"dtype": "f32", "shape": arg_type["shape"]}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} for {name}: {env[name]}")
                continue

            if expr["type"] == "slice":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined tensor {tensor_name} for {expr['type']}")
                    return False, env

                input_shape = tensor_type["shape"]
                slice_specs = expr["specs"]

                output_shape = []
                for i, spec in enumerate(slice_specs):
                    if i >= len(input_shape):
                        tensorlang.print(message=f"[TYPE CHECKER] error: Slice dimension {i} exceeds tensor dimensions {len(input_shape)}")
                        return False, env

                    dim_size = input_shape[i]

                    if spec["type"] == "index":
                        continue
                    if spec["type"] == "full_slice":
                        output_shape.append(dim_size)
                    elif spec["type"] == "slice":
                        start = spec.get("start", 0)
                        end = spec.get("end", dim_size)
                        if start is None:
                            start = 0
                        if end is None:
                            end = dim_size
                        if start < 0:
                            start = max(0, dim_size + start)
                        if end < 0:
                            end = max(0, dim_size + end)
                        start = min(start, dim_size)
                        end = min(end, dim_size)
                        output_shape.append(max(0, end - start))

                for i in range(len(slice_specs), len(input_shape)):
                    output_shape.append(input_shape[i])

                env[name] = {"dtype": "f32", "shape": tuple(output_shape) if output_shape else ()}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "layer_norm":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined tensor {tensor_name} for layer_norm")
                    return False, env

                input_shape = tensor_type["shape"]
                axis = expr.get("axis")
                if axis is None:
                    axis = len(input_shape) - 1
                if axis < 0 or axis >= len(input_shape):
                    tensorlang.print(message=f"[TYPE CHECKER] error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                    return False, env

                env[name] = {"dtype": "f32", "shape": input_shape}
                if DEBUG_MODE:
                    print(f"Type {expr['type']} Assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "batch_norm":
                tensor_name = expr["tensor"]
                running_mean_name = expr["running_mean"]
                running_var_name = expr["running_var"]

                tensor_type = _resolve_tensor_type(env, tensor_name)
                mean_type = _resolve_tensor_type(env, running_mean_name)
                var_type = _resolve_tensor_type(env, running_var_name)

                if tensor_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined tensor {tensor_name} for batch_norm")
                    return False, env
                if mean_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined running_mean {running_mean_name} for batch_norm")
                    return False, env
                if var_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined running_var {running_var_name} for batch_norm")
                    return False, env

                input_shape = tensor_type["shape"]
                mean_shape = mean_type["shape"]
                var_shape = var_type["shape"]

                if len(input_shape) < 2:
                    tensorlang.print(message="[TYPE CHECKER] error: BatchNorm input must be at least 2D")
                    return False, env

                num_features = input_shape[1]
                expected_stats_shape = (num_features,)

                if mean_shape != expected_stats_shape:
                    tensorlang.print(message=f"[TYPE CHECKER] error: BatchNorm running_mean shape {mean_shape} != expected {expected_stats_shape}")
                    return False, env
                if var_shape != expected_stats_shape:
                    tensorlang.print(message=f"[TYPE CHECKER] error: BatchNorm running_var shape {var_shape} != expected {expected_stats_shape}")
                    return False, env

                env[name] = {"dtype": "f32", "shape": input_shape}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "instance_norm":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined tensor {tensor_name} for instance_norm")
                    return False, env
                env[name] = {"dtype": "f32", "shape": tensor_type["shape"]}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] in ("max", "min", "argmax", "argmin"):
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined tensor {tensor_name} for {expr['type']}")
                    return False, env

                input_shape = tensor_type["shape"]
                axis = expr.get("axis")
                if axis is None:
                    output_shape = ()
                else:
                    if axis < 0 or axis >= len(input_shape):
                        tensorlang.print(message=f"[TYPE CHECKER] error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                        return False, env
                    output_shape = tuple(dim for i, dim in enumerate(input_shape) if i != axis)
                    if not output_shape:
                        output_shape = ()

                env[name] = {"dtype": "f32", "shape": output_shape}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "transpose":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined tensor {tensor_name} for transpose")
                    return False, env

                input_shape = tensor_type["shape"]
                axes = expr.get("axes")
                if axes is None:
                    output_shape = tuple(reversed(input_shape))
                else:
                    if len(axes) != len(input_shape):
                        tensorlang.print(message="[TYPE CHECKER] error: Transpose axes length mismatch")
                        return False, env
                    if set(axes) != set(range(len(input_shape))):
                        tensorlang.print(message="[TYPE CHECKER] error: Transpose axes must be permutation")
                        return False, env
                    output_shape = tuple(input_shape[i] for i in axes)

                env[name] = {"dtype": "f32", "shape": output_shape}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "reshape":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined tensor {tensor_name} for reshape")
                    return False, env

                input_shape = tensor_type["shape"]
                new_shape = expr["new_shape"]

                input_elements = int(np.prod([int(dim) for dim in input_shape])) if input_shape else 1
                new_elements = int(np.prod([int(dim) for dim in new_shape])) if new_shape else 1

                if input_elements != new_elements:
                    tensorlang.print(message="[TYPE CHECKER] error: Reshape element count mismatch")
                    return False, env

                env[name] = {"dtype": "f32", "shape": tuple(new_shape)}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "concat":
                tensor_names = expr["tensors"]
                axis = expr["axis"]

                tensor_types = []
                for tensor_name in tensor_names:
                    tensor_type = _resolve_tensor_type(env, tensor_name)
                    if tensor_type is None:
                        tensorlang.print(message=f"[TYPE CHECKER] error: Undefined tensor {tensor_name} for concat")
                        return False, env
                    tensor_types.append(tensor_type)

                first_shape = tensor_types[0]["shape"]
                if axis < 0 or axis >= len(first_shape):
                    tensorlang.print(message="[TYPE CHECKER] error: Concat axis out of bounds")
                    return False, env

                concat_dim_size = 0
                for tensor_type in tensor_types:
                    shape = tensor_type["shape"]
                    if len(shape) != len(first_shape):
                        tensorlang.print(message="[TYPE CHECKER] error: Concat tensor dimension count mismatch")
                        return False, env
                    for dim_idx, (dim1, dim2) in enumerate(zip(first_shape, shape)):
                        if dim_idx != axis and dim1 != dim2:
                            tensorlang.print(message="[TYPE CHECKER] error: Concat tensor shape mismatch")
                            return False, env
                    concat_dim_size += shape[axis]

                output_shape = list(first_shape)
                output_shape[axis] = concat_dim_size
                env[name] = {"dtype": "f32", "shape": tuple(output_shape)}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            # Generic argument-based operations.
            arg_names = expr.get("args", [])
            if arg_names:
                args = []
                missing_args = []

                for arg_name in arg_names:
                    arg_type = env.get(arg_name)
                    if arg_type is None and arg_name.endswith("_grad"):
                        base_name = arg_name[:-5]
                        if base_name in env:
                            arg_type = env[base_name].copy()
                            if DEBUG_INFO:
                                print(f"[INFO] Inferred type for {arg_name} from {base_name}: {arg_type}")
                    if arg_type is None:
                        missing_args.append(arg_name)
                    args.append(arg_type)

                if DEBUG_INFO:
                    print(f"[INFO] Checking args for {name}: {args}")

                if missing_args:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Undefined args for {name}: {missing_args}")
                    return False, env

                if expr["type"] == "matmul":
                    if args[0]["shape"][1] != args[1]["shape"][0]:
                        tensorlang.print(message=_shape_error(
                            "matmul", name, args[0]["shape"], args[1]["shape"],
                            arg_names,
                            hint=(f"Inner dimensions must match: left has {args[0]['shape'][1]} "
                                  f"columns but right has {args[1]['shape'][0]} rows")))
                        return False, env
                    env[name] = {"dtype": "f32", "shape": (args[0]["shape"][0], args[1]["shape"][1])}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                    continue

                if expr["type"] in ("add", "minus", "mult", "div"):
                    shape1, shape2 = args[0]["shape"], args[1]["shape"]

                    if shape1 == shape2:
                        output_shape = shape1
                    elif _is_scalar_shape(shape1):
                        output_shape = shape2
                    elif _is_scalar_shape(shape2):
                        output_shape = shape1
                    elif len(shape1) == 2 and len(shape2) == 1:
                        if shape1[0] == shape2[0] or shape1[1] == shape2[0]:
                            output_shape = shape1
                        else:
                            tensorlang.print(message=_shape_error(
                                expr["type"], name, shape1, shape2, arg_names,
                                hint=_MATRIX_VECTOR_HINT))
                            return False, env
                    elif len(shape2) == 2 and len(shape1) == 1:
                        if shape2[0] == shape1[0] or shape2[1] == shape1[0]:
                            output_shape = shape2
                        else:
                            tensorlang.print(message=_shape_error(
                                expr["type"], name, shape1, shape2, arg_names,
                                hint=_MATRIX_VECTOR_HINT))
                            return False, env
                    else:
                        ndim = max(len(shape1), len(shape2))
                        padded_shape1 = (1,) * (ndim - len(shape1)) + shape1
                        padded_shape2 = (1,) * (ndim - len(shape2)) + shape2

                        output_shape = []
                        for d1, d2 in zip(padded_shape1, padded_shape2):
                            if d1 == d2:
                                output_shape.append(d1)
                            elif d1 == 1:
                                output_shape.append(d2)
                            elif d2 == 1:
                                output_shape.append(d1)
                            else:
                                tensorlang.print(message=_shape_error(
                                    expr["type"], name, shape1, shape2, arg_names,
                                    hint=_ELEMENTWISE_HINT))
                                return False, env
                        output_shape = tuple(output_shape)

                    env[name] = {"dtype": "f32", "shape": output_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                    continue

                if expr["type"] in ("greater", "less", "equal"):
                    shape1, shape2 = args[0]["shape"], args[1]["shape"]
                    orig_shape1, orig_shape2 = shape1, shape2
                    if len(shape1) < len(shape2):
                        shape1, shape2 = shape2, shape1

                    if len(shape2) == 0:
                        output_shape = shape1
                    else:
                        output_shape = []
                        for d1, d2 in zip(shape1[-len(shape2):], shape2):
                            if d1 == d2 or d2 == 1:
                                output_shape.append(d1)
                            else:
                                tensorlang.print(message=_shape_error(
                                    expr["type"], name, orig_shape1, orig_shape2, arg_names,
                                    hint=_ELEMENTWISE_HINT))
                                return False, env
                        output_shape = shape1[:-len(shape2)] + tuple(output_shape)

                    env[name] = {"dtype": "f32", "shape": output_shape}
                    if DEBUG_INFO:
                        tensorlang.print(message=f"[TYPE CHECKER] Type {expr['type']} assigned for {name}: {env[name]}")
                    continue

                if expr["type"] == "linear":
                    if len(args) != 3:
                        tensorlang.print(message="[TYPE CHECKER] error: Linear requires exactly 3 arguments")
                        return False, env

                    input_tensor, weight_tensor, bias_tensor = args
                    input_shape = input_tensor["shape"]
                    weight_shape = weight_tensor["shape"]
                    bias_shape = bias_tensor["shape"]

                    if len(input_shape) == 1:
                        in_features = input_shape[0]
                        if len(weight_shape) != 2 or weight_shape[0] != in_features:
                            tensorlang.print(message="[TYPE CHECKER] error: Linear weight shape mismatch")
                            return False, env
                        out_features = weight_shape[1]
                        if len(bias_shape) != 1 or bias_shape[0] != out_features:
                            tensorlang.print(message="[TYPE CHECKER] error: Linear bias shape mismatch")
                            return False, env
                        output_shape = (out_features,)

                    elif len(input_shape) == 2:
                        batch_size, in_features = input_shape
                        if len(weight_shape) != 2 or weight_shape[0] != in_features:
                            tensorlang.print(message="[TYPE CHECKER] error: Linear weight shape mismatch")
                            return False, env
                        out_features = weight_shape[1]
                        if len(bias_shape) != 1 or bias_shape[0] != out_features:
                            tensorlang.print(message="[TYPE CHECKER] error: Linear bias shape mismatch")
                            return False, env
                        output_shape = (batch_size, out_features)

                    else:
                        tensorlang.print(message="[TYPE CHECKER] error: Linear input must be 1D or 2D tensor")
                        return False, env

                    env[name] = {"dtype": "f32", "shape": output_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                    continue

                if expr["type"] in ("cross_entropy", "mse_loss"):
                    if len(args) != 2:
                        tensorlang.print(message=f"[TYPE CHECKER] error: {expr['type']} requires exactly 2 arguments")
                        return False, env

                    predictions_tensor, targets_tensor = args
                    pred_shape = predictions_tensor["shape"]
                    target_shape = targets_tensor["shape"]

                    if expr["type"] == "cross_entropy":
                        if len(pred_shape) != 2:
                            tensorlang.print(message="[TYPE CHECKER] error: Cross entropy predictions must be 2D")
                            return False, env
                        batch_size, _num_classes = pred_shape
                        if not (
                            (len(target_shape) == 1 and target_shape[0] == batch_size)
                            or (len(target_shape) == 2 and target_shape == pred_shape)
                        ):
                            tensorlang.print(message="[TYPE CHECKER] error: Cross entropy target shape incompatible")
                            return False, env
                        output_shape = ()

                    else:
                        if pred_shape != target_shape:
                            tensorlang.print(message="[TYPE CHECKER] error: MSE loss shape mismatch")
                            return False, env
                        output_shape = ()

                    env[name] = {"dtype": "f32", "shape": output_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                    continue

                tensorlang.print(message=f"[TYPE CHECKER] error: Unhandled operation '{expr['type']}' for {name}")
                return False, env

            tensorlang.print(message=f"[TYPE CHECKER] error: Unrecognized expr type for {name}: {expr.get('type') if isinstance(expr, dict) else expr}")
            return False, env

    # ================================================================
    # PASS 2: Register gradient types for backward statements
    # ================================================================
    if DEBUG_MODE:
        tensorlang.print(message=f"[TYPE CHECKER] Starting PASS 2, AST has {len(ast)} nodes")

    backward_count = 0
    for node in ast:
        if node.get("type") != "backward":
            continue

        backward_count += 1
        loss_name = node.get("loss_tensor")

        if DEBUG_MODE:
            tensorlang.print(message=f"[TYPE CHECKER] Found backward statement for '{loss_name}'")

        requires_grad_count = 0
        for prev_node in ast:
            if prev_node.get("type") != "let":
                continue

            if not prev_node.get("requires_grad", False):
                continue

            requires_grad_count += 1
            tensor_name = prev_node["name"]

            if DEBUG_MODE:
                tensorlang.print(message=f"[TYPE CHECKER] Found requires_grad tensor: '{tensor_name}'")

            if tensor_name in env:
                grad_name = f"{tensor_name}_grad"
                env[grad_name] = {
                    "dtype": env[tensor_name]["dtype"],
                    "shape": env[tensor_name]["shape"],
                }
                if DEBUG_MODE:
                    tensorlang.print(message=f"[TYPE CHECKER] Registered gradient type: {grad_name}: {env[grad_name]}")
            else:
                tensorlang.print(message=f"[TYPE CHECKER] ERROR: '{tensor_name}' not in env!")

        if DEBUG_MODE:
            tensorlang.print(message=f"[TYPE CHECKER] Found {requires_grad_count} requires_grad tensors")

    if DEBUG_MODE:
        tensorlang.print(message=f"[TYPE CHECKER] Found {backward_count} backward statements")
        tensorlang.print(message=f"[TYPE CHECKER] Final environment keys: {list(env.keys())}")

    return True, env
