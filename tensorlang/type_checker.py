"""
TensorLang Type Checker

File: tensorlang/type_checker.py
"""

import copy
import copy
import difflib
import numpy as np

from tensorlang.tensor_lang import TensorLang
from tensorlang.constants import OPTIMIZER_NAMES, promote_dtype

tensorlang = TensorLang()

def _is_scalar_shape(shape):
    return len(shape) == 0 or all(d == 1 for d in shape)

def _normalize_axis(axis, ndim):
    if axis < 0:
        axis += ndim
    return axis

_SYNTHETIC_PREFIXES = ("__tmp_", "__if_cond_")


def _is_synthetic(name):
    """Compiler-generated temporaries (flattened nested calls, condition results)."""
    return isinstance(name, str) and name.startswith(_SYNTHETIC_PREFIXES)


def _name_label(name):
    """'x' for a user name; a readable phrase for a compiler temporary."""
    if _is_synthetic(name):
        return f"a nested expression ('{name}')"
    return f"'{name}'"


def _render_expr(expr):
    """Compact source-like text for an expression node, for error messages."""
    if not isinstance(expr, dict):
        return str(expr)
    t = expr.get("type")
    if t == "name":
        return str(expr.get("name"))
    if t == "number":
        return str(expr.get("value"))
    if t == "fill":
        return f"fill({expr.get('value')}, {tuple(expr.get('shape', ()))})"
    if t == "load":
        return f'load("{expr.get("file_path")}")'
    if t == "tensor_literal":
        return "[...]"
    args = expr.get("args")
    if isinstance(args, list):
        return f"{t}({', '.join(_render_expr(a) for a in args)})"
    if "tensor" in expr:
        return f"{t}({expr['tensor']})"
    return f"{t}(...)"


def _describe_binding(name, expr):
    """How an error refers to the let being checked."""
    if _is_synthetic(name):
        return f"the nested expression `{_render_expr(expr)}`"
    if expr is None:
        return f"'{name}'"
    return f"`let {name} = {_render_expr(expr)}`"


def _fmt_type(entry):
    return f"{entry.get('dtype')} {tuple(entry.get('shape', ()))}"


def _undefined_message(missing, env, name=None, expr=None, where=None):
    """[E-UNDEFINED-TENSOR]: say which expression uses what, and suggest close names."""
    missing = list(missing)
    shown = ", ".join(f"'{m}'" for m in missing)
    verb = "is" if len(missing) == 1 else "are"
    subject = where if where is not None else _describe_binding(name, expr)
    msg = (f"[TYPE CHECKER] error: [E-UNDEFINED-TENSOR] {subject} uses {shown}, "
           f"which {verb} not defined at this point.")
    pool = [n for n in env if not _is_synthetic(n)]
    hints = []
    for m in missing:
        close = difflib.get_close_matches(str(m), pool, n=3, cutoff=0.6)
        if close:
            hints.append(f"Did you mean {' or '.join(repr(c) for c in close)} instead of '{m}'?")
    if hints:
        msg += " " + " ".join(hints)
    return msg


def _scalar_literal_message(name, expr):
    value = expr.get("value")
    if _is_synthetic(name):
        return (f"[TYPE CHECKER] error: [E-SCALAR-LITERAL] the numeric literal {value} cannot be "
                f"used as an operand here. Declare a tensor instead, e.g. "
                f"`let c: Tensor[f32, (1,)] = [{value}]`.")
    return (f"[TYPE CHECKER] error: [E-SCALAR-LITERAL] '{name}' is bound to the bare number "
            f"{value}, but TensorLang has no scalar values. Declare a tensor instead, e.g. "
            f"`let {name}: Tensor[f32, (1,)] = [{value}]` (or fill({value}, <shape>)).")


def _has_zero_dim(shape):
    try:
        return any(int(d) == 0 for d in shape)
    except (TypeError, ValueError):
        return False


def _empty_condition_message(text, name, shape):
    return (f"[TYPE CHECKER] error: [E-CONDITION-EMPTY] the condition `{text}` involves an empty "
            f"tensor ({_name_label(name)} has shape {tuple(shape)}). A condition needs at least "
            f"one element.")


def _shape_error(op, name, shape1, shape2, arg_names=None, hint=None):
    if arg_names and len(arg_names) == 2:
        operands = (f"{_name_label(arg_names[0])} {tuple(shape1)} vs "
                    f"{_name_label(arg_names[1])} {tuple(shape2)}")
    else:
        operands = f"left {tuple(shape1)} vs right {tuple(shape2)}"
    msg = f"[TYPE CHECKER] error: [E-SHAPE-MISMATCH] {op}: shape mismatch computing {_name_label(name)}: {operands}"
    if hint:
        msg += f". {hint}"
    return msg


_ELEMENTWISE_HINT = (
    "Shapes must match, one operand must be a scalar, or they must broadcast "
    "(trailing dimensions equal or 1)"
)
_MATRIX_VECTOR_HINT = (
    "A 1D operand combined with an N-D operand must have length equal to "
    "one of the N-D operand's dimensions"
)


_COMPARISON_OPS = ("greater", "less", "equal", "greater_equal", "less_equal", "not_equal")
_COMPARISON_HINT = (
    "comparison kernels support equal shapes, or a (rows, cols) tensor against a (cols,) "
    "or (1, cols) row"
)


_COMPARISON_DTYPE_HINT = (
    "comparison kernels only support f32 data (f64 is stored as f32), not f16; "
    "a numeric literal is f32, so comparing an f16 tensor with a literal is rejected too"
)


def _comparison_has_f16(dtype1, dtype2):
    return "f16" in (dtype1, dtype2)


def _comparison_supported(shape1, shape2):
    """True if the CUDA comparison kernels can run these operand shapes.

    Shapes with non-integer (generic) dimensions are not judged here.
    """
    try:
        s1 = tuple(int(d) for d in shape1)
        s2 = tuple(int(d) for d in shape2)
    except (TypeError, ValueError):
        return True
    if s1 == s2:
        return True
    if len(s1) == 2 and len(s2) in (1, 2):
        cols = s1[1]
        return s2 == (cols,) or s2 == (1, cols)
    return False


_UNSUPPORTED_DTYPE_HINT = (
    "f64 has no CUDA kernel support yet (operations raise a KeyError or read it as f32); "
    "declare the tensor as f32"
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
    def _count_backward(nodes):
        count = 0
        for n in nodes:
            if n.get("type") == "backward":
                count += 1
            if n.get("type") == "for":
                count += _count_backward(n.get("body", []))
            if n.get("type") == "if":
                count += _count_backward(n.get("then", []))
                for e in n.get("elifs", []):
                    count += _count_backward(e.get("body", []))
                if n.get("else") is not None:
                    count += _count_backward(n.get("else", []))
        return count

    # ================================================================
    # PASS 1: Type check all statements
    # ================================================================
    for node in ast:
        if node.get("type") == "if":
            for synth in list(node.get("_condition_synthetics", [])) + [
                s for e in node.get("elifs", []) for s in e.get("_condition_synthetics", [])
            ]:
                success, env = type_checker([synth], env, DEBUG_INFO, DEBUG_MODE)
                if not success:
                    return False, env

            for cond_node in [node] + node.get("elifs", []):
                cond = cond_node.get("condition")
                if isinstance(cond, dict) and cond.get("type") == "bool":
                    cond_node["_condition_constant"] = bool(cond.get("value"))
                    cond_node["_condition_shape"] = (1,)
                elif isinstance(cond, dict) and cond.get("type") == "name":
                    cond_name = cond["name"]
                    if cond_name not in env:
                        tensorlang.print(message=_undefined_message([cond_name], env, where="the `if`/`elif` condition"))
                        return False, env
                    shape = env[cond_name]["shape"]
                    if _has_zero_dim(shape):
                        tensorlang.print(message=_empty_condition_message(cond_name, cond_name, shape))
                        return False, env
                    cond_node["_condition_text"] = cond_name
                    # Tensor-valued conditions are allowed. Truthiness is all-nonzero.
                    cond_node["_condition_name"] = cond_name
                    cond_node["_condition_shape"] = shape
                elif isinstance(cond, dict) and cond.get("type") in ("any", "all"):
                    tensor_name = cond["tensor"]
                    tensor_type = _resolve_tensor_type(env, tensor_name)
                    if tensor_type is None:
                        tensorlang.print(message=_undefined_message([tensor_name], env, where=f"the `{cond['type']}({tensor_name})` condition"))
                        return False, env
                    if _has_zero_dim(tensor_type.get("shape", ())):
                        tensorlang.print(message=_empty_condition_message(
                            f"{cond['type']}({tensor_name})", tensor_name, tensor_type.get("shape", ())))
                        return False, env
                    cond_node["_condition_text"] = f"{cond['type']}({tensor_name})"
                    synth_name = f"__if_cond_{len(env)}"
                    env[synth_name] = {"dtype": "f32", "shape": ()}
                    cond_node["_condition_name"] = synth_name
                    cond_node["_condition_shape"] = ()
                    cond_node["_condition_expr"] = {"type": cond["type"], "tensor": tensor_name}

                elif isinstance(cond, dict) and cond.get("type") in ("any", "all"):
                    tensor_name = cond["tensor"]
                    tensor_type = _resolve_tensor_type(env, tensor_name)
                    if tensor_type is None:
                        tensorlang.print(message=_undefined_message([tensor_name], env, where=f"the `{cond['type']}({tensor_name})` condition"))
                        return False, env
                    synth_name = f"__if_cond_{len(env)}"
                    env[synth_name] = {"dtype": "f32", "shape": ()}
                    cond_node["_condition_name"] = synth_name
                    cond_node["_condition_shape"] = ()
                    cond_node["_condition_expr"] = {"type": cond["type"], "tensor": tensor_name}

                elif isinstance(cond, dict) and cond.get("type") == "comparison":
                    left = cond.get("left")
                    right = cond.get("right")
                    if not (isinstance(left, dict) and left.get("type") == "name"):
                        tensorlang.print(message="[TYPE CHECKER] error: [E-CONDITION-OPERAND] the left side of an if/elif comparison must be a tensor name or a numeric literal")
                        return False, env
                    if not (isinstance(right, dict) and right.get("type") == "name"):
                        tensorlang.print(message="[TYPE CHECKER] error: [E-CONDITION-OPERAND] the right side of an if/elif comparison must be a tensor name or a numeric literal")
                        return False, env
                    left_name = left["name"]
                    right_name = right["name"]
                    if left_name not in env or right_name not in env:
                        tensorlang.print(message=_undefined_message(
                            [n for n in (left_name, right_name) if n not in env], env,
                            where=f"the condition `{_name_label(left_name) if _is_synthetic(left_name) else left_name} {cond.get('op')} {_name_label(right_name) if _is_synthetic(right_name) else right_name}`"))
                        return False, env
                    op = cond.get("op")
                    if op not in (">", "<", "==", ">=", "<=", "!="):
                        tensorlang.print(message=f"[TYPE CHECKER] error: Unsupported if comparison operator '{op}'")
                        return False, env
                    if not _comparison_supported(env[left_name]["shape"], env[right_name]["shape"]):
                        tensorlang.print(message=f"[TYPE CHECKER] error: [E-SHAPE-MISMATCH] if/elif condition: cannot compare {_name_label(left_name)} {tuple(env[left_name]['shape'])} with {_name_label(right_name)} {tuple(env[right_name]['shape'])}. {_COMPARISON_HINT}")
                        return False, env
                    if _comparison_has_f16(env[left_name]["dtype"], env[right_name]["dtype"]):
                        tensorlang.print(message=(
                            f"[TYPE CHECKER] error: [E-COMPARISON-DTYPE] if/elif condition: cannot compare "
                            f"{_name_label(left_name)} ({env[left_name]['dtype']}) with "
                            f"{_name_label(right_name)} ({env[right_name]['dtype']}). {_COMPARISON_DTYPE_HINT}"))
                        return False, env
                    try:
                        out_shape = tuple(np.broadcast_shapes(env[left_name]["shape"], env[right_name]["shape"]))
                    except ValueError:
                        tensorlang.print(message=f"[TYPE CHECKER] error: [E-SHAPE-MISMATCH] if/elif condition: cannot compare {_name_label(left_name)} {tuple(env[left_name]['shape'])} with {_name_label(right_name)} {tuple(env[right_name]['shape'])}")
                        return False, env
                    left_text = cond.get("left_text") or ("<expression>" if _is_synthetic(left_name) else left_name)
                    right_text = cond.get("right_text") or ("<expression>" if _is_synthetic(right_name) else right_name)
                    cond_text = f"{left_text} {op} {right_text}"
                    for operand in (left_name, right_name):
                        if _has_zero_dim(env[operand]["shape"]):
                            tensorlang.print(message=_empty_condition_message(
                                cond_text, operand, env[operand]["shape"]))
                            return False, env
                    cond_node["_condition_text"] = cond_text
                    # Tensor-valued conditions are allowed. Truthiness is all-nonzero.
                    synth_name = f"__if_cond_{len(env)}"
                    env[synth_name] = {"dtype": "f32", "shape": out_shape}
                    cond_node["_condition_name"] = synth_name
                    cond_node["_condition_shape"] = out_shape
                    cond_node["_condition_expr"] = {
                        "type": {">": "greater", "<": "less", "==": "equal",
                                 ">=": "greater_equal", "<=": "less_equal",
                                 "!=": "not_equal"}[op],
                        "args": [left_name, right_name],
                    }
                else:
                    tensorlang.print(message="[TYPE CHECKER] error: Unsupported if condition")
                    return False, env

            branch_specs = [node.get("then", [])] + [e.get("body", []) for e in node.get("elifs", [])]
            if node.get("else") is not None:
                branch_specs.append(node.get("else"))

            branch_labels = ["the `if` branch"]
            branch_labels += [f"`elif` branch {i + 1}" for i in range(len(node.get("elifs", [])))]
            if node.get("else") is not None:
                branch_labels.append("the `else` branch")

            branch_envs = []
            for branch in branch_specs:
                branch_env = copy.deepcopy(env)
                success, branch_env = type_checker(branch, branch_env, DEBUG_INFO, DEBUG_MODE)
                if not success:
                    return False, env
                branch_envs.append(branch_env)

            # Enforce matching shapes/dtypes for names appearing in multiple branches.
            all_names = set()
            for branch_env in branch_envs:
                all_names.update(branch_env.keys())
            for name in all_names:
                present = [(branch_labels[i], branch_env[name])
                           for i, branch_env in enumerate(branch_envs) if name in branch_env]
                if len(present) > 1:
                    first_label, first = present[0]
                    for other_label, other in present[1:]:
                        if (other.get("dtype") != first.get("dtype")
                                or tuple(other.get("shape", ())) != tuple(first.get("shape", ()))):
                            def _how(label, entry):
                                verb = "leaves it as" if env.get(name) == entry else "defines it as"
                                return f"{label} {verb} {_fmt_type(entry)}"
                            tensorlang.print(
                                message=f"[TYPE CHECKER] error: [E-BRANCH-MISMATCH] branch shape/dtype mismatch for "
                                        f"'{name}': {_how(first_label, first)}, but {_how(other_label, other)}. "
                                        f"A name used after an if/elif/else must have the same shape and dtype "
                                        f"in every branch that sets it."
                            )
                            return False, env

            # Merge branch results into the enclosing environment.
            for name in all_names:
                for branch_env in branch_envs:
                    if name in branch_env:
                        env[name] = branch_env[name]
                        break
            continue

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
        if isinstance(expr, dict) and expr.get("type") == "number":
            tensorlang.print(message=_scalar_literal_message(name, expr))
            return False, env

        arg_names = expr.get("args", []) if isinstance(expr, dict) else []
        args = [env.get(a) for a in arg_names if a in env]

        if DEBUG_MODE:
            tensorlang.print(message=f"[TYPE CHECKER] Arg Names: {arg_names}")
            tensorlang.print(message=f"[TYPE CHECKER] Args: {args}")

        # Comparison operand shapes must be ones the CUDA kernels can run (checked even when the
        # result has a declared type).
        if (isinstance(expr, dict) and expr.get("type") in _COMPARISON_OPS
                and len(arg_names) == 2 and all(a in env for a in arg_names)):
            _s1, _s2 = env[arg_names[0]]["shape"], env[arg_names[1]]["shape"]
            if not _comparison_supported(_s1, _s2):
                tensorlang.print(message=_shape_error(
                    expr["type"], name, _s1, _s2, arg_names, hint=_COMPARISON_HINT))
                return False, env
            _d1, _d2 = env[arg_names[0]]["dtype"], env[arg_names[1]]["dtype"]
            if _comparison_has_f16(_d1, _d2):
                tensorlang.print(message=(
                    f"[TYPE CHECKER] error: [E-COMPARISON-DTYPE] {expr['type']}: cannot compare "
                    f"{_name_label(arg_names[0])} ({_d1}) with {_name_label(arg_names[1])} ({_d2}) "
                    f"computing {_name_label(name)}. {_COMPARISON_DTYPE_HINT}"))
                return False, env

        # Annotated type: trust the declared type.
        if ty and isinstance(ty, dict) and "dtype" in ty:
            if ty["dtype"] == "f64":
                tensorlang.print(message=(
                    f"[TYPE CHECKER] error: [E-UNSUPPORTED-DTYPE] {_name_label(name)} is declared as f64, "
                    f"which is not supported. {_UNSUPPORTED_DTYPE_HINT}"))
                return False, env
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
            "any", "all",
            "greater", "less", "equal", "greater_equal", "less_equal", "not_equal",
            "linear", "layer_norm", "batch_norm", "instance_norm",
            "cross_entropy", "mse_loss",
            "transpose", "reshape", "concat", "slice",
            "sgd", "adam", "sgd_update", "adam_update",
        ]:
            if expr["type"] == "fill":
                if expr.get("shape_like") is not None:
                    like = expr["shape_like"]
                    if like not in env:
                        tensorlang.print(message=_undefined_message([like], env, name, expr))
                        return False, env
                    expr["shape"] = env[like]["shape"]    # same object: kernels test shape1 == shape2
                fill_dtype = "f32"
                if ty and isinstance(ty, dict) and "dtype" in ty:
                    fill_dtype = ty["dtype"]
                env[name] = {"dtype": fill_dtype, "shape": expr["shape"]}
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
                        if loaded_data.dtype == np.float16:
                            inferred_dtype = "f16"
                        elif loaded_data.dtype == np.float64:
                            inferred_dtype = "f64"
                        else:
                            inferred_dtype = "f32"

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
                        if data_array.dtype == np.float16:
                            inferred_dtype = "f16"
                        elif data_array.dtype == np.float64:
                            inferred_dtype = "f64"
                        else:
                            inferred_dtype = "f32"
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
                        inferred_dtype = "f32"

                    else:
                        tensorlang.print(message=f"[TYPE CHECKER] error: Unsupported file format: {suffix}")
                        return False, env

                    env[name] = {"dtype": inferred_dtype, "shape": inferred_shape}
                    if DEBUG_MODE:
                        tensorlang.print(message=f"[TYPE CHECKER] {name}: inferred shape {inferred_shape} from {suffix.upper()} file")

                except FileNotFoundError:
                    tensorlang.print(message=f"[TYPE CHECKER] error: File not found during type inference: {file_path}")
                    return False, env
                except (OSError, UnicodeError, ValueError) as e:
                    tensorlang.print(message=f"[TYPE CHECKER] error: Could not infer shape from {file_path}: {e}")
                    return False, env
                continue

            if expr["type"] in ("any", "all"):
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=_undefined_message([tensor_name], env, name, expr))
                    return False, env

                input_shape = tensor_type["shape"]
                axis = expr.get("axis")
                if axis is None:
                    output_shape = ()
                else:
                    axis = _normalize_axis(axis, len(input_shape))
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
                    axis = _normalize_axis(axis, len(input_shape))
                    if axis < 0 or axis >= len(input_shape):
                        print(f"Type error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                        return False, env
                    output_shape = tuple(dim for i, dim in enumerate(input_shape) if i != axis)
                    if not output_shape:
                        output_shape = ()

                env[name] = {"dtype": tensor_type["dtype"], "shape": output_shape}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "softmax":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=_undefined_message([tensor_name], env, name, expr))
                    return False, env
                input_shape = tensor_type["shape"]
                axis = expr.get("axis")
                if axis is not None:
                    axis = _normalize_axis(axis, len(input_shape))
                    if axis < 0 or axis >= len(input_shape):
                        tensorlang.print(message=f"[TYPE CHECKER] error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                        return False, env
                env[name] = {"dtype": tensor_type["dtype"], "shape": tensor_type["shape"]}
                if DEBUG_INFO:
                    print(f"[INFO] Tensor {tensor_name} {name}:({expr['type']}):{env[name]}")
                continue

            if expr["type"] in ("relu", "sigmoid", "tanh", "gelu", "swish"):
                arg_name = expr["args"][0]
                arg_type = _resolve_tensor_type(env, arg_name)
                if arg_type is None:
                    print(f"Type error: Undefined tensor {arg_name} for {expr['type']}")
                    return False, env
                env[name] = {"dtype": arg_type["dtype"], "shape": arg_type["shape"]}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} for {name}: {env[name]}")
                continue

            if expr["type"] == "slice":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=_undefined_message([tensor_name], env, name, expr))
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

                env[name] = {"dtype": tensor_type["dtype"], "shape": tuple(output_shape) if output_shape else ()}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "layer_norm":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=_undefined_message([tensor_name], env, name, expr))
                    return False, env

                input_shape = tensor_type["shape"]
                axis = expr.get("axis")
                if axis is None:
                    axis = len(input_shape) - 1
                else:
                    axis = _normalize_axis(axis, len(input_shape))
                if axis < 0 or axis >= len(input_shape):
                    tensorlang.print(message=f"[TYPE CHECKER] error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                    return False, env

                env[name] = {"dtype": tensor_type["dtype"], "shape": input_shape}
                if DEBUG_MODE:
                    print(f"Type {expr['type']} Assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "batch_norm":
                # v0.5.6: batch_norm gamma/beta
                tensor_name = expr["tensor"]
                gamma_name = expr.get("gamma")
                beta_name = expr.get("beta")
                roles = [("input", tensor_name), ("running_mean", expr["running_mean"]),
                         ("running_var", expr["running_var"])]
                if gamma_name is not None:
                    roles += [("gamma", gamma_name), ("beta", beta_name)]

                missing = [n for _, n in roles if _resolve_tensor_type(env, n) is None]
                if missing:
                    tensorlang.print(message=_undefined_message(missing, env, name, expr))
                    return False, env

                tensor_type = _resolve_tensor_type(env, tensor_name)
                input_shape = tuple(tensor_type["shape"])
                if len(input_shape) < 2:
                    tensorlang.print(message=(
                        f"[TYPE CHECKER] error: [E-SHAPE-MISMATCH] batch_norm: the input {_name_label(tensor_name)} "
                        f"has shape {input_shape}, but batch_norm needs at least 2 dimensions "
                        f"(batch, channels, ...); the channel is dimension 1"))
                    return False, env

                expected_stats_shape = (input_shape[1],)
                for role, tname in roles[1:]:
                    t = _resolve_tensor_type(env, tname)
                    if tuple(t["shape"]) != expected_stats_shape:
                        tensorlang.print(message=(
                            f"[TYPE CHECKER] error: [E-SHAPE-MISMATCH] batch_norm: {role} {_name_label(tname)} "
                            f"has shape {tuple(t['shape'])}, expected {expected_stats_shape}: one value per "
                            f"channel (dimension 1 of the input, shape {input_shape})"))
                        return False, env
                for role, tname in roles[1:]:
                    t = _resolve_tensor_type(env, tname)
                    if t.get("dtype") != "f32":
                        tensorlang.print(message=(
                            f"[TYPE CHECKER] error: [E-BATCH-NORM-DTYPE] batch_norm: {role} {_name_label(tname)} "
                            f"is {t.get('dtype')}, but running_mean, running_var, gamma and beta must be f32 "
                            f"(the kernel reads them as float)"))
                        return False, env

                env[name] = {"dtype": tensor_type["dtype"], "shape": input_shape}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "instance_norm":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=_undefined_message([tensor_name], env, name, expr))
                    return False, env
                env[name] = {"dtype": tensor_type["dtype"], "shape": tensor_type["shape"]}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] in ("max", "min", "argmax", "argmin"):
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=_undefined_message([tensor_name], env, name, expr))
                    return False, env

                input_shape = tensor_type["shape"]
                axis = expr.get("axis")
                if axis is None:
                    output_shape = ()
                else:
                    axis = _normalize_axis(axis, len(input_shape))
                    if axis < 0 or axis >= len(input_shape):
                        tensorlang.print(message=f"[TYPE CHECKER] error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                        return False, env
                    output_shape = tuple(dim for i, dim in enumerate(input_shape) if i != axis)
                    if not output_shape:
                        output_shape = ()

                env[name] = {"dtype": tensor_type["dtype"], "shape": output_shape}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "transpose":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=_undefined_message([tensor_name], env, name, expr))
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

                env[name] = {"dtype": tensor_type["dtype"], "shape": output_shape}
                if DEBUG_INFO:
                    print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                continue

            if expr["type"] == "reshape":
                tensor_name = expr["tensor"]
                tensor_type = _resolve_tensor_type(env, tensor_name)
                if tensor_type is None:
                    tensorlang.print(message=_undefined_message([tensor_name], env, name, expr))
                    return False, env

                input_shape = tensor_type["shape"]
                new_shape = expr["new_shape"]

                input_elements = int(np.prod([int(dim) for dim in input_shape])) if input_shape else 1
                new_elements = int(np.prod([int(dim) for dim in new_shape])) if new_shape else 1

                if input_elements != new_elements:
                    tensorlang.print(message="[TYPE CHECKER] error: Reshape element count mismatch")
                    return False, env

                env[name] = {"dtype": tensor_type["dtype"], "shape": tuple(new_shape)}
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
                        tensorlang.print(message=_undefined_message([tensor_name], env, name, expr))
                        return False, env
                    tensor_types.append(tensor_type)

                first_shape = tensor_types[0]["shape"]
                axis = _normalize_axis(axis, len(first_shape))
                if axis < 0 or axis >= len(first_shape):
                    tensorlang.print(message="[TYPE CHECKER] error: Concat axis out of bounds")
                    return False, env

                concat_dim_size = 0
                concat_dtype = tensor_types[0]["dtype"]
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
                    concat_dtype = promote_dtype(concat_dtype, tensor_type["dtype"])

                output_shape = list(first_shape)
                output_shape[axis] = concat_dim_size
                env[name] = {"dtype": concat_dtype, "shape": tuple(output_shape)}
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
                    tensorlang.print(message=_undefined_message(missing_args, env, name, expr))
                    return False, env

                if expr["type"] == "matmul":
                    a_shape = args[0]["shape"]
                    b_shape = args[1]["shape"]

                    if len(a_shape) < 2 or len(b_shape) < 2:
                        tensorlang.print(message="[TYPE CHECKER] error: Matmul requires at least 2D tensors")
                        return False, env

                    if a_shape[-1] != b_shape[-2]:
                        tensorlang.print(message=_shape_error(
                            "matmul", name, a_shape, b_shape,
                            arg_names,
                            hint=(f"Inner dimensions must match: left has {a_shape[-1]} "
                                  f"columns but right has {b_shape[-2]} rows")))
                        return False, env

                    # Broadcast batch dimensions (N-D matmul: (..., m, k) @ (..., k, n))
                    batch_a = a_shape[:-2]
                    batch_b = b_shape[:-2]
                    ndim = max(len(batch_a), len(batch_b))
                    padded_a = (1,) * (ndim - len(batch_a)) + batch_a
                    padded_b = (1,) * (ndim - len(batch_b)) + batch_b

                    batch_out = []
                    for d1, d2 in zip(padded_a, padded_b):
                        if d1 == d2:
                            batch_out.append(d1)
                        elif d1 == 1:
                            batch_out.append(d2)
                        elif d2 == 1:
                            batch_out.append(d1)
                        else:
                            tensorlang.print(message="[TYPE CHECKER] error: Matmul batch dimensions must broadcast")
                            return False, env

                    output_shape = tuple(batch_out) + (a_shape[-2], b_shape[-1])
                    matmul_dtype = promote_dtype(args[0]["dtype"], args[1]["dtype"])
                    env[name] = {"dtype": matmul_dtype, "shape": output_shape}
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
                    else:
                        # Non-standard: 1D vector whose size matches a dim of an N-D tensor
                        non_standard = False
                        one_is_1d_nd = False  # True when one operand is 1D and other is N-D (ndim>=2)

                        if len(shape1) == 1 and len(shape2) >= 2:
                            one_is_1d_nd = True
                            if shape1[0] in shape2:
                                output_shape = shape2
                                non_standard = True
                        elif len(shape2) == 1 and len(shape1) >= 2:
                            one_is_1d_nd = True
                            if shape2[0] in shape1:
                                output_shape = shape1
                                non_standard = True

                        if one_is_1d_nd and not non_standard:
                            # 1D size doesn't match any dim of N-D operand → matrix-vector hint
                            tensorlang.print(message=_shape_error(
                                expr["type"], name, shape1, shape2, arg_names,
                                hint=_MATRIX_VECTOR_HINT))
                            return False, env

                        if not non_standard and not one_is_1d_nd:
                            # Standard NumPy broadcasting
                            ndim = max(len(shape1), len(shape2))
                            padded_shape1 = (1,) * (ndim - len(shape1)) + tuple(shape1)
                            padded_shape2 = (1,) * (ndim - len(shape2)) + tuple(shape2)

                            output_shape = []
                            broadcast_ok = True
                            for d1, d2 in zip(padded_shape1, padded_shape2):
                                if d1 == d2:
                                    output_shape.append(d1)
                                elif d1 == 1:
                                    output_shape.append(d2)
                                elif d2 == 1:
                                    output_shape.append(d1)
                                else:
                                    broadcast_ok = False
                                    break
                            if not broadcast_ok:
                                tensorlang.print(message=_shape_error(
                                    expr["type"], name, shape1, shape2, arg_names,
                                    hint=_ELEMENTWISE_HINT))
                                return False, env
                            output_shape = tuple(output_shape)

                    elem_dtype = promote_dtype(args[0]["dtype"], args[1]["dtype"])
                    env[name] = {"dtype": elem_dtype, "shape": output_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")
                    continue


                if expr["type"] in ("greater", "less", "equal", "greater_equal", "less_equal", "not_equal"):
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

                    # Comparison results are 0/1 values; keep f32.
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

                    if len(input_shape) == 0:
                        tensorlang.print(message="[TYPE CHECKER] error: Linear input must be at least 1D")
                        return False, env

                    in_features = input_shape[-1]
                    if len(weight_shape) != 2 or weight_shape[0] != in_features:
                        tensorlang.print(message="[TYPE CHECKER] error: Linear weight shape mismatch")
                        return False, env
                    out_features = weight_shape[1]
                    if len(bias_shape) != 1 or bias_shape[0] != out_features:
                        tensorlang.print(message="[TYPE CHECKER] error: Linear bias shape mismatch")
                        return False, env

                    output_shape = input_shape[:-1] + (out_features,)

                    linear_dtype = promote_dtype(
                        promote_dtype(input_tensor["dtype"], weight_tensor["dtype"]),
                        bias_tensor["dtype"],
                    )
                    env[name] = {"dtype": linear_dtype, "shape": output_shape}
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
                        if len(pred_shape) < 2:
                            tensorlang.print(message="[TYPE CHECKER] error: Cross entropy predictions must be at least 2D")
                            return False, env
                        # Last dim is num_classes; target is class indices (pred_shape[:-1]) or one-hot (pred_shape)
                        if len(target_shape) == len(pred_shape) - 1:
                            if target_shape != pred_shape[:-1]:
                                tensorlang.print(message="[TYPE CHECKER] error: Cross entropy target shape incompatible")
                                return False, env
                        elif len(target_shape) == len(pred_shape):
                            if target_shape != pred_shape:
                                tensorlang.print(message="[TYPE CHECKER] error: Cross entropy target shape incompatible")
                                return False, env
                        else:
                            tensorlang.print(message="[TYPE CHECKER] error: Cross entropy target shape incompatible")
                            return False, env
                        output_shape = ()

                    else:
                        if pred_shape != target_shape:
                            tensorlang.print(message="[TYPE CHECKER] error: MSE loss shape mismatch")
                            return False, env
                        output_shape = ()

                    loss_dtype = promote_dtype(predictions_tensor["dtype"], targets_tensor["dtype"])
                    env[name] = {"dtype": loss_dtype, "shape": output_shape}
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
