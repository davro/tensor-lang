# tensorlang/fusion.py
"""Conservative elementwise kernel fusion pass.

Fuses consecutive elementwise unary/binary let-bindings into a single
``fused_elementwise`` node.  The pass is purely a codegen optimization:
autograd still records each original op, and the fused kernel writes every
intermediate output so the backward pass keeps working.
"""

import os

FUSION_ENABLED = os.environ.get("TENSORLANG_DISABLE_FUSION", "").lower() not in (
    "1", "true", "yes",
)

_FUSIBLE = {
    "add", "minus", "mult", "div",
    "relu", "sigmoid", "tanh", "gelu", "swish",
}


def _is_let_expr(node):
    return (
        isinstance(node, dict)
        and node.get("type") == "let"
        and isinstance(node.get("expr"), dict)
    )


def _expr_args(expr):
    """Return the arg-name list for an elementwise expr, or None."""
    if not isinstance(expr, dict):
        return None
    if expr.get("type") in _FUSIBLE and isinstance(expr.get("args"), list):
        return expr["args"]
    return None


def _first_arg_name(arg):
    if isinstance(arg, str):
        return arg
    if isinstance(arg, dict):
        return arg.get("name")
    return None


def _no_broadcast(node, env):
    """True only if every input arg has the same shape as the output."""
    expr = node.get("expr", {})
    out_shape = tuple(env[node["name"]]["shape"])
    args = expr.get("args", [])
    for a in args:
        name = a if isinstance(a, str) else (a.get("name") if isinstance(a, dict) else None)
        if name is None or name not in env:
            return False
        if tuple(env[name]["shape"]) != out_shape:
            return False
    return True


def fuse_elementwise_chains(nodes, env):
    if not FUSION_ENABLED or not nodes:
        return list(nodes)

    result = []
    i = 0
    n = len(nodes)

    while i < n:
        if not _is_let_expr(nodes[i]):
            result.append(nodes[i])
            i += 1
            continue

        expr0 = nodes[i]["expr"]
        if expr0.get("type") not in _FUSIBLE or not _no_broadcast(nodes[i], env):
            result.append(nodes[i])
            i += 1
            continue

        chain = [nodes[i]]
        j = i
        while j + 1 < n:
            nxt = nodes[j + 1]
            if not _is_let_expr(nxt):
                break
            nxt_expr = nxt["expr"]
            if nxt_expr.get("type") not in _FUSIBLE:
                break
            if not _no_broadcast(nxt, env):
                break
            args = _expr_args(nxt_expr)
            if not args:
                break
            prev_name = chain[-1]["name"]
            if _first_arg_name(args[0]) != prev_name:
                break
            try:
                if tuple(env[prev_name]["shape"]) != tuple(env[nxt["name"]]["shape"]):
                    break
            except (KeyError, TypeError):
                break
            chain.append(nxt)
            j += 1

        if len(chain) >= 2:
            ops = []
            for node in chain:
                e = node["expr"]
                args = [a if isinstance(a, str) else a.get("name") for a in e["args"]]
                ops.append((e["type"], node["name"], args))
            fused_node = {
                "type": "let",
                "name": chain[-1]["name"],
                "ty": chain[-1].get("ty"),
                "expr": {"type": "fused_elementwise", "chain": ops},
                "requires_grad": chain[-1].get("requires_grad", False),
                "tree": None,
            }
            result.append(fused_node)
            i = j + 1
        else:
            result.append(nodes[i])
            i += 1

    return result


