# tensorlang/fusion.py
"""Conservative elementwise kernel fusion pass.

Fuses consecutive elementwise unary/binary let-bindings into a single
``fused_elementwise`` node.  Supports both non-broadcast and broadcasting
chains (broadcasting fusion uses per-element index remapping).  The pass is
purely a codegen optimization: autograd still records each original op, and
the fused kernel writes every intermediate output so the backward pass
keeps working.
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

def _can_broadcast(shape1, shape2):
    ndim = max(len(shape1), len(shape2))
    s1 = (1,) * (ndim - len(shape1)) + tuple(shape1)
    s2 = (1,) * (ndim - len(shape2)) + tuple(shape2)
    for d1, d2 in zip(s1, s2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            return False
    return True

def _broadcast_ok(node, env):
    """True if every input can broadcast to the output shape."""
    expr = node.get("expr", {})
    out_shape = tuple(env[node["name"]]["shape"])
    args = expr.get("args", [])
    for a in args:
        name = a if isinstance(a, str) else (a.get("name") if isinstance(a, dict) else None)
        if name is None or name not in env:
            return False
        if not _can_broadcast(env[name]["shape"], out_shape):
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
        if expr0.get("type") not in _FUSIBLE or not _broadcast_ok(nodes[i], env):
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
            if not _broadcast_ok(nxt, env):
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

            final_name = chain[-1]["name"]
            final_shape = tuple(env[final_name]["shape"])
            broadcast = False
            for node in chain:
                e = node["expr"]
                for a in e["args"]:
                    name = a if isinstance(a, str) else a.get("name")
                    if tuple(env[name]["shape"]) != final_shape:
                        broadcast = True

            fused_node = {
                "type": "let",
                "name": final_name,
                "ty": chain[-1].get("ty"),
                "expr": {"type": "fused_elementwise", "chain": ops, "broadcast": broadcast},
                "requires_grad": chain[-1].get("requires_grad", False),
                "tree": None,
            }
            result.append(fused_node)
            i = j + 1
        else:
            result.append(nodes[i])
            i += 1

    return result
