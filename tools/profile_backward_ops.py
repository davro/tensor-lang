#!/usr/bin/env python3
"""Where does the backward pass spend its time, op by op?

    TENSORLANG_GPU_BACKWARD=1 python3 tools/profile_backward_ops.py build/micro_gpt_perf.tl
    python3 tools/profile_backward_ops.py build/micro_gpt_perf.tl --top 25 --skip 2

Runs the program once and times every call of ComputationGraph._backward_node, grouped by the node's op type, plus the work
backward() does outside the nodes: the zero-filled gradient for every tensor in the graph, the np.all checks (the zero-gradient skip
before each node and the overflow scan at the end), np.isfinite, and what is left (the loop itself and the `output_grad == 0`
comparison made before each node).

Nothing in the repo is edited: the timers are patched in for this process only. The first --skip backward passes (default 1) are
left out, because the first call of the GPU backward builds the device and its kernels once. Run from anywhere; the program path is
resolved before the tool changes to the repo root, which tensorlang.lark needs; it must be inside the repo, because the compiler
builds its cache directory from the program path.

Reading the table: "elems/call" is the size of the gradient arriving at the node. A node that takes hundreds of microseconds on a
few thousand elements is bound by per-call overhead, not by arithmetic; moving it to the GPU would not help unless the data also
stays there. The timers add about a microsecond per node, so very small ops read slightly high.
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("program", help="a .tl file with a training loop")
    ap.add_argument("--top", type=int, default=15, help="op types to show (default 15)")
    ap.add_argument("--skip", type=int, default=1, help="backward passes to leave out as warm-up (default 1)")
    a = ap.parse_args()

    program = Path(a.program).resolve()
    if not program.is_file():
        print(f"not found: {a.program}", file=sys.stderr)
        return 1
    try:
        # compile_and_execute builds its cache directory as Path("cache") / <program path>; an absolute path would make that
        # the program file itself and fail, so hand it a path relative to the repo root (where we chdir to)
        program_arg = str(program.relative_to(ROOT))
    except ValueError:
        print(f"{program} is outside the repo; the compiler's cache path needs a program inside it (copy it under build/)",
              file=sys.stderr)
        return 1
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    import numpy as real_np

    from tensorlang import autograd

    G = autograd.ComputationGraph
    stats = defaultdict(lambda: [0, 0.0, 0])        # op type -> [calls, seconds, gradient elements]
    outside = defaultdict(lambda: [0, 0.0])         # label -> [calls, seconds]
    state = {"passes": 0, "collecting": False, "in_backward": False, "depth": 0,
             "backward_s": 0.0, "skipped_s": 0.0, "nodes": 0}

    orig_node = G._backward_node
    orig_backward = G.backward

    def timed_node(self, node):
        if not state["collecting"]:
            return orig_node(self, node)
        grad = self.gradients.get(node.output_name)
        elems = getattr(grad, "size", 0)
        state["depth"] += 1
        t0 = time.perf_counter()
        try:
            return orig_node(self, node)
        finally:
            dt = time.perf_counter() - t0
            state["depth"] -= 1
            s = stats[node.op_type]
            s[0] += 1
            s[1] += dt
            s[2] += elems
            state["nodes"] += 1

    def timed_backward(self, *args, **kwargs):
        state["passes"] += 1
        state["collecting"] = state["passes"] > a.skip
        state["in_backward"] = True
        t0 = time.perf_counter()
        try:
            return orig_backward(self, *args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            state["in_backward"] = False
            state["backward_s" if state["collecting"] else "skipped_s"] += dt

    def timed_np(fn, label):
        def wrapper(*args, **kwargs):
            if not (state["collecting"] and state["in_backward"]) or state["depth"]:
                return fn(*args, **kwargs)            # inside a node (counted there) or outside a timed pass
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                o = outside[label]
                o[0] += 1
                o[1] += time.perf_counter() - t0
        return wrapper

    class NumpyProxy:
        """The numpy module, with a few functions timed when backward() itself (not a node) calls them."""
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

    proxy = NumpyProxy(real_np)
    for fname in ("zeros_like", "all", "isfinite"):
        setattr(proxy, fname, timed_np(getattr(real_np, fname), fname))

    G._backward_node = timed_node
    G.backward = timed_backward
    autograd.np = proxy

    from tensorlang.compiler import TensorCompiler

    wall0 = time.perf_counter()
    ok = TensorCompiler().compile_and_execute(program_arg)
    wall = time.perf_counter() - wall0

    G._backward_node = orig_node
    G.backward = orig_backward
    autograd.np = real_np

    measured = state["passes"] - min(state["passes"], a.skip)
    if not measured:
        print(f"\nonly {state['passes']} backward pass(es) ran; nothing left after --skip {a.skip}", file=sys.stderr)
        return 1

    total = state["backward_s"]
    in_nodes = sum(v[1] for v in stats.values())
    named = sum(v[1] for v in outside.values())
    other = max(total - in_nodes - named, 0.0)
    used = dict(getattr(autograd._gpu_backward, "STATS", {}))   # backward calls that really ran on the GPU, all passes

    def on_gpu(op):
        kind = "matmul" if op in ("matmul", "mm", "bmm") else "gelu" if op == "gelu" else None
        return bool(kind and used.get(kind))

    print()
    print("=" * 78)
    print(f"backward by op type: {program.name}   (run ok={ok}, wall {wall:.2f} s)")
    print("=" * 78)
    print(f"TENSORLANG_GPU_BACKWARD={os.environ.get('TENSORLANG_GPU_BACKWARD', '(unset)')}"
          f"   passes timed: {measured} (first {min(state['passes'], a.skip)} skipped, {state['skipped_s']:.3f} s)"
          f"   nodes: {state['nodes']}")
    print(f"backward total {total:.3f} s   ({total / measured * 1e3:.1f} ms per pass, "
          f"{state['nodes'] / measured:.0f} nodes per pass)\n")

    print(f"{'op type':<22}{'calls':>7}{'total s':>9}{'share':>8}{'us/call':>9}{'elems/call':>12}")
    print("-" * 67)
    rows = sorted(stats.items(), key=lambda kv: -kv[1][1])
    for op, (calls, secs, elems) in rows[:a.top]:
        mark = " *" if on_gpu(op) else ""
        print(f"{op + mark:<22}{calls:>7}{secs:>9.3f}{secs / total:>8.1%}{secs / calls * 1e6:>9.0f}{elems / calls:>12.0f}")
    if len(rows) > a.top:
        rest = rows[a.top:]
        secs = sum(v[1] for _, v in rest)
        print(f"{'(' + str(len(rest)) + ' more op types)':<22}{sum(v[0] for _, v in rest):>7}{secs:>9.3f}{secs / total:>8.1%}")
    print("-" * 67)
    print(f"{'inside nodes, total':<22}{state['nodes']:>7}{in_nodes:>9.3f}{in_nodes / total:>8.1%}")
    if any(used.values()):
        print(f"  * ran on the GPU backward: {used.get('matmul', 0)} matmul and {used.get('gelu', 0)} gelu calls over all passes;"
              " everything else ran in NumPy")
    elif os.environ.get("TENSORLANG_GPU_BACKWARD", "") not in ("", "0"):
        print("  GPU backward was requested but not used (see any message above); everything ran in NumPy")

    print("\noutside the nodes, inside backward():")
    labels = {"zeros_like": "gradient for every tensor (np.zeros_like)",
              "all": "np.all (zero-skip check, overflow scan)",
              "isfinite": "np.isfinite (overflow scan)"}
    for key, text in labels.items():
        calls, secs = outside.get(key, (0, 0.0))
        print(f"  {text:<44}{calls:>7}{secs:>9.3f}{secs / total:>8.1%}")
    print(f"  {'the rest (loop, `== 0` before each node, ...)':<44}{'':>7}{other:>9.3f}{other / total:>8.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
