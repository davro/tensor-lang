#!/usr/bin/env python3
"""Compare the GPU run of the micro-GPT probe with the float64 reference, parameter by parameter.

  python3 tensorlang.py --test --filter probe_micro_gpt_train --verify-tensors     # writes the cache
  python3 tools/compare_micro_gpt.py [--cache cache/tests/probe_micro_gpt_train.tl] [--weights tests/fixtures/micro_gpt]

For every parameter: the largest |GPU - float64| after the final update, that error as a multiple of the runner's
tolerance (1e-8 + 1e-5*|value|), and, for the worst element, its float64 gradient at step 1 and its value.
How to read it: Adam divides the gradient by its own size, so a gradient with relative error e moves the weight by
about lr*e whatever its size. Errors that are large only where the step-1 gradient is tiny are Adam amplifying float32
noise; errors that are large where the gradient is large point at a real difference in a forward or backward kernel.
"""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_micro_gpt_train as g

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache/tests/probe_micro_gpt_train.tl")
ap.add_argument("--weights", default="tests/fixtures/micro_gpt")
ap.add_argument("--grads", action="store_true",
                help="compare the saved <param>_grad.npy of a one-step, no-optimizer run with the float64 gradients")
a = ap.parse_args()
cfg = json.load(open(os.path.join(a.weights, "config.json")))
args = argparse.Namespace(**cfg)
P0 = {k: np.load(os.path.join(a.weights, k + ".npy")) for k in g.make_params(args)}
tok, tgt = g.make_data(args)
losses, P64 = g.train(P0, tok, tgt, args, np.float64)
_, G1 = g.fwd_bwd({k: v.astype(np.float64) for k, v in P0.items()}, tok, tgt, args.layers, 1 / math.sqrt(args.dim))

if a.grads:
    print(f"{'parameter':10s}{'norm-wise error':>17s}{'max |err|/max|g|':>19s}{'max|g|':>10s}{'Adam step-1 err':>17s}")
    bad, miss, worst_step = [], [], 0.0
    for k in P0:
        f = os.path.join(a.cache, k + "_grad.npy")
        if not os.path.isfile(f): miss.append(k); continue
        gpu = np.load(f).astype(np.float64); ref = G1[k]; gm = np.abs(ref).max()
        if gm < 1e-10: print(f"{k:10s}{'(zero gradient)':>17s}{np.abs(gpu).max():>19.2e}{gm:>10.1e}   |gpu| is noise, expected"); continue
        nw = np.linalg.norm(gpu - ref) / np.linalg.norm(ref); mx = np.abs(gpu - ref).max() / gm
        # what Adam's first step (m_hat/sqrt(v_hat) = g/(|g|+eps)) would do with these two gradients
        step = args.lr * np.abs(gpu / (np.abs(gpu) + 1e-8) - ref / (np.abs(ref) + 1e-8)).max(); worst_step = max(worst_step, step)
        print(f"{k:10s}{nw:17.2e}{mx:19.2e}{gm:10.2e}{step:17.2e}"); bad += [k] if nw > 1e-4 else []
    print(f"\nfloat32 noise is ~1e-6 norm-wise; {len(bad)} parameter(s) above 1e-4" + (": " + ", ".join(bad) if bad else ""))
    print(f"largest weight error Adam would make in its FIRST step from these gradients: {worst_step:.2e} "
          f"(runner tolerance for a weight of 0.01 is {1e-8 + 1e-5 * 0.01:.1e})")
    if miss: print("no _grad.npy for:", ", ".join(miss[:6]), "(run with --cache-layers)")
    sys.exit(0)

rows, missing = [], []
for k in P0:
    f = os.path.join(a.cache, k + ".npy")
    if not os.path.isfile(f): missing.append(k); continue
    gpu = np.load(f).astype(np.float64); ref = P64[k]
    d = np.abs(gpu - ref); tol = 1e-8 + 1e-5 * np.abs(ref); i = np.unravel_index(np.argmax(d / tol), d.shape)
    rows.append(((d / tol).max(), k, d.max(), ref[i], G1[k][i], i))
lf = os.path.join(a.cache, "loss.npy")
if os.path.isfile(lf):
    gl = float(np.load(lf)); print(f"loss of the last step: GPU {gl:.9f}  float64 {losses[-1]:.9f}  diff {abs(gl - losses[-1]):.2e}  "
                                  f"({abs(gl - losses[-1]) / (1e-8 + 1e-5 * abs(losses[-1])):.3f} of the tolerance)")
print(f"{'parameter':10s}{'max |diff|':>12s}{'x tolerance':>13s}{'ref value':>12s}{'step-1 grad':>13s}  element")
for mg, k, mx, rv, gr, i in sorted(rows, reverse=True)[:12]:
    print(f"{k:10s}{mx:12.2e}{mg:13.2f}{rv:12.5f}{gr:13.2e}  {tuple(int(x) for x in i)}")
zero = [k for k in P0 if np.abs(G1[k]).max() < 1e-10]      # e.g. the key biases: softmax ignores a per-row constant
over = [k for mg, k, *_ in rows if mg > 1 and k not in zero]
if zero: print(f"\nzero true gradient (Adam turns float32 noise into a full step, so these are EXPECTED to differ): {', '.join(zero)}")
print(f"{len(over)} of {len(rows) - len([k for k in zero if k in dict((r[1], 0) for r in rows)])} other parameters exceed the runner tolerance"
      + (f": {', '.join(over)}" if over else ""))
if missing: print("not in the cache (did the test run with --cache-layers?):", ", ".join(missing[:6]), "..." if len(missing) > 6 else "")
