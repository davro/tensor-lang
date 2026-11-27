#!/usr/bin/env python3
"""Does data-parallel training give the same result as training on the whole batch?

    python3 tools/ddp_check.py                         # 2 ranks on 2 GPUs, both backward modes
    python3 tools/ddp_check.py --ranks 4 --same-gpu    # development: 4 ranks on one GPU
    python3 tools/ddp_check.py --backward gpu          # only with TENSORLANG_GPU_BACKWARD=1
    python3 tools/ddp_check.py --keep                  # leave build/ddp_check and the cache outputs for inspection

It builds a small regression problem (a two-layer network with a GELU, mean-squared-error loss, plain SGD), writes the
whole batch and its shards to build/ddp_check/, and generates two programs that differ only in file paths and batch size:

    full.tl   one process, the whole batch
    w{N}.tl   N ranks, each loading its own shard with {rank} placeholders (and, deliberately, a different initial weight
              on every rank but the first, so the start-up broadcast of rank 0's parameters is part of what is checked)

It runs both and compares the final weights:

    ranks identical      every rank ends with bit-identical weights (they applied the same averaged gradients)
    ddp vs single        N ranks on shards match one process on the whole batch (float32 rounding only)
    vs float64 reference one process against a NumPy float64 run; informational, since the GELU variant may differ slightly
    weights moved        the training did change the weights, so the comparisons mean something

With equal shards and a mean loss, the average of the shard gradients is exactly the gradient of the whole batch, so any
real difference between the first two means gradients, parameters or the optimizer step went wrong on some rank.
Exit status 0 if every check passes. Run it from the repository root (it changes there itself).
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

N_ROWS, D_IN, D_HID = 64, 8, 16
STEPS, LR = 5, 0.1
SEED = 1234
TOL_DDP_VS_SINGLE = 1e-4          # norm-wise relative difference: float32 rounding after a few steps
TOL_VS_REFERENCE = 5e-3           # looser, and informational: the GPU GELU may be a slightly different approximation

_erf = np.vectorize(math.erf)


# ----------------------------------------------------------------------------------------------------------------
# The problem, and a float64 reference for it
# ----------------------------------------------------------------------------------------------------------------

def make_problem(seed=SEED, n=N_ROWS):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, D_IN)).astype(np.float32)
    true_w = rng.standard_normal((D_IN, 1)) * 0.5
    y = (np.tanh(x @ true_w) + 0.05 * rng.standard_normal((n, 1))).astype(np.float32)
    w1 = (rng.standard_normal((D_IN, D_HID)) * 0.4).astype(np.float32)
    w2 = (rng.standard_normal((D_HID, 1)) * 0.4).astype(np.float32)
    return x, y, w1, w2


def gelu(z):
    return 0.5 * z * (1.0 + _erf(z / math.sqrt(2.0)))


def gelu_grad(z):
    return 0.5 * (1.0 + _erf(z / math.sqrt(2.0))) + z * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def loss_and_gradients(x, y, w1, w2):
    """Mean squared error of gelu(x @ w1) @ w2, and its gradients with respect to w1 and w2."""
    pre = x @ w1
    act = gelu(pre)
    diff = act @ w2 - y
    d_pred = 2.0 * diff / diff.size
    g_w2 = act.T @ d_pred
    g_w1 = x.T @ ((d_pred @ w2.T) * gelu_grad(pre))
    return float((diff ** 2).mean()), g_w1, g_w2


def reference_train(x, y, w1, w2, steps=STEPS, lr=LR):
    """Plain SGD in float64 on the whole batch."""
    x, y, w1, w2 = (np.asarray(a, dtype=np.float64) for a in (x, y, w1, w2))
    for _ in range(steps):
        _, g1, g2 = loss_and_gradients(x, y, w1, w2)
        w1, w2 = w1 - lr * g1, w2 - lr * g2
    return w1, w2


# ----------------------------------------------------------------------------------------------------------------
# Files and programs
# ----------------------------------------------------------------------------------------------------------------

def program_text(directory, rows, steps=STEPS, lr=LR):
    """A TensorLang program training the network on `rows` rows loaded from `directory` ({rank} placeholders)."""
    return f'''let x: Tensor[f32, ({rows}, {D_IN})] = load("{directory}/x_{{rank}}.npy")
let y_true: Tensor[f32, ({rows}, 1)] = load("{directory}/y_{{rank}}.npy")
let w1: Tensor[f32, ({D_IN}, {D_HID})] = load("{directory}/w1_{{rank}}.npy") with grad
let w2: Tensor[f32, ({D_HID}, 1)] = load("{directory}/w2_{{rank}}.npy") with grad

for step in range({steps}) {{
    let h = gelu(matmul(x, w1))
    let pred = matmul(h, w2)
    let loss = mse_loss(pred, y_true)
    backward(loss)
    sgd(w1, w1_grad, {lr})
    sgd(w2, w2_grad, {lr})
}}

save(w1, "w1_final.npy")
save(w2, "w2_final.npy")
'''


def write_problem(root, ranks, steps=STEPS, lr=LR):
    """Write data and programs under root/build/ddp_check. Returns (full program path, ddp program path), relative to root."""
    if N_ROWS % ranks:
        raise ValueError(f"{N_ROWS} rows do not split evenly over {ranks} ranks")
    x, y, w1, w2 = make_problem()
    base = Path(root) / "build" / "ddp_check"
    if base.exists():
        shutil.rmtree(base)
    full, sharded = base / "full", base / f"w{ranks}"
    full.mkdir(parents=True)
    sharded.mkdir(parents=True)

    for name, arr in (("x", x), ("y", y), ("w1", w1), ("w2", w2)):
        np.save(full / f"{name}_0.npy", arr)
    rows = N_ROWS // ranks
    rng = np.random.default_rng(SEED + 1)
    for r in range(ranks):
        np.save(sharded / f"x_{r}.npy", x[r * rows:(r + 1) * rows])
        np.save(sharded / f"y_{r}.npy", y[r * rows:(r + 1) * rows])
        # every rank but 0 starts from different weights on purpose: the broadcast must make them rank 0's
        np.save(sharded / f"w1_{r}.npy", w1 if r == 0 else w1 + (0.1 * rng.standard_normal(w1.shape)).astype(np.float32))
        np.save(sharded / f"w2_{r}.npy", w2 if r == 0 else w2 + (0.1 * rng.standard_normal(w2.shape)).astype(np.float32))

    rel = Path("build") / "ddp_check"
    full_tl, ddp_tl = base / "full.tl", base / f"w{ranks}.tl"
    full_tl.write_text(program_text(str(rel / "full"), N_ROWS, steps, lr))
    ddp_tl.write_text(program_text(str(rel / f"w{ranks}"), rows, steps, lr))
    return rel / "full.tl", rel / f"w{ranks}.tl"


# ----------------------------------------------------------------------------------------------------------------
# Running and comparing
# ----------------------------------------------------------------------------------------------------------------

def rel_diff(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def load_final(directory):
    return np.load(Path(directory) / "w1_final.npy"), np.load(Path(directory) / "w2_final.npy")


def run_program(root, args, env, run=subprocess.run):
    return run([sys.executable, "tensorlang.py", *args], cwd=root, env=env, capture_output=True, text=True)


def check(root, ranks, same_gpu, backward, run=subprocess.run, base_env=None):
    """Run one mode (backward in 'numpy' or 'gpu') and return a list of (label, passed, detail, informational)."""
    root = Path(root)
    full_tl, ddp_tl = write_problem(root, ranks)
    env = dict(os.environ if base_env is None else base_env)
    env.pop("TENSORLANG_GPU_BACKWARD", None)
    if backward == "gpu":
        env["TENSORLANG_GPU_BACKWARD"] = "1"
    cache = root / "cache" / "build" / "ddp_check"
    shutil.rmtree(cache, ignore_errors=True)

    results = []
    single = run_program(root, [str(full_tl)], env, run)
    if single.returncode != 0:
        return [("single-process run", False, f"exit code {single.returncode}\n{(single.stdout + single.stderr)[-1500:]}", False)]
    flags = ["--ddp", "--nproc", str(ranks)] + (["--ddp-same-gpu"] if same_gpu else [])
    multi = run_program(root, [*flags, str(ddp_tl)], env, run)
    if multi.returncode != 0:
        return [("data-parallel run", False, f"exit code {multi.returncode}\n{(multi.stdout + multi.stderr)[-1500:]}", False)]

    try:
        want = load_final(cache / "full.tl")
        per_rank = [load_final(cache / f"w{ranks}.tl" / f"ws{ranks}" / f"rank{r}") for r in range(ranks)]
    except FileNotFoundError as exc:
        return [("outputs", False, f"a run exited normally but did not write {exc.filename}; did it reach the "
                                   "save() statements, and did a rank write somewhere else?", False)]

    drift = [r for r in range(1, ranks)
             if any(per_rank[r][i].tobytes() != per_rank[0][i].tobytes() for i in (0, 1))]
    results.append(("ranks identical", not drift,
                    "all ranks ended with identical weights" if not drift else
                    f"rank(s) {drift} differ from rank 0 (max |dw1| "
                    f"{max(float(np.abs(per_rank[r][0] - per_rank[0][0]).max()) for r in drift):.3g})", False))

    worst = max(max(rel_diff(per_rank[r][i], want[i]) for i in (0, 1)) for r in range(ranks))
    results.append(("ddp vs single", worst <= TOL_DDP_VS_SINGLE,
                    f"worst relative difference {worst:.2e} (tolerance {TOL_DDP_VS_SINGLE:.0e})", False))

    x, y, w1, w2 = make_problem()
    ref = reference_train(x, y, w1, w2)
    far = max(rel_diff(want[0], ref[0]), rel_diff(want[1], ref[1]))
    results.append(("vs float64 reference", far <= TOL_VS_REFERENCE,
                    f"single-process vs NumPy float64: {far:.2e} (tolerance {TOL_VS_REFERENCE:.0e}); "
                    "a larger value can just mean a different GELU approximation", True))

    moved = min(rel_diff(want[0], w1), rel_diff(want[1], w2))
    results.append(("weights moved", moved > 1e-3, f"smallest relative change from the start: {moved:.2e}", False))
    return results


def main(argv=None, run=subprocess.run):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ranks", type=int, default=2, help="number of data-parallel ranks (default 2; must divide %d)" % N_ROWS)
    ap.add_argument("--same-gpu", action="store_true", help="run every rank on one GPU (development)")
    ap.add_argument("--backward", choices=("both", "gpu", "numpy"), default="both",
                    help="which backward to check (default both: NumPy, then the GPU backward)")
    ap.add_argument("--keep", action="store_true", help="keep build/ddp_check and its cache outputs")
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    root = Path(a.root)
    modes = ["numpy", "gpu"] if a.backward == "both" else [a.backward]
    failed = False
    for mode in modes:
        print(f"\n== {a.ranks} rank(s), {mode} backward ==")
        for label, ok, detail, info in check(root, a.ranks, a.same_gpu, mode, run=run):
            status = "PASS" if ok else ("note" if info else "FAIL")
            print(f"  {status:<5}{label:<22}{detail}")
            failed = failed or (not ok and not info)
    if not a.keep:
        shutil.rmtree(root / "build" / "ddp_check", ignore_errors=True)
        shutil.rmtree(root / "cache" / "build" / "ddp_check", ignore_errors=True)
    print("\n" + ("FAILED" if failed else "all checks passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
