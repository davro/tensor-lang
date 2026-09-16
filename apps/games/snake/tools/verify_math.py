#!/usr/bin/env python3
"""
Hand-derives train.tl's forward+backward pass in plain NumPy and
finite-difference-checks every gradient — same purpose as
connect_four/tools/verify_math.py: this validates the DESIGN (is this
architecture/loss actually differentiable the way train.tl assumes?),
independent of whether TensorLang's own CUDA kernels implement each op
correctly (that part still needs real GPU hardware — see NOTES.md).

No GPU needed:
    python3 apps/games/snake/tools/verify_math.py
"""
import numpy as np

rng = np.random.default_rng(0)

BATCH = 8
IN, H1, H2, OUT = 32, 16, 10, 4  # small sizes for a fast, thorough check

x = rng.standard_normal((BATCH, IN)).astype(np.float64)
y_true = np.zeros((BATCH, OUT))
y_true[np.arange(BATCH), rng.integers(0, OUT, BATCH)] = 1.0

w1 = rng.standard_normal((IN, H1)) * 0.3
b1 = rng.standard_normal((H1,)) * 0.1
w2 = rng.standard_normal((H1, H2)) * 0.3
b2 = rng.standard_normal((H2,)) * 0.1
w3 = rng.standard_normal((H2, OUT)) * 0.3
b3 = rng.standard_normal((OUT,)) * 0.1

params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def forward(p):
    h1_pre = x @ p["w1"] + p["b1"]
    h1 = np.maximum(h1_pre, 0)
    h2_pre = h1 @ p["w2"] + p["b2"]
    h2 = np.maximum(h2_pre, 0)
    y_pre = h2 @ p["w3"] + p["b3"]
    y_pre_shift = y_pre - y_pre.max(axis=1, keepdims=True)
    e = np.exp(y_pre_shift)
    probs = e / e.sum(axis=1, keepdims=True)
    # mse_loss: mean over ALL elements of (probs - y_true)**2, matching
    # train.tl's mse_loss(probs, y_true) — see HANDOVER.md's autograd
    # section for the un-normalized-forward/1-over-total-elements-
    # backward convention every app here relies on.
    loss = np.mean((probs - y_true) ** 2)
    cache = (h1_pre, h1, h2_pre, h2, y_pre, probs)
    return loss, cache


def backward(p, cache):
    h1_pre, h1, h2_pre, h2, y_pre, probs = cache
    n = probs.size

    dprobs = 2.0 * (probs - y_true) / n
    # Softmax Jacobian-vector product: dy_pre_i = probs_i * (dprobs_i - sum_j probs_j*dprobs_j)
    dot = np.sum(probs * dprobs, axis=1, keepdims=True)
    dy_pre = probs * (dprobs - dot)

    dw3 = h2.T @ dy_pre
    db3 = dy_pre.sum(axis=0)
    dh2 = dy_pre @ p["w3"].T

    dh2_pre = dh2 * (h2_pre > 0)
    dw2 = h1.T @ dh2_pre
    db2 = dh2_pre.sum(axis=0)
    dh1 = dh2_pre @ p["w2"].T

    dh1_pre = dh1 * (h1_pre > 0)
    dw1 = x.T @ dh1_pre
    db1 = dh1_pre.sum(axis=0)

    return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2, "w3": dw3, "b3": db3}


def finite_diff_check(name, eps=1e-5):
    p = {k: v.copy() for k, v in params.items()}
    _, cache = forward(p)
    grads = backward(p, cache)

    arr = p[name]
    analytic = grads[name]
    max_rel_err = 0.0
    # Sample a handful of entries rather than every element — plenty to
    # catch a wrong-shape or wrong-sign bug, at a fraction of the cost.
    flat_idx = rng.choice(arr.size, size=min(20, arr.size), replace=False)
    for idx in flat_idx:
        coord = np.unravel_index(idx, arr.shape)
        orig = arr[coord]
        arr[coord] = orig + eps
        loss_plus, _ = forward(p)
        arr[coord] = orig - eps
        loss_minus, _ = forward(p)
        arr[coord] = orig
        numeric = (loss_plus - loss_minus) / (2 * eps)
        a = analytic[coord]
        rel_err = abs(numeric - a) / max(abs(numeric), abs(a), 1e-8)
        max_rel_err = max(max_rel_err, rel_err)
    print(f"  {name}: max relative error over {len(flat_idx)} sampled entries = {max_rel_err:.2e}")
    return max_rel_err


def main():
    print("Finite-difference gradient check for train.tl's architecture "
          f"(batch={BATCH}, {IN}-{H1}-{H2}-{OUT}):")
    worst = 0.0
    for name in params:
        worst = max(worst, finite_diff_check(name))
    print()
    if worst < 1e-4:
        print(f"OK — worst-case relative error {worst:.2e} is well within finite-difference noise.")
    else:
        print(f"WARNING — worst-case relative error {worst:.2e} is higher than expected; "
              "check the backward pass above against train.tl.")


if __name__ == "__main__":
    main()
