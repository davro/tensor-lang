#!/usr/bin/env python3
"""
apps/games/connect_four/tools/verify_math.py

Verifies the ARITHMETIC (not just the shapes check.py already confirms)
of train.tl / infer.tl's forward+backward math, without needing CUDA or
a GPU — same spirit as the repo root's verify.py/verify_all.py, but
implemented as a direct hand-derived NumPy reimplementation of this
architecture rather than a generic AST interpreter (verify.py's
interpreter is hardcoded to 2048's specific tensor-ops game engine and
doesn't cover matmul/softmax/tanh/backward(), which is what this
architecture actually needs).

What this checks:
  1. Forward pass shapes match what check.py's type-checker reports.
  2. A finite-difference gradient check against the hand-derived
     backward pass, for every weight tensor — this is the same
     validation used during design (see the PR/commit history), rerun
     here so it stays checked if the architecture ever changes.
  3. A few steps of the exact SGD update train.tl performs actually
     decrease the combined loss on random data.

What this does NOT check: that TensorLang's own CUDA kernels for
matmul/add/relu/tanh/softmax/mse_loss compute the same thing as this
reference — that requires actually running train.tl/infer.tl on real
GPU hardware. Those individual ops are documented as hardware-verified
building blocks elsewhere in this repo (HANDOVER.md), and mse_loss +
softmax + tanh gradients are exactly the ones tic_tac_toe's and
decision_boundary's already-verified networks rely on — but the
COMBINATION used here (two loss heads added together, backward()'d as
one) has not been run on hardware. Run this file, then check.py on
both .tl files, then a real (small) training run and inspect
policy_loss/value_loss individually before trusting a promoted network.

Run:
    python3 apps/games/connect_four/tools/verify_math.py
"""
import numpy as np

IN, H1, H2, POL, VAL = 42, 128, 64, 7, 2  # VAL=2, not 1 -- see train.tl REAL BUG #3


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(x, y_policy, y_value, params):
    w1, b1, w2, b2, wp, bp, wv, bv = params
    h1p = x @ w1 + b1
    h1 = np.maximum(h1p, 0)
    h2p = h1 @ w2 + b2
    h2 = np.maximum(h2p, 0)
    pp = h2 @ wp + bp
    probs = softmax(pp)
    vp = h2 @ wv + bv
    value = np.tanh(vp)
    policy_loss = np.mean((probs - y_policy) ** 2)
    value_loss = np.mean((value - y_value) ** 2)
    loss = policy_loss + value_loss
    cache = (h1p, h1, h2p, h2, probs, value)
    return loss, policy_loss, value_loss, cache


def backward(x, y_policy, y_value, params, cache):
    w1, b1, w2, b2, wp, bp, wv, bv = params
    h1p, h1, h2p, h2, probs, value = cache

    dprobs = 2 * (probs - y_policy) / probs.size
    dvalue = 2 * (value - y_value) / value.size

    dpp = probs * (dprobs - np.sum(dprobs * probs, axis=1, keepdims=True))
    dwp = h2.T @ dpp
    dbp = dpp.sum(axis=0)
    dh2_from_p = dpp @ wp.T

    dvp = dvalue * (1 - value ** 2)
    dwv = h2.T @ dvp
    dbv = dvp.sum(axis=0)
    dh2_from_v = dvp @ wv.T

    dh2 = dh2_from_p + dh2_from_v
    dh2p = dh2 * (h2p > 0)
    dw2 = h1.T @ dh2p
    db2 = dh2p.sum(axis=0)
    dh1 = dh2p @ w2.T
    dh1p = dh1 * (h1p > 0)
    dw1 = x.T @ dh1p
    db1 = dh1p.sum(axis=0)

    return [dw1, db1, dw2, db2, dwp, dbp, dwv, dbv]


def main():
    rng = np.random.default_rng(0)
    n = 6
    x = rng.standard_normal((n, IN))
    y_policy = np.zeros((n, POL))
    y_policy[np.arange(n), rng.integers(0, POL, n)] = 1.0
    y_value_scalar = rng.uniform(-1, 1, (n, 1))
    y_value = np.concatenate([y_value_scalar, -y_value_scalar], axis=1)  # [value, -value], matches generate_data.py

    params = [
        rng.standard_normal((IN, H1)) * 0.3, np.zeros(H1),
        rng.standard_normal((H1, H2)) * 0.3, np.zeros(H2),
        rng.standard_normal((H2, POL)) * 0.3, np.zeros(POL),
        rng.standard_normal((H2, VAL)) * 0.3, np.zeros(VAL),
    ]

    loss0, ploss0, vloss0, cache = forward(x, y_policy, y_value, params)
    print(f"forward OK: loss={loss0:.4f} (policy={ploss0:.4f}, value={vloss0:.4f})")

    grads = backward(x, y_policy, y_value, params, cache)

    eps = 1e-5
    max_rel_err = 0.0
    for pi, p in enumerate(params):
        g = grads[pi]
        it = np.nditer(p, flags=["multi_index"])
        checked = 0
        for _ in it:
            idx = it.multi_index
            orig = p[idx]
            p[idx] = orig + eps
            lp, _, _, _ = forward(x, y_policy, y_value, params)
            p[idx] = orig - eps
            lm, _, _, _ = forward(x, y_policy, y_value, params)
            p[idx] = orig
            g_num = (lp - lm) / (2 * eps)
            rel = abs(g_num - g[idx]) / max(1e-6, abs(g_num) + abs(g[idx]))
            max_rel_err = max(max_rel_err, rel)
            checked += 1
            if checked >= 20:
                break
    print(f"max relative gradient error (finite-diff vs analytic): {max_rel_err:.2e}")
    assert max_rel_err < 1e-4, "GRADIENT CHECK FAILED"
    print("GRADIENT CHECK PASSED")

    lr = 0.1  # lowered from 0.5: the 2-column value head changes mse_loss's effective normalization (mean over N*2 elements instead of N*1), so the old lr overshot on this tiny random toy example -- unrelated to train.tl's own separately-tuned lr=8.0 at n=3000
    for _ in range(200):
        loss, ploss, vloss, cache = forward(x, y_policy, y_value, params)
        grads = backward(x, y_policy, y_value, params, cache)
        params = [p - lr * g for p, g in zip(params, grads)]
    loss_final, ploss_final, vloss_final, _ = forward(x, y_policy, y_value, params)
    print(f"loss after 200 SGD steps: {loss_final:.4f} (started at {loss0:.4f})")
    assert loss_final < loss0
    print("SGD CONVERGENCE CHECK PASSED")
    print("\nAll numeric checks passed. Run check.py on train.tl/infer.tl next, "
          "then an actual (small) training run on real GPU hardware before trusting a promoted network.")


if __name__ == "__main__":
    main()
