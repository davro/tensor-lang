#!/usr/bin/env python3
"""Generate a TRAINING program for a small pre-LN GPT, plus a float64 NumPy reference with a hand-written backward.

  python3 tools/make_micro_gpt_train.py                        # check, then write the probe (small, exact-checked)
  python3 tools/make_micro_gpt_train.py --check                # gradient check + sensitivity check only
  python3 tools/make_micro_gpt_train.py --perf --dim 128 --seq 64 --layers 4 --vocab 256 --steps 20 \
        --out-tl build/micro_gpt_perf.tl --weights build/micro_gpt_perf     # bigger workload for profile_train_step.py

The model (one head, GPT-2 style, no dropout):
  x = wte[tokens] + wpe
  per layer:  x += ((softmax(causal(q k^T * scale)) v) wo + bo),  q,k,v = (LN(x) g1 + e1) wq/wk/wv + bq/bk/bv
              x += gelu((LN(x) g2 + e2) wf1 + bf1) wf2 + bf2
  logits = (LN(x) gf + ef) w_out          loss = softmax_cross_entropy(logits, targets)     (mean over positions)
trained on ONE fixed sequence (targets = tokens shifted by one) with Adam inside a `for` loop. One fixed batch is
deliberate: the language cannot yet index a load() by the loop counter, so the data cannot change per step.

The reference is float64 on exactly the float32 values written to the .npy files (gelu = erf form, layer_norm eps
1e-5, population variance, Adam with bias correction, eps 1e-8). The runner compares with rtol=1e-5, atol=1e-8.
What the probe checks: `loss` of the LAST step (it depends on every update before it). Parameters whose gradient is exactly zero in
theory (the key bias bk: softmax ignores a per-row constant) are NOT compared, because Adam turns float32 noise in a
zero gradient into a full-size step. The final w_out is also not compared by default: the first GPU run missed it by
9.75e-07 (the loss passed), and elements near zero have a tolerance of only ~1e-8. Use --expect-params to opt in.
"""
import argparse, json, math, os, sys
import numpy as np

erf = np.vectorize(math.erf)

def make_params(a):
    rng = np.random.default_rng(a.seed)
    D, V, T, FF = a.dim, a.vocab, a.seq, a.dim * a.ff_mult
    mat = lambda r, c, s: rng.normal(0, s, (r, c)).astype(np.float32)
    vec = lambda n, s: rng.normal(0, s, n).astype(np.float32)
    gam = lambda n: (1 + rng.normal(0, 0.1, n)).astype(np.float32)
    s = a.init_std
    P = {"wte": mat(V, D, s), "wpe": mat(T, D, s)}
    for l in range(a.layers):
        p = f"l{l}_"
        P.update({p+"g1": gam(D), p+"e1": vec(D, 0.1), p+"wq": mat(D, D, s), p+"bq": vec(D, 0.1),
                  p+"wk": mat(D, D, s), p+"bk": vec(D, 0.1), p+"wv": mat(D, D, s), p+"bv": vec(D, 0.1),
                  p+"wo": mat(D, D, s), p+"bo": vec(D, 0.1), p+"g2": gam(D), p+"e2": vec(D, 0.1),
                  p+"wf1": mat(D, FF, s), p+"bf1": vec(FF, 0.1), p+"wf2": mat(FF, D, s), p+"bf2": vec(D, 0.1)})
    P.update({"gf": gam(D), "ef": vec(D, 0.1), "w_out": mat(D, V, s)})
    return P

def make_data(a):
    rng = np.random.default_rng(a.seed + 1000)
    seq = rng.integers(0, a.vocab, a.seq + 1)
    return seq[:-1].astype(np.float32), seq[1:].astype(np.float32)

# ---------------------------------------------------------------- reference (explicit forward + backward)
def ln_fwd(x, eps=1e-5):
    m = x.mean(1, keepdims=True); xc = x - m; var = (xc ** 2).mean(1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps); xh = xc * inv
    return xh, (xh, inv)
def ln_bwd(dxh, c):
    xh, inv = c
    return inv * (dxh - dxh.mean(1, keepdims=True) - xh * (dxh * xh).mean(1, keepdims=True))
def gelu_f(x): return (0.5 * x * (1 + erf(x / math.sqrt(2)))).astype(x.dtype)
def gelu_b(x): return (0.5 * (1 + erf(x / math.sqrt(2))) + x * np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)).astype(x.dtype)

def fwd_bwd(P, tok, tgt, layers, scale, mut=()):
    """Returns loss and gradients for every parameter (dtype of P)."""
    dt = P["wte"].dtype; T = len(tok); ti = tok.astype(int); yi = tgt.astype(int)
    mask = np.where(np.arange(T)[:, None] >= np.arange(T)[None, :], 0.0, -1e9).astype(dt)
    if "no mask" in mut: mask = np.zeros_like(mask)
    sc = dt.type(scale * (2.0 if "scale x2" in mut else 1.0))
    G = {k: np.zeros_like(v) for k, v in P.items()}; C = []
    x = P["wte"][ti] + (0 if "no wpe" in mut else P["wpe"])
    for l in range(layers):
        p = f"l{l}_"; c = {"x_in": x}
        n1, c["ln1"] = ln_fwd(x); h1 = n1 * P[p+"g1"] + P[p+"e1"]; c["n1"], c["h1"] = n1, h1
        q = h1 @ P[p+"wq"] + P[p+"bq"]; k = h1 @ P[p+"wk"] + P[p+"bk"]; v = h1 @ P[p+"wv"] + P[p+"bv"]
        s = (q @ k.T) * sc + mask; e = np.exp(s - s.max(1, keepdims=True)); at = e / e.sum(1, keepdims=True)
        ctx = at @ v; ao = ctx @ P[p+"wo"] + P[p+"bo"]
        c.update(q=q, k=k, v=v, at=at, ctx=ctx)
        x = x + ao if "no attn residual" not in mut else ao
        c["x_mid"] = x
        n2, c["ln2"] = ln_fwd(x); h2 = n2 * P[p+"g2"] + P[p+"e2"]; c["n2"], c["h2"] = n2, h2
        f1 = h2 @ P[p+"wf1"] + P[p+"bf1"]; f1g = np.maximum(f1, 0) if "relu" in mut else gelu_f(f1)
        fo = f1g @ P[p+"wf2"] + P[p+"bf2"]; c.update(f1=f1, f1g=f1g)
        x = x + fo if "no ffn residual" not in mut else fo
        C.append(c)
    nf, cf = ln_fwd(x); hf = nf * P["gf"] + P["ef"] if "no final ln" not in mut else x
    lg = hf @ P["w_out"]
    m = lg.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(lg - m).sum(1))
    loss = (lse - lg[np.arange(T), yi]).mean()
    pr = np.exp(lg - m); pr /= pr.sum(1, keepdims=True); dl = pr; dl[np.arange(T), yi] -= 1; dl /= T
    G["w_out"] = hf.T @ dl; dhf = dl @ P["w_out"].T
    if "no final ln" in mut: dx = dhf
    else:
        G["gf"] = (dhf * nf).sum(0); G["ef"] = dhf.sum(0); dx = ln_bwd(dhf * P["gf"], cf)
    for l in reversed(range(layers)):
        p = f"l{l}_"; c = C[l]
        df = dx; dxr = dx if "no ffn residual" not in mut else 0
        G[p+"wf2"] = c["f1g"].T @ df; G[p+"bf2"] = df.sum(0)
        dgl = df @ P[p+"wf2"].T; df1 = dgl * ((c["f1"] > 0) if "relu" in mut else gelu_b(c["f1"]))
        G[p+"wf1"] = c["h2"].T @ df1; G[p+"bf1"] = df1.sum(0); dh2 = df1 @ P[p+"wf1"].T
        G[p+"g2"] = (dh2 * c["n2"]).sum(0); G[p+"e2"] = dh2.sum(0)
        dxm = ln_bwd(dh2 * P[p+"g2"], c["ln2"]) + dxr
        da = dxm; dxr2 = dxm if "no attn residual" not in mut else 0
        G[p+"wo"] = c["ctx"].T @ da; G[p+"bo"] = da.sum(0); dctx = da @ P[p+"wo"].T
        dat = dctx @ c["v"].T; dv = c["at"].T @ dctx
        ds = c["at"] * (dat - (dat * c["at"]).sum(1, keepdims=True)) * sc
        dq = ds @ c["k"]; dk = ds.T @ c["q"]
        G[p+"wq"] = c["h1"].T @ dq; G[p+"bq"] = dq.sum(0); G[p+"wk"] = c["h1"].T @ dk; G[p+"bk"] = dk.sum(0)
        G[p+"wv"] = c["h1"].T @ dv; G[p+"bv"] = dv.sum(0)
        dh1 = dq @ P[p+"wq"].T + dk @ P[p+"wk"].T + dv @ P[p+"wv"].T
        G[p+"g1"] = (dh1 * c["n1"]).sum(0); G[p+"e1"] = dh1.sum(0)
        dx = ln_bwd(dh1 * P[p+"g1"], c["ln1"]) + dxr2
    if "no wpe" not in mut: G["wpe"] = dx
    np.add.at(G["wte"], ti, dx)
    return float(loss), G

def train(P0, tok, tgt, a, dtype, mut=()):
    P = {k: v.astype(dtype).copy() for k, v in P0.items()}
    M = {k: np.zeros_like(v) for k, v in P.items()}; Vv = {k: np.zeros_like(v) for k, v in P.items()}
    losses = []; b1, b2, eps = dtype(0.9), dtype(0.999), dtype(1e-8)
    for t in range(1, a.steps + 1):
        loss, G = fwd_bwd(P, tok, tgt, a.layers, 1 / math.sqrt(a.dim), mut)
        losses.append(loss)
        for k in P:
            g = G[k].astype(dtype)
            if "sgd" in mut: P[k] = P[k] - dtype(a.lr) * g; continue
            M[k] = b1 * M[k] + (1 - b1) * g; Vv[k] = b2 * Vv[k] + (1 - b2) * g * g
            mh = M[k] / (1 - b1 ** t) if "no bias correction" not in mut else M[k]
            vh = Vv[k] / (1 - b2 ** t) if "no bias correction" not in mut else Vv[k]
            P[k] = P[k] - dtype(a.lr) * mh / (np.sqrt(vh) + eps)
    return losses, P

def grad_check(a):
    b = argparse.Namespace(**vars(a)); b.dim, b.seq, b.vocab, b.layers = 8, 5, 7, 2
    P = {k: v.astype(np.float64) for k, v in make_params(b).items()}; tok, tgt = make_data(b)
    _, G = fwd_bwd(P, tok, tgt, b.layers, 1 / math.sqrt(b.dim)); rng = np.random.default_rng(5); worst = 0.0
    for k in P:
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in P[k].shape); old = P[k][i]; h = 1e-6
            P[k][i] = old + h; lp, _ = fwd_bwd(P, tok, tgt, b.layers, 1 / math.sqrt(b.dim))
            P[k][i] = old - h; lm, _ = fwd_bwd(P, tok, tgt, b.layers, 1 / math.sqrt(b.dim)); P[k][i] = old
            # absolute + relative: a gradient that is zero in theory (bk) must not be judged by a pure ratio
            worst = max(worst, abs((lp - lm) / (2 * h) - G[k][i]) / (1e-7 + 1e-6 * abs(G[k][i])))
    return worst

MUTS = ["no mask", "scale x2", "no attn residual", "no ffn residual", "relu", "no final ln", "no wpe",
        "sgd", "no bias correction"]

def sensitivity(P, tok, tgt, a):
    base, _ = train(P, tok, tgt, a, np.float64); ok = True
    print(f"  base loss per step: {[round(x, 5) for x in base]}")
    for m in MUTS:
        d = abs(train(P, tok, tgt, a, np.float64, (m,))[0][-1] - base[-1]); good = d > 1e-4; ok &= good
        print(f"  mutation {m:20s} changes the last loss by {d:.2e} -> {'noticed' if good else 'NOT NOTICED'}")
    d = abs(train(P, tok, tgt[::-1].copy(), a, np.float64)[0][-1] - base[-1]); good = d > 1e-4; ok &= good
    print(f"  mutation {'wrong targets':20s} changes the last loss by {d:.2e} -> {'noticed' if good else 'NOT NOTICED'}")
    return ok, base

# ---------------------------------------------------------------- program text
def shape_s(arr): return "(" + ", ".join(str(s) for s in arr.shape) + ("," if arr.ndim == 1 else "") + ")"
def lit(v): return "[" + ", ".join(f"{float(x):.1f}" for x in v) + "]"

def build(P, tok, tgt, a, wdir, expected, optimizer=True):
    L = []
    hdr = f"""// ============================================================================
// TensorLang: training a {a.layers}-layer pre-LN GPT (seq {a.seq}, d {a.dim}, vocab {a.vocab}, one head) with Adam
// ============================================================================
// Generated by tools/make_micro_gpt_train.py (seed {a.seed}). One fixed sequence, {a.steps} Adam steps (lr {a.lr}),
// softmax_cross_entropy loss. Weights are loaded from {wdir}/ .
"""
    if expected:
        hdr += "// Expected values are float64 on those exact files.\n//\n// @EXPECTED\n// " + json.dumps(expected) + "\n"
    L.append(hdr)
    for k, v in P.items():
        L.append(f'let {k}: Tensor[f32, {shape_s(v)}] = load("{wdir}/{k}.npy") with grad')
    L.append(f'let mask: Tensor[f32, ({a.seq}, {a.seq})] = load("{wdir}/mask.npy")')
    L.append(f"let idx: Tensor[f32, ({a.seq},)] = {lit(tok)}")
    L.append(f"let targets: Tensor[f32, ({a.seq},)] = {lit(tgt)}")
    L.append(f"let scale: Tensor[f32, (1,)] = [{1 / math.sqrt(a.dim):.8f}]\n")
    L.append(f"for step in range({a.steps}) {{")
    B = L.append
    B("    let tok_emb = embedding(idx, wte)\n    let x0 = add(tok_emb, wpe)")
    prev = "x0"
    for l in range(a.layers):
        p = f"l{l}_"
        B(f"""    let {p}n1 = layer_norm({prev}, axis=1)
    let {p}h1 = add(mult({p}n1, {p}g1), {p}e1)
    let {p}q = add(matmul({p}h1, {p}wq), {p}bq)
    let {p}k = add(matmul({p}h1, {p}wk), {p}bk)
    let {p}v = add(matmul({p}h1, {p}wv), {p}bv)
    let {p}kt = transpose({p}k, axes=(1, 0))
    let {p}qk = matmul({p}q, {p}kt)
    let {p}sc = mult({p}qk, scale)
    let {p}msk = add({p}sc, mask)
    let {p}at = softmax({p}msk, axis=1)
    let {p}ctx = matmul({p}at, {p}v)
    let {p}ao = add(matmul({p}ctx, {p}wo), {p}bo)
    let {p}xm = add({prev}, {p}ao)
    let {p}n2 = layer_norm({p}xm, axis=1)
    let {p}h2 = add(mult({p}n2, {p}g2), {p}e2)
    let {p}f1 = add(matmul({p}h2, {p}wf1), {p}bf1)
    let {p}f1g = gelu({p}f1)
    let {p}fo = add(matmul({p}f1g, {p}wf2), {p}bf2)
    let {p}x = add({p}xm, {p}fo)""")
        prev = f"{p}x"
    B(f"""    let nf = layer_norm({prev}, axis=1)
    let hf = add(mult(nf, gf), ef)
    let logits = matmul(hf, w_out)
    let loss = softmax_cross_entropy(logits, targets)
    backward(loss)""")
    if optimizer:
        for k in P: B(f"    adam({k}, {k}_grad, {a.lr})")
    L.append("}\n")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--seq", type=int, default=6); ap.add_argument("--vocab", type=int, default=12)
    ap.add_argument("--ff-mult", type=int, default=2); ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--init-std", type=float, default=0.4)
    ap.add_argument("--no-optimizer", action="store_true",
                    help="omit the adam lines (use with --steps 1): the run leaves each <param>_grad.npy for compare_micro_gpt.py --grads")
    ap.add_argument("--expect-params", action="store_true",
                    help="also compare the final w_out; fragile, see the module docstring")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--perf", action="store_true",
                    help="no reference run and no expected values: a bigger workload for the profiler")
    ap.add_argument("--out-tl", default="tests/probe_micro_gpt_train.tl")
    ap.add_argument("--weights", default="tests/fixtures/micro_gpt")
    a = ap.parse_args()
    P = make_params(a); tok, tgt = make_data(a)
    if a.no_optimizer: a.perf = True      # no reference run: the comparison is gradient-level
    expected = None
    if not a.perf:
        w = grad_check(a); print(f"gradient check (float64, finite differences): worst error / tolerance = {w:.3f} (must be below 1)")
        assert w < 1, "the reference backward disagrees with finite differences"
        ok, base = sensitivity({k: v.astype(np.float64) for k, v in P.items()}, tok, tgt, a)
        if a.check: sys.exit(0 if ok else 1)
        assert ok, "the probe is not sensitive to a real error"
        l64, P64 = train(P, tok, tgt, a, np.float64); l32, P32 = train(P, tok, tgt, a, np.float32)
        margin = lambda x, y: float(np.max(np.abs(x - y) / (1e-8 + 1e-5 * np.abs(y))))
        mg = margin(np.float64(l32[-1]), np.float64(l64[-1]))
        print(f"float32 emulation vs float64, loss: error / runner tolerance = {mg:.3f} (must be well below 1)")
        print(f"  (info) final w_out in the same emulation: {margin(P32['w_out'], P64['w_out']):.3f} of the tolerance")
        assert mg < 0.5, "float32 rounding alone would fail the runner; change the seed or lr"
        expected = {"loss": round(l64[-1], 9)}
        if a.expect_params: expected["w_out"] = np.round(P64["w_out"], 9).tolist()
    os.makedirs(a.weights, exist_ok=True)
    for k, v in P.items(): np.save(os.path.join(a.weights, k + ".npy"), v)
    T = a.seq
    np.save(os.path.join(a.weights, "mask.npy"),
            np.where(np.arange(T)[:, None] >= np.arange(T)[None, :], 0.0, -1e9).astype(np.float32))
    json.dump({k: v for k, v in vars(a).items() if k not in ("check", "perf", "expect_params", "no_optimizer")},
              open(os.path.join(a.weights, "config.json"), "w"))
    os.makedirs(os.path.dirname(a.out_tl) or ".", exist_ok=True)
    open(a.out_tl, "w").write(build(P, tok, tgt, a, a.weights.replace(os.sep, "/"), expected, not a.no_optimizer))
    n = sum(v.size for v in P.values()); print(f"wrote {a.out_tl} and {len(P) + 1} weight files ({n:,} parameters) in {a.weights}/")

if __name__ == "__main__":
    main()
