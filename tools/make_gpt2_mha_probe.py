#!/usr/bin/env python3
"""Generate tests/probe_gpt2_multihead.tl: a one-layer pre-LN GPT-2 block with TWO heads, a fused QKV
projection split by slicing, and token/position `embedding` -- the shapes a real GPT-2 checkpoint uses.

    seq 4, d 4, heads 2 (head_dim 2), ffn 8, vocab 8

Like make_gpt2_probe.py: seeded DISTINCT weights written as 2-decimal literals, expected logits computed in
float64 on those exact decimals, and a mutation check proving that each plausible multi-head bug (wrong
head split, wrong merge, k/v swapped, softmax over the wrong axis, ...) changes the logits. A probe that no
bug can fail is worthless (see the degenerate probe in the v0.5.7 notes), so the script refuses to write
unless every mutation is noticed and the logits are not near zero.

    python3 tools/make_gpt2_mha_probe.py            # search a seed, check, write tests/probe_gpt2_multihead.tl
    python3 tools/make_gpt2_mha_probe.py --seed 5   # fixed seed
"""
import argparse
import json
import math
import sys

import numpy as np

SEQ, D, H, FFN, VOCAB = 4, 4, 2, 8, 8
HD = D // H
SCALE = 0.70710678          # literal for 1/sqrt(head_dim=2); the reference uses the same decimal
TOKENS = [2, 5, 2, 7]       # repeated token 2: exercises embedding gather
POSITIONS = [0, 1, 2, 3]
EPS = 1e-5                  # TensorLang layer_norm default
MIN_CHANGE = 2e-2           # a mutation must move some logit by at least this much
OUT = "tests/probe_gpt2_multihead.tl"

_erf = np.vectorize(math.erf)


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x * 0.70710678))


def layer_norm(x):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)   # biased, as the kernel
    return (x - mu) / np.sqrt(var + EPS)


def softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def make_params(seed):
    rng = np.random.default_rng(seed)

    def mat(r, c, s=0.5):
        return np.round(rng.normal(0, s, (r, c)), 2)

    def vec(n, s=0.15):
        return np.round(rng.normal(0, s, n), 2)

    def gamma(n):
        return np.round(1.0 + rng.normal(0, 0.15, n), 2)

    return dict(
        w_tok=mat(VOCAB, D), w_pos=mat(SEQ, D),
        w_qkv=mat(D, 3 * D), b_qkv=vec(3 * D),
        w_o=mat(D, D), b_o=vec(D),
        w_f1=mat(D, FFN), b_f1=vec(FFN), w_f2=mat(FFN, D), b_f2=vec(D),
        w_out=mat(D, VOCAB), b_out=vec(VOCAB),
        g1=gamma(D), e1=vec(D), g2=gamma(D), e2=vec(D), gf=gamma(D), ef=vec(D),
    )


def forward(p, mut=None, scale=SCALE, heads=H):
    """float64 reference. `mut` names one deliberate bug (None = the correct block)."""
    hd = D // heads
    x = p["w_tok"][TOKENS] + p["w_pos"][POSITIONS]
    h1 = layer_norm(x) * p["g1"] + p["e1"]
    qkv = h1 @ p["w_qkv"] + p["b_qkv"]
    q, k, v = qkv[:, 0:D], qkv[:, D:2 * D], qkv[:, 2 * D:3 * D]
    if mut == "swap_kv":
        k, v = v, k

    def split(t):
        if mut == "split_no_transpose":                       # reshape straight to (heads, seq, hd)
            return t.reshape(heads, SEQ, hd)
        return t.reshape(SEQ, heads, hd).transpose(1, 0, 2)

    qh, kh, vh = split(q), split(k), split(v)
    s = (qh @ kh.transpose(0, 2, 1)) * scale
    if mut != "no_mask":
        row = np.arange(SEQ)[:, None]
        col = np.arange(SEQ)[None, :]
        s = s + (1.0 - (row - col >= 0)) * -1e9
    a = softmax(s, axis=1 if mut == "softmax_wrong_axis" else 2)
    c = a @ vh
    if mut == "merge_no_transpose":                           # reshape (heads, seq, hd) -> (seq, d) directly
        c = c.reshape(SEQ, D)
    else:
        c = c.transpose(1, 0, 2).reshape(SEQ, D)
    attn = c @ p["w_o"] + p["b_o"]
    x1 = attn if mut == "no_attn_residual" else x + attn
    h2 = layer_norm(x1) * p["g2"] + p["e2"]
    f = gelu(h2 @ p["w_f1"] + p["b_f1"]) @ p["w_f2"] + p["b_f2"]
    x2 = f if mut == "no_ffn_residual" else x1 + f
    hf = layer_norm(x2) * p["gf"] + p["ef"]
    return hf @ p["w_out"] + p["b_out"]


MUTATIONS = ["no_mask", "split_no_transpose", "merge_no_transpose", "swap_kv", "softmax_wrong_axis",
             "no_attn_residual", "no_ffn_residual", "wrong_scale", "one_head"]


def run_mutation(p, m):
    if m == "wrong_scale":
        return forward(p, scale=1.0)
    if m == "one_head":
        return forward(p, heads=1, scale=0.5)   # same weights, treated as a single 4-wide head
    return forward(p, mut=m)


def check(p, verbose=True):
    good = forward(p)
    ok = True
    lo, hi = np.abs(good).min(), np.abs(good).max()
    if verbose:
        print(f"logits min|.|={lo:.3f}  max|.|={hi:.3f}")
    if lo < 0.1 or hi > 4.0:
        ok = False
    for m in MUTATIONS:
        change = np.abs(run_mutation(p, m) - good).max()
        noticed = change >= MIN_CHANGE
        ok &= bool(noticed)
        if verbose:
            print(f"  {m:20s} max change {change:.3e}  -> {'noticed' if noticed else 'MISSED'}")
    # float32 emulation: cast the parameters to float32 and compare, as the GPU will see the literals
    p32 = {k: v.astype(np.float32).astype(np.float64) for k, v in p.items()}
    err = np.abs(forward(p32) - good).max()
    if verbose:
        print(f"float32-rounded literals vs exact decimals: max err {err:.2e}")
    return ok


def lit(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return "[" + ", ".join(f"{v:.2f}" for v in a) + "]"
    return "[" + ",\n      ".join(lit(r) for r in a) + "]"


def shape(a):
    return "(" + ", ".join(str(d) for d in a.shape) + ("," if a.ndim == 1 else "") + ")"


def emit(p, seed):
    expected = np.round(forward(p), 7).tolist()
    L = []
    w = L.append
    w("// tests/probe_gpt2_multihead.tl")
    w("// " + "=" * 76)
    w(f"// TensorLang: one-layer pre-LN GPT-2 block with TWO heads (seq {SEQ}, d {D}, heads {H}, head_dim {HD}, ffn {FFN}, vocab {VOCAB})")
    w("// " + "=" * 76)
    w("// What this adds over probe_gpt2_forward: the shapes a real checkpoint uses. Token and position")
    w("// `embedding` (token 2 repeats), a FUSED QKV projection split by slicing, the head split")
    w("// reshape -> transpose(1,0,2), batched (heads, seq, seq) attention with a broadcast causal mask,")
    w("// softmax over the last axis, and the head merge transpose(1,0,2) -> reshape.")
    w("// Weights are seeded DISTINCT 2-decimal literals; expected logits are float64 on those decimals.")
    w("// Each plausible head-handling bug changes the logits (see make_gpt2_mha_probe.py).")
    w(f"// Generated by tools/make_gpt2_mha_probe.py (seed {seed}).")
    w("//")
    w("// @EXPECTED")
    w("// " + json.dumps({"logits": expected}))
    w("")
    w(f"let tokens: Tensor[f32, ({SEQ},)] = {lit(np.array(TOKENS, dtype=float))}")
    w(f"let positions: Tensor[f32, ({SEQ},)] = {lit(np.array(POSITIONS, dtype=float))}")
    for name in ["w_tok", "w_pos", "w_qkv", "w_o", "w_f1", "w_f2", "w_out"]:
        w(f"let {name}: Tensor[f32, {shape(p[name])}] = {lit(p[name])}")
    for name in ["b_qkv", "b_o", "b_f1", "b_f2", "b_out", "g1", "e1", "g2", "e2", "gf", "ef"]:
        w(f"let {name}: Tensor[f32, {shape(p[name])}] = {lit(p[name])}")
    w(f"let scale: Tensor[f32, (1,)] = [{SCALE}]")
    w("let neg: Tensor[f32, (1,)] = [-1000000000.0]")
    w("let one: Tensor[f32, (1,)] = [1.0]")
    w(f"let row: Tensor[f32, ({SEQ}, 1)] = [" + ", ".join(f"[{i}.0]" for i in range(SEQ)) + "]")
    w(f"let col: Tensor[f32, (1, {SEQ})] = [[" + ", ".join(f"{i}.0" for i in range(SEQ)) + "]]")
    w("""
// --- embeddings (gather) ---
let tok_emb = embedding(tokens, w_tok)
let pos_emb = embedding(positions, w_pos)
let x0 = add(tok_emb, pos_emb)

// --- attention block (pre-LN), fused QKV, two heads ---
let n1 = layer_norm(x0, axis=1)
let h1 = add(mult(n1, g1), e1)
let qkv_raw = matmul(h1, w_qkv)
let qkv = add(qkv_raw, b_qkv)
let q2 = qkv[:, 0:%(d)d]
let k2 = qkv[:, %(d)d:%(d2)d]
let v2 = qkv[:, %(d2)d:%(d3)d]
let q3 = reshape(q2, (%(s)d, %(h)d, %(hd)d))
let k3 = reshape(k2, (%(s)d, %(h)d, %(hd)d))
let v3 = reshape(v2, (%(s)d, %(h)d, %(hd)d))
let qh = transpose(q3, axes=(1, 0, 2))
let kh = transpose(k3, axes=(1, 0, 2))
let vh = transpose(v3, axes=(1, 0, 2))
let kt = transpose(kh, axes=(0, 2, 1))
let qk = matmul(qh, kt)
let scores = mult(qk, scale)

// causal mask (seq, seq), broadcast over the heads axis: 0 where row >= col, -1e9 where blocked
let diff = minus(row, col)
let allowed = diff >= 0.0
let blocked = minus(one, allowed)
let mask = mult(blocked, neg)
let masked = add(scores, mask)
let attnw = softmax(masked, axis=2)
let ctxh = matmul(attnw, vh)
let ctxt = transpose(ctxh, axes=(1, 0, 2))
let ctx = reshape(ctxt, (%(s)d, %(d)d))
let proj = matmul(ctx, w_o)
let attn_out = add(proj, b_o)
let x1 = add(x0, attn_out)

// --- feed-forward block (pre-LN) ---
let n2 = layer_norm(x1, axis=1)
let h2 = add(mult(n2, g2), e2)
let f1_raw = matmul(h2, w_f1)
let f1 = add(f1_raw, b_f1)
let f1g = gelu(f1)
let f2_raw = matmul(f1g, w_f2)
let ffn_out = add(f2_raw, b_f2)
let x2 = add(x1, ffn_out)

// --- final layer norm and output projection ---
let nf = layer_norm(x2, axis=1)
let hf = add(mult(nf, gf), ef)
let out_raw = matmul(hf, w_out)
let logits = add(out_raw, b_out)

logits
""" % dict(d=D, d2=2 * D, d3=3 * D, s=SEQ, h=H, hd=HD))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    seeds = [args.seed] if args.seed is not None else range(1, 200)
    for seed in seeds:
        p = make_params(seed)
        if check(p, verbose=False):
            print(f"seed {seed}:")
            check(p, verbose=True)
            with open(args.out, "w") as f:
                f.write(emit(p, seed))
            print(f"wrote {args.out}")
            return
        if args.seed is not None:
            print(f"seed {seed} rejected:")
            check(p, verbose=True)
    sys.exit("no seed passed the checks")


if __name__ == "__main__":
    main()
