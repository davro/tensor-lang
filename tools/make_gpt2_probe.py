#!/usr/bin/env python3
"""Generate tests/probe_gpt2_forward.tl: a one-layer pre-LN GPT-2-style forward pass with
NON-DEGENERATE seeded weights written as 2-decimal literals, plus its NumPy reference.

  python3 make_gpt2_probe.py [OUT.tl]     # sensitivity check, then write the probe (default tests/probe_gpt2_forward.tl)
  python3 make_gpt2_probe.py --check      # sensitivity check only
  python3 make_gpt2_probe.py --search     # find a seed with large logits and float32 margin under the runner tolerance

Shapes: seq=4, d=4, vocab=8, ffn=8, one head. Architecture (GPT-2 pre-LN):
  x0 = onehot @ w_tok + w_pos
  x1 = x0 + (softmax(causal(0.5 * q k^T)) v) @ w_o + b_o,     q,k,v from LN1(x0)
  x2 = x1 + gelu(LN2(x1) @ w_f1 + b_f1) @ w_f2 + b_f2
  logits = LNf(x2) @ w_out + b_out
The reference is float64 on the exact decimals in the file (gelu = erf form, layer_norm eps = 1e-5,
population variance), which is what TensorLang computes. The runner compares with rtol=1e-5, atol=1e-8.
"""
import json, math, sys
import numpy as np

SEED = 32   # chosen with --search: all |logits| >= 0.15 and float32 error is ~4% of the runner tolerance
SEQ, D, V, FF = 4, 4, 8, 8
TOKENS = [2, 5, 2, 7]
CAUSAL = np.arange(SEQ)[:, None] - np.arange(SEQ)[None, :] >= 0
W = B = G = ONEHOT = None

def init(seed):
    global W, B, G, ONEHOT
    rng = np.random.default_rng(seed)
    mat = lambda r, c, s=0.5: np.round(rng.normal(0, s, (r, c)), 2)
    vec = lambda n, s=0.2: np.round(rng.normal(0, s, n), 2)
    gam = lambda n: np.round(1 + rng.normal(0, 0.15, n), 2)
    W = dict(w_tok=mat(V, D), w_pos=mat(SEQ, D), w_q=mat(D, D), w_k=mat(D, D), w_v=mat(D, D), w_o=mat(D, D),
             w_f1=mat(D, FF), w_f2=mat(FF, D), w_out=mat(D, V))
    B = dict(b_q=vec(D), b_k=vec(D), b_v=vec(D), b_o=vec(D), b_f1=vec(FF), b_f2=vec(D), b_out=vec(V))
    G = dict(g1=gam(D), e1=vec(D), g2=gam(D), e2=vec(D), gf=gam(D), ef=vec(D))
    ONEHOT = np.zeros((SEQ, V)); ONEHOT[np.arange(SEQ), TOKENS] = 1

erf64 = np.vectorize(math.erf)
def ln(x, dt=np.float64):
    m = x.mean(1, keepdims=True); v = ((x - m) ** 2).mean(1, keepdims=True)
    return (x - m) / np.sqrt(v + dt(1e-5))
def gelu(x): return 0.5 * x * (1 + erf64(x / math.sqrt(2)))
def gelu_tanh(x): return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
def softmax(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def forward(mask=True, scale=0.5, uniform_attn=False, attn_residual=True, ffn_residual=True,
            act=gelu, bias_before_proj=False, final_ln=True):
    x0 = ONEHOT @ W["w_tok"] + W["w_pos"]
    h1 = ln(x0) * G["g1"] + G["e1"]
    q, k, v = h1 @ W["w_q"] + B["b_q"], h1 @ W["w_k"] + B["b_k"], h1 @ W["w_v"] + B["b_v"]
    s = (q @ k.T) * scale + (np.where(CAUSAL, 0.0, -1e9) if mask else 0.0)
    a = softmax(s) if not uniform_attn else np.full((SEQ, SEQ), 1.0 / SEQ)
    ctx = a @ v
    ap = (ctx + B["b_o"]) @ W["w_o"] if bias_before_proj else ctx @ W["w_o"] + B["b_o"]
    x1 = x0 + ap if attn_residual else ap
    h2 = ln(x1) * G["g2"] + G["e2"]
    f = act(h2 @ W["w_f1"] + B["b_f1"]) @ W["w_f2"] + B["b_f2"]
    x2 = x1 + f if ffn_residual else f
    hf = ln(x2) * G["gf"] + G["ef"] if final_ln else x2
    return hf @ W["w_out"] + B["b_out"]

def forward_f32():
    """Float32 emulation: a proxy for the GPU run (the real kernels differ by a few ulp)."""
    c = lambda a: np.asarray(a, dtype=np.float32)
    Wf, Bf, Gf, oh = ({k: c(v) for k, v in d.items()} for d in (W, B, G, {"o": ONEHOT}))
    oh = oh["o"]; f = np.float32
    def ln32(x):
        m = x.mean(1, keepdims=True, dtype=f); v = ((x - m) ** 2).mean(1, keepdims=True, dtype=f)
        return (x - m) / np.sqrt(v + f(1e-5))
    erf32 = np.vectorize(lambda z: math.erf(float(z)), otypes=[np.float32])
    gelu32 = lambda x: f(.5) * x * (f(1) + erf32(x / f(math.sqrt(2))))
    def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    x0 = oh @ Wf["w_tok"] + Wf["w_pos"]; h1 = ln32(x0) * Gf["g1"] + Gf["e1"]
    q, k, v = h1 @ Wf["w_q"] + Bf["b_q"], h1 @ Wf["w_k"] + Bf["b_k"], h1 @ Wf["w_v"] + Bf["b_v"]
    s = (q @ k.T) * f(.5) + np.where(CAUSAL, f(0), f(-1e9))
    x1 = x0 + (sm(s) @ v) @ Wf["w_o"] + Bf["b_o"]
    h2 = ln32(x1) * Gf["g2"] + Gf["e2"]
    x2 = x1 + gelu32(h2 @ Wf["w_f1"] + Bf["b_f1"]) @ Wf["w_f2"] + Bf["b_f2"]
    return (ln32(x2) * Gf["gf"] + Gf["ef"]) @ Wf["w_out"] + Bf["b_out"]

MUTATIONS = {"no causal mask": dict(mask=False), "scale 0.5 -> 1.0": dict(scale=1.0),
             "uniform attention": dict(uniform_attn=True), "no attention residual": dict(attn_residual=False),
             "no FFN residual": dict(ffn_residual=False), "gelu -> relu": dict(act=lambda z: np.maximum(z, 0)),
             "bias before projection": dict(bias_before_proj=True), "no final layer norm": dict(final_ln=False)}

def sensitivity(verbose=True):
    base = forward(); ok = True
    if verbose: print(f"seed {SEED}: logits min|.|={np.abs(base).min():.3f}  max|.|={np.abs(base).max():.3f}")
    for n, kw in MUTATIONS.items():
        d = np.abs(forward(**kw) - base).max(); good = d > 1e-4; ok &= good
        if verbose: print(f"  {n:26s} max change {d:.2e}  -> {'noticed' if good else 'NOT NOTICED'}")
    d = np.abs(forward(act=gelu_tanh) - base).max()
    if verbose: print(f"  {'gelu erf -> tanh form':26s} max change {d:.2e}  (legitimate variant; a real-weights run must use the matching form)")
    return ok

def margin():
    base = forward(); d = np.abs(forward_f32() - base)
    return (d / (1e-8 + 1e-5 * np.abs(base))).max()        # < 1 passes the runner

def lit(a):
    if a.ndim == 1: return "[" + ", ".join(f"{x:.2f}" for x in a) + "]"
    return "[" + ",\n      ".join("[" + ", ".join(f"{x:.2f}" for x in r) + "]" for r in a) + "]"

def build():
    L = []
    def t(name, a):
        shape = f"({a.shape[0]}, {a.shape[1]})" if a.ndim == 2 else f"({a.shape[0]},)"
        L.append(f"let {name}: Tensor[f32, {shape}] = {lit(a)}")
    exp = forward()
    L.append(f"""// tests/probe_gpt2_forward.tl
// ============================================================================
// TensorLang: one-layer pre-LN GPT-2-style forward pass (seq 4, d 4, vocab 8, ffn 8, one head)
// ============================================================================
// Built only from existing operations. Weights are seeded, DISTINCT values written as literals, the
// input is a real one-hot (tokens {TOKENS}), and every intermediate is O(1), so a wrong mask, scale,
// residual, activation or bias placement changes the logits (see make_gpt2_probe.py --check).
// Generated by make_gpt2_probe.py (seed {SEED}); expected values are float64 on the exact decimals below.
//
// @EXPECTED
// {json.dumps({"logits": np.round(exp, 7).tolist()})}
""")
    t("x_tok", ONEHOT)
    for d in (W, B, G):
        for k_, v_ in d.items(): t(k_, v_)
    L.append("""let scale: Tensor[f32, (1,)] = [0.5]
let neg: Tensor[f32, (1,)] = [-1000000000.0]
let one: Tensor[f32, (1,)] = [1.0]
let row: Tensor[f32, (4, 1)] = [[0.0], [1.0], [2.0], [3.0]]
let col: Tensor[f32, (1, 4)] = [[0.0, 1.0, 2.0, 3.0]]

// --- embeddings ---
let tok_emb = matmul(x_tok, w_tok)
let x0 = add(tok_emb, w_pos)

// --- attention block (pre-LN) ---
let n1 = layer_norm(x0, axis=1)
let h1 = add(mult(n1, g1), e1)
let q_raw = matmul(h1, w_q)
let q = add(q_raw, b_q)
let k_raw = matmul(h1, w_k)
let k = add(k_raw, b_k)
let v_raw = matmul(h1, w_v)
let v = add(v_raw, b_v)
let kt = transpose(k, axes=(1, 0))
let qk = matmul(q, kt)
let scores = mult(qk, scale)

// causal mask: 0 where row >= col, -1e9 where blocked
let diff = minus(row, col)
let allowed = diff >= 0.0
let blocked = minus(one, allowed)
let mask = mult(blocked, neg)
let masked = add(scores, mask)
let attnw = softmax(masked, axis=1)
let ctx = matmul(attnw, v)
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
""")
    return "\n".join(L)

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--search" in sys.argv:
        best = None
        for seed in range(1, 500):
            SEED = seed; init(seed)
            if np.abs(forward()).min() < 0.15 or not sensitivity(False): continue
            m = margin()
            if best is None or m < best[1]: best = (seed, m, np.abs(forward()).min())
        print("best (seed, f32-vs-f64 error / runner tolerance, min|logit|):", best); sys.exit(0)
    init(SEED)
    ok = sensitivity()
    if "--check" in sys.argv: sys.exit(0 if ok else 1)
    assert ok, "probe is not sensitive to a real error"
    print(f"float32 emulation vs expected: error / runner tolerance = {margin():.3f} (needs to be well below 1)")
    out = args[0] if args else "tests/probe_gpt2_forward.tl"
    open(out, "w").write(build()); print("wrote", out)
