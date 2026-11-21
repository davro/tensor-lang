#!/usr/bin/env python3
import argparse, json, os, math
import numpy as np

EPS = 1e-5

def ld(wdir, name):
    return np.load(os.path.join(wdir, name + ".npy"))

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

def capture_f1(wdir, cfg, ids, layer):
    """Return the MLP hidden (post-GELU) vector for the last token at `layer`."""
    L, D, H = cfg["n_layer"], cfg["n_embd"], cfg["n_head"]
    HD, S = D // H, len(ids)
    def Ld(n): return ld(wdir, n).astype(np.float64)
    wte = Ld("wte")
    x = wte[ids] + Ld("wpe")[:S]
    causal = np.tril(np.ones((S, S), dtype=bool))
    for i in range(L):
        p = lambda n: Ld(f"h{i}_{n}")
        # attention (same as reference_logits)
        h = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x-x.mean(-1,keepdims=True))**2).mean(-1,keepdims=True)+EPS)
        h = h * p("ln1_g") + p("ln1_b")
        qkv = h @ p("qkv_w") + p("qkv_b")
        q,k,v = (t.reshape(S,H,HD).transpose(1,0,2) for t in (qkv[:,:D], qkv[:,D:2*D], qkv[:,2*D:]))
        s = (q @ k.transpose(0,2,1)) / math.sqrt(HD)
        s = np.where(causal, s, -1e9)
        e = np.exp(s - s.max(-1, keepdims=True))
        a = e / e.sum(-1, keepdims=True)
        c = (a @ v).transpose(1,0,2).reshape(S,D)
        x = x + c @ p("proj_w") + p("proj_b")
        if i == layer:
            # x is now post-attention residual (x1); compute MLP input h2
            xn = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x-x.mean(-1,keepdims=True))**2).mean(-1,keepdims=True)+EPS)
            h2 = xn * p("ln2_g") + p("ln2_b")
            f1_pre = h2 @ p("fc_w") + p("fc_b")
            f1 = gelu(f1_pre)
            return f1[-1].copy(), x.copy()
        # finish layer
        xn = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x-x.mean(-1,keepdims=True))**2).mean(-1,keepdims=True)+EPS)
        h2 = xn * p("ln2_g") + p("ln2_b")
        f1 = gelu(h2 @ p("fc_w") + p("fc_b"))
        x = x + f1 @ p("out_w") + p("out_b")
    raise RuntimeError("layer out of range")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="build/gpt2_edited")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--prompt-ids", default="464,3139,286,4881,318,262")
    ap.add_argument("--target-token", type=int, required=True, help="token id to push toward")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.weights, "config.json")))
    ids = [int(t) for t in args.prompt_ids.split(",")]

    # baseline prediction
    # (reuse reference_logits logic by importing? We'll just compute argmax quickly)
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", "tools/make_gpt2_demo.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    base = m.reference_logits(args.weights, cfg, ids)
    cur = int(base[-1].argmax())
    print(f"baseline next token: {cur}")

    f1, _ = capture_f1(args.weights, cfg, ids, args.layer)
    wte = ld(args.weights, "wte").astype(np.float64)
    delta = (wte[args.target_token] - wte[cur]).astype(np.float64) * args.alpha

    out_w_path = os.path.join(args.weights, f"h{args.layer}_out_w.npy")
    out_b_path = os.path.join(args.weights, f"h{args.layer}_out_b.npy")
    out_w = np.load(out_w_path).astype(np.float64)  # (F, D)
    out_b = np.load(out_b_path).astype(np.float64)  # (D,)

    denom = float(f1 @ f1) + 1e-6
    out_w = out_w + np.outer(f1, delta) / denom
    out_b = out_b + delta * 1.0

    np.save(out_w_path, out_w.astype(np.float32))
    np.save(out_b_path, out_b.astype(np.float32))
    print(f"edited layer {args.layer}: out_w/out_b updated (alpha={args.alpha})")

    # quick check
    new = m.reference_logits(args.weights, cfg, ids)
    print(f"edited next token: {int(new[-1].argmax())}")

if __name__ == "__main__":
    main()

