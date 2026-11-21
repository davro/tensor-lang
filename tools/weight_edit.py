#!/usr/bin/env python3
"""Small weight-space programming tool for the TensorLang GPT-2 demo.

Examples:
  python3 tools/weight_edit.py --weights build/gpt2 --out build/gpt2_edited \\
      --prompt "Paris is the capital of" --search " the" --replace " Germany" --auto

  python3 tools/weight_edit.py --weights build/gpt2 --out build/gpt2_edited \\
      --prompt "Paris is the capital of" --search " the" --replace " Mordor" --auto
"""
import argparse, json, os, shutil, tempfile
import numpy as np

EPS = 1e-5

def load_tok(wdir):
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(os.path.join(wdir, "tokenizer.json"))
    except Exception:
        return None

def ld(wdir, name):
    return np.load(os.path.join(wdir, name + ".npy"))

def first_id(tok, text):
    ids = tok.encode(text).ids
    if not ids:
        raise SystemExit(f"could not tokenize {text!r}")
    if len(ids) > 1:
        print(f"  note: {text!r} -> {ids}, using first id {ids[0]}")
    return ids[0]

def reference_logits(wdir, cfg, ids):
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", "tools/make_gpt2_demo.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.reference_logits(wdir, cfg, ids)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def capture_f1(wdir, cfg, ids, layer):
    L, D, H = cfg["n_layer"], cfg["n_embd"], cfg["n_head"]
    HD, S = D // H, len(ids)
    def Ld(n): return ld(wdir, n).astype(np.float64)
    wte = Ld("wte")
    x = wte[ids] + Ld("wpe")[:S]
    causal = np.tril(np.ones((S, S), dtype=bool))
    for i in range(L):
        p = lambda n: Ld(f"h{i}_{n}")
        h = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x-x.mean(-1,keepdims=True))**2).mean(-1,keepdims=True)+EPS)
        h = h * p("ln1_g") + p("ln1_b")
        qkv = h @ p("qkv_w") + p("qkv_b")
        q,k,v = (t.reshape(S,H,HD).transpose(1,0,2) for t in (qkv[:,:D], qkv[:,D:2*D], qkv[:,2*D:]))
        s = (q @ k.transpose(0,2,1)) / np.sqrt(HD)
        s = np.where(causal, s, -1e9)
        e = np.exp(s - s.max(-1, keepdims=True))
        a = e / e.sum(-1, keepdims=True)
        c = (a @ v).transpose(1,0,2).reshape(S,D)
        x = x + c @ p("proj_w") + p("proj_b")
        if i == layer:
            xn = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x-x.mean(-1,keepdims=True))**2).mean(-1,keepdims=True)+EPS)
            h2 = xn * p("ln2_g") + p("ln2_b")
            f1 = gelu(h2 @ p("fc_w") + p("fc_b"))
            return f1[-1].copy()
        xn = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x-x.mean(-1,keepdims=True))**2).mean(-1,keepdims=True)+EPS)
        h2 = xn * p("ln2_g") + p("ln2_b")
        f1 = gelu(h2 @ p("fc_w") + p("fc_b"))
        x = x + f1 @ p("out_w") + p("out_b")
    raise RuntimeError("layer out of range")

def apply_edit(wdir, layer, ids, search_id, replace_id, alpha, mode):
    cfg = json.load(open(os.path.join(wdir, "config.json")))
    wte = ld(wdir, "wte").astype(np.float64)
    delta = (wte[replace_id] - wte[search_id]).astype(np.float64) * alpha

    out_w_path = os.path.join(wdir, f"h{layer}_out_w.npy")
    out_b_path = os.path.join(wdir, f"h{layer}_out_b.npy")
    out_w = np.load(out_w_path).astype(np.float64)
    out_b = np.load(out_b_path).astype(np.float64)

    if mode == "bias":
        out_b = out_b + delta
    else:
        f1 = capture_f1(wdir, cfg, ids, layer)
        denom = float(f1 @ f1) + 1e-6
        out_w = out_w + np.outer(f1, delta) / denom
        out_b = out_b + delta * 0.05 * alpha

    np.save(out_w_path, out_w.astype(np.float32))
    np.save(out_b_path, out_b.astype(np.float32))

def auto_search(weights, cfg, ids, search_id, replace_id, mode):
    print("auto-searching layer/alpha ...")
    for layer in range(5, 10):
        for alpha in [1.0, 2.0, 4.0, 8.0]:
            tmp = tempfile.mkdtemp(prefix="ws_auto_")
            try:
                shutil.copytree(weights, tmp, dirs_exist_ok=True)
                apply_edit(tmp, layer, ids, search_id, replace_id, alpha, mode)
                edited = reference_logits(tmp, cfg, ids)
                if int(edited[-1].argmax()) == replace_id:
                    print(f"  found: layer={layer} alpha={alpha}")
                    return layer, alpha
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="build/gpt2")
    ap.add_argument("--out", default="build/gpt2_edited")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--search", default=" the")
    ap.add_argument("--replace", default=" Germany")
    ap.add_argument("--search-id", type=int, default=None)
    ap.add_argument("--replace-id", type=int, default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--mode", choices=["bias", "weight"], default="bias")
    ap.add_argument("--auto", action="store_true", help="search layer/alpha automatically")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.weights, "config.json")))
    tok = load_tok(args.weights)

    if args.ids is not None:
        ids = [int(t) for t in args.ids.split(",")]
    elif args.prompt is not None:
        if tok is None:
            raise SystemExit("tokenizer not found; pass --ids")
        ids = tok.encode(args.prompt).ids
    else:
        ids = [464, 3139, 286, 4881, 318]

    search_id = args.search_id if args.search_id is not None else (first_id(tok, args.search) if tok else None)
    replace_id = args.replace_id if args.replace_id is not None else (first_id(tok, args.replace) if tok else None)
    if search_id is None or replace_id is None:
        raise SystemExit("need tokenizer or --search-id/--replace-id")

    if args.auto:
        layer, alpha = auto_search(args.weights, cfg, ids, search_id, replace_id, args.mode)
        if layer is None:
            raise SystemExit("auto-search failed: no layer/alpha flipped the prediction")
        layers = [layer]
        args.alpha = alpha
    elif args.layers:
        layers = [int(x) for x in args.layers.split(",")]
        args.alpha = args.alpha if args.alpha is not None else 4.0
    elif args.layer is not None:
        layers = [args.layer]
        args.alpha = args.alpha if args.alpha is not None else 4.0
    else:
        layers = [7]
        args.alpha = args.alpha if args.alpha is not None else 4.0

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    shutil.copytree(args.weights, args.out)

    base = reference_logits(args.weights, cfg, ids)
    print(f"prompt ids: {ids}")
    print(f"search: {args.search!r} -> id {search_id}")
    print(f"replace: {args.replace!r} -> id {replace_id}")
    print(f"baseline next token: {int(base[-1].argmax())}")

    for L in layers:
        apply_edit(args.out, L, ids, search_id, replace_id, args.alpha, args.mode)
        print(f"edited layer {L} ({args.mode}, alpha={args.alpha})")

    edited = reference_logits(args.out, cfg, ids)
    print(f"edited   next token: {int(edited[-1].argmax())}")

if __name__ == "__main__":
    main()

