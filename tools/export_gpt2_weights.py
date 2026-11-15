#!/usr/bin/env python3
"""Export GPT-2 weights to one .npy file per tensor (TensorLang's load() reads one array per file).

    pip install safetensors huggingface_hub tokenizers
    python3 tools/export_gpt2_weights.py                      # openai-community/gpt2 -> build/gpt2/
    python3 tools/export_gpt2_weights.py --repo gpt2 --out build/gpt2

Writes (all float32):
    wte.npy (vocab, d)   wpe.npy (n_ctx, d)   lnf_g.npy lnf_b.npy
    h{i}_ln1_g h{i}_ln1_b  h{i}_qkv_w (d, 3d) h{i}_qkv_b  h{i}_proj_w (d, d) h{i}_proj_b
    h{i}_ln2_g h{i}_ln2_b  h{i}_fc_w (d, 4d)  h{i}_fc_b   h{i}_out_w (4d, d) h{i}_out_b
plus config.json and tokenizer.json. GPT-2's Conv1D weights are stored (in, out), so the forward pass is
x @ W + b with no transpose. The output projection reuses wte (tied weights): the program transposes it.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="openai-community/gpt2")
    ap.add_argument("--out", default="build/gpt2")
    args = ap.parse_args()
    try:
        from safetensors.numpy import load_file
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
        try:
            from safetensors.numpy import load_file
        except ImportError:
            sys.exit("missing packages: pip install safetensors huggingface_hub tokenizers")
    if hf_hub_download is None and not os.path.isdir(args.repo):
        sys.exit("downloading needs: pip install huggingface_hub  (or pass a local directory as --repo)")

    os.makedirs(args.out, exist_ok=True)
    if os.path.isdir(args.repo):            # a local checkpoint directory (offline use, tests)
        cfg_path, weights_path, tok_path = (os.path.join(args.repo, n) for n in
                                            ("config.json", "model.safetensors", "tokenizer.json"))
    else:
        cfg_path = hf_hub_download(args.repo, "config.json")
        weights_path = hf_hub_download(args.repo, "model.safetensors")
        tok_path = hf_hub_download(args.repo, "tokenizer.json")
    hf_cfg = json.load(open(cfg_path))
    raw = load_file(weights_path)
    # some checkpoints prefix every key with "transformer."
    w = {(k[len("transformer."):] if k.startswith("transformer.") else k): v for k, v in raw.items()}

    n_layer = int(hf_cfg.get("n_layer", hf_cfg.get("num_hidden_layers")))
    n_embd = int(hf_cfg.get("n_embd", hf_cfg.get("hidden_size")))
    n_head = int(hf_cfg.get("n_head", hf_cfg.get("num_attention_heads")))
    vocab = w["wte.weight"].shape[0]
    n_ctx = w["wpe.weight"].shape[0]
    config = dict(n_layer=n_layer, n_embd=n_embd, n_head=n_head, vocab=int(vocab), n_ctx=int(n_ctx),
                  layer_norm_epsilon=float(hf_cfg.get("layer_norm_epsilon", 1e-5)))

    def put(name, arr):
        np.save(os.path.join(args.out, name + ".npy"), np.ascontiguousarray(arr, dtype=np.float32))

    put("wte", w["wte.weight"])
    put("wpe", w["wpe.weight"])
    put("lnf_g", w["ln_f.weight"])
    put("lnf_b", w["ln_f.bias"])
    names = [("ln_1.weight", "ln1_g"), ("ln_1.bias", "ln1_b"),
             ("attn.c_attn.weight", "qkv_w"), ("attn.c_attn.bias", "qkv_b"),
             ("attn.c_proj.weight", "proj_w"), ("attn.c_proj.bias", "proj_b"),
             ("ln_2.weight", "ln2_g"), ("ln_2.bias", "ln2_b"),
             ("mlp.c_fc.weight", "fc_w"), ("mlp.c_fc.bias", "fc_b"),
             ("mlp.c_proj.weight", "out_w"), ("mlp.c_proj.bias", "out_b")]
    for i in range(n_layer):
        for src, dst in names:
            put(f"h{i}_{dst}", w[f"h.{i}.{src}"])
    if os.path.exists(tok_path):
        shutil.copy(tok_path, os.path.join(args.out, "tokenizer.json"))
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    total = sum(v.size for k, v in w.items() if not k.endswith((".attn.bias", ".attn.masked_bias")))
    print(f"exported {n_layer} layers, d={n_embd}, heads={n_head}, vocab={vocab}, ctx={n_ctx} "
          f"({total/1e6:.1f}M parameters) to {args.out}/")
    print(json.dumps(config))


if __name__ == "__main__":
    main()
