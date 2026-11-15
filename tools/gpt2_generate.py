#!/usr/bin/env python3
"""Greedy text generation driven from the host, with TensorLang running the model.

    python3 tools/gpt2_generate.py --prompt "The capital of France is" --tokens 20

How it stays fast: one program is compiled for a FIXED length N = prompt + new tokens. Unused positions are
padded; causal masking means they cannot influence earlier positions, so the prediction for a sequence of
t real tokens is the argmax at row t-1. The token ids are read from a .npy file, so the program text never
changes between steps and the compiled kernels are reused (a new program length costs ~20 s to compile;
a repeat run costs a few seconds, mostly loading the weights).

Each step is a separate run of tensorlang.py: the host appends the predicted token and runs again.
There is no KV cache, so every step recomputes the whole sequence.

    --check-reference   also run the float64 NumPy reference greedily and report whether the tokens match
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_gpt2_demo as demo  # noqa: E402


def generate(cfg, ids, n_new, run_step, on_token=None, pad_id=0):
    """Greedy loop. run_step(padded_ids) -> per-position argmax vector (length N)."""
    n = len(ids) + n_new
    seq = list(ids)
    times = []
    for _ in range(n_new):
        padded = seq + [pad_id] * (n - len(seq))
        t0 = time.time()
        nxt = np.asarray(run_step(padded)).reshape(-1)
        times.append(time.time() - t0)
        tok = int(nxt[len(seq) - 1])
        seq.append(tok)
        if on_token:
            on_token(tok, times[-1])
    return seq, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="build/gpt2")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=20, help="new tokens to generate")
    ap.add_argument("--program", default="build/gpt2_generate.tl")
    ap.add_argument("--tokens-file", default="build/gpt2_tokens.npy")
    ap.add_argument("--next-file", default="build/gpt2_next.npy")
    ap.add_argument("--check-reference", action="store_true")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.weights, "tokenizer.json"))
    cfg = json.load(open(os.path.join(args.weights, "config.json")))
    ids = tok.encode(args.prompt).ids
    n = len(ids) + args.tokens
    if n > cfg["n_ctx"]:
        sys.exit(f"{n} tokens exceed the context of {cfg['n_ctx']}")

    os.makedirs(os.path.dirname(args.program) or ".", exist_ok=True)
    with open(args.program, "w") as f:
        f.write(demo.emit_program(cfg, [0] * n, args.weights, None,
                                  tokens_file=args.tokens_file, next_path=args.next_file))
    print(f"program: {args.program} (fixed length {n}); tokens read from {args.tokens_file}")

    def run_step(padded):
        np.save(args.tokens_file, np.array(padded, dtype=np.float32))
        r = subprocess.run([sys.executable, "tensorlang.py", args.program], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"tensorlang.py failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        # the runner redirects save() under cache/<program>/
        for path in (os.path.join("cache", args.program, args.next_file), args.next_file):
            if os.path.exists(path):
                return np.load(path)
        sys.exit(f"could not find the saved output {args.next_file} (looked under cache/{args.program}/)")

    print(args.prompt, end="", flush=True)

    def show(t, secs):
        print(tok.decode([t]), end="", flush=True)

    t0 = time.time()
    seq, times = generate(cfg, ids, args.tokens, run_step, on_token=show)
    print(f"\n\n{args.tokens} tokens in {time.time() - t0:.1f}s; per step: first {times[0]:.1f}s "
          f"(includes compile on a cold cache), then median {np.median(times[1:]) if len(times) > 1 else float('nan'):.1f}s")

    if args.check_reference:
        ref = list(ids)
        for _ in range(args.tokens):
            ref.append(int(demo.reference_logits(args.weights, cfg, ref)[-1].argmax()))
        same = ref == seq
        print("reference (float64 NumPy) greedy tokens match:", "YES" if same else "NO")
        if not same:
            first = next(i for i, (a, b) in enumerate(zip(ref, seq)) if a != b)
            print(f"  first difference at token {first - len(ids)}: TensorLang {tok.decode([seq[first]])!r} "
                  f"vs reference {tok.decode([ref[first]])!r}  (a near-tie can legitimately flip)")
            print("  reference:", repr(tok.decode(ref)))
        sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
