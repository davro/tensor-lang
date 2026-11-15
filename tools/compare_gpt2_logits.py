#!/usr/bin/env python3
"""Compare the logits a TensorLang GPT-2 program saved against the float64 reference.

    python3 tools/compare_gpt2_logits.py [--logits build/gpt2_logits.npy] [--ref build/gpt2_ref_logits.npy]

Passes when there are no NaN/inf values, the argmax agrees at every position, and the largest absolute
difference is below --tol. The tolerance is a judgement call: GPT-2 logits are large (tens in magnitude)
and float32 accumulates rounding over 12 layers, so look at the reported numbers, not only the verdict.
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", default="build/gpt2_logits.npy")
    ap.add_argument("--ref", default="build/gpt2_ref_logits.npy")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--program", default="build/gpt2_forward.tl", help="program name, for locating its cache/ directory")
    ap.add_argument("--tokenizer", default="build/gpt2/tokenizer.json")
    args = ap.parse_args()
    path = args.logits
    if not os.path.exists(path):
        # the runner redirects save() under cache/<program>/ ; look there too
        alt = os.path.join("cache", args.program, path)
        if os.path.exists(alt):
            path = alt
    got = np.load(path).astype(np.float64)
    ref = np.load(args.ref).astype(np.float64)
    if got.shape != ref.shape:
        sys.exit(f"FAIL: shape {got.shape} vs reference {ref.shape}")
    if not np.isfinite(got).all():
        sys.exit(f"FAIL: {(~np.isfinite(got)).sum()} NaN/inf values in the TensorLang logits")
    diff = np.abs(got - ref)
    print(f"logits {got.shape}: |ref| mean {np.abs(ref).mean():.2f} max {np.abs(ref).max():.2f}")
    print(f"max abs diff {diff.max():.3e}   mean abs diff {diff.mean():.3e}")
    agree = got.argmax(axis=1) == ref.argmax(axis=1)
    print(f"argmax agrees at {int(agree.sum())}/{len(agree)} positions")
    dec = None
    if os.path.exists(args.tokenizer):
        try:
            from tokenizers import Tokenizer
            dec = Tokenizer.from_file(args.tokenizer).decode
        except ImportError:
            pass
    k = 5
    tg, tr = np.argsort(got[-1])[::-1][:k], np.argsort(ref[-1])[::-1][:k]
    show = (lambda ids: [(int(t), dec([int(t)])) for t in ids]) if dec else (lambda ids: [int(t) for t in ids])
    print("last position top-5   TensorLang:", show(tg))
    print("                      reference :", show(tr))
    ok = bool(agree.all()) and diff.max() < args.tol
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
