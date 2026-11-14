#!/usr/bin/env python3
"""Find the first tensor where a TensorLang run diverges from NumPy.

Reads the weights straight from the .tl file that was run (so it cannot drift from the generator),
recomputes every named intermediate in float64, and compares each with
cache/tests/<name>.tl/<tensor>.npy in program order.

  python3 compare_probe_intermediates.py [tests/probe_gpt2_forward.tl] [--cache-dir cache/tests/probe_gpt2_forward.tl]
"""
import json, math, re, sys
from pathlib import Path
import numpy as np

tl = Path(next((a for a in sys.argv[1:] if a.endswith(".tl")), "tests/probe_gpt2_forward.tl"))
cache = Path(sys.argv[sys.argv.index("--cache-dir") + 1]) if "--cache-dir" in sys.argv else Path("cache") / tl.parent.name / tl.name
code = tl.read_text()

def lit(name):
    # a literal may span several lines: read up to the next blank line or `let`
    m = re.search(rf"^let {name}: Tensor\[f32, \([^)]*\)\] = (.*?)(?=^\s*$|^let |\Z)", code, re.S | re.M)
    return np.array(json.loads(m.group(1)), dtype=np.float64)

P = {n: lit(n) for n in ["x_tok", "w_tok", "w_pos", "w_q", "w_k", "w_v", "w_o", "w_f1", "w_f2", "w_out",
                         "b_q", "b_k", "b_v", "b_o", "b_f1", "b_f2", "b_out", "g1", "e1", "g2", "e2", "gf", "ef"]}
erf = np.vectorize(math.erf)
def ln(x): m = x.mean(1, keepdims=True); v = ((x - m) ** 2).mean(1, keepdims=True); return (x - m) / np.sqrt(v + 1e-5)
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

R = {}
R["tok_emb"] = P["x_tok"] @ P["w_tok"]; R["x0"] = R["tok_emb"] + P["w_pos"]
R["n1"] = ln(R["x0"]); R["h1"] = R["n1"] * P["g1"] + P["e1"]
for s in "qkv":
    R[f"{s}_raw"] = R["h1"] @ P[f"w_{s}"]; R[s] = R[f"{s}_raw"] + P[f"b_{s}"]
R["kt"] = R["k"].T; R["qk"] = R["q"] @ R["kt"]; R["scores"] = R["qk"] * 0.5
row = np.arange(4.0)[:, None]; col = np.arange(4.0)[None, :]
R["diff"] = row - col; R["allowed"] = (R["diff"] >= 0).astype(float); R["blocked"] = 1.0 - R["allowed"]
R["mask"] = R["blocked"] * -1e9; R["masked"] = R["scores"] + R["mask"]; R["attnw"] = sm(R["masked"])
R["ctx"] = R["attnw"] @ R["v"]; R["proj"] = R["ctx"] @ P["w_o"]; R["attn_out"] = R["proj"] + P["b_o"]
R["x1"] = R["x0"] + R["attn_out"]
R["n2"] = ln(R["x1"]); R["h2"] = R["n2"] * P["g2"] + P["e2"]
R["f1_raw"] = R["h2"] @ P["w_f1"]; R["f1"] = R["f1_raw"] + P["b_f1"]
R["f1g"] = 0.5 * R["f1"] * (1 + erf(R["f1"] / math.sqrt(2)))
R["f2_raw"] = R["f1g"] @ P["w_f2"]; R["ffn_out"] = R["f2_raw"] + P["b_f2"]; R["x2"] = R["x1"] + R["ffn_out"]
R["nf"] = ln(R["x2"]); R["hf"] = R["nf"] * P["gf"] + P["ef"]
R["out_raw"] = R["hf"] @ P["w_out"]; R["logits"] = R["out_raw"] + P["b_out"]

print(f"cache: {cache}\n{'tensor':10s} {'shape':10s} {'max |diff|':>12s}  status")
first = None; missing = []
for name, ref in R.items():
    f = cache / f"{name}.npy"
    if not f.exists(): missing.append(name); continue
    got = np.load(f).astype(np.float64)
    if got.shape != ref.shape:
        status = f"SHAPE {got.shape} vs {ref.shape}"; d = float("nan"); bad = True
    else:
        d = np.abs(got - ref).max(); bad = not np.allclose(got, ref, rtol=1e-4, atol=1e-5)
        status = "DIVERGES" if bad else "ok"
    if bad and first is None: first = name; status += "   <-- first divergence"
    print(f"{name:10s} {str(ref.shape):10s} {d:12.3e}  {status}")
if missing: print("\nnot found in cache (not saved?):", ", ".join(missing))
print("\nFIRST DIVERGENCE:", first or "none (all compared tensors agree)")
sys.exit(1 if first else 0)
