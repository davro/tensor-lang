# Weight-Space Programming in TensorLang

## Overview

TensorLang can load and execute *edited* GPT-2 weights, enabling experiments in
weight-space programming: modifying a trained model's parameters directly and
observing the effect through the same compiled forward pass. All edits are
verified against a float64 NumPy reference with <2e-4 max abs diff.

A small tool (`tools/weight_edit.py`) applies a search/replace edit to one or
more MLP layers. By default it edits only the MLP bias (a single vector
addition); `--mode weight` also applies a rank-one update to the
down-projection. `--auto` searches layers 5–9 and alphas [1, 2, 4, 8] for
the first combination that flips the prediction.

```bash
# One-liner demo
./tools/demo_weight_edit.sh

# Custom edit
python3 tools/weight_edit.py \
  --weights build/gpt2 --out build/gpt2_edited \
  --prompt "Paris is the capital of" \
  --search " the" --replace " England" \
  --auto
```

## Tools

| Tool | Purpose |
|---|---|
| `tools/weight_edit.py` | Search/replace weight editor (bias or rank-one mode, with `--auto` layer/alpha search) |
| `tools/demo_weight_edit.sh` | One-liner demo: edit → compile → run → compare |
| `scripts/extract_readme_pairs.py` | Extract `(goal, prefix, target)` pairs from README.md code blocks |
| `scripts/lark_scorer.py` | Score generated text against TensorLang's Lark grammar (0.0–1.0) |
| `scripts/lark_auto_search.py` | Lark-guided auto-search for (layer, alpha) combinations |

## What works: single-token fact replacement

Editing a single MLP bias vector (layer 7, α=4.0) reliably flips a factual
prediction while preserving fluency and specificity.

| Prompt | Search → Replace | Layer / α | Greedy continuation | Coherence |
|---|---|---|---|---|
| "Paris is the capital of" | the → England | 7 / 4.0 | "England and Wales, England and…" | ✅ Fluent |
| "The best programming language is" | Python → Rust | 5 / 8.0 | "Rust, and Rust is the…" | ✅ Fluent |

**Numerical agreement:** TensorLang's compiled forward pass agrees with the
float64 NumPy reference to <2e-4 max abs diff with identical argmax and top-5
after editing.

**Specificity:** 3 of 4 unrelated prompts ("Hello world", "Machine learning is",
"The capital of France is") are unaffected. One ("The sky is") shifts from
"the" → "falling", indicating the bias vector has some collateral effect on
adjective distributions.

## What doesn't work: syntax-level code generation

An automated Lark-guided search (layers 5–9, alphas [1, 2, 4, 8]) was run
to find edits that push GPT-2 toward generating valid TensorLang syntax. The
search used actual TensorLang prefixes from README.md code blocks as context
and scored generated continuations with the Lark parser.

**Result:** No improvement. The base model's distribution is too far from
valid TensorLang for bias edits to bridge the gap:

| Alpha | Mean Lark score | Interpretation |
|---|---|---|
| 1.0–2.0 | 0.636 | Edit too weak; model generates baseline English |
| 4.0+ | 0.333–0.385 | Edit active but **corrupts** output (lower syntax score) |

This is a well-documented negative result. The edit either does nothing
(low alpha) or makes things worse (high alpha). GPT-2 (124M) lacks the
code-generation priors needed for weight-space syntax editing.

## Limitations

1. **Single-token replacements only.** Multi-token targets (e.g. "Mordor" = 2
   tokens) cause degenerate repetition because the bias only targets the first
   token. The model gets stuck emitting the first token repeatedly.

2. **The `--search` token must match the baseline prediction.** If you search
   for " France" but the model predicts " the", the edit has no target to flip.
   Use `--auto` to handle this automatically.

3. **Multi-fact edits interfere.** Averaging deltas for multiple facts (e.g.
   Germany + Italy + Spain) produces a diffuse vector that doesn't flip any
   specific prompt. Sequential application works better but can be too strong.

4. **No syntax generation on GPT-2.** The base model's distribution is too far
   from valid TensorLang. This requires a stronger base model (see Next Steps).

## Next Steps

Per the project roadmap (Phase 5), the path forward is:

1. **Upgrade base model to Qwen2.5-0.5B** — stronger code generation priors,
   better instruction following, still CPU-friendly.
2. **Abstract module mapping** — decouple the search engine from GPT-2 tensor
   names (`transformer.h.{layer}.mlp.c_proj`) to support different
   architectures (`model.layers.{layer}.mlp.down_proj`).
3. **Contrastive value vectors** — compute deltas from prompt pairs instead of
   raw token embeddings (ROME-style) for cleaner flips at lower alpha.
4. **Multi-token support** — edit multiple positions or use a proper
   least-squares solve for the value vector.


## Reproducing the demo

```bash
# 1. Export base GPT-2 weights (~500 MB download)
python3 tools/export_gpt2_weights.py

# 2. Apply the England edit (layer 7, alpha 4.0, bias-only)
rm -rf build/gpt2_edited && cp -r build/gpt2 build/gpt2_edited
python3 tools/weight_edit.py \
  --weights build/gpt2 --out build/gpt2_edited \
  --prompt "Paris is the capital of" \
  --search " the" --replace " England" \
  --layer 7 --alpha 4.0 --mode bias

# 3. Generate TensorLang program and run
python3 tools/make_gpt2_demo.py --weights build/gpt2_edited \
  --prompt "Paris is the capital of" --ref-generate 4
python3 tensorlang.py build/gpt2_forward.tl
python3 tools/compare_gpt2_logits.py
```

Expected: `next_token = [4492.]` (England), PASS (<2e-4 diff).

## Files

| File | Purpose |
|---|---|
| `tools/weight_edit.py` | Search/replace weight editor |
| `tools/demo_weight_edit.sh` | One-liner demo |
| `scripts/extract_readme_pairs.py` | README → (goal, prefix, target) pairs |
| `scripts/lark_scorer.py` | Lark grammar scorer (0.0–1.0) |
| `scripts/lark_auto_search.py` | Lark-guided auto-search |

