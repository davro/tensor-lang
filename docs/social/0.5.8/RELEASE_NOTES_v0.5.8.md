# TensorLang v0.5.8 — `embedding`, and real GPT-2 running end to end

A release driven by one demo: run the real 124M-parameter GPT-2 small from a TensorLang program and check
it against a float64 NumPy reference. It matches. Getting there needed one new operation, `embedding`
(gather), and the rest was existing operations plus probes that proved each piece at toy size first.
Suite: 311 `.tl` tests + 31 Python tests, all passing on a GTX 1080 Ti (Pascal, sm_61).
*(Confirm the counts with a full `source build.sh --test` before tagging.)*

## New

* **`embedding(indices, weight)`.** Gathers rows: `weight` is `(vocab, d)`, `indices` is `f32` of any rank
  >= 1, the result is `indices.shape + (d,)` in the weight's dtype (`f16` or `f32`).
  * Indices are stored as floats because there is no integer dtype (the same convention as
    `cross_entropy` targets). `f32` is exact up to 2^24, far beyond any vocabulary; `f16` indices are
    rejected, since `f16` cannot hold ids above 2048.
  * Run-time check: ids are validated on the host before the launch. A non-integer or out-of-range id raises
    `[E-EMBEDDING-INDEX]` with the first bad position. The kernel also writes NaN for an out-of-range id
    instead of reading outside the table.
  * Backward is a scatter-add into the weight, so a repeated id accumulates. Nothing flows to the indices.
  * New diagnostics: `[E-EMBEDDING-SHAPE]`, `[E-EMBEDDING-DTYPE]`, `[E-EMBEDDING-INDEX]`.
  * Tests: `embedding_gather`, `embedding_grad`, `embedding_token_position`,
    `diagnostics_embedding_{weight_rank,index_dtype,index_range}`.
* **A two-head GPT-2 block, verified** (`probe_gpt2_multihead`). The shapes a real checkpoint uses: token and
  position `embedding`, a fused QKV projection split by slicing, head split (`reshape` then
  `transpose(1, 0, 2)`), batched `(heads, seq, seq)` attention with the causal mask broadcast over the heads
  axis, softmax over the last axis, and the head merge. Matches float64 NumPy.
  `tools/make_gpt2_mha_probe.py` generates it and refuses to write unless nine plausible head-handling bugs
  (no mask, split or merge without the transpose, k and v swapped, softmax over the wrong axis, a dropped
  residual, wrong scale, one head instead of two) each change the logits.
* **tanh-GELU, from existing operations** (`gelu_tanh`). GPT-2 checkpoints were trained with the tanh
  approximation; the erf `gelu` differs from it by a few 1e-4. It is a user-defined function (`mult`, `add`,
  `tanh`), so there is no new kernel; forward values and the gradient through the composed operations match
  float64 NumPy. There is no import mechanism, so copy the function into your program.
* **Real GPT-2 small, run from a `.tl` program** (`tools/`):
  * `export_gpt2_weights.py` downloads the checkpoint and writes one `.npy` per tensor (`load()` reads one
    array per file); `make_gpt2_demo.py` generates the unrolled 12-layer program and a float64 reference;
    `compare_gpt2_logits.py` compares them; `gpt2_demo.sh` runs the first two.
  * Result on a GTX 1080 Ti, prompt "The capital of France is" (5 tokens): max absolute logit difference
    **1.1e-4** on logits averaging 83 in magnitude, argmax identical at all 5 positions, identical top-5 at
    the last position.
  * `gpt2_generate.py` generates text with the loop on the host: 20 greedy tokens, **identical to the float64
    reference**. One program is compiled for a fixed length (prompt plus new tokens) and the token ids are read
    from a `.npy`, so the compile cache is reused. Padding at the end is exact, because causal masking keeps
    padded positions from influencing earlier ones.
  * Timing: compiling a new sequence length costs about 16–18 s; a repeat run costs about 3 s (mostly process
    start and loading ~500 MB of weights), so generation runs at about 2.9 s per token.

## Fixed

* `load()` printed one `[AST] build expression: ...` line per call regardless of the debug flags. With a real
  model that is 148 lines of noise. It is now silent unless `DEBUG_INFO` is on.

## Documentation

* `verify.py` was listed under Development Tools but does not exist; removed.
* The "No gather / `embedding`" limitation is resolved; `embedding` is in the Operations table and the three
  new codes are in the Diagnostics table.

## Known limitations carried forward

* A rebind whose new value has a different shape (`tokens = concat(tokens, new)`) is skipped with a
  diagnostic at run time; the type checker does not check rebinds. Generation therefore runs on the host.
* `fill` takes one scalar and is not accepted as an operand inside another call.
* Defining the same name twice with `let` passes type checking and then fails (or is unsupported) later; use
  distinct names. A diagnostic is planned.
* f16 operands in comparison kernels are rejected, not supported.
* An optimiser step inside an `if` branch does not honour the loss-scaling overflow skip.
* `batch_norm` does not compute batch statistics or update running statistics; there is no `axis=` option and
  `instance_norm` is unchanged.
* No `isnan()` operation; the dynamic loss-scale growth interval is not configurable; full f64 support is not
  planned.

## New limitations and things to know

* `load()` reads one array per file (`.npz` takes the array named `data`, else the first), and the filename
  must be a string literal, so a 12-layer model is generated and unrolled rather than looped over a path.
* Generation has no KV cache and each token is a separate run, so every step recomputes the whole sequence and
  reloads the weights. It is a demonstration, not a fast path.
* `embedding` is now a keyword. It is still accepted as a variable name (checked), but `embedding(` is the call.
* The NumPy reference used for the GPT-2 comparison is a reimplementation, not HuggingFace. It agrees with an
  independent per-head loop implementation to 1e-15, and its greedy text is fluent, but it has not been
  checked against HuggingFace transformers.
* GPT-2 small greedy decoding repeats itself ("the capital of the French Republic, and the capital of the
  French Republic ..."). That is the model, and the reference does the same.
