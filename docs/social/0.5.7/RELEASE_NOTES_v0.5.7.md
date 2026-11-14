# TensorLang v0.5.7 — a transformer block, and the broadcast bug it found

A release driven by one experiment: write a one-layer GPT-2-style forward pass in TensorLang from
existing operations only, and check every number against NumPy. It works, and in doing so it exposed
a silent wrong-answer bug in broadcasting that exists in **every earlier version, including v0.5.6**.
Suite: 303 `.tl` tests + 31 Python tests, all passing on 2× GTX 1080 Ti (Pascal, sm_61).

## Behaviour changes to be aware of

* **A square 2-D tensor combined with a 1-D vector now broadcasts per column, as in NumPy.**
  `add`, `minus`, `mult` and `div` of an `(n, n)` tensor with an `(n,)` vector read the vector
  per row instead of per column, in both the unfused and the fused kernel. A square weight matrix
  with a bias, or a layer-norm `gamma`/`beta` on a square activation, therefore gave plausible but
  wrong numbers with no error. Non-square shapes (`(rows, cols)` with `(cols,)`, rows != cols)
  were always correct, which is why it went unnoticed. Programs that ran through the old kernel
  change their output.
  * The backward pass already reduced these gradients per column (it sums over axis 0), so for
    these operations the forward values and the gradients disagreed. (Read from the code; not
    re-checked on a trained model.)
  * Three existing tests had pinned the wrong values and were corrected: `fusion_broadcast` and
    `fusion_broadcast_guard` (`[[1.6, 1.6], [3.7, 3.7]]` became `[[1.6, 1.7], [3.6, 3.7]]`) and
    `mlp` (its 4×4 hidden layer plus a 4-element bias; `output` changed by up to 0.16). All of
    them were passing with a numerically wrong answer.
  * To check your own programs: `python3 tools/scan_square_vector_broadcast.py` lists every
    square-matrix / vector operation (it found four tests and no apps or examples here), and
    `python3 tools/check_broadcast_kernels.py` compares the generated kernels with NumPy without
    a GPU.

## New

* **A transformer forward pass, verified.** A one-layer pre-LN GPT-2-style block (token and
  positional embedding, layer norm, causal attention, residuals, gelu FFN, final layer norm, output
  projection) at toy size (seq 4, d 4, vocab 8) is written with existing operations: batched
  `matmul`, `transpose(axes=...)`, `softmax(axis=)`, `layer_norm`, `gelu` and broadcasting. With
  seeded distinct weights and a real one-hot input it matches a float64 NumPy reference
  (`probe_gpt2_forward`), and attention gradients match NumPy (`probe_attention_block`: `out`,
  `q_grad`, `k_grad`, `v_grad`). The causal mask needs no new operation: `(rows, 1) - (1, cols)`
  broadcast, `>= 0.0`, then an additive `-1e9`.
* **Probe tests** (`tests/probe_*.tl`): the causal mask, layer norm, gelu, broadcasting, a one-hot
  embedding with gradient accumulation for a repeated token, the shape-stability guard, attention
  with gradients, and the forward pass.
* **`tools/make_gpt2_probe.py`** generates the forward-pass probe and its float64 expected values,
  and checks that a deliberately broken block (no mask, wrong scale, a missing residual, ReLU for
  GELU, the bias before the projection, no final norm) changes the logits.
  **`tools/compare_probe_intermediates.py`** finds the first tensor where a run diverges from NumPy.
* **`broadcast_vector_square`**: the regression test for the fix, covering `add`, `minus`, `mult`,
  `div`, a fused chain, and a non-square control.

## Fixed

* The broadcast bug above.
* `probe_gpt2_forward` was first written with weights that were constant down each column. Its true
  logits were exactly zero, and its "match" with NumPy compared two rounding errors (about 1e-8)
  that no change to the block could move. A bug in the block, and this one, would have passed it.
  The README now records this under Known limitations.

## Known limitations carried forward

* No gather / `embedding`: token ids cannot be turned into rows in-language (a one-hot matrix
  prepared outside works with `matmul`).
* A rebind whose new value has a different shape (`tokens = concat(tokens, new)`) is skipped with a
  diagnostic at run time; the type checker does not check rebinds. Generation runs on the host.
* `fill` takes one scalar and is not accepted as an operand inside another call.
* Defining the same name twice with `let` passes type checking and then fails (or is unsupported)
  later; use distinct names. A diagnostic is planned.
* `gelu` is the erf form; GPT-2 uses the tanh approximation, so a real-weights comparison needs the
  tanh form built from existing operations.
* f16 operands in comparison kernels are rejected, not supported.
* An optimiser step inside an `if` branch does not honour the loss-scaling overflow skip.
* `batch_norm` does not compute batch statistics or update running statistics; there is no `axis=`
  option and `instance_norm` is unchanged.
* No `isnan()` operation; the dynamic loss-scale growth interval is not configurable; full f64
  support is not planned.
