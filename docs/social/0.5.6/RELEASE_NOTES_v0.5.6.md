# TensorLang v0.5.6 — closing the v0.5.5 gaps

A release about the gaps v0.5.5 left open: shapes and dtypes the kernels cannot run are now
compile errors instead of crashes or wrong answers, `batch_norm` gets `gamma`/`beta` and correct
results on N-D input, and two loop bugs found by a new acceptance app are fixed.
Suite: 294 `.tl` tests + 31 Python tests, all passing on 2× GTX 1080 Ti (Pascal, sm_61).

## Behaviour changes to be aware of

* **A comparison with an unsupported shape pair is `[E-SHAPE-MISMATCH]`.** The kernels run equal
  shapes, or a `(rows, cols)` tensor against a `(cols,)` or `(1, cols)` row. Anything else used to
  crash or read past a buffer. This covers comparison operations (annotated `let` included) and
  `if`/`elif` conditions.
* **f16 operands in a comparison are `[E-COMPARISON-DTYPE]`**, including an f16 tensor compared
  with a numeric literal. The comparison kernels are generated for f32 only.
* **A tensor declared as `f64` is `[E-UNSUPPORTED-DTYPE]`.** f64 never had code generation:
  operations raised `KeyError: 'f64'` and comparisons misread the data. Declare tensors as `f32`.
  The two `apps/trading` indicator apps now declare their window as `f32`.
* **An undeclared `load()` of a float64 `.npy` or `.npz` is inferred as `f32`** and converted on
  load (`--debug-info` prints a note). It used to be inferred as `f64` and failed with
  `KeyError: 'f64'` at the first operation that used it. A float16 file still loads as `f16`, and
  an annotated `load()` still converts to the declared type.
* **`batch_norm` on 3-D and higher input uses each element's own channel** (channel at dimension
  1). The old N-D forward took the channel as `idx % C`, which is wrong for NCHW, with no error.
  Its gradients now match its forward; before, the backward used batch statistics while the forward
  uses the given ones, so gradients were near zero in simple cases. Running statistics, `gamma`
  and `beta` must be f32 with shape `(C,)` (`[E-BATCH-NORM-DTYPE]`).
* **An `if` after a loop's first `backward()` now runs.** It used to be skipped silently, so a
  conditional optimiser step never happened. v0.5.5's notes said `backward()` works in any
  branch or loop; this case was not covered.
* **Statements after a loop that contains an `if` now run their own kernels** (they ran the wrong
  kernel before).

## New

* **`batch_norm(x, mean, var [, gamma, beta] [, eps=...])`:** an optional affine pair (both or
  neither). Gradients: `dx = dy*gamma/sqrt(var+eps)`, `dgamma = sum(dy*xhat)`, `dbeta = sum(dy)`.
* **New diagnostic codes:** `[E-COMPARISON-DTYPE]`, `[E-UNSUPPORTED-DTYPE]`, `[E-BATCH-NORM-DTYPE]`.
* **`apps/examples/adaptive_lr`:** an acceptance app with an `if loss > threshold` learning-rate
  step and two losses with separate optimisers, pinned against hand-derived values. It found the
  two loop bugs above. Run it with `python3 tensorlang.py --app examples/adaptive_lr --test`.
* **Sweep tools:** `tools/scan_comparison_shapes.py`, `scan_comparison_dtypes.py`,
  `scan_loop_ifs.py`, `scan_load_f64.py`, and `check_indicators_f64.py`.

## Fixed

* The two loop bugs above, and the two `batch_norm` bugs above. Regression tests:
  `for_loop_if_after_backward_sgd`, `for_loop_if_then_statement_after`, `batch_norm_nchw`,
  `autograd_batchnorm_given_stats`, `load_f64_inferred`.
* The README's NumPy interop notes now describe the real behaviour: `load_numpy` copies
  non-contiguous arrays (they are not shared), and its buffers do not feed `.tl` programs.

## Known limitations carried forward

* f16 operands in comparison kernels are rejected, not supported.
* An optimiser step inside an `if` branch does not honour the loss-scaling overflow skip (f16 with
  dynamic loss scaling); the same already held for a `backward()` and `sgd` in one branch.
* `batch_norm` does not compute batch statistics or update running statistics; there is no `axis=`
  option and `instance_norm` is unchanged.
* No `isnan()` operation, so `NaN` cannot be tested for on purpose.
* The dynamic loss-scale growth interval is not configurable.
* Full f64 support is not planned.
