# TensorLang v0.5.5 — robustness

A release about making what already exists reliable: clearer errors, defined semantics at the
edges, and a set of silent-wrong-answer bugs in branch and loop autograd that are now fixed.
Suite: 268 `.tl` tests + 31 Python tests, all passing on 2× GTX 1080 Ti (Pascal, sm_61).

## Behaviour changes to be aware of

* **A bare number can no longer be bound with `let`** (`let s = 3.0`). It used to be skipped
  silently and only failed later, as an undefined tensor. It is now `[E-SCALAR-LITERAL]`. Use a
  `(1,)` tensor: `let s: Tensor[f32, (1,)] = [3.0]`.
* **`NaN` in an `if`/`elif` condition is a run-time error** (`[E-NAN-IN-CONDITION]`). Comparisons
  with `NaN` are false (except `!=`), so a `NaN` loss used to silently take the `else` branch.
* **An empty tensor in a condition is a compile error** (`[E-CONDITION-EMPTY]`).
* **Multi-pass gradients:** a `backward()` pass only overwrites the gradients of tensors its loss
  depends on. Other parameters keep the gradient from the last pass that reached them. There is
  no implicit zeroing.
* **Parse errors** print a short coded message with the source line and a caret, and no Python
  traceback unless `--debug`.
* **`@EXPECT_FAILURE`** in tests now pins required output: `// @EXPECT_FAILURE: [E-CODE]`.
  A bare marker still works but is reported.
* Tests are named by feature (`control_flow_*`, `for_loop_*`, `autograd_*`), not by version.

## New

* **Stable diagnostic codes:** `[E-SHAPE-MISMATCH]`, `[E-BRANCH-MISMATCH]`, `[E-UNDEFINED-TENSOR]`,
  `[E-CONDITION-OPERAND]`, `[E-SCALAR-LITERAL]`, `[E-SYNTAX]`, `[E-CONDITION-EMPTY]`,
  `[E-NAN-IN-CONDITION]`. Messages name the expression involved (not `__tmp_N`), suggest close
  names for typos, and name the branches in a branch mismatch.
* **Comparison operations as values:** `greater_equal`, `less_equal`, `not_equal`, and the
  operators `>`, `<`, `==`, `>=`, `<=`, `!=` as expressions (`let m = x >= y`, `let m = x != 2.0`).
  Chained comparisons (`a > b > c`) are a syntax error.
* **`backward()` works in any branch** (`if`, `elif`, `else`), in nested branches, and several times
  in one branch or one `for` loop body, each running where it appears.

## Fixed (all of these produced wrong results or failed without a useful message)

* A numeric literal in an `elif` condition was never evaluated: its fill kernel was generated but
  not executed.
* A literal compared against a tensor of more than one element read past its buffer or crashed
  code generation (the comparison kernels cannot broadcast a `(1,)` operand). A literal now takes
  the shape of the tensor it is compared with.
* Top-level `if` statements and `if` inside loop bodies ran their branch without its autograd
  path, so a backward pass included the nodes of every branch (`w_grad` came out as `[22, 30]`
  instead of `[6, 8]`).
* A branch `backward()` through a value computed outside the branch gave silent zero gradients.
* Only the first `backward()` in a branch or loop body ran; later ones were skipped silently.
* `backward()` followed by `sgd` in a branch stepped with a stale gradient.
* Sibling branches shared one kernel scope, so same-named kernels were duplicate CUDA symbols.
* The test runner no longer reuses a test's old cache directory.

## Known limitations carried forward (planned for v0.5.6)

* The comparison kernels run equal shapes, or `(rows, cols)` against `(cols,)`; other
  combinations are not rejected yet and f16 operands are untested.
* `batch_norm` has no `gamma`/`beta`.
* No `isnan()` operation, so `NaN` cannot be tested for on purpose.
