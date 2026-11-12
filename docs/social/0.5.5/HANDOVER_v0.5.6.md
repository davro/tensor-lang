# TensorLang — handover for the v0.5.6 chat

Paste this file, `README.md`, and fresh collector output (list at the bottom) into the new chat.

## Where things stand
* v0.5.5 is released (robustness: diagnostics, condition semantics, comparison operators as values,
  `backward()` in branches/loops, multi-pass gradient rules). 268 `.tl` + 31 Python tests pass on
  2× GTX 1080 Ti (Pascal, sm_61). README is current and describes all of it.
* To do in v0.5.6, in this order:
  1. **Comparison-shape guard** (small).
  2. **Phase 4: `batch_norm` gamma/beta**, autograd for them, axis validation (kernels).
  3. Application acceptance test (`apps/`): `if loss > threshold` LR step + two losses with separate
     optimisers, branch taken in some iterations and not others.
  4. Backlog: self-contained `load_*` tests; loss-scale growth interval; `isnan()`; general N-D
     broadcast and f16 in comparison kernels; check `verify.py` for the 3 new comparison ops;
     README's `adam([w1, w2], lr=..., eps=1e-8)` example does not parse (see below).

## The environment
* User: davro, machine `bigred`, repo `~/workspace/tensor-lang`, `python-env` venv (Python 3.12),
  2× GTX 1080 Ti. Commands: `python3 tensorlang.py --test --filter NAME`, `bash build.sh --test`
  (clears `cache/`, runs the `.tl` suite, then pytest). Work is committed to `main` and tagged.
* **The assistant has no GPU.** It cannot run nvcc or the compiler. Everything GPU-side is checked by
  the user; the assistant verifies parsing/type-checking/Python logic off-GPU.

## The working process (it worked; keep it)
1. **Tests first.** Hand-derive expected values AND recompute them independently in NumPy. Confirm a
   new test fails for the expected reason before writing the fix. Never commit a red test: park it in
   an `exploratory/` folder until the behaviour is decided.
2. **Deliver changes as anchored, idempotent Python scripts**, not line-number patches (`git apply`
   failed once on a trailing-newline difference). Pattern: each edit is an exact-text anchor that must
   match exactly once; validate every file (`ast.parse`) before writing any; skip files already
   patched (marker string); `--dry-run`; run twice = no-op. See `apply_phase*.py` in the phase bundles.
3. **Off-GPU replay harnesses.** Fake `pycuda` (device memory = numpy arrays in a small `Alloc`
   class), stub missing modules (`tensorlang.fusion`, `tensorlang.kernel_generator`), subclass the real
   mixins, and drive the REAL `_generate_kernels_for_nodes`, `_execute_if_node`,
   `_execute_backward_pass`, `_execute_for_loop` with a numpy stand-in for kernel launches. This
   reproduced the user's GPU failures exactly before each fix. Files: `verify_*_harness.py` in the
   phase bundles (re-create from scratch if lost; the technique matters more than the files).
4. **Sweeps before tightening behaviour:** `scan_scalar_lets.py`, `scan_comparison_shapes.py` (run on the
   user's repo; exit 1 = hits). Do the same before any new compile error.
5. **Negative tests pin a diagnostic code**: `// @EXPECT_FAILURE: [E-CODE]` (marker must stand alone on
   a comment line; stack several; prose mentioning the marker is ignored). Messages can be reworded
   freely; codes are stable.
6. **README as a drop-in file**, edited by anchored replacements with assertions; check TOC anchors and
   fence balance after.
7. Be honest about what is unverified: say "not run on a GPU" and why. One earlier claim ("stale kernel
   cache") was a guess presented as fact and had to be retracted.

## Pitfalls hit (avoid)
* Shell: the sandbox tool runs `sh` — `<( )` and `${PIPESTATUS}` fail; use `bash -c`. `diff` exits 1
  when files differ, which silently breaks `&&` chains.
* `git apply` with an end-of-file hunk breaks on a stray trailing newline — hence the scripts.
* A test can typecheck and still fail at code generation or run time; type-check passing is not enough.
* The user's `build.sh --test` clears the cache; `--filter` runs now clear only that test's own cache dir.

## Facts about the codebase (learned the hard way)
* **Class layout:** `TensorCompiler(CodegenMixin, ExecutionMixin, IOMixin, InteropMixin)`. MRO means
  ExecutionMixin wins over IOMixin; `compiler.py` defines its own `_load_data_file`. `io.py` is now just
  `prod` and `_eval_scalar`.
* **Grammar** (`tensorlang.lark`): `NUMBER` is `-?\d+(\.\d+)?` (no exponents: `1e-8` does not parse);
  reductions take a plain NAME (`sum(mult(a,b))` does NOT parse — split into lets); tensor literals are
  1-D/2-D only (build 3-D with `reshape`); there are no optimizer rules in the grammar: `sgd(w, w_grad,
  0.01)` parses as an ordinary call; `#` pragma lines (`# loss_scale: ...`) are stripped by the compiler
  before parsing; comparison is an `expr` alternative (6 benign shift/reduce conflicts, resolved as
  shift; chains rejected in `ast_builder`); one older `[` (slicing) conflict is pre-existing.
* **Comparison kernels** (`generators/comparisons.py`): only equal shapes, or a `(rows, cols)` operand
  against a `(cols,)`/`(1, cols)` row; hard-coded f32; the type checker accepts more than that. A numeric
  literal is lowered to a `fill` with `shape_like` = the other operand, so the equal-shape path is used.
* **Conditions:** truthiness = all-nonzero; `-0.0` and f16 underflow are falsy; inf is fine; NaN →
  `[E-NAN-IN-CONDITION]` raised at run time by `_reject_nan_condition` (checks the OPERANDS, fetched
  fresh from the GPU, because the 0/1 result has already lost the NaN); empty → `[E-CONDITION-EMPTY]`.
  `TensorLangRuntimeError` (execution.py) is caught in `compile_and_execute`.
* **Branches:** every recorded autograd node carries `branch_path` (tuple of scope labels like
  `if3_then`, `if5_elif0`); a node is active if its path is a prefix of the executing branch's path.
  Each branch has its own kernel scope (`ifN_then`, `ifN_elifK`, `ifN_else`). ALL `if` execution must go
  through `_execute_if_node` (compiler.py and the loop executor used to have inline copies that dropped
  the branch path). A backward pass in a branch syncs every device tensor and runs at its position.
* **Loops:** `_execute_for_loop` handles the FIRST `backward()` inline (loss scaling, overflow, dynamic
  scale) and runs any further ones via `_execute_backward_pass`.
* **Gradient rules (Option B):** `ComputationGraph.grads_written`, `reachable_tensors()`,
  `keeps_previous_gradient()`: a pass overwrites `<name>_grad` only for tensors its loss reaches; never-
  reached tensors still get zeros once; `grads_written` is reset at the start of each compile. No implicit
  zeroing (an optimizer step on an unreached parameter reuses its old gradient).
* **Diagnostics:** helpers in `type_checker.py` (`_undefined_message`, `_describe_binding`,
  `_render_expr`, `_name_label`, `_empty_condition_message`, `_scalar_literal_message`); parse errors in
  `compiler.py::_format_parse_error`. Synthetic names: `__tmp_N` (flattened nested calls), `__if_cond_N`.
* **Known leftovers:** the `any()/all()` condition block in `type_checker.py` is duplicated (the second
  copy is unreachable); `_count_backward` in `type_checker.py` is unused; `verify.py` (NumPy interpreter)
  and `check.py` were never seen by the assistant.

## v0.5.6 task notes

### 1. Comparison-shape guard
* Allowed: `shape1 == shape2` (same rank/dims), or `len(shape1) == 2` and `shape2 in {(cols,), (1, cols)}`
  where `cols = shape1[1]`. Anything else: `[E-SHAPE-MISMATCH]` with a hint ("comparison kernels support
  equal shapes, or a (rows, cols) tensor against a (cols,) row"). The helper `supported()` in
  `scan_comparison_shapes.py` is the reference rule.
* Apply to comparison `let` ops (type_checker op branch for greater/less/equal/greater_equal/less_equal/
  not_equal) AND to `if/elif` conditions (these use `np.broadcast_shapes`, looser). Literals already
  match their partner's shape.
* Sweep result on the user's repo (before the guard): 716 comparisons, 275 files, 0 suspicious. Open
  question: reject f16 operands too, or only document.
* Tests: one `diagnostics_*` negative per rejected pattern (`(3,)` vs `(1,)`, row vs matrix, rank 3 vs
  rank 2) pinned to the code.

### 2. `batch_norm` gamma/beta (Phase 4)
* Today: `batch_norm(x, running_mean, running_var)`; uses running stats; NO affine; simplified NCHW
  indexing for N-D; gradients simplified; `instance_norm` flattens spatial dims. README "Known
  limitations" says so.
* Needs a design decision first: signature (`batch_norm(x, mean, var, gamma, beta)` vs optional kwargs),
  eps, axis (channel at dim 1 today), dtype (f16), whether gamma/beta are `with grad` parameters.
* Work: kernel (`generators/nn.py`), type checker (shapes of gamma/beta = channel dim, axis validation,
  codes), codegen/compiler op chains (top-level compile path in `compiler.py` AND generic path in
  `codegen.py` — ops are duplicated in both), autograd (`autograd.py`: grads for x, gamma, beta), tests
  (forward values vs NumPy, autograd values, negative tests for bad shapes/axis), README.
* Files the assistant needs (it has only the ones in the collector list below): see "Collector".

### 3. Application acceptance test
Small app under `apps/` (needs `app.toml`; see README "Applications"), regression-tested via
`python3 tensorlang.py --app ... --test`. Use `if loss > threshold` (comparison with a literal) and two
losses with separate `sgd` steps, branch taken in some iterations and not others. Check optimizer
syntax against `tests/optimizer_adam_linear.tl` first (README example is wrong).

## Release housekeeping already decided
Tag `v0.5.5` (annotated), version string in `tensorlang/tensor_lang.py`, GitHub release from
`RELEASE_NOTES_v0.5.5.md`. Scratch files (`code.txt`, `collector.py`, `phase*/`, `optionb/`, `*.patch`)
are not committed; the two `scan_*.py` sweeps are worth keeping under `tools/`.

## Collector — send this next time
Extend the user's `collector.py` file list with everything the next tasks touch:

```python
files = [
    "tensorlang.lark",
    "tensorlang/ast_builder.py", "tensorlang/type_checker.py", "tensorlang/codegen.py",
    "tensorlang/compiler.py", "tensorlang/execution.py", "tensorlang/autograd.py", "tensorlang/io.py",
    "tensorlang/constants.py", "tensorlang/tensor_lang.py", "tensorlang/fusion.py",
    "tensorlang/kernel_generator.py", "tensorlang/interop.py", "tensorlang/test_runner.py",
    "tensorlang/generators/base.py", "tensorlang/generators/nn.py", "tensorlang/generators/shapes.py",
    "tensorlang/generators/comparisons.py", "tensorlang/generators/elementwise.py",
    "tensorlang/generators/reductions.py", "tensorlang/generators/unary.py",
    "tensorlang/generators/optimizers.py", "check.py", "verify.py", "build.sh",
]
```
Plus, for Phase 4: `tests/batch_norm.tl`, `tests/autograd_batchnorm.tl`, `tests/instance_norm.tl`,
`tests/autograd_instancenorm.tl`, `tests/normalization.tl`, `tests/optimizer_adam_linear.tl`.
