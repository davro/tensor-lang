# TensorLang Release Notes

Per-release history of what was done, found and fixed. For what is planned next, see [Future Work](README.md#future-work) in the README.

Newest release first.

## v0.5.10 (released) — training that scales

* Done: data-parallel training, `python3 tensorlang.py --ddp --nproc N file.tl` (`--ddp-same-gpu` for one card). The launcher starts one worker per GPU; after every `backward()` the workers average their gradients and share the overflow flag, and at start-up every rank takes rank 0's `with grad` tensors. `{rank}` and `{world_size}` work in `load()` and `save()` paths. The exchange is a host-side TCP backend (`tensorlang/dist.py`): ranks that disagree, a dead peer or a timeout fail on every rank with a message instead of hanging. See `docs/data_parallel.md`
* Verified: `tools/ddp_check.py` trains a two-layer GELU network on the whole batch and on shards. Two ranks on two GTX 1080 Ti differ from one process by 5.4e-08 (NumPy backward) and 5.6e-08 (GPU backward), and every rank ends with bit-identical weights; four ranks on one GPU: 4.4e-08 and 5.7e-08. Not yet checked under data parallelism: Adam, `layer_norm`, `softmax`, `embedding`, and the speed-up with the gradient exchange included
* Done: a shared kernel build for data-parallel runs (one `nvcc`, a file lock, the library moved into place before its source) and per-rank output directories (`cache/<file>/ws{N}/rank{r}/`); a rank that generates different code stops the run
* Done: the launcher gives each worker its share of the CPU's BLAS threads. Two NumPy-backward workers that each used all 8 cores took 15.8 s per worker; with the cap, 4.5 s, against 4.7 s for one process. Two workers with the GPU backward ran at single-process speed (1.84 s and 2.03 s against about 1.85 s), before the gradient exchange was added
* Done: the GPU backward reuses its device buffers instead of allocating and freeing five per call (`TENSORLANG_GPU_BACKWARD_POOL=0` turns this off): a `matmul` backward 0.26 ms faster, the 20-step micro-GPT run 2.03 s to 1.85 s
* Done: the autograd pass zero-fills with `np.zeros`, tests for a zero gradient with `.any()` and checks overflow with an array method. Results are identical on 600 random graphs; the time outside the nodes fell from 12.0 to 9.7 ms per pass and the backward from 40.5 to 37.8 ms per pass
* Found and fixed: `tensorlang.py` imported the compiler at the top, and the compiler creates its CUDA context on import. Every invocation therefore created a context on whichever GPU `CUDA_VISIBLE_DEVICES` named at that moment (GPU 0 by default) before `--gpu` was applied, and a `--ddp` parent would have held a context of its own. The compiler, the test runner and the app runner are now imported where they are used; `--list-gpus` no longer needs pycuda
* Done: `tools/profile_backward_ops.py` (backward time by op type, and the work outside the nodes), `tools/bench_gpu_backward.py`, `tools/ddp_check.py`, collector presets for the new modules and for `tests/python`; 162 new Python unit tests (172 in `tests/python`)
* Done: the README's per-release history moved to `RELEASES.md`; testing, mixed precision, applications and data parallelism have their own guides in `docs/`
* Found, not fixed: the NumPy `gelu` backward uses `np.vectorize(math.erf)`: 13 ms per call on 32K elements, 34 times the GPU path, and 30% of the NumPy-backward run. Passing an absolute program path to the compiler fails (the cache directory becomes the program file). `tools/make_micro_gpt_train.py` writes into `tests/` by default, so a gradient-check run overwrote the committed probe test (restored); use `--out-tl` and `--weights` for check runs
* Moved to v0.5.11: Stage 2 of the GPU backward (gradients resident on the device, then NCCL for data parallelism), the multi-head training program for GPT-2 small, `load()` indexed by the loop counter, elementwise `sqrt` / `rsqrt` / `exp` / `log`

## v0.5.9 (released) — training you can trust

* Done: `softmax_cross_entropy(logits, targets)` — fused, stable, N-D, `(softmax - onehot)/N` gradient, four diagnostic codes; four tests
* Found and documented: `softmax` followed by `cross_entropy` back-propagates a wrong gradient (`cross_entropy` takes probabilities but its backward is the logits gradient). `cross_entropy` is unchanged
* Fixed: Adam's bias correction `1 - beta^t` was computed with a float32 `powf` on the device and was off by up to 6e-5 relative for `beta2 = 0.999`; it is now computed on the host in float64. The stored expected value of `optimizer_adam_linear` was a product of the old kernel and is now the float64 reference
* Fixed: Adam's `m` and `v` buffers were allocated, zero-uploaded and discarded every step (`setdefault` evaluates its default eagerly)
* Fixed: a loop downloaded every tensor to the host about three times per step before `backward()`; it now downloads only what the host lacks (`TENSORLANG_HOST_SYNC=check` found no stale host copy in the suite)
* Done: parser tables are cached; `load()` no longer copies data that already has the target dtype
* Done (opt-in): `TENSORLANG_GPU_BACKWARD=1` runs the `matmul` and `gelu` backward on the GPU. 20 Adam steps of a 4-layer, 868K-parameter GPT: 5.5 s to 1.9 s with the other fixes
* Done: tools to profile, generate and verify training programs (`docs/training.md`); a 162M-parameter GPT ran three Adam steps on one 11 GB card (about 10.8 s per step, random weights, not compared with a reference)
* Found, not fixed: `--cache-layers` does not overwrite an existing cache file; the README described an `opt.step()` syntax that the grammar does not accept (the Optimizers, loss-scaling and FP16 master-weight sections now use the statement form)

## v0.5.8 (released) — `embedding`, and real GPT-2 running end to end

**Phase B — language changes**
* Done: `embedding(indices, weight)` — gather rows, scatter-add gradient, run-time index check, three diagnostic codes (indices are `f32`: there is no integer dtype)
* Done: a two-head probe (`probe_gpt2_multihead`: fused QKV by slicing, head split and merge, batched attention, `embedding`) with a generator that refuses to write unless nine head-handling bugs each change the logits
* Done: tanh-GELU built from existing operations (`gelu_tanh`), values and gradient checked against float64 NumPy
* Fixed: `load()` printed a line per call regardless of the debug flags

**Phase C — demo**
* Done (C1): the real GPT-2 small (124M parameters) runs from a generated program; logits match a float64 reference to 1.1e-4, argmax identical
* Done (C2): host-driven greedy generation with a fixed-length program and the token ids read from a `.npy` (compile once: ~21 s, then ~2.9 s per token); 20 tokens identical to the reference
* Found: a run costs ~3 s even when warm, of which reading all the weight files is only ~0.4 s, so the rest is the load and copy path and process start (to profile)

## v0.5.7 (released) — a transformer forward pass, and the broadcast bug it found

**Phase A — probes (done)**
* Done: the pieces of a transformer block are expressible with existing operations (no attention primitive): batched `matmul`, `transpose(axes=...)`, `softmax(axis=)`, `layer_norm`, `gelu`, `slice`, `concat`, N-D broadcasting; a one-layer forward pass compiles and runs
* Done: the causal mask recipe (`(rows, 1) - (1, cols)` broadcast, `>= 0.0`, additive `-1e9`) matches its exact expected matrix on the GPU
* Found: in-language generation is blocked (a growing rebind is skipped at run time); there is no gather; `fill` is scalar-only and not an operand; redefining a `let` name is accepted by the type checker and then fails later; constant test inputs hide errors (see [Known limitations](README.md#known-limitations))
* Found: the first forward-pass probe was degenerate. Its weights were constant down each column, so the true logits are exactly zero and the GPU and NumPy values (about 1e-8) were two roundoff results; no mutation of the block changed them. It is replaced by a generated probe with seeded distinct weights, a real one-hot input and a built-in sensitivity check (the mask, scale, residual, bias placement and activation each move the logits by at least 0.1)
* Done: the replacement probe matches its float64 reference on the GPU, and `probe_attention_block` matches NumPy for `out` and all three gradients
* Fixed: the probe exposed a silent wrong-answer bug that exists in every earlier version, including v0.5.6. `add`, `minus`, `mult` and `div` of a square `(n, n)` tensor with an `(n,)` vector read the vector per row instead of per column, in both the unfused and the fused kernel (non-square shapes were always correct). Square weights with a bias, or a layer-norm gamma/beta, gave plausible but wrong numbers with no error. Three existing tests had pinned the wrong values (`fusion_broadcast`, `fusion_broadcast_guard`, `mlp`) and were corrected; regression test `broadcast_vector_square`. `tools/scan_square_vector_broadcast.py` lists the affected programs in a codebase and `tools/check_broadcast_kernels.py` checks the kernels against NumPy without a GPU

## v0.5.6 (released) — close the gaps found in v0.5.5

**Phase 1 — comparison and dtype guards (done)**
* Done: comparison operand shapes the kernels cannot run (anything other than equal shapes, or a `(rows, cols)` tensor against a `(cols,)` or `(1, cols)` row) are `[E-SHAPE-MISMATCH]`, for comparison `let` operations (annotated ones included) and for `if`/`elif` conditions; `tools/scan_comparison_shapes.py` found no affected comparison in the existing tests, apps or examples
* Done: f16 operands in comparisons, including against a numeric literal, are `[E-COMPARISON-DTYPE]`: the kernels are generated for f32 only. No existing test or app compared f16 (`tools/scan_comparison_dtypes.py`). Real f16 comparison support is deferred
* Done: a tensor declared as `f64` is `[E-UNSUPPORTED-DTYPE]`. f64 never had code generation support (operations raised `KeyError: 'f64'`) and comparisons misread it silently. The two `apps/trading` indicator apps now declare their window as `f32`
* Done: an undeclared `load()` of a float64 `.npy` or `.npz` is inferred as f32 and converted on load. It used to be inferred as f64 and failed with `KeyError: 'f64'` at the first operation that used it. NumPy writes float64 by default, so rejecting it would have been hostile. `tools/scan_load_f64.py` lists the undeclared loads in a codebase that this affects (none in the existing tests, apps or examples). Test: `load_f64_inferred`

**Phase 2 — one normalisation slice (done)**
* Done: `batch_norm(x, mean, var, gamma, beta)` with optional `gamma`/`beta` (both or neither), type checking with stable codes (`[E-SHAPE-MISMATCH]`, `[E-UNDEFINED-TENSOR]`, `[E-BATCH-NORM-DTYPE]`) and gradients for `x`, `gamma` and `beta`
* Found and fixed by it: the N-D forward took each element's channel as `idx % C`, which is wrong for NCHW (only half the elements got the right channel), and the backward used batch statistics while the forward uses the given ones (a gradient near zero where it should be 0.5 to 4). Both gave wrong results with no error. Regression tests: `batch_norm_nchw`, `autograd_batchnorm_given_stats`
* Deferred: training-mode batch statistics and running-statistics updates, an `axis=` option, and an `instance_norm` rework

**Phase 3 — application acceptance test (done)**
* Done: `apps/examples/adaptive_lr` — an `if loss > threshold` learning-rate step and two losses with separate optimisers, with the branch taken in some iterations and not others, pinned against hand-derived values
* Found and fixed by it: an `if` placed after the first `backward()` in a `for` body was silently never executed (a branch `sgd` did not run), and every statement after a loop containing an `if` ran the wrong kernel (the loop's kernels were miscounted). Regression tests: `for_loop_if_after_backward_sgd`, `for_loop_if_then_statement_after`
* Open: an optimiser step inside an `if` does not honour the loss-scaling overflow skip (f16 with dynamic loss scaling); the same already held for a `backward()` and `sgd` in one branch

**Also under consideration**
* Make `load_basic`, `load_dynamic`, and `load_inferred` self-contained with committed fixtures
* Dynamic loss scaling: make the growth interval configurable (otherwise deferred)

## v0.5.5 (released) — robustness, not breadth

Focus: harden what already exists rather than add another broad parity release.

**Phase 0 — test harness and regression guards (done)**
* Stable diagnostic codes in the type checker
* `@EXPECT_FAILURE` pins required output text; existing negative tests migrated
* Test runner clears each test's own cache directory before running it
* Regression tests for the seams that broke during v0.5.4: branch consistency, scalar literals in `elif`, `load()` and tensor literals in loop bodies, f16 `load()`, alias semantics, `if` inside `for`

**Phase 1 — small cleanups (done)**
* Removed dead, shadowed duplicates in `io.py` (`_ensure_gpu_alloc`, `_allocate_zeros`, `_load_data_file`)
* Removed a duplicated `_inline_user_functions_in_nodes` call
* `autograd_multiple_backward` now uses a nonzero learning rate so it can detect wrong gradients
* Multi-pass gradient semantics: a `backward()` pass no longer zeroes the gradients of parameters its loss does not reach (found by the deferred-update test)

**Phase 2 — diagnostics (done)**
* Errors name the expression involved instead of compiler temporaries (`__tmp_N`)
* `[E-SCALAR-LITERAL]` distinguishes a bare number from an undefined tensor; a bare numeric `let` is now an error instead of being skipped silently
* "Did you mean" suggestions for undefined names; branch mismatches name the branches and both types
* Parse errors show the source line with a caret (`[E-SYNTAX]`), with no traceback unless `--debug`
* Still open: better debug output for loop and branch execution

**Phase 3 — control-flow edge cases (done)**
* Done: condition semantics (`NaN` is a run-time error, an empty tensor is a compile error, `-0.0` and f16 underflow are falsy)
* Done: `greater_equal`, `less_equal`, `not_equal` and the operators `>`, `<`, `==`, `>=`, `<=`, `!=` as values
* Done: numeric literals in comparisons take the shape of the other operand (the comparison kernels cannot broadcast a `(1,)` operand)
* Done: `backward()` in an `elif` and in nested branches, plus bugs this exposed: a loss through a value computed outside the branch gave silent zero gradients; only the first `backward()` in a branch ran; the pass ran after the branch's other statements instead of in order; top-level `if` statements (and `if` inside loop bodies) ran their branch without its autograd path, so a backward pass included the nodes of every branch; and sibling branches shared one kernel scope, so same-named kernels collided as duplicate CUDA symbols (each branch now has its own scope)
* Done: a `for` loop body now runs every `backward()` it contains, in order (it used to run only the first and skip the rest silently)
* Found along the way, carried to v0.5.6: general N-D broadcasting in the comparison kernels (currently equal shapes or `(rows, cols)` vs `(cols,)`); f16 operands in comparison kernels; an explicit `isnan()` operation, since a deliberate check for `NaN` is impossible while any condition on a `NaN` is an error
