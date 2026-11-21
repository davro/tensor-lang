# TensorLang v0.5.9

Theme: **training is correct, and a lot less wasteful.** A new fused loss for training, two optimizer fixes, an opt-in GPU
backward for `matmul` and `gelu`, and a set of measurements and tools that found most of it. 319 `.tl` tests and the
Python tests pass on 2x GTX 1080 Ti (Pascal, sm_61); v0.5.8 had 311.

## New

* **`softmax_cross_entropy(logits, targets)`**: fused and numerically stable (`logsumexp - logit[target]`, mean over rows).
  N-D logits (last dimension = classes), f32 or f16; targets are `f32` class indices of shape `logits.shape[:-1]`; the result
  is an f32 scalar. Gradient `(softmax - onehot)/N`. CUDA forward, NumPy backward, float64 gradient check against finite
  differences. Errors: `E-SCE-ARGS`, `E-SCE-SHAPE`, `E-SCE-DTYPE`, and a run-time `E-SCE-TARGET` for targets outside
  `[0, classes)`. Logits of magnitude 1000 give the exact loss (`tests/softmax_cross_entropy_stable.tl`).
* **GPU backward for `matmul` and `gelu` (opt-in, stage 1).** `TENSORLANG_GPU_BACKWARD=1` runs those two backward ops on the
  GPU (tiled transposed-gemm kernel, `erff`-based gelu derivative); anything it does not handle falls back to NumPy.
  `=2` also prints a usage summary at exit; `TENSORLANG_GPU_BACKWARD_MIN=<n>` skips small operands. Gradients agree with the
  float64 reference to about 1e-6 norm-wise. Operands still go up and gradients come back per op (stage 2 will keep them
  on the device).
* **Tools:** `tools/profile_train_step.py` (where a run spends its time: GPU wait, copies, backward, with `--cprofile`),
  `tools/make_micro_gpt_train.py` (generates a training program for a small pre-LN GPT plus a float64 NumPy reference with a
  hand-written backward, a finite-difference check and a ten-bug mutation check), `tools/compare_micro_gpt.py` (GPU
  gradients and final weights against that reference), `tools/check_adam_bias.py`.
* **Tests (8 new):** `embedding` inside a `for` loop, inside a user function, and in a training loop with a scatter-add
  gradient; four for `softmax_cross_entropy`; `probe_micro_gpt_train` (2-layer GPT, three Adam steps, loss matches float64
  to 5e-8).

## Fixed

* **Adam bias correction.** The kernels computed `1.0f - powf(beta2, t)`; for `beta2 = 0.999` that is about 1e-3 x t, so the
  `powf` rounding error became a relative error of up to 6e-5 (measured on the 1080 Ti: 0.0010000467 against an exact
  0.0009999871) in every weight's step. The corrections are now computed once per launch on the host in float64. On the
  micro-GPT probe the last-loss error fell from 4.5e-6 to 5e-8 and the largest weight error from 1.6e-5 to 4e-7. The stored
  expected value of `optimizer_adam_linear` was a product of the old kernel (2.5e-5 off the exact answer); it is now the
  float64 reference, 0.0991352973. Earlier versions trained with Adam correctly to about 1e-5 relative, not exactly.
* **Adam state was allocated every step.** `optimizer_state.setdefault(key, {allocate...})` evaluates its default on every
  call, so each step allocated, zero-uploaded and discarded two GPU buffers per parameter (2,760 allocations in 20 steps of a
  69-parameter model, instead of 138). Results were never affected. Now allocated once, zeroed on the device.
* **Loops copied every tensor to the host about three times per step** before `backward()`. Each loop now downloads only
  what the host does not already have (`TENSORLANG_HOST_SYNC=check` over the full suite found 0 stale copies; `=full` restores
  the old behaviour). Device-to-host traffic on the micro-GPT fell from 557 MB to 163 MB over 20 steps.

## Faster

* Parser tables are cached (`Lark(cache=True)`): `compile_and_execute` on a small program 0.50 s -> 0.07 s, a whole run
  1.4 s -> 1.0 s.
* `load()` no longer copies data that already has the target dtype.
* 20 Adam steps on a 4-layer, 868K-parameter GPT (warm, GTX 1080 Ti): **5.5 s -> 1.9 s** (0.27 s -> 0.095 s per step) from the
  GPU backward, the Adam state fix and the download change together.

## Experiments (tools, not language features)

* **Weight-space editing demo** (`tools/weight_edit.py`, `tools/demo_weight_edit.sh`; details in
  `docs/weight_space_programming.md`). The edits are made in Python on exported GPT-2 weights; TensorLang's part is running the
  edited model through the same compiled forward pass, which agrees with a float64 NumPy reference to under 2e-4 max abs
  with identical argmax and top-5. Example: adding a vector to one MLP bias (layer 7, alpha 4) turns the greedy continuation
  of "Paris is the capital of" into "England". This is a constant steering vector added at every position, not a located
  fact edit, so it can move unrelated prompts (1 of 4 in the small check) and can repeat the target token.
* **A negative result, kept on purpose.** A Lark-guided search (`scripts/lark_auto_search.py`, scored by
  `scripts/lark_scorer.py`) over layers 5-9 and alphas 1, 2, 4, 8 found no bias edit that makes GPT-2 emit more valid
  TensorLang: weak edits changed nothing and strong ones lowered the mean parse score.

## Known issues and limits

* **`cross_entropy` must not follow `softmax` when training.** It takes probabilities but its backward is the logits
  gradient `(p - onehot)/N`, so `softmax` -> `cross_entropy` back-propagates a wrong gradient (about 2.5x too small on the
  test case). Use `softmax_cross_entropy`. `cross_entropy` itself is unchanged; the README now says so.
* `--cache-layers` does not overwrite an existing cache file, so repeated direct runs can leave stale `.npy` files (the test
  runner starts clean).
* A training loop sees one fixed batch: `load()` cannot yet be indexed by the loop counter.
* The GPU backward is opt-in and stage 1; per-launch `cudaDeviceSynchronize` is unchanged (it is hidden behind the per-kernel
  host copies, which still wait for the GPU).
* Scale check, not a result: a randomly initialised 12-layer, 768-wide, 50,257-token GPT (162M parameters) ran three Adam
  steps on one 11 GB card at about 10.8 s per step (warm). It was not compared with a reference at that size.