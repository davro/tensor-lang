# Training in TensorLang

How to write a training loop, which loss and optimizer to use, the opt-in speed options, and how to check that a training
program is right. The language reference for loops is in the README ([Training Loops](../README.md#training-loops)); this page
covers what you need to train something real. Numbers were measured on a GTX 1080 Ti (Pascal, sm_61).

## A training step

Mark parameters `with grad`, compute a loss, call `backward(loss)`, and update. The gradient of a parameter `W` is available
as `W_grad`.

```text
let x: Tensor[f32, (4, 3)] = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
let target: Tensor[f32, (4,)] = [0.0, 1.0, 1.0, 0.0]
let W: Tensor[f32, (3, 2)] = [[0.1, -0.2], [0.3, 0.2], [-0.1, 0.4]] with grad
let lr: Tensor[f32, (1,)] = [0.5]

for step in range(3) {
    let logits = matmul(x, W)
    let loss = softmax_cross_entropy(logits, target)
    backward(loss)
    let W_updated = minus(W, mult(lr, W_grad))
    W = W_updated
}
```

This is `tests/softmax_cross_entropy_train.tl`; the loss falls from 0.7996 to 0.6245 over the three steps. The loop count must
be an integer literal, and a rebind needs a name on the right-hand side (`W = W_updated`).

## Losses

| Loss | Takes | Use it for |
|---|---|---|
| `softmax_cross_entropy(logits, targets)` | logits, N-D (last dimension = classes); `f32` class indices of shape `logits.shape[:-1]` | **Classification and language models.** Fused and numerically stable. Gradient `(softmax - onehot) / N`. Result: an f32 scalar, the mean over rows |
| `mse_loss(pred, target)` | same-shaped tensors | Regression. The result is a `(1,1)` tensor |
| `cross_entropy(probs, targets)` | **probabilities**, class-index targets | Evaluation on probabilities. **Not for training after `softmax`** |

**Do not train with `softmax` followed by `cross_entropy`.** `cross_entropy` takes probabilities, but its backward is the
gradient with respect to *logits*, `(p - onehot)/N`. After a `softmax`, that wrong gradient is pushed through the softmax's
backward, and the loss decreases for a while while the weights move the wrong way (about 2.5 times too small on the test
case, and a cosine of 0.71 with the true gradient on a random 5-class example). The loss value is correct, so nothing looks
broken. `softmax_cross_entropy` takes the logits directly. It also survives logits of magnitude 1000, where `softmax` followed
by `-log` would overflow or hit the probability clamp (`tests/softmax_cross_entropy_stable.tl`).

`softmax_cross_entropy` errors: `E-SCE-ARGS` (needs two arguments), `E-SCE-SHAPE` (logits need at least two dimensions;
targets must have `logits.shape[:-1]`), `E-SCE-DTYPE` (targets must be f32, because f16 cannot represent ids above 2048;
logits f32 or f16), and the run-time `E-SCE-TARGET` (a target that is not an integer in `[0, classes)`).

## Optimizers

The optimizer is a statement in the loop body, after `backward()`. Arguments are positional:

```text
sgd(W, W_grad, 0.1)
adam(W, W_grad, 0.001)                        # lr
adam(W, W_grad, 0.001, 0.9, 0.999, 1e-8)      # lr, beta1, beta2, eps (these are the defaults)
```

For the example above, replace the two update lines with `adam(W, W_grad, 0.1)` (and drop `lr`).

* Adam's `m` and `v` buffers are allocated once per parameter and persist across iterations.
* For an `f16` parameter the optimizer keeps float32 master weights and float32 `m` and `v`, and casts back to f16.
* The `let opt = adam([w1, w2], ...)` / `opt.step()` form that older README text described is not accepted by the current
  grammar. Use the statement form.

**Adam and tight tolerances.** Adam divides each gradient by its own size, so float32 noise on a near-zero gradient becomes a
full-size step. A parameter whose true gradient is exactly zero (for example the key bias in attention: softmax ignores a
constant added to every score) is moved by about `lr` per step in a direction set by rounding noise. So do not compare final
weights element by element at `rtol=1e-5`. Compare the loss, or compare gradients (see below).

## Speed options (opt-in)

These are environment variables; nothing changes unless you set them.

| Variable | Effect |
|---|---|
| `TENSORLANG_GPU_BACKWARD=1` | Runs the `matmul` and `gelu` backward on the GPU. Everything else still runs in NumPy on the CPU. `=2` also prints a usage line at exit. `TENSORLANG_GPU_BACKWARD_MIN=<n>` skips operands smaller than n elements |
| `TENSORLANG_HOST_SYNC=full` | Restores the old behaviour of re-downloading every tensor before each loop's backward pass. The default downloads only what the host does not have. `=check` does the full download and reports any stale host copy at exit |

On a 4-layer GPT (868K parameters, sequence 64, width 128), 20 Adam steps, warm: **5.5 s with the NumPy backward, 1.9 s** with
`TENSORLANG_GPU_BACKWARD=1` and the other fixes in v0.5.9. The backward alone went from about 3.5 s to 0.85 s. The GPU backward is
"stage 1": operands go up and gradients come back for each `matmul`, which still costs about a fifth of the run. Gradients agree
with a float64 reference to about 1e-6 norm-wise. The backward `matmul` uses a simple tiled kernel, not cuBLAS.

### Where does the time go?

```bash
TENSORLANG_GPU_BACKWARD=1 python3 tools/profile_train_step.py my_program.tl            # run twice: the first run may compile
TENSORLANG_GPU_BACKWARD=1 python3 tools/profile_train_step.py my_program.tl --cprofile 25
```

It prints GPU wait, device-to-host and host-to-device copies, and the backward (with its own copies shown underneath); "rest"
is everything else (launches, optimizer kernels, loading, printing). Run with debug flags off.

## Checking that a training program is right

* **`tools/make_micro_gpt_train.py`** generates a training program for a small pre-LN GPT with a float64 NumPy reference that has
  its own hand-written backward. Before it writes anything it checks the reference against finite differences, runs ten
  deliberate bugs (no causal mask, wrong scale, a missing residual, relu for gelu, SGD for Adam, and so on) and refuses to write
  unless each one changes the result, and checks that float32 rounding alone would not fail the runner. The default writes
  `tests/probe_micro_gpt_train.tl`. `--perf` writes a larger program without a reference, for profiling:
  `--perf --dim 128 --seq 64 --layers 4 --vocab 256 --ff-mult 4 --steps 20 --out-tl build/micro_gpt_perf.tl --weights build/micro_gpt_perf`.
* **`tools/compare_micro_gpt.py`** compares what the GPU produced with the reference. For gradients, generate with
  `--no-optimizer --steps 1`, run with `--cache-layers`, then `compare_micro_gpt.py --grads`: float32 noise is about 1e-6
  norm-wise, and anything above 1e-4 is a real difference. Delete the cache folder before re-running, because `--cache-layers`
  does not overwrite files that already exist.
* **`tools/check_adam_bias.py`** measures the accuracy of the Adam bias-correction terms on your GPU.

## What runs today, and what does not

* A 162M-parameter, 12-layer GPT (width 768, vocabulary 50,257, sequence 64) with random weights ran three Adam steps on one
  11 GB card at about 10.8 s per step (warm). That is a stress test, not a trained model, and it was not compared with a
  reference at that size. Memory is about 16 bytes per parameter for the weights, gradients and Adam state.
* **One fixed batch.** `load()` cannot be indexed by the loop counter, so a loop cannot see a different batch each step.
* **Backward is mostly CPU.** Only `matmul` and `gelu` can run on the GPU, so every other op's backward runs in NumPy.
  Forward results are still copied to the host after each kernel.
* **No multi-head training generator yet.** The generator above uses one head and separate q, k and v; GPT-2 has 12 heads and
  a fused QKV weight, so fine-tuning GPT-2 small needs that extension.
* With `--cache-layers`, the last iteration's loop-body tensors (`loss`, `<param>_grad`) are saved, and that is what the test
  runner compares. In a loop the gradient is `<param>_grad`; `<param>.grad` is only available after a top-level `backward()`.

## Tools

| Tool | Purpose |
|---|---|
| `tools/profile_train_step.py` | Where a run spends its time |
| `tools/make_micro_gpt_train.py` | Training-program generator with a float64 reference and a mutation check |
| `tools/compare_micro_gpt.py` | GPU results against the reference (final weights, or `--grads`) |
| `tools/check_adam_bias.py` | Accuracy of Adam's bias correction on your GPU |
