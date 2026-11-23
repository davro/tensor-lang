# Mixed-precision training in TensorLang

Two features make FP16 training stable: **loss scaling**, which keeps small gradients from underflowing, and **FP32 master weights**, which keep the optimizer's arithmetic out of FP16. Both work inside `for` loops. The short version is in the README ([Mixed Precision](../README.md#mixed-precision)); the optimizers themselves are covered in [training.md](training.md) and the README's [Optimizers](../README.md#optimizers).

## Loss Scaling

TensorLang supports **static and dynamic loss scaling** for mixed-precision (FP16) training stability. Loss scaling prevents gradient underflow by scaling the loss (and thus all gradients) by a large factor during backward, then unscaling before the optimizer step.

### How It Works

1. During `backward()`, the loss gradient is seeded with `scale` instead of `1.0`
2. All gradients are scaled by `scale` during backpropagation
3. Before the optimizer step, gradients are divided by `scale` (unscaled)
4. If the scale exceeds FP16 max (~65504), gradients are automatically upcast to FP32
5. **Dynamic mode:** overflow (inf/nan in gradients) is detected after backward; scale is halved and optimizer step is skipped; after N successful steps, scale is doubled

### Configuration

**Via pragma (in `.tl` file):**

```text
# loss_scale: 65536
let x: Tensor[f16, (4,)] = [1.0, 2.0, 3.0, 4.0] with grad
let y = mult(x, x)
let loss = sum(y)
backward(loss)
```

**Auto-detect:**

```text
# loss_scale: auto
```

Auto-detect checks if any tensor in the graph is FP16 and selects `65536.0`; otherwise uses `1.0`.

**Dynamic scaling:**

```text
# loss_scale: dynamic [init_scale] [patience]
```

Enables dynamic scaling with optional initial scale (default: auto-detected) and patience (default: 2000 successful steps before doubling).

**Via Python API:**

```python
compiler = TensorCompiler(loss_scale=65536.0)
compiler = TensorCompiler(dynamic_loss_scale=True, loss_scale_patience=100)
compiler.compile_and_execute("model.tl")
```

**Default:** `1.0` (no scaling). Existing tests and FP32 programs are unaffected.

### Supported Scales

| Dtype | Default Scale | Notes |
|---|---|---|
| `f32` | `1.0` | No scaling needed |
| `f16` | `65536.0` (2^16) | Auto-detected or explicit; gradients upcast to FP32 |

### Loss Scaling in Training Loops

Loss scaling works inside `for` loops. The scale is applied per-backward-call and unscaled before the optimizer step:

```text
# loss_scale: 65536
let w: Tensor[f16, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

for epoch in range(10) {
    let y = matmul(x, w)
    let loss = mse_loss(y, y_true)
    backward(loss)
    sgd(w, w_grad, 0.01)
}
```

### Limitations

* **Multiple `backward()` statements** — the interaction of loss scaling with more than one `backward()` in a file is not yet tested.
* **FP16 master weights** — supported (see [FP16 Master Weights](#fp16-master-weights)).

## FP16 Master Weights

TensorLang supports **FP32 master weights** for FP16 parameters. When an FP16 tensor has `with grad`, the compiler automatically allocates an FP32 master weight buffer. The optimizer (Adam/SGD) computes updates in FP32 (m, v buffers in FP32) and casts the result back to FP16 for the forward pass.

### How It Works

1. When an FP16 tensor is declared with `with grad`, an FP32 master weight buffer (`{name}_master`) is allocated
2. Master is initialized from the FP16 param (cast FP16 → FP32)
3. During the optimizer statement (`sgd` or `adam`), the optimizer updates the master weight (FP32) with FP32 m/v buffers
4. After update, master is cast to FP16 and written to the param buffer
5. The forward pass uses the FP16 param as before

### Usage

No configuration needed — master weights are automatic for FP16 params with grad:

```text
let w: Tensor[f16, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

for epoch in range(100) {
    let y = matmul(x, w)
    let loss = mse_loss(y, y_true)
    backward(loss)
    adam(w, w_grad, 0.01)  // updates master (FP32), casts to w (FP16)
}
```

FP32 params continue to use the original (non-master) code path. No extra memory or overhead for FP32 training.

### Benefits

* **Better precision** — optimizer state (m, v) computed in FP32 instead of FP16
* **No underflow** — weight updates in FP32 prevent FP16 underflow in small gradients
* **Transparent** — no API changes; automatic for FP16 params with grad
* **Compatible with loss scaling** — works with both static and dynamic loss scaling
