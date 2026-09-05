// apps/examples/decision_boundary/main.tl
//
// Trains a tiny 2-8-1 tanh/sigmoid MLP (with bias) on XOR for ONE chunk
// of epochs, then evaluates the (now-updated) weights over a fixed
// visualization grid and saves that snapshot for the pygame viewer.
//
// Why chunked instead of one long-running loop:
//   - save() targets must be string literals, so a single loop can't
//     write a new filename per epoch.
//   - rebind (w = w_updated) requires a fixed-shape GPU buffer per
//     iteration, so growing a snapshot-history tensor inside the loop
//     isn't supported either.
//   Instead: run this whole file once per chunk. Weights load from and
//   save back to the same cache path each time, so training resumes
//   where the previous chunk left off. tools/train_and_snapshot.py
//   drives the repeated invocation and collects each chunk's frame.
//
// Paths:
//   - load("cache/apps/examples/decision_boundary/main.tl/...") reads
//     back exactly what THIS file's own save() calls wrote last chunk
//     (save() is auto-prefixed with cache/<this-file's-path>/ by the
//     compiler; load() is not, so the load path has to spell that out).
//   - load("apps/examples/decision_boundary/data/grid.npy") reads the
//     static evaluation grid, generated once by tools/init_state.py and
//     checked into the app dir (not the gitignored cache/ tree).
//
// Why bias terms matter here: without them, matmul([0,0], w1) is
// exactly 0 for every hidden unit regardless of w1, so tanh(0)=0 and
// the (0,0) corner's prediction is permanently stuck at sigmoid(0)=0.5
// — a hard MSE floor of 0.25/4=0.0625 that no amount of training on w1/w2
// alone can get under. b1/b2 let the network shift past the origin.

let x:      Tensor[f32, (4, 2)] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
let y_true: Tensor[f32, (4, 1)] = [[0.0], [1.0], [1.0], [0.0]]
let lr:     Tensor[f32, (1, 1)] = [[0.5]]
// Same-shaped learning-rate vectors for the 1D bias grads. Needed because
// (1) the type checker's mult broadcast rule for a 2D-shape x 1D-shape pair
// only handles it correctly when the 1D side has size 1, and (2) reshape()
// does its own standalone env lookup with no special-casing for _grad-suffixed
// names (unlike add/minus/mult/div, which do infer a grad tensor's shape from
// its base tensor) — so reshaping b1_grad/b2_grad isn't an option either.
// Keeping both mult operands the same 1D shape sidesteps both issues.
let lr_b1: Tensor[f32, (8,)] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
let lr_b2: Tensor[f32, (1,)] = [0.5]

let w1: Tensor[f32, (2, 8)] = load("cache/apps/examples/decision_boundary/main.tl/weights/w1.npy") with grad
let w2: Tensor[f32, (8, 1)] = load("cache/apps/examples/decision_boundary/main.tl/weights/w2.npy") with grad
let b1: Tensor[f32, (8,)]   = load("cache/apps/examples/decision_boundary/main.tl/weights/b1.npy") with grad
let b2: Tensor[f32, (1,)]   = load("cache/apps/examples/decision_boundary/main.tl/weights/b2.npy") with grad

for epoch in range(25) {
    let h_pre        = matmul(x, w1)
    let h_pre_biased = add(h_pre, b1)
    let h            = tanh(h_pre_biased)
    let y_pre        = matmul(h, w2)
    let y_pre_biased = add(y_pre, b2)
    let pred         = sigmoid(y_pre_biased)
    let loss         = mse_loss(pred, y_true)
    backward(loss)
    let w1_step     = mult(lr, w1_grad)
    let w2_step     = mult(lr, w2_grad)
    let b1_step     = mult(lr_b1, b1_grad)
    let b2_step     = mult(lr_b2, b2_grad)
    let w1_updated  = minus(w1, w1_step)
    let w2_updated  = minus(w2, w2_step)
    let b1_updated  = minus(b1, b1_step)
    let b2_updated  = minus(b2, b2_step)
    w1 = w1_updated
    w2 = w2_updated
    b1 = b1_updated
    b2 = b2_updated
}

// Post-loop eval on the training set — distinct names, per the
// established convention, to avoid CUDA kernel symbol collisions with
// the identically-shaped tensors declared inside the loop body above.
// Note: sigmoid/tanh only accept a bare NAME argument (not a nested
// call like tanh(matmul(...))), so each step needs its own let.
let h_pre_final        = matmul(x, w1)
let h_pre_final_biased = add(h_pre_final, b1)
let h_final            = tanh(h_pre_final_biased)
let y_pre_final        = matmul(h_final, w2)
let y_pre_final_biased = add(y_pre_final, b2)
let pred_final         = sigmoid(y_pre_final_biased)
let loss_final         = mse_loss(pred_final, y_true)

// Evaluate the same post-chunk weights over the fixed visualization grid
let grid: Tensor[f32, (1600, 2)] = load("apps/examples/decision_boundary/data/grid.npy")
let h_pre_grid        = matmul(grid, w1)
let h_pre_grid_biased = add(h_pre_grid, b1)
let h_grid            = tanh(h_pre_grid_biased)
let y_pre_grid        = matmul(h_grid, w2)
let y_pre_grid_biased = add(y_pre_grid, b2)
let pred_grid         = sigmoid(y_pre_grid_biased)

// Persist weights for the next chunk, and this chunk's outputs for the driver
save(w1, "weights/w1.npy")
save(w2, "weights/w2.npy")
save(b1, "weights/b1.npy")
save(b2, "weights/b2.npy")
save(pred_grid, "frames/frame.npy")
save(loss_final, "frames/loss.npy")