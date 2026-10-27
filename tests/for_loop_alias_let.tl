// ============================================================================
// TensorLang: for loop — alias let tracking across rebinds
// ============================================================================
// Verifies that `let y = w` inside a loop re-resolves each iteration,
// so y tracks w even after w is rebound to new storage.

let w: Tensor[f32, (1, 1)] = fill(0.0, (1, 1))

for i in range(3) {
    let inc: Tensor[f32, (1, 1)] = fill(1.0, (1, 1))
    let w_new = add(w, inc)
    w = w_new
    let y = w
    save(y, "y.npy")
}

save(w, "w.npy")

// @EXPECTED
// {
//   "w": [[3.0]]
// }

