// ============================================================================
// TensorLang: for loop — save() alias stability
// ============================================================================
// Verifies that save() on an alias captures the current storage, not a
// stale pointer from a previous iteration.

let w: Tensor[f32, (1, 1)] = fill(0.0, (1, 1))

for i in range(3) {
    let inc: Tensor[f32, (1, 1)] = fill(1.0, (1, 1))
    let w_new = add(w, inc)
    let y = w_new
    w = w_new
    save(y, "y.npy")
    save(w, "w.npy")
}

// @EXPECTED
// {
//   "w": [[3.0]]
// }

