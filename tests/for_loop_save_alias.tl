let w: Tensor[f32, (1, 1)] = [[0.5]] with grad
let lr: Tensor[f32, (1, 1)] = [[0.1]]

for epoch in range(3) {
    let g: Tensor[f32, (1, 1)] = fill(1.0, (1, 1))   // stand-in gradient
    let step = mult(lr, g)
    let w_new = minus(w, step)
    w = w_new
    save(w, "w.npy")
}

// @EXPECTED
// {
//   "w":  [[0.2]],
//   "lr": [[0.1]]
// }
