let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad
let b: Tensor[f32, (2, 2)] = [[0.1, 0.2], [0.1, 0.2]]
let y_true: Tensor[f32, (2, 2)] = [[1.6, 1.7], [3.6, 3.7]]
let lr: Tensor[f32, (1,)] = [0.01]

for epoch in range(3) {
    let h = relu(add(matmul(x, w), b))
    let loss = mse_loss(h, y_true)
    backward(loss)
    let w_updated = minus(w, mult(lr, w_grad))
    w = w_updated
}

let h_final = relu(add(matmul(x, w), b))

// @EXPECTED
// {"h_final": [[1.6, 1.7], [3.6, 3.7]]}
