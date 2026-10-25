// tests/linear_autograd_basic.tl
let x: Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]

let w: Tensor[f32, (1, 1)] = [[0.5]] with grad
let b: Tensor[f32, (1,)] = [0.0] with grad
let lr: Tensor[f32, (1, 1)] = [[0.05]]

for epoch in range(5) {
    let y_pred = linear(x, w, b)
    let loss = mse_loss(y_pred, y_true)
    backward(loss)
    let w_updated = minus(w, mult(lr, w_grad))
    let b_updated = minus(b, mult(lr, b_grad))
    w = w_updated
    b = b_updated
}

let y_pred_final = linear(x, w, b)
let loss_final = mse_loss(y_pred_final, y_true)

// @EXPECTED
// {"loss_final": 0.030034389346837997,
//  "w": [[1.8556015625]],
//  "b": [0.42401015625]}

