// ============================================================================
// TensorLang: inline expressions — binary op nesting
// ============================================================================
// Description:
//   Tests that binary op arguments accept full inline_expr subtrees.
//   The weight update step uses a single inline expression:
//
//     let w_updated = minus(w, mult(lr, w_grad))
//
//   The AST builder flattens this into:
//     let __tmp_0  = mult(lr, w_grad)
//     let w_updated = minus(w, __tmp_0)
//
//   Expected values are identical to for_loop_basic.tl — same computation,
//   written with inline syntax instead of intermediate let bindings.
//
// @EXPECTED
// {
//   "loss_final": 0.0164794921875,
//   "w":          [[2.046875]]
// }

let x:      Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let w:      Tensor[f32, (1, 1)] = [[0.5]] with grad
let lr:     Tensor[f32, (1, 1)] = [[0.1]]

for epoch in range(5) {
    let y_pred    = matmul(x, w)
    let loss      = mse_loss(y_pred, y_true)
    backward(loss)
    let w_updated = minus(w, mult(lr, w_grad))
    w = w_updated
}

let y_pred_final = matmul(x, w)
let loss_final   = mse_loss(y_pred_final, y_true)