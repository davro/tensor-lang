// ============================================================================
// TensorLang: two backward() statements inside a for-loop body
// ============================================================================
// Same numbers as autograd_branch_two_backward, inside a loop. The loop executor used to run only
// the first backward() of a body (w2_grad stayed zero); every one now runs, in order. Both
// gradients equal x = [1, 2]. Expected values derived by hand.
//
// @EXPECTED
// {"w1_grad": [1.0, 2.0], "w2_grad": [1.0, 2.0]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let w1: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let w2: Tensor[f32, (2,)] = [5.0, 6.0] with grad
for i in range(2) {
    let p1 = mult(x, w1)
    let loss1 = sum(p1)
    backward(loss1)
    let p2 = mult(x, w2)
    let loss2 = sum(p2)
    backward(loss2)
}
