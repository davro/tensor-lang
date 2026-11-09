// ============================================================================
// TensorLang: two backward() statements inside one branch
// ============================================================================
// Each pass runs in order. w1 = [3,4], w2 = [5,6], x = [1,2]:
//   pass 1: loss1 = sum(x*w1) -> w1_grad = x = [1, 2]
//   pass 2: loss2 = sum(x*w2) -> w2_grad = x = [1, 2]   (w1_grad is kept, see multi-pass rules)
// If only the first backward ran, w2_grad would still be zero. Expected values derived by hand.
//
// @EXPECTED
// {"w1_grad": [1.0, 2.0], "w2_grad": [1.0, 2.0]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let w1: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let w2: Tensor[f32, (2,)] = [5.0, 6.0] with grad
let yes: Tensor[f32, (1,)] = [1.0]
if yes {
    let p1 = mult(x, w1)
    let loss1 = sum(p1)
    backward(loss1)
    let p2 = mult(x, w2)
    let loss2 = sum(p2)
    backward(loss2)
}
