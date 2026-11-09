// ============================================================================
// TensorLang: a branch that is not taken never runs its backward()
// ============================================================================
// A top-level backward pass first sets w_grad = x = [1, 2]. The branch below would replace it
// with 2w = [6, 8] if its backward pass ran, but its condition is false, so w_grad must still
// be [1, 2]. Expected value derived by hand.
//
// @EXPECTED
// {"w_grad": [1.0, 2.0]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let w: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let no: Tensor[f32, (1,)] = [0.0]
let a = mult(x, w)
let loss0 = sum(a)
backward(loss0)
if no {
    let z = mult(w, w)
    let loss = sum(z)
    backward(loss)
}
