// ============================================================================
// TensorLang: a branch backward through a value computed BEFORE the if
// ============================================================================
// a = x*w is a kernel output at top level. The branch only computes loss = sum(a) and runs
// backward: d loss/d w = x = [1, 2]. The autograd reference needs a's value, which lives on
// the GPU, so it must be synced even though the branch itself never computes it.
// Expected value derived by hand.
//
// @EXPECTED
// {"w_grad": [1.0, 2.0]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let w: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let yes: Tensor[f32, (1,)] = [1.0]
let no: Tensor[f32, (1,)] = [0.0]
let a = mult(x, w)
if yes {
    let loss = sum(a)
    backward(loss)
}
