// ============================================================================
// TensorLang: backward() two levels deep (then -> then)
// ============================================================================
// The outer branch computes a = x*w with a kernel; the inner branch takes the loss of it.
// loss = sum(a) -> d loss/d w = x = [1, 2]. Also exercises using a value that was
// computed in the OUTER branch from the inner branch's backward pass.
// Expected value derived by hand.
//
// @EXPECTED
// {"w_grad": [1.0, 2.0]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let w: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let yes: Tensor[f32, (1,)] = [1.0]
let no: Tensor[f32, (1,)] = [0.0]
if yes {
    let a = mult(x, w)
    if yes {
        let loss = sum(a)
        backward(loss)
    } else {
        let sq = mult(a, a)
        let loss = sum(sq)
        backward(loss)
    }
} else {
    let loss = sum(w)
    backward(loss)
}
