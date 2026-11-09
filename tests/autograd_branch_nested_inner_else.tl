// ============================================================================
// TensorLang: backward() two levels deep (then -> else)
// ============================================================================
// a = x*w = [3, 8]. The inner else takes loss = sum(a*a), so d loss/d w = 2*a*x
// = 2*[3,8]*[1,2] = [6, 32]. Expected value derived by hand.
//
// @EXPECTED
// {"w_grad": [6.0, 32.0]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let w: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let yes: Tensor[f32, (1,)] = [1.0]
let no: Tensor[f32, (1,)] = [0.0]
if yes {
    let a = mult(x, w)
    if no {
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
