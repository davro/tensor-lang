// ============================================================================
// TensorLang: backward() in an if nested inside an elif branch
// ============================================================================
// The first condition is false, the elif is taken, and inside it an inner if runs the
// backward pass: loss = sum(x*w) -> d loss/d w = x = [1, 2]. Expected value derived by hand.
//
// @EXPECTED
// {"w_grad": [1.0, 2.0]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let w: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let yes: Tensor[f32, (1,)] = [1.0]
let no: Tensor[f32, (1,)] = [0.0]
if no {
    let loss = sum(w)
    backward(loss)
} elif yes {
    let a = mult(x, w)
    if yes {
        let loss = sum(a)
        backward(loss)
    }
} else {
    let dbl = add(w, w)
    let loss = sum(dbl)
    backward(loss)
}
