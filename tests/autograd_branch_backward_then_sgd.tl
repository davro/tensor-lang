// ============================================================================
// TensorLang: backward() followed by an sgd step inside a branch
// ============================================================================
// loss = sum(x*w) -> w_grad = x = [1, 2]; sgd with lr 0.01 gives
// w = [3,4] - 0.01*[1,2] = [2.99, 3.98]. If the backward pass were deferred until after
// the sgd statement, the step would use a zero gradient and w would stay [3, 4].
// Expected value derived by hand.
//
// @EXPECTED
// {"result": [2.99, 3.98]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let w: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let yes: Tensor[f32, (1,)] = [1.0]
let no: Tensor[f32, (1,)] = [0.0]
if yes {
    let p = mult(x, w)
    let loss = sum(p)
    backward(loss)
    sgd(w, w_grad, 0.01)
}
let result = w
