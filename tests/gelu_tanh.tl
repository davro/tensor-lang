// GPT-2's tanh-approximation GELU built from existing operations (no new kernel).
// Real GPT-2 checkpoints were trained with this form; the erf `gelu` differs from it by a few 1e-4 (4e-4 at these inputs),
// which matters when comparing logits against a reference with real weights.
// Inputs span the saturated tails (+-6), the bend (+-1 .. 3) and zero. Forward values and the gradient
// through the composed operations are checked against float64 NumPy.
//
// @EXPECTED
// {"y": [[-0.0, -0.0036374, -0.0454023, -0.158808, -0.154286, 0.0], [0.345714, 0.841192, 1.3995716, 1.9545977, 2.9963626, 6.0]], "x_grad": [[-0.0, -0.0115842, -0.0860993, -0.0829641, 0.1326301, 0.5], [0.8673699, 1.0829641, 1.1277108, 1.0860993, 1.0115842, 1.0]]}

fn gelu_tanh(x: Tensor[M, N]) -> Tensor[M, N] {
    // GPT-2's gelu_new: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
    let c0: Tensor[f32, (1,)] = [0.7978845608]
    let c1: Tensor[f32, (1,)] = [0.044715]
    let half: Tensor[f32, (1,)] = [0.5]
    let one: Tensor[f32, (1,)] = [1.0]
    let x2 = mult(x, x)
    let x3 = mult(x2, x)
    let cube = mult(x3, c1)
    let inner = add(x, cube)
    let arg = mult(inner, c0)
    let t = tanh(arg)
    let onept = add(t, one)
    let hx = mult(x, half)
    let r = mult(hx, onept)
    return r
}

let x: Tensor[f32, (2, 6)] = [[-6.0, -3.0, -2.0, -1.0, -0.5, 0.0], [0.5, 1.0, 1.5, 2.0, 3.0, 6.0]] with grad
let y = gelu_tanh(x)
let loss = sum(y)
backward(loss)
