// ============================================================================
// TensorLang: comparison results have zero gradient, other paths still flow
// ============================================================================
// w = [1,2,3], y = [2,2,2], m = w >= y = [0,1,1], p = m * w = [0,2,3], loss = 5.
// d loss / d w = m (through the mult) + w * 0 (through the comparison) = [0,1,1].
// Expected value derived by hand.
//
// @EXPECTED
// {"w_grad": [0.0, 1.0, 1.0]}
let w: Tensor[f32, (3,)] = [1.0, 2.0, 3.0] with grad
let y: Tensor[f32, (3,)] = [2.0, 2.0, 2.0]
let m = w >= y
let p = mult(m, w)
let loss = sum(p)
backward(loss)
