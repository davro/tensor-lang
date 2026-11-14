// tests/probe_attention_block.tl
// Probe 1: single-head causal attention, forward + backward, seq=4, d=4.
// Inputs are distinct literals so a wrong mask, scale or matmul orientation changes the answer
// (constant fills would give 0.3 everywhere whatever the mask). Row 0 attends only to
// itself, so out[0] must equal v[0]. Expected values: NumPy float64, gradients checked
// against central finite differences.
//
// @EXPECTED
// {"out": [[1.0, 2.0, 3.0, 4.0], [0.158648504, 1.36898638, 2.57932425, 1.05526977], [0.664016102, 0.675443252, 2.25648332, 0.88820043], [-0.351747419, 0.392146359, 1.95595395, -1.77464344]], "q_grad": [[0.0, 0.0, 0.0, 0.0], [-0.560527625, -1.40131906, 1.12105525, -0.560527625], [-0.638723981, -0.722472703, 0.528016034, -0.638723981], [-0.227161569, -0.685286822, 0.594473369, -0.219328902]], "k_grad": [[0.276731929, 1.23223993, 1.5491196, 0.480724664], [-0.670453629, -1.04132941, -1.07560913, -0.994129523], [0.376750922, -0.209186742, -0.453928806, 0.492517748], [0.0169707778, 0.0182762222, -0.0195816667, 0.0208871111]], "v_grad": [[2.16028394, 2.16028394, 2.16028394, 2.16028394], [1.50055768, 1.50055768, 1.50055768, 1.50055768], [0.32447637, 0.32447637, 0.32447637, 0.32447637], [0.0146820152, 0.0146820152, 0.0146820152, 0.0146820152]]}

let q: Tensor[f32, (4, 4)] = [[0.10, 0.20, 0.30, 0.40], [0.50, -0.60, 0.70, 0.80], [-0.90, 1.00, 1.10, -1.20], [1.30, 1.40, -1.50, 1.60]] with grad
let k: Tensor[f32, (4, 4)] = [[0.40, -0.30, 0.20, 0.10], [0.80, 0.70, -0.60, 0.50], [1.20, -1.10, 1.00, 0.90], [-1.60, 1.50, 1.40, -1.30]] with grad
let v: Tensor[f32, (4, 4)] = [[1.00, 2.00, 3.00, 4.00], [-1.00, 0.50, 2.00, -3.00], [2.50, -2.00, 1.00, 0.00], [0.00, 1.00, -1.00, 2.00]] with grad
let scale: Tensor[f32, (1,)] = [0.5]

let kt = transpose(k, axes=(1, 0))
let scores = matmul(q, kt)
let scaled = mult(scores, scale)

// causal mask: 0 where row >= col, -1e9 where blocked
let r: Tensor[f32, (4, 1)] = [[0.0], [1.0], [2.0], [3.0]]
let c: Tensor[f32, (1, 4)] = [[0.0, 1.0, 2.0, 3.0]]
let diff = minus(r, c)
let allowed = diff >= 0.0
let one: Tensor[f32, (1,)] = [1.0]
let blocked = minus(one, allowed)
let neg: Tensor[f32, (1,)] = [-1000000000.0]
let mask = mult(blocked, neg)

let masked = add(scaled, mask)
let attn = softmax(masked, axis=1)
let out = matmul(attn, v)

let loss = sum(out)
backward(loss)

out
