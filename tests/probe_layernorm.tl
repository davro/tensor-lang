// ============================================================================
// Probe: layer_norm with genuine column variation.
// ============================================================================
// Each row varies across columns → non-zero variance → should normalize.
//
// @EXPECTED
// {
//   "out": [
//     [-1.34164, -0.447214, 0.447214, 1.34164],
//     [-1.34164, -0.447214, 0.447214, 1.34164]
//   ]
// }

let x: Tensor[f32, (2, 4)] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]

let gamma: Tensor[f32, (4,)] = fill(1.0, (4,))
let beta: Tensor[f32, (4,)] = fill(0.0, (4,))

let xn = layer_norm(x, axis=1)
let out = add(mult(xn, gamma), beta)

out


