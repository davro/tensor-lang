// Probe: gelu activation
let x: Tensor[f32, (2, 4)] = [[-2.0, -1.0, 0.0, 1.0], [2.0, 3.0, -0.5, 0.5]]
let out = gelu(x)
out

// @EXPECTED
// {"out": [[...]]}

