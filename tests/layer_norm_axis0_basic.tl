// tests/layer_norm_axis0_basic.tl
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let y = layer_norm(x, axis=0, eps=1.0)

// @EXPECTED
// {"y": [[-0.70710678, -0.70710678], [0.70710678, 0.70710678]]}

