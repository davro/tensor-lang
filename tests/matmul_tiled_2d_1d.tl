let x: Tensor[f32, (3, 4)] = fill(1.0, (3, 4))
let w: Tensor[f32, (4, 1)] = fill(1.0, (4, 1))
let z = matmul(x, w)

// @EXPECTED
// {"z": [[4.0], [4.0], [4.0]]}

