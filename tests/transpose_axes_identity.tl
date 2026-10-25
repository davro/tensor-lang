// tests/transpose_axes_identity.tl
let a: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let b = transpose(a, axes=(0, 1))

// @EXPECTED
// {"b": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}

