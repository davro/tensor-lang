// tests/transpose_axes_explicit.tl
let a: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let b = transpose(a, axes=(1, 0))

// @EXPECTED
// {"b": [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]}

