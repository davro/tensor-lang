// tests/transpose_default_basic.tl
let a: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let b = transpose(a)

// @EXPECTED
// {"b": [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]}

