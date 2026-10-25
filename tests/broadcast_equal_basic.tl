// tests/broadcast_equal_basic.tl
let a: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let b: Tensor[f32, (3,)] = [2.0, 2.0, 2.0]
let c = equal(a, b)

// @EXPECTED
// {"c": [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]}

