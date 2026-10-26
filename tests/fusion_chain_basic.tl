let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]]
let b: Tensor[f32, (2, 2)] = [[0.1, 0.2], [0.1, 0.2]]

let h = relu(add(matmul(x, w), b))

// @EXPECTED
// {"h": [[1.6, 1.7], [3.6, 3.7]]}