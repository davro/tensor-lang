let x : Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let y : Tensor[f32, (3, 2)] = [[1.0, 0.0], [0.0, -1.0], [-1.0, 1.0]]
let z = matmul(x, y)
let w = relu(z)