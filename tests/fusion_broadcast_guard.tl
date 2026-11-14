let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]]
let b: Tensor[f32, (2,)] = [0.1, 0.2]

let h = relu(add(matmul(x, w), b))

// Broadcast: the (2,) bias is added per column (NumPy). Through v0.5.6 a square matrix read it per ROW, so this
// file used to expect [[1.6, 1.6], [3.7, 3.7]]. matmul(x, w) = [[1.5, 1.5], [3.5, 3.5]]; + [0.1, 0.2] per column.
// @EXPECTED
// {"h": [[1.6, 1.7], [3.6, 3.7]]}