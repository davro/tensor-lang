let x: Tensor[f16, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let y: Tensor[f16, (2, 2)] = [[1.0, 1.0], [1.0, 1.0]]
let z = add(x, y)
save(z, "z.npy")

// @EXPECTED
// {"z": [[2.0, 3.0], [4.0, 5.0]]}

