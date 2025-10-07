fn add_tensors(a: Tensor[M, N], b: Tensor[M, N]) -> Tensor[M, N] {
    return add(a, b)
}

let x: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let y: Tensor[f32, (2, 3)] = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
let result = add_tensors(x, y)
result