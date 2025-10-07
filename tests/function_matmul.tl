// Matrix multiplication function
fn matmul_wrapper(a: Tensor[M, N], b: Tensor[N, P]) -> Tensor[M, P] {
    return matmul(a, b)
}

let a: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let b: Tensor[f32, (3, 2)] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
let c = matmul_wrapper(a, b)
c
