// tests/function_broadcast_add.tl
// Tests broadcasting inside a function with row-wise addition

fn add_bias_rows(data: Tensor[M, N], bias: Tensor[M, 1]) -> Tensor[M, N] {
    return add(data, bias)
}

let features: Tensor[f32, (3, 4)] = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
]

let row_bias: Tensor[f32, (3,)] = [100.0, 200.0, 300.0]

let adjusted = add_bias_rows(features, row_bias)
adjusted
