// Normalize a tensor (mean=0, std=1)
fn normalize(x: Tensor[M, N]) -> Tensor[M, N] {
    let x_mean = mean(x, axis=1)
    let centered = minus(x, x_mean)
    return centered
}

let data: Tensor[f32, (3, 4)] = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
]
let normalized = normalize(data)
normalized
