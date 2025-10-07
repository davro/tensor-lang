let a : Tensor[f32, (2, 2)] = fill(2.0, (2, 2))
let b : Tensor[f32, (2, 2)] = fill(3.0, (2, 2))
let c = minus(a, b)
