let a: Tensor[f32, (3, 1)] = [[1.0], [2.0], [3.0]]
let b: Tensor[f32, (1, 4)] = [[10.0, 20.0, 30.0, 40.0]]
let c = minus(a, b)