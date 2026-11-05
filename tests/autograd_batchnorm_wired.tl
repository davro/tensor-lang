// @EXPECTED
// {"y": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]}
let x: Tensor[f32, (2, 4)] = fill(1.0, (2, 4)) with grad
let running_mean: Tensor[f32, (4,)] = fill(0.0, (4,))
let running_var: Tensor[f32, (4,)] = fill(1.0, (4,))

let y = batch_norm(x, running_mean, running_var)
let loss = sum(y)
backward(loss)
y
