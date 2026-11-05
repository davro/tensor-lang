// @EXPECTED
// {"y": [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]}
let x: Tensor[f32, (2, 2, 2)] = fill(1.0, (2, 2, 2)) with grad

let y = instance_norm(x)
let loss = sum(y)
backward(loss)
y
