// @EXPECTED
// {"z": [3.0, 4.0], "x_grad": [1.0, 1.0]}
let x: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let w: Tensor[f32, (2,)] = [1.0, 2.0] with grad
let z: Tensor[f32, (2,)] = [0.0, 0.0]
if x > w {
    let z = x
    let loss = sum(z)
    backward(loss)
} else {
    let z = w
}
z
