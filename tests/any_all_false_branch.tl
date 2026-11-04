// @EXPECTED
// {"z": [0.0]}
let data: Tensor[f32, (5,)] = [0.0, 0.0, 0.0, 0.0, 0.0]
let x: Tensor[f32, (1,)] = [0.0]
let y: Tensor[f32, (1,)] = [1.0]
let z: Tensor[f32, (1,)] = [0.0]
if any(data) {
    let z = y
} else {
    let z = x
}
z
