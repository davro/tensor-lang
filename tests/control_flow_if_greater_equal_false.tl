// @EXPECTED
// {"z": [10.0, 20.0]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let y: Tensor[f32, (2,)] = [10.0, 20.0]
let z: Tensor[f32, (2,)] = [0.0, 0.0]
if x >= y {
    let z = x
} else {
    let z = y
}
z
