// @EXPECTED
// {"z": [2.0]}
let x: Tensor[f32, (1,)] = [1.0]
let y: Tensor[f32, (1,)] = [2.0]
let z: Tensor[f32, (1,)] = [0.0]
if x > y {
    let z = x
} elif x < y {
    let z = y
} else {
    let z = x
}
z
