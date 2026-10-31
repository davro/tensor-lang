let x: Tensor[f32, (1,)] = [2.0]
let y: Tensor[f32, (1,)] = [1.0]
let z: Tensor[f32, (1,)] = [0.0]
if true {
    let z = x
} else {
    let z = y
}
if false {
    let z = x
} else {
    let z = y
}
z

// @EXPECTED
// {"z": [1.0]}
