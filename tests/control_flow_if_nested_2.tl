// @EXPECTED
// {"z": [3.0]}
let x: Tensor[f32, (1,)] = [3.0]
let y: Tensor[f32, (1,)] = [2.0]
let z: Tensor[f32, (1,)] = [0.0]
if x > y {
    if y > x {
        let z = y
    } else {
        let z = x
    }
} else {
    let z = y
}
z
