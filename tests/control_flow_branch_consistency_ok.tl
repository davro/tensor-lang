// @EXPECTED
// {"result": [[2.0, 2.0], [2.0, 2.0]]}
let x: Tensor[f32, (2, 2)] = fill(1.0, (2, 2))

if x > 0.0 {
    let y: Tensor[f32, (2, 2)] = add(x, x)
} else {
    let y: Tensor[f32, (2, 2)] = mult(x, x)
}

let result = y
