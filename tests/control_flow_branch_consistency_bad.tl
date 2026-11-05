// @EXPECT_FAILURE
let x: Tensor[f32, (2, 2)] = fill(1.0, (2, 2))

if x > 0.0 {
    let y: Tensor[f32, (2, 2)] = add(x, x)
} else {
    let y: Tensor[f32, (1,)] = fill(1.0, (1,))
}

let result = y
