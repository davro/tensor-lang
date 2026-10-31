let x: Tensor[f32, (1,)] = [0.0]
let y: Tensor[f32, (1,)] = [1.0]
for i in range(3) {
    if x < y {
        let x = y
    } else {
        let x = x
    }
}
x

// @EXPECTED
// {"x": [1.0]}
