// tests/argmin_full_basic.tl
let x: Tensor[f32, (6,)] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
let y = argmin(x)

// @EXPECTED
// {"y": 1.0}

