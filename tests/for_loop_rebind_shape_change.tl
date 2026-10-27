let w: Tensor[f32, (1, 1)] = fill(0.0, (1, 1))

for i in range(2) {
    let w_new: Tensor[f32, (2, 2)] = fill(1.0, (2, 2))
    w = w_new   // shape mismatch: (1,1) vs (2,2)
}

// @EXPECTED
// { "w": [[0.0]] }

