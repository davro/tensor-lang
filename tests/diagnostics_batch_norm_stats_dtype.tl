// ============================================================================
// TensorLang: mean, var, gamma and beta must be f32
// ============================================================================
// The kernel reads them as float*, so an f16 tensor would be misread.
// @EXPECT_FAILURE: [E-BATCH-NORM-DTYPE]
let x: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let m: Tensor[f16, (3,)] = [0.0, 0.0, 0.0]
let v: Tensor[f32, (3,)] = [1.0, 1.0, 1.0]
let y = batch_norm(x, m, v)
