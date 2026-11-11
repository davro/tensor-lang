// ============================================================================
// TensorLang: batch_norm needs at least a batch and a channel dimension
// ============================================================================
// @EXPECT_FAILURE: [E-SHAPE-MISMATCH]
// @EXPECT_FAILURE: at least 2
let x: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let m: Tensor[f32, (3,)] = [0.0, 0.0, 0.0]
let v: Tensor[f32, (3,)] = [1.0, 1.0, 1.0]
let y = batch_norm(x, m, v)
