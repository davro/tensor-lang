// ============================================================================
// TensorLang: gamma without beta is a syntax error: they come as a pair
// ============================================================================
// @EXPECT_FAILURE: [E-SYNTAX]
let x: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let m: Tensor[f32, (3,)] = [0.0, 0.0, 0.0]
let v: Tensor[f32, (3,)] = [1.0, 1.0, 1.0]
let g: Tensor[f32, (3,)] = [1.0, 1.0, 1.0]
let y = batch_norm(x, m, v, g)
