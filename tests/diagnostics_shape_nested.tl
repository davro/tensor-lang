// ============================================================================
// TensorLang diagnostics: a shape error in a nested call says so
// ============================================================================
// The inner mult is fine; the outer add combines a (3,) result with a (4,).
// @EXPECT_FAILURE: [E-SHAPE-MISMATCH]
// @EXPECT_FAILURE: a nested expression
let a: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let b: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let c: Tensor[f32, (4,)] = [1.0, 2.0, 3.0, 4.0]
let z = add(mult(a, b), c)
