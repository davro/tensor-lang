// ============================================================================
// TensorLang: a literal against a nested expression needs its own let
// ============================================================================
// In a `let`, a literal is given the shape of a NAMED tensor. There is nothing to
// take a shape from when the other side is a nested call, so that is an error.
// @EXPECT_FAILURE: [E-SCALAR-LITERAL]
// @EXPECT_FAILURE: named tensor
let a: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let m = 1.5 < add(a, a)
