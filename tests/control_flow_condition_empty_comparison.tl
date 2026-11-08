// ============================================================================
// TensorLang: an empty tensor on one side of a comparison condition
// ============================================================================
// x[1:1] has shape (0,); comparing it with a literal broadcasts fine, so the
// empty-tensor check (not a shape error) is what must fire.
// @EXPECT_FAILURE: [E-CONDITION-EMPTY]
// @EXPECT_FAILURE: the condition `e > 0.5`
let x: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let e = x[1:1]
let z: Tensor[f32, (1,)] = [0.0]
if e > 0.5 {
    let z = z
}
z
