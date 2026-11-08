// ============================================================================
// TensorLang: greater_equal / less_equal / not_equal as general operations
// ============================================================================
// x = [1,2,3], y = [2,2,2]
//   greater_equal -> [0,1,1]   less_equal -> [1,1,0]   not_equal -> [1,0,1]
// Results are f32 0/1 tensors. Expected values derived by hand.
//
// @EXPECTED
// {"ge": [0.0, 1.0, 1.0], "le": [1.0, 1.0, 0.0], "ne": [1.0, 0.0, 1.0]}
let x: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let y: Tensor[f32, (3,)] = [2.0, 2.0, 2.0]
let ge = greater_equal(x, y)
let le = less_equal(x, y)
let ne = not_equal(x, y)
