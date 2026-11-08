// ============================================================================
// TensorLang: the new comparison operations broadcast like greater/less/equal
// ============================================================================
// x = [[1,2],[3,4]] against y = [2,4] (broadcast over rows)
//   greater_equal -> [[0,0],[1,1]]   less_equal -> [[1,1],[0,1]]   not_equal -> [[1,1],[1,0]]
// Expected values derived by hand.
//
// @EXPECTED
// {"ge": [[0.0, 0.0], [1.0, 1.0]], "le": [[1.0, 1.0], [0.0, 1.0]], "ne": [[1.0, 1.0], [1.0, 0.0]]}
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let y: Tensor[f32, (2,)] = [2.0, 4.0]
let ge = greater_equal(x, y)
let le = less_equal(x, y)
let ne = not_equal(x, y)
