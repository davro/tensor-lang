// ============================================================================
// TensorLang: a (rows, cols) tensor against a (1, cols) row, all six comparisons
// ============================================================================
// x = [[1,2,3],[4,5,6]] against r = [[2,4,3]] (a (1, 3) row, broadcast over rows).
// The kernels accept (cols,) and (1, cols) rows alike; the type checker rejects other mixes.
//   greater       -> [[0,0,0],[1,1,1]]   less       -> [[1,1,0],[0,0,0]]   equal   -> [[0,0,1],[0,0,0]]
//   greater_equal -> [[0,0,1],[1,1,1]]   less_equal -> [[1,1,1],[0,0,0]]   not_equal -> [[1,1,0],[1,1,1]]
// Expected values derived by hand and recomputed in NumPy.
//
// @EXPECTED
// {"gt": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], "lt": [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], "eq": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], "ge": [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], "le": [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], "ne": [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]}
let x: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let r: Tensor[f32, (1, 3)] = [[2.0, 4.0, 3.0]]
let gt = greater(x, r)
let lt = less(x, r)
let eq = equal(x, r)
let ge = greater_equal(x, r)
let le = less_equal(x, r)
let ne = not_equal(x, r)
