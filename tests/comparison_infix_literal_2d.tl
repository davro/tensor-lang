// ============================================================================
// TensorLang: a literal compared against 2-D and 3-D tensors
// ============================================================================
// The comparison kernels only run equal shapes (or (rows, cols) vs (cols,)), so a
// literal must take the shape of the tensor it is compared with. A (1,) literal
// would read past its buffer here and give wrong values in column 1.
//   x > 3.5  with x = [[1,2],[3,4]]  -> [[0,0],[0,1]]
//   v > 4.5  with v = 1..8 as (2,2,2) -> [[[0,0],[0,0]],[[1,1],[1,1]]]
// Expected values derived by hand.
//
// @EXPECTED
// {"m2": [[0.0, 0.0], [0.0, 1.0]], "m3": [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]}
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let flat: Tensor[f32, (2, 4)] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
let v = reshape(flat, (2, 2, 2))
let m2 = x > 3.5
let m3 = v > 4.5
