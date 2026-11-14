// tests/broadcast_vector_square.tl
// ============================================================================
// TensorLang: a 2-D tensor plus a 1-D vector aligns the vector with the LAST axis (one value per column),
// also when the matrix is square
// ============================================================================
// NumPy-style broadcasting: (rows, cols) op (cols,) uses w[j] for column j. With rows == cols the unfused
// kernel (binary_broadcast) and the fused kernel both used w[i] (per row), so square weights plus a bias or a
// layer-norm gamma/beta gave wrong results with no error. Covers add, minus, mult, div, a fused mult -> add
// chain, and a non-square control that was always right. Distinct values: a transposed read changes every row.
//
// @EXPECTED
// {"sq_add": [[1.25, 2.5, 1.25, 5.0], [2.25, 3.5, 2.25, 6.0], [3.25, 4.5, 3.25, 7.0], [4.25, 5.5, 4.25, 8.0]], "sq_minus": [[-0.75, -1.5, 0.25, -3.0], [0.25, -0.5, 1.25, -2.0], [1.25, 0.5, 2.25, -1.0], [2.25, 1.5, 3.25, 0.0]], "sq_mult": [[0.25, 1.0, 0.375, 4.0], [1.25, 3.0, 0.875, 8.0], [2.25, 5.0, 1.375, 12.0], [3.25, 7.0, 1.875, 16.0]], "sq_div": [[0.25, 0.25, 1.5, 0.25], [1.25, 0.75, 3.5, 0.5], [2.25, 1.25, 5.5, 0.75], [3.25, 1.75, 7.5, 1.0]], "chain_mult": [[0.25, 1.0, 0.375, 4.0], [1.25, 3.0, 0.875, 8.0], [2.25, 5.0, 1.375, 12.0], [3.25, 7.0, 1.875, 16.0]], "chain_add": [[1.25, 3.0, 0.875, 8.0], [2.25, 5.0, 1.375, 12.0], [3.25, 7.0, 1.875, 16.0], [4.25, 9.0, 2.375, 20.0]], "ctrl_add": [[1.5, 3.0, 2.0, 6.0], [3.5, 5.0, 4.0, 8.0], [5.5, 7.0, 6.0, 10.0]]}

let m_sq: Tensor[f32, (4, 4)] = [[0.25, 0.5, 0.75, 1],
      [1.25, 1.5, 1.75, 2],
      [2.25, 2.5, 2.75, 3],
      [3.25, 3.5, 3.75, 4]]
let m_rect: Tensor[f32, (3, 4)] = [[0.5, 1, 1.5, 2],
      [2.5, 3, 3.5, 4],
      [4.5, 5, 5.5, 6]]
let w: Tensor[f32, (4,)] = [1, 2, 0.5, 4]

let sq_add = add(m_sq, w)
let sq_minus = minus(m_sq, w)
let sq_mult = mult(m_sq, w)
let sq_div = div(m_sq, w)

// a fused chain: the second op's first argument is the first op's result
let chain_mult = mult(m_sq, w)
let chain_add = add(chain_mult, w)

let ctrl_add = add(m_rect, w)

ctrl_add
