// embedding(indices, weight): gather rows. Distinct row contents, a repeated id, and a 2-D index tensor.
let table: Tensor[f32, (5, 3)] = [[0.25, 0.5, 0.75], [1.25, 1.5, 1.75], [2.25, 2.5, 2.75], [3.25, 3.5, 3.75], [4.25, 4.5, 4.75]]
let ids: Tensor[f32, (4,)] = [2.0, 0.0, 2.0, 4.0]
let rows = embedding(ids, table)
let ids2: Tensor[f32, (2, 3)] = [[1.0, 1.0, 3.0], [0.0, 4.0, 2.0]]
let rows2 = embedding(ids2, table)

// @EXPECTED
// {"rows": [[2.25, 2.5, 2.75], [0.25, 0.5, 0.75], [2.25, 2.5, 2.75], [4.25, 4.5, 4.75]], "rows2": [[[1.25, 1.5, 1.75], [1.25, 1.5, 1.75], [3.25, 3.5, 3.75]], [[0.25, 0.5, 0.75], [4.25, 4.5, 4.75], [2.25, 2.5, 2.75]]]}
