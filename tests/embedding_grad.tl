// Gradient of embedding is a scatter-add into the weight: the repeated id 2 accumulates both rows of wts,
// the unused rows 1 and 3 get exactly zero. The indices receive no gradient.
let table: Tensor[f32, (5, 3)] = [[0.25, 0.5, 0.75], [1.25, 1.5, 1.75], [2.25, 2.5, 2.75], [3.25, 3.5, 3.75], [4.25, 4.5, 4.75]] with grad
let ids: Tensor[f32, (4,)] = [2.0, 0.0, 2.0, 4.0]
let wts: Tensor[f32, (4, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
let rows = embedding(ids, table)
let weighted = mult(rows, wts)
let loss = sum(weighted)
backward(loss)

// @EXPECTED
// {"table_grad": [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [8.0, 10.0, 12.0], [0.0, 0.0, 0.0], [10.0, 11.0, 12.0]]}
