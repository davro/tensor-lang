// ============================================================================
// TensorLang: embedding inside a for loop
// ============================================================================
// Description: embedding() in a for-loop body, accumulating the gathered rows
//
// Expected:
// - the gather is the same on every iteration: rows 2, 0, 3 of the table
// - three iterations, so acc = 3 * [[5,6],[1,2],[7,8]]
//
// @EXPECTED
// {
//   "acc": [[15.0, 18.0], [3.0, 6.0], [21.0, 24.0]]
// }

let table: Tensor[f32, (4, 2)] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
let idx: Tensor[f32, (3,)] = [2.0, 0.0, 3.0]
let acc: Tensor[f32, (3, 2)] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

for i in range(3) {
    let e = embedding(idx, table)
    let acc_new = add(acc, e)
    acc = acc_new
}
