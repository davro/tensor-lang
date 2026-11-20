// ============================================================================
// TensorLang: embedding gradient inside a training loop, through a function
// ============================================================================
// Description: embedding() called from a user-defined function inside a for loop,
//              with backward() and a manual SGD step on the table
//
// Expected:
// - index 2 appears twice, so its gradient accumulates (scatter-add):
//   iteration 1: grad row 2 = 4 * [5, 6], grad row 0 = 2 * [1, 2], rows 1 and 3 zero
// - the table is rebound after every step, so iteration 2 gathers the updated rows
// - loss and table_grad are those of the last iteration, table is the final table
//
// @EXPECTED
// {
//   "loss": 47.12,
//   "table_grad": [[1.6, 3.2], [0.0, 0.0], [12.0, 14.4], [0.0, 0.0]],
//   "table": [[0.64, 1.28], [3.0, 4.0], [1.8, 2.16], [7.0, 8.0]]
// }

fn lookup(ids: Tensor[f32, (3,)], w: Tensor[f32, (4, 2)]) -> Tensor[f32, (3, 2)] {
    return embedding(ids, w)
}

let table: Tensor[f32, (4, 2)] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]] with grad
let idx: Tensor[f32, (3,)] = [2.0, 0.0, 2.0]
let lr: Tensor[f32, (1,)] = [0.1]

for step in range(2) {
    let e = lookup(idx, table)
    let sq = mult(e, e)
    let loss = sum(sq)
    backward(loss)
    let table_updated = minus(table, mult(lr, table_grad))
    table = table_updated
}
