// ============================================================================
// TensorLang: embedding inside a user-defined function
// ============================================================================
// Description: embedding() in a function body; the call is inlined at the call site
//
// Expected:
// - rows 2, 0, 2 of the table, then scaled by 2
//
// @EXPECTED
// {
//   "out": [[10.0, 12.0], [2.0, 4.0], [10.0, 12.0]]
// }

fn lookup(ids: Tensor[f32, (3,)], w: Tensor[f32, (4, 2)]) -> Tensor[f32, (3, 2)] {
    let rows = embedding(ids, w)
    return add(rows, rows)
}

let table: Tensor[f32, (4, 2)] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
let idx: Tensor[f32, (3,)] = [2.0, 0.0, 2.0]
let out = lookup(idx, table)
