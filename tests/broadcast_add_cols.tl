// tests/broadcast_add_cols.tl
// Tests broadcasting a 1D tensor across rows (existing behavior)
let data: Tensor[f32, (3, 4)] = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
]
let col_vals: Tensor[f32, (4,)] = [1.0, 2.0, 3.0, 4.0]
let result = add(data, col_vals)
result
