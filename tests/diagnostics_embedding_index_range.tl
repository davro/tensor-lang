// run-time check: id 7 is outside a table of 5 rows. Must be an error, not a silent out-of-bounds read.
let table: Tensor[f32, (5, 2)] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
let ids: Tensor[f32, (2,)] = [1.0, 7.0]
let r = embedding(ids, table)

// @EXPECT_FAILURE: [E-EMBEDDING-INDEX]
