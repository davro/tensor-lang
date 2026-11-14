// indices must be f32: f16 cannot hold token ids above 2048 exactly
let table: Tensor[f32, (3, 2)] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
let ids: Tensor[f16, (2,)] = [1.0, 2.0]
let r = embedding(ids, table)

// @EXPECT_FAILURE: [E-EMBEDDING-DTYPE]
