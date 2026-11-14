// weight must be 2-D (vocab, dim)
let table: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let ids: Tensor[f32, (2,)] = [1.0, 2.0]
let r = embedding(ids, table)

// @EXPECT_FAILURE: [E-EMBEDDING-SHAPE]
