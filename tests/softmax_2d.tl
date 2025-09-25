// Test 2D softmax activation function (row-wise softmax)
// Each row gets normalized independently to sum to 1.0
// Row 1: [1, 2] -> exp: [2.718, 7.389] -> softmax: [0.269, 0.731]
// Row 2: [3, 4] -> exp: [20.086, 54.598] -> softmax: [0.269, 0.731]

let data: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let result = softmax(data, axis=1)
result
