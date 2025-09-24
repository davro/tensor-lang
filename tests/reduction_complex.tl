// Test combining reductions with other operations
// Create a matrix, apply ReLU, then get row sums

let data: Tensor[f32, (3, 4)] = [[-1.0, 2.0, -3.0, 4.0],
                                 [5.0, -6.0, 7.0, -8.0],
                                 [-9.0, 10.0, -11.0, 12.0]]

let positive = relu(data)               // Apply ReLU (zeros out negatives)
let row_sums = sum(positive, axis=1)    // Sum each row
row_sums
