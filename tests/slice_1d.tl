// Test 1D tensor slicing
// Extract middle elements from vector [1,2,3,4,5]
// Slice [1:4] should return [2,3,4]

let data: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let result = data[1:4]
result
