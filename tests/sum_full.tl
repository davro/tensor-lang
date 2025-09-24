// Test full tensor sum (all elements)
// Expected result: 15.0 (1+2+3+4+5)

let data: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let total = sum(data)
total
