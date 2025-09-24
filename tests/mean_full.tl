
// Test full tensor mean (all elements)
// Expected result: 3.0 ((1+2+3+4+5)/5 = 15/5 = 3.0)

let data: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let average = mean(data)
average
