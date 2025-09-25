// Test comparison for masking - common ML pattern
// Create data, find values > 0, then multiply original data by mask
// This simulates conditional processing based on sign

let data: Tensor[f32, (4,)] = [-1.0, 2.0, -3.0, 4.0]
let zeros: Tensor[f32, (4,)] = [0.0, 0.0, 0.0, 0.0]
let mask = greater(data, zeros)    // Create boolean mask (values > 0)
let filtered = mult(data, mask)    // Apply mask (zero out negative values)
filtered
