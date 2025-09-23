// Z-score normalization simulation
let data : Tensor[f32, (4, 3)] = [
  [100.0, 85.0, 92.0],
  [110.0, 78.0, 88.0],
  [95.0, 92.0, 85.0],
  [105.0, 88.0, 90.0]
]
let mean : Tensor[f32, (3)] = [102.5, 85.75, 88.75]
let std : Tensor[f32, (3)] = [6.45, 5.74, 2.99]
let centered = minus(data, mean)
let normalized = div(centered, std)
