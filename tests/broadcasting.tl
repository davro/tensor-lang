// Scale a 2D image by different factors per channel
let image : Tensor[f32, (3, 4)] = [
  [0.2, 0.4, 0.6, 0.8],
  [0.1, 0.3, 0.5, 0.7],
  [0.9, 0.7, 0.5, 0.3]
]
let channel_scale : Tensor[f32, (4)] = [1.2, 0.8, 1.5, 0.9]
let scaled = mult(image, channel_scale)
