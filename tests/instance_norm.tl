// Test instance normalization
// Each sample gets normalized independently (like layer norm)
// Useful for style transfer and image generation tasks

let input: Tensor[f32, (3, 4)] = [[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [10.0, 15.0, 20.0, 25.0]]

let normalized = instance_norm(input)
normalized
