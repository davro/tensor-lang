// Compare different activation patterns
let logits : Tensor[f32, (3, 4)] = [
  [-2.0, -1.0, 1.0, 2.0],
  [-0.5, 0.0, 0.5, 1.5],
  [-1.5, 0.5, 1.0, 2.5]
]
let relu_out = relu(logits)

// Leaky ReLU approximation: 0.1*x for negative, x for positive
let scale_factor : Tensor[f32, (3, 4)] = fill(0.1, (3, 4))
let scaled = mult(logits, scale_factor)
let negative_mask : Tensor[f32, (3, 4)] = fill(-1.0, (3, 4))
let negative_scaled = mult(scaled, negative_mask)
let leaky_relu = add(relu_out, negative_scaled)
let positive_scaled = relu(scaled)
let final_leaky = add(leaky_relu, positive_scaled)