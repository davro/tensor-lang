//
// @EXPECTED
// {
//   "input": [[0.5], [0.5]],
//   "output": [[1.5], [3.5], [5.5]]
// }
//

// Test basic save functionality
let weights: Tensor[f32, (3, 2)] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
save(weights, "outputs/initial_weights.npy")

let input: Tensor[f32, (2, 1)] = [[0.5], [0.5]]
let output = matmul(weights, input)
save(output, "outputs/forward_pass.npy")

// Output for validation
output