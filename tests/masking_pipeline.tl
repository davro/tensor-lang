// ============================================================================
// TensorLang: Masking Pipeline
// ============================================================================
// Description: Tests operation pipeline
//
// Architecture Pattern:
//   'filtered' output
//
// Expected Output Variables:
//   filtered
//
// @EXPECTED
// {
//   "filtered": [
//     0.0,
//     2.0,
//     0.0,
//     4.0
//   ]
// }

let data: Tensor[f32, (4,)] = [-1.0, 2.0, -3.0, 4.0]
let zeros: Tensor[f32, (4,)] = [0.0, 0.0, 0.0, 0.0]
let mask = greater(data, zeros)    // Create boolean mask (values > 0)
let filtered = mult(data, mask)    // Apply mask (zero out negative values)
filtered
