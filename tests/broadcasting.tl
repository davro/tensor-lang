// ============================================================================
// TensorLang: Broadcasting
// ============================================================================
// Description: Tests broadcasting behavior
//
// Architecture Pattern:
//   'scaled' output
//
// Expected Output Variables:
//   scaled
//
// @EXPECTED
// {
//   "scaled": [
//     [
//       0.24,
//       0.32,
//       0.9,
//       0.72
//     ],
//     [
//       0.12,
//       0.24,
//       0.75,
//       0.63
//     ],
//     [
//       1.08,
//       0.56,
//       0.75,
//       0.27
//     ]
//   ]
// }

let image : Tensor[f32, (3, 4)] = [
  [0.2, 0.4, 0.6, 0.8],
  [0.1, 0.3, 0.5, 0.7],
  [0.9, 0.7, 0.5, 0.3]
]
let channel_scale : Tensor[f32, (4)] = [1.2, 0.8, 1.5, 0.9]
let scaled = mult(image, channel_scale)
