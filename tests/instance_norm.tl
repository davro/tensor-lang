// ============================================================================
// TensorLang: Instance Norm
// ============================================================================
// Description: Tests instance normalization
//
// Architecture Pattern:
//   'normalized' output
//
// Expected Output Variables:
//   normalized
//
// @EXPECTED
// {
//   "normalized": [
//     [
//       -1.3416355,
//       -0.44721183,
//       0.44721183,
//       1.3416355
//     ],
//     [
//       -1.3416355,
//       -0.44721183,
//       0.44721183,
//       1.3416355
//     ],
//     [
//       -1.3416406,
//       -0.44721353,
//       0.44721353,
//       1.3416406
//     ]
//   ]
// }

let input: Tensor[f32, (3, 4)] = [[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [10.0, 15.0, 20.0, 25.0]]

let normalized = instance_norm(input)
normalized