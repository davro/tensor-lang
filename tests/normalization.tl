// ============================================================================
// TensorLang: Normalization
// ============================================================================
// Description: Tests normalization techniques
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
//       -0.3875969,
//       -0.13066202,
//       1.0869565
//     ],
//     [
//       1.1627908,
//       -1.3501743,
//       -0.25083613
//     ],
//     [
//       -1.1627908,
//       1.0888503,
//       -1.2541806
//     ],
//     [
//       0.3875969,
//       0.39198607,
//       0.4180602
//     ]
//   ]
// }

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
