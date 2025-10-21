// ============================================================================
// TensorLang: Portfolio
// ============================================================================
// Description: Tests portfolio computations
//
// Architecture Pattern:
//   'risk_adjusted' output
//
// Expected Output Variables:
//   risk_adjusted
//
// @EXPECTED
// {
//   "risk_adjusted": [
//     [
//       0.08791666
//     ],
//     [
//       0.08090909
//     ],
//     [
//       0.0825
//     ],
//     [
//       0.09714287
//     ],
//     [
//       0.07666666
//     ]
//   ]
// }

let returns : Tensor[f32, (5, 4)] = [
  [0.12, 0.08, 0.15, 0.06],
  [0.08, 0.10, 0.12, 0.05],
  [0.06, 0.07, 0.09, 0.04],
  [0.15, 0.12, 0.18, 0.08],
  [0.05, 0.06, 0.04, 0.03]
]
let weights : Tensor[f32, (4, 1)] = [[0.3], [0.25], [0.25], [0.2]]
let portfolio_returns = matmul(returns, weights)

let risk_factor : Tensor[f32, (5, 1)] = [[1.2], [1.1], [0.8], [1.4], [0.6]]
let risk_adjusted = div(portfolio_returns, risk_factor)
