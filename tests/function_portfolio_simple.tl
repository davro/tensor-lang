// ============================================================================
// TensorLang: Function Portfolio Simple
// ============================================================================
// Description: Tests custom function definition
//
// Architecture Pattern:
//   'port_returns' output
//
// Expected Output Variables:
//   port_returns
//
// @EXPECTED
// {
//   "port_returns": [
//     [
//       0.10550001
//     ],
//     [
//       0.089
//     ],
//     [
//       0.066
//     ]
//   ]
// }

// Calculate portfolio returns
fn portfolio_returns(returns: Tensor[M, N], weights: Tensor[N, P]) -> Tensor[M, P] {
    return matmul(returns, weights)
}

let returns: Tensor[f32, (3, 4)] = [
    [0.12, 0.08, 0.15, 0.06],
    [0.08, 0.10, 0.12, 0.05],
    [0.06, 0.07, 0.09, 0.04]
]
let weights: Tensor[f32, (4, 1)] = [[0.3], [0.25], [0.25], [0.2]]
let port_returns = portfolio_returns(returns, weights)
port_returns
