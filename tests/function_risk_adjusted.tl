// Portfolio returns function
fn portfolio_returns(returns: Tensor[M, N], weights: Tensor[N, P]) -> Tensor[M, P] {
    return matmul(returns, weights)
}

// Risk-adjusted returns function
fn risk_adjusted(returns: Tensor[M, N], weights: Tensor[N, P], risk: Tensor[M, 1]) -> Tensor[M, P] {
    let port = portfolio_returns(returns, weights)
    return div(port, risk)
}

let returns: Tensor[f32, (5, 4)] = [
    [0.12, 0.08, 0.15, 0.06],
    [0.08, 0.10, 0.12, 0.05],
    [0.06, 0.07, 0.09, 0.04],
    [0.15, 0.12, 0.18, 0.08],
    [0.05, 0.06, 0.04, 0.03]
]
let weights: Tensor[f32, (4, 1)] = [[0.3], [0.25], [0.25], [0.2]]
let risk: Tensor[f32, (5, 1)] = [[1.2], [1.1], [0.8], [1.4], [0.6]]
let adjusted = risk_adjusted(returns, weights, risk)
adjusted