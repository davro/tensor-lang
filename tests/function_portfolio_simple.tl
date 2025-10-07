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
