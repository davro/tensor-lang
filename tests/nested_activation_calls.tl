// ============================================================================
// TensorLang: Nested activation calls
// ============================================================================
// Description:
//   Regression test for a compiler gotcha: activation functions
//   (tanh/sigmoid/relu) used to require a NAME as their argument, so
//   `tanh(matmul(x, w1))` failed to parse — you had to write
//   `let pre = matmul(x, w1); let h = tanh(pre)` instead.
//
//   The grammar and AST builder now accept a full inline_expr, and
//   flatten_expr_args() hoists the nested call into a synthetic
//   __tmp_N binding, e.g.:
//     let __tmp_0 = matmul(x, w1)
//     let h_tanh  = tanh(__tmp_0)
//
//   This test covers all three activations that have real kernel
//   codegen (relu/sigmoid/tanh) nested inside both a unary (matmul)
//   and binary+unary (add + relu) combination.
//
// @EXPECTED
// {
//   "h_tanh": [
//     [0.9640275835990906, 0.9640275835990906],
//     [0.9640275835990906, 0.9640275835990906]
//   ],
//   "h_sigmoid": [
//     [0.8807970285415649, 0.8807970285415649],
//     [0.8807970285415649, 0.8807970285415649]
//   ],
//   "h_relu_bias": [
//     [0.0, 0.0],
//     [0.0, 0.0]
//   ]
// }

let x:  Tensor[f32, (2, 2)] = [[2.0, 2.0], [2.0, 2.0]]
let w1: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]]
let b:  Tensor[f32, (2,)]   = [-3.0, -3.0]

// Unary nested inside unary: tanh(matmul(x, w1))
let h_tanh = tanh(matmul(x, w1))

// Unary nested inside unary: sigmoid(matmul(x, w1))
let h_sigmoid = sigmoid(matmul(x, w1))

// Binary nested inside unary: relu(add(matmul(x, w1), b))
let h_relu_bias = relu(add(matmul(x, w1), b))
