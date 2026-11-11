// ============================================================================
// TensorLang: batch_norm on an NCHW tensor takes its statistics per channel (dim 1)
// ============================================================================
// x[n, c, h, w] = 12n + 4c + 2h + w + 1, shape (2, 3, 2, 2); channel c has mean/var from the lists below.
// normalized = (x - mean[c]) / sqrt(var[c] + 1e-5). A kernel that indexes the channel as `idx % 3` gets it wrong.
// Recomputed independently in NumPy.
//
// @EXPECTED
// {"normalized": [[[[-1.4999925, -0.4999975], [0.4999975, 1.4999925]], [[-0.74999906, -0.24999969], [0.24999969, 0.74999906]], [[-0.49999972, -0.16666657], [0.16666657, 0.49999972]]], [[[10.499948, 11.499943], [12.499938, 13.499933]], [[5.2499934, 5.7499928], [6.2499922, 6.7499916]], [[3.4999981, 3.8333312], [4.1666644, 4.4999975]]]]}
let flat: Tensor[f32, (2, 12)] = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]]
let input = reshape(flat, (2, 3, 2, 2))
let running_mean: Tensor[f32, (3,)] = [2.5, 6.5, 10.5]
let running_var: Tensor[f32, (3,)] = [1.0, 4.0, 9.0]
let normalized = batch_norm(input, running_mean, running_var)
