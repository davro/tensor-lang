// tests/probe_gelu.tl
// ============================================================================
// TensorLang: gelu (erf form), a handful of values including 0 and negatives
// ============================================================================
// Expected: 0.5 * x * (1 + erf(x / sqrt(2))), computed with math.erf. gelu(0) = 0.
//
// @EXPECTED
// {"out": [[-0.0455003, -0.1586553, 0.0, 0.8413447], [1.9544997, 2.9959503, -0.1542688, 0.3457312]]}

let x: Tensor[f32, (2, 4)] = [[-2.0, -1.0, 0.0, 1.0], [2.0, 3.0, -0.5, 0.5]]
let out = gelu(x)

out