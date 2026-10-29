// ============================================================================
// TensorLang: N-D Reductions
// ============================================================================
//
// @EXPECTED
// {"b": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]}

let x: Tensor[f32, (2, 3, 4)] = fill(1.0, (2, 3, 4))
let s0 = sum(x, axis=0)
let s1 = sum(x, axis=1)
let s2 = sum(x, axis=2)
let m = mean(x, axis=-1)
let mx = max(x, axis=1)
let mn = min(x, axis=2)
let a = argmax(x, axis=2)
let b = argmin(x, axis=1)
