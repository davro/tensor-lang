// Probe 2: isolate causal-mask construction.

let r: Tensor[f32, (4, 1)] = [[0.0], [1.0], [2.0], [3.0]]
let c: Tensor[f32, (1, 4)] = [[0.0, 1.0, 2.0, 3.0]]
let diff = minus(r, c)
let allowed = diff >= 0.0
let one: Tensor[f32, (1,)] = [1.0]
let blocked = minus(one, allowed)
let neg: Tensor[f32, (1,)] = [-1000000000.0]
let mask = mult(blocked, neg)

mask

// @EXPECTED
// {"mask": [[...]]}

