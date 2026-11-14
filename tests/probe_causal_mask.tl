// tests/probe_causal_mask.tl
// Probe 2: the causal mask alone. Questions: does minus((4,1), (1,4)) broadcast to (4,4) at run
// time (the type checker accepts it), and does `diff >= 0.0` give 0/1? If Probe 1 fails, run this first.
//
// @EXPECTED
// {"mask": [[-0.0, -1000000000.0, -1000000000.0, -1000000000.0], [-0.0, -0.0, -1000000000.0, -1000000000.0], [-0.0, -0.0, -0.0, -1000000000.0], [-0.0, -0.0, -0.0, -0.0]]}

let r: Tensor[f32, (4, 1)] = [[0.0], [1.0], [2.0], [3.0]]
let c: Tensor[f32, (1, 4)] = [[0.0, 1.0, 2.0, 3.0]]
let diff = minus(r, c)
let allowed = diff >= 0.0
let one: Tensor[f32, (1,)] = [1.0]
let blocked = minus(one, allowed)
let neg: Tensor[f32, (1,)] = [-1000000000.0]
let mask = mult(blocked, neg)

mask