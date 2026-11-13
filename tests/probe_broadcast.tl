// Probe: broadcasting add at runtime.
let base = fill(1.0, (2, 4))
let delta: Tensor[f32, (1, 4)] = [[0.0, 1.0, 2.0, 3.0]]
let out = add(base, delta)
out

// @EXPECTED
// {"out": [[1.0, 2.0, 3.0, 4.0],
//          [1.0, 2.0, 3.0, 4.0]]}

