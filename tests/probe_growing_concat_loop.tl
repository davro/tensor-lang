// Probe 4: confirm the documented shape-stability guard behavior.
// Expected: the rebind is skipped with a diagnostic (shape changes from
// (1,2) to (1,3)), so `tokens` retains its initial (1,2) shape.
// This documents the boundary: in-language autoregressive generation via
// growing concat is blocked by design.

let tokens: Tensor[f32, (1, 2)] = [[1.0, 2.0]]

for i in range(3) {
    let new: Tensor[f32, (1, 1)] = [[3.0]]
    let grown = concat(tokens, new, axis=1)
    tokens = grown
}

tokens

// @EXPECTED
// (run once to capture: either the initial (1,2) value with a skip
// diagnostic, or an error — both are informative findings)

