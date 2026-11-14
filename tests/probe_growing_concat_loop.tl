// tests/probe_growing_concat_loop.tl
// ============================================================================
// TensorLang: a rebind that changes shape is skipped (shape-stability guard)
// ============================================================================
// README: rebinds with mismatched shapes are skipped with a diagnostic at run time; the type checker does not
// check rebinds. `tokens` therefore stays (1, 2) = [[1, 2]] after three iterations. This pins that behaviour
// (observed on the GPU: three skip diagnostics, tokens = [[1, 2]]). If the guard is ever replaced by real
// shape-changing rebinds or by a compile error, this test is the one to update.
//
// @EXPECTED
// {"tokens": [[1.0, 2.0]]}

let tokens: Tensor[f32, (1, 2)] = [[1.0, 2.0]]

for i in range(3) {
    let new: Tensor[f32, (1, 1)] = [[3.0]]
    let grown = concat(tokens, new, axis=1)
    tokens = grown
}

tokens