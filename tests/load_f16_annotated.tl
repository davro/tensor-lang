// ============================================================================
// TensorLang: load() of a float16 .npy into an annotated f16 tensor
// ============================================================================
// Fixture tests/load_f16_data.npy is float16 [[1.5,2.5],[3.5,4.5]] (all exactly
// representable in f16). y = x + x = [[3,5],[7,9]].
// Expected value derived by hand.
//
// @EXPECTED
// {"y": [[3.0, 5.0], [7.0, 9.0]]}
let x: Tensor[f16, (2, 2)] = load("tests/load_f16_data.npy")
let y = add(x, x)
y
