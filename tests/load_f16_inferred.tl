// ============================================================================
// TensorLang: load() of a float16 .npy with dtype/shape inferred from the file
// ============================================================================
// No annotation: the compiler must infer f16 and (2, 2) from the file.
// Same data as load_f16_annotated -> y = [[3,5],[7,9]].
// Expected value derived by hand.
//
// @EXPECTED
// {"y": [[3.0, 5.0], [7.0, 9.0]]}
let x = load("tests/load_f16_data.npy")
let y = add(x, x)
y
