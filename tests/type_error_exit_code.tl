// ============================================================================
// TensorLang: Type-check failure exits non-zero
// ============================================================================
// Description:
//   Regression test for a compiler bug where compile_and_execute() had
//   no `else` branch for a failed type check (and several internal
//   kernel-generation errors returned `False` with nothing checking
//   it) — so the process printed a "[TYPE CHECKER] error: ..." line
//   and then exited with code 0 anyway, indistinguishable from success
//   to any script or CI job checking $?.
//
//   This file is intentionally invalid — matmul shape mismatch,
//   (3,4) @ (5,6) — and produces no usable output. The test passes if
//   and only if tensorlang exits non-zero AND its output contains every
//   @EXPECT_FAILURE substring below (a stable diagnostic code plus a
//   phrase), so it cannot pass on an unrelated failure. See
//   TestRunner.parse_expect_failure() / run_single_test(): unlike every
//   other test in this directory, it does not use an @EXPECTED block and
//   does not run to a successful save().
//
// @EXPECT_FAILURE: [E-SHAPE-MISMATCH]
// @EXPECT_FAILURE: Inner dimensions must match

let x: Tensor[f32, (3, 4)] = [[1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0]]
let y: Tensor[f32, (5, 6)] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
let z = matmul(x, y)
