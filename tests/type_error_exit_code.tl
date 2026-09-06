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
//   and only if tensorlang exits non-zero. See TestRunner.
//   is_expect_failure_test() / run_single_test() for how @EXPECT_FAILURE
//   is handled: unlike every other test in this directory, it does not
//   use an @EXPECTED block and does not run to a successful save().
//
// @EXPECT_FAILURE

let x: Tensor[f32, (3, 4)] = [[1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0]]
let y: Tensor[f32, (5, 6)] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
let z = matmul(x, y)
