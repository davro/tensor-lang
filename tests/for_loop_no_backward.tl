// ============================================================================
// TensorLang: for loop — no backward(), pure forward accumulation
// ============================================================================
// Description:
//   Runs a for loop with no backward() call — pure iterative computation.
//   Each iteration adds a fixed increment to a running result via rebind.
//   Tests that the executor handles the no-backward loop body path correctly,
//   which is a separate code branch from the training loop case.
//
// Architecture:
//   result = add(result, increment)   (4 times)
//   increment = fill(2.0, (1,1))
//   result starts at fill(0.0, (1,1))
//
// Step-by-step:
//   iter 1: result = [[2.0]]
//   iter 2: result = [[4.0]]
//   iter 3: result = [[6.0]]
//   iter 4: result = [[8.0]]
//
// @EXPECTED
// {
//   "result": [[8.0]]
// }

let result:    Tensor[f32, (1, 1)] = [[0.0]]
let increment: Tensor[f32, (1, 1)] = [[2.0]]

for step in range(4) {
    let result_new = add(result, increment)
    result = result_new
}
