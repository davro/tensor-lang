// ============================================================================
// TensorLang: for loop — no backward(), pure forward accumulation
// ============================================================================
// Description:
//   Runs a for loop with no backward() call — pure iterative computation.
//   Each iteration adds a fixed increment to a running accumulator via rebind.
//   Tests that the executor handles the no-backward loop body path correctly,
//   which is a separate code branch from the training loop case.
//
// Architecture:
//   acc = add(acc, increment)   (4 times, increment = [[2.0]])
//   acc starts at [[0.0]], ends at [[8.0]]
//
// Step-by-step:
//   iter 1: acc = [[2.0]]
//   iter 2: acc = [[4.0]]
//   iter 3: acc = [[6.0]]
//   iter 4: acc = [[8.0]]
//
// Post-loop: add(acc, zero) forces the final rebound value into the .npy
// cache under 'result_final'. Using a distinct name avoids the GPU
// allocation conflict where top-level 'acc' still points to [[0.0]].
//
// @EXPECTED
// {
//   "result_final": [[8.0]]
// }

let acc:       Tensor[f32, (1, 1)] = [[0.0]]
let increment: Tensor[f32, (1, 1)] = [[2.0]]
let zero:      Tensor[f32, (1, 1)] = [[0.0]]

for step in range(4) {
    let acc_new = add(acc, increment)
    acc = acc_new
}

// Post-loop: read the rebound 'acc' buffer and save under 'result_final'.
// 'acc' GPU pointer was rebound each iteration to acc_new's buffer,
// so add(acc, zero) reads the final accumulated value.
let result_final = add(acc, zero)