// ============================================================================
// TensorLang: an if/else with backward() inside a for-loop body (EXPLORATORY)
// ============================================================================
// An `if` inside a loop body used to be executed by an inline copy that never resolved the
// branch's autograd path, so the backward pass included the nodes of both branches. The else
// branch is taken in both iterations: z = w + w, loss = sum(z), so w_grad = [2, 2]. With the
// bug the then-branch nodes would also contribute and give [14, 18]. Not simulated off-GPU:
// the loop executor is not covered by my local harness. Expected value derived by hand.
//
// @EXPECTED
// {"w_grad": [2.0, 2.0]}
let w: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let no: Tensor[f32, (1,)] = [0.0]
for i in range(2) {
    if no {
        let z = mult(w, w)
        let loss = sum(z)
        backward(loss)
    } else {
        let z = add(w, w)
        let loss = sum(z)
        backward(loss)
    }
}
