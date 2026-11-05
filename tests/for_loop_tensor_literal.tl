// ============================================================================
// TensorLang: tensor literal declared inside a for-loop body
// ============================================================================
// Regression: the compiler pre-allocates every env name before execution, and
// _ensure_gpu_alloc returned early without uploading, so loop-local literals
// (and load()) never reached the GPU. 3 iterations of acc += [[1,2],[3,4]]
// -> [[3,6],[9,12]]. Expected value derived by hand.
//
// @EXPECTED
// {"result": [[3.0, 6.0], [9.0, 12.0]]}
let acc: Tensor[f32, (2, 2)] = fill(0.0, (2, 2))

for i in range(3) {
    let inc: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
    let acc_new = add(acc, inc)
    acc = acc_new
}

let result = acc
