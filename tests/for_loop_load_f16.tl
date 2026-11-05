// ============================================================================
// TensorLang: f16 load() inside a for-loop body
// ============================================================================
// Combines the two earlier bugs: loop-body load reaching the GPU, and a
// non-f32 dtype. 2 iterations of acc += x with x = [[1.5,2.5],[3.5,4.5]]
// -> [[3,5],[7,9]]. Expected value derived by hand.
//
// @EXPECTED
// {"result": [[3.0, 5.0], [7.0, 9.0]]}
let acc: Tensor[f16, (2, 2)] = [[0.0, 0.0], [0.0, 0.0]]

for i in range(2) {
    let x: Tensor[f16, (2, 2)] = load("tests/load_f16_data.npy")
    let acc_new = add(acc, x)
    acc = acc_new
}

let result = acc
