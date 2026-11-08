// ============================================================================
// TensorLang: an infix comparison inside a for-loop body
// ============================================================================
// m = x >= y = [0,1,1] every iteration; acc accumulates over 2 iterations
// -> [0,2,2]. Expected value derived by hand.
//
// @EXPECTED
// {"result": [0.0, 2.0, 2.0]}
let x: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let y: Tensor[f32, (3,)] = [2.0, 2.0, 2.0]
let acc: Tensor[f32, (3,)] = [0.0, 0.0, 0.0]

for i in range(2) {
    let m = x >= y
    let acc_new = add(acc, m)
    acc = acc_new
}

let result = acc
