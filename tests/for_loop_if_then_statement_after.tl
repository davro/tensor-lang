// ============================================================================
// TensorLang: statements after a loop that contains an `if` run their own kernels
// ============================================================================
// The kernels of an `if` inside a loop body are in the global kernel list, so a statement after the loop
// must be positioned after them. It used to run a kernel from inside the loop instead, so `result` was never
// computed. No backward() here: this isolates the kernel positions.
// result = x * x = [1, 4, 9]
//
// @EXPECTED
// {"result": [1.0, 4.0, 9.0]}
let x: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]

for i in range(2) {
    let y = add(x, x)
    if y > 1.0 {
        let z: Tensor[f32, (3,)] = mult(y, y)
    }
}

let result = mult(x, x)
