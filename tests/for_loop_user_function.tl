// @EXPECTED
// {"result": [[3.0, 3.0], [3.0, 3.0]]}
fn add_bias(x: Tensor[M, N], b: Tensor[M, N]) -> Tensor[M, N] {
    let y = add(x, b)
    return y
}

let acc: Tensor[f32, (2, 2)] = fill(0.0, (2, 2))

for i in range(3) {
    let inc: Tensor[f32, (2, 2)] = fill(1.0, (2, 2))
    let acc_new = add_bias(acc, inc)
    acc = acc_new
}

let result = acc
