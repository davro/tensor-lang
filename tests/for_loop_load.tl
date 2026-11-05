// @EXPECTED
// {"result": [[4.0, 8.0], [12.0, 16.0]]}
let acc: Tensor[f32, (2, 2)] = fill(0.0, (2, 2))

for i in range(2) {
    let x: Tensor[f32, (2, 2)] = load("tests/for_loop_load_data.npy")
    let y = add(x, x)
    let acc_new = add(acc, y)
    acc = acc_new
}

let result = acc
