// @EXPECTED
// {"acc": [3.0]}
let acc: Tensor[f32, (1,)] = [0.0]
for i in range(3) {
    let inc = fill(1.0, (1,))
    let acc_new = add(acc, inc)
    acc = acc_new
    let has = any(acc)
    let allnz = all(acc)
}
acc
