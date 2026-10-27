let acc: Tensor[f32, (1, 1)] = [[0.0]]

for i in range(3) {
    for j in range(4) {
        let inc: Tensor[f32, (1, 1)] = fill(1.0, (1, 1))
        let acc_new = add(acc, inc)
        acc = acc_new
    }
    save(acc, "acc_outer.npy")
}

// @EXPECTED
// {
//   "acc": [[12.0]]
// }
