// ============================================================================
// TensorLang: for loop body parity regression test
// ============================================================================
let running_mean_placeholder: Tensor[f32, (2,)] = [3.5, 3.5]
let running_var_placeholder: Tensor[f32, (2,)] = [2.25, 2.25]
let x: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let w: Tensor[f32, (3, 2)] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] with grad
let two_mat: Tensor[f32, (2, 2)] = [[2.0, 2.0], [2.0, 2.0]]
let lr: Tensor[f32, (1, 1)] = [[0.01]]

for step in range(3) {
    let h = matmul(x, w)
    let hn = layer_norm(h, axis=1)
    let hb = batch_norm(h, running_mean_placeholder, running_var_placeholder)
    let hi = instance_norm(h)
    let hr = reshape(h, (2, 2))
    let ht = transpose(h)
    let hc = concat(h, h, axis=0)
    let hg = greater(h, two_mat)
    let hl = less(h, two_mat)
    let he = equal(h, h)
    let hs = h[0:1, :]
    let sm = softmax(h, axis=1)
    let s = sum(h, axis=1)
    let m = mean(h, axis=0)
    let mx = max(h, axis=1)
    let mn = min(h, axis=0)
    let am = argmax(h, axis=1)
    let an = argmin(h, axis=0)
    let g = gelu(h)
    let sw = swish(h)
    let loss = mse_loss(h, h)
    backward(loss)
    let gs = mult(lr, w_grad)
    let wu = minus(w, gs)
    w = wu
}

let h_final = matmul(x, w)
let loss_final = mse_loss(h_final, h_final)
loss_final

// @EXPECTED
// {"loss_final": 0.0}
