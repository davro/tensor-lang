// Probe 1: single-head attention block, causal mask, forward + backward.
// seq=4, d=4. Constant fills for determinism.

let q: Tensor[f32, (4, 4)] = fill(0.1, (4, 4)) with grad
let k: Tensor[f32, (4, 4)] = fill(0.2, (4, 4)) with grad
let v: Tensor[f32, (4, 4)] = fill(0.3, (4, 4)) with grad

let scale: Tensor[f32, (1,)] = [0.5]

let kt = transpose(k, axes=(1, 0))
let scores = matmul(q, kt)
let scaled = mult(scores, scale)

// causal mask via arithmetic broadcasting + literal comparison (operator form)
let r: Tensor[f32, (4, 1)] = [[0.0], [1.0], [2.0], [3.0]]
let c: Tensor[f32, (1, 4)] = [[0.0, 1.0, 2.0, 3.0]]
let diff = minus(r, c)
let allowed = diff >= 0.0
let one: Tensor[f32, (1,)] = [1.0]
let blocked = minus(one, allowed)
let neg: Tensor[f32, (1,)] = [-1000000000.0]
let mask = mult(blocked, neg)

let masked = add(scaled, mask)
let attn = softmax(masked, axis=1)
let out = matmul(attn, v)

let loss = sum(out)
backward(loss)

out

// @EXPECTED
// {"out": [[...]]}

