// Probe 5: tiny GPT-2-style forward pass (1 layer, seq=4, d=4, vocab=8).
// w_tok, w_pos: tensor literals (column variation for non-zero layer_norm).
// w_out: tensor literal (column variation so final projection is non-zero).
// Rest: uniform fills (acceptable — they produce zero intermediate
// contributions but don't zero out the final logits).

let x_tok: Tensor[f32, (4, 8)] = fill(1.0, (4, 8))

// --- weights ---
let w_tok: Tensor[f32, (8, 4)] = [[0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4]]

let w_pos: Tensor[f32, (4, 4)] = [[0.05, 0.1, 0.15, 0.2],
                                    [0.05, 0.1, 0.15, 0.2],
                                    [0.05, 0.1, 0.15, 0.2],
                                    [0.05, 0.1, 0.15, 0.2]]

let w_q: Tensor[f32, (4, 4)] = fill(0.1, (4, 4))
let w_k: Tensor[f32, (4, 4)] = fill(0.2, (4, 4))
let w_v: Tensor[f32, (4, 4)] = fill(0.3, (4, 4))
let w_o: Tensor[f32, (4, 4)] = fill(0.1, (4, 4))
let w_ff1: Tensor[f32, (4, 16)] = fill(0.1, (4, 16))
let w_ff2: Tensor[f32, (16, 4)] = fill(0.1, (16, 4))

// w_out varies across columns → non-zero logits
let w_out: Tensor[f32, (4, 8)] = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]

// --- biases ---
let b_q: Tensor[f32, (4,)] = fill(0.0, (4,))
let b_k: Tensor[f32, (4,)] = fill(0.0, (4,))
let b_v: Tensor[f32, (4,)] = fill(0.0, (4,))
let b_o: Tensor[f32, (4,)] = fill(0.0, (4,))
let b_ff1: Tensor[f32, (16,)] = fill(0.0, (16,))
let b_ff2: Tensor[f32, (4,)] = fill(0.0, (4,))
let b_out: Tensor[f32, (8,)] = fill(0.0, (8,))

let gamma1: Tensor[f32, (4,)] = fill(1.0, (4,))
let beta1: Tensor[f32, (4,)] = fill(0.0, (4,))

// --- embedding ---
let tok_emb = matmul(x_tok, w_tok)
let x0 = add(tok_emb, w_pos)

// --- layer norm 1 ---
let xn1 = layer_norm(x0, axis=1)
let x1 = add(mult(xn1, gamma1), beta1)

// --- attention ---
let q = add(matmul(x1, w_q), b_q)
let k = add(matmul(x1, w_k), b_k)
let v = add(matmul(x1, w_v), b_v)

let scale: Tensor[f32, (1,)] = [0.5]
let kt = transpose(k, axes=(1, 0))
let scores = mult(matmul(q, kt), scale)

let r: Tensor[f32, (4, 1)] = [[0.0], [1.0], [2.0], [3.0]]
let cidx: Tensor[f32, (1, 4)] = [[0.0, 1.0, 2.0, 3.0]]
let diff = minus(r, cidx)
let allowed = diff >= 0.0
let one: Tensor[f32, (1,)] = [1.0]
let blocked = minus(one, allowed)
let neg: Tensor[f32, (1,)] = [-1000000000.0]
let mask = mult(blocked, neg)
let masked = add(scores, mask)
let attnw = softmax(masked, axis=1)
let attn_raw = matmul(attnw, v)
let attn_b = add(attn_raw, b_o)
let attn_proj = matmul(attn_b, w_o)

// residual 1
let x2 = add(x1, attn_proj)

// --- layer norm 2 ---
let xn2 = layer_norm(x2, axis=1)
let x3 = add(mult(xn2, gamma1), beta1)

// --- FFN ---
let ff1 = add(matmul(x3, w_ff1), b_ff1)
let ff1g = gelu(ff1)
let ff2 = add(matmul(ff1g, w_ff2), b_ff2)

// residual 2
let x4 = add(x3, ff2)

// --- output projection ---
let logits = add(matmul(x4, w_out), b_out)

logits

// @EXPECTED
// {"logits": [[-1.4901161e-08, -2.9802322e-08, 0.0, -5.9604645e-08, 0.0, 0.0, 1.1920929e-07, -1.1920929e-07],
//             [-1.4901161e-08, -2.9802322e-08, 0.0, -5.9604645e-08, 0.0, 0.0, 1.1920929e-07, -1.1920929e-07],
//             [-1.4901161e-08, -2.9802322e-08, 0.0, -5.9604645e-08, 0.0, 0.0, 1.1920929e-07, -1.1920929e-07],
//             [-1.4901161e-08, -2.9802322e-08, 0.0, -5.9604645e-08, 0.0, 0.0, 1.1920929e-07, -1.1920929e-07]]}

