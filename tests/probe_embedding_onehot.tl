// Probe 3: embedding lookup as one-hot @ W. Forward + backward.
// The one-hot is prepared outside (a (seq, vocab) f32 matrix). This probe
// confirms matmul carries gradients through a one-hot input, which is the
// property that makes the external-preparation workaround viable for a
// forward-only demo. Generation still needs a real embedding/gather op.

let one_hot: Tensor[f32, (4, 8)] = fill(0.0, (4, 8)) with grad
// row 0 -> token 2, row 1 -> token 5, etc. (hand-set via literals is tedious;
// use a uniform fill for the probe and accept a degenerate embedding)
let w_emb: Tensor[f32, (8, 4)] = fill(0.1, (8, 4)) with grad

let emb = matmul(one_hot, w_emb)

let loss = sum(emb)
backward(loss)

emb

// @EXPECTED
// {"emb": [[0.0, 0.0, 0.0, 0.0], ...]}

