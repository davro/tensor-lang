// ============================================================================
// TensorLang: backward() inside the if branch of an if/elif/else
// ============================================================================
// Every branch defines the same names (z, loss) with a different operation, so the
// autograd graph holds three z nodes and three loss nodes; only the executed branch's
// nodes may contribute. w = [3,4], x = [1,2]:
//   if   : z = w*w -> loss = sum(z) -> d loss/d w = 2w = [6, 8]
//   elif : z = w*x -> loss = sum(z) -> d loss/d w = x  = [1, 2]
//   else : z = w+w -> loss = sum(z) -> d loss/d w = 2  = [2, 2]
// Expected value derived by hand.
//
// @EXPECTED
// {"w_grad": [6.0, 8.0]}
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let w: Tensor[f32, (2,)] = [3.0, 4.0] with grad
let yes: Tensor[f32, (1,)] = [1.0]
let no: Tensor[f32, (1,)] = [0.0]
let c1: Tensor[f32, (1,)] = [1.0]
let c2: Tensor[f32, (1,)] = [1.0]
if c1 {
    let z = mult(w, w)
    let loss = sum(z)
    backward(loss)
} elif c2 {
    let z = mult(w, x)
    let loss = sum(z)
    backward(loss)
} else {
    let z = add(w, w)
    let loss = sum(z)
    backward(loss)
}
