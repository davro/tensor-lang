// Token + positional embedding as in GPT-2: x = wte[tokens] + wpe[positions]. Token 5 repeats at positions 0 and 2.
let wte: Tensor[f32, (6, 4)] = [[0.0, 0.25, 0.5, 0.75], [1.0, 1.25, 1.5, 1.75], [2.0, 2.25, 2.5, 2.75], [3.0, 3.25, 3.5, 3.75], [4.0, 4.25, 4.5, 4.75], [5.0, 5.25, 5.5, 5.75]]
let wpe: Tensor[f32, (3, 4)] = [[8.0, 8.5, 9.0, 9.5], [16.0, 16.5, 17.0, 17.5], [24.0, 24.5, 25.0, 25.5]]
let tokens: Tensor[f32, (3,)] = [5.0, 1.0, 5.0]
let positions: Tensor[f32, (3,)] = [0.0, 1.0, 2.0]
let tok_emb = embedding(tokens, wte)
let pos_emb = embedding(positions, wpe)
let x = add(tok_emb, pos_emb)

// @EXPECTED
// {"x": [[13.0, 13.75, 14.5, 15.25], [17.0, 17.75, 18.5, 19.25], [29.0, 29.75, 30.5, 31.25]]}
