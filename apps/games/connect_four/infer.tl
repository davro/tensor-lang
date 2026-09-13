// apps/games/connect_four/infer.tl
//
// Single-board forward pass through the trained policy+value network.
// tools/agent.py writes the current board to
// cache/apps/games/connect_four/infer.tl/board.npy before every AI turn,
// invokes this file as a fresh subprocess, then reads back:
//   - policy.npy (1, 7): softmax probabilities over the 7 columns
//   - value.npy  (1, 1): tanh-scaled position evaluation in [-1, 1]
//     from the perspective of whoever is about to move (the AI)
// Masking illegal (full) columns happens in Python, not here, same
// reasoning as tic_tac_toe's infer.tl: the driver already knows which
// columns are full, so the network itself stays simple, and a fresh
// subprocess per move keeps GPU memory from accumulating across a long
// play session.
//
// Board encoding matches train.tl exactly: +1 = mover's own disc,
// -1 = opponent's disc, 0 = empty, always from the perspective of
// whoever is about to move. Row-major, 42 cells (6 rows x 7 columns).
//
// The value head has 2 output units, not 1 — see train.tl's "REAL BUG
// #3" comment: a width-1 value head produces a (1,1) tensor at this
// file's batch size of 1, which a type-checker heuristic misclassifies
// as a scalar and crashes the compiler. value.npy here is (1,2);
// tools/agent.py uses (col0 - col1)/2 as the actual value estimate.
//
// Run directly, for manual testing:
//   python3 tensorlang.py apps/games/connect_four/infer.tl

let board: Tensor[f32, (1, 42)] = load("cache/apps/games/connect_four/infer.tl/board.npy")

let w1: Tensor[f32, (42, 128)] = load("apps/games/connect_four/weights/w1.npy")
let b1: Tensor[f32, (128,)]    = load("apps/games/connect_four/weights/b1.npy")
let w2: Tensor[f32, (128, 64)] = load("apps/games/connect_four/weights/w2.npy")
let b2: Tensor[f32, (64,)]     = load("apps/games/connect_four/weights/b2.npy")
let wp: Tensor[f32, (64, 7)]   = load("apps/games/connect_four/weights/wp.npy")
let bp: Tensor[f32, (7,)]      = load("apps/games/connect_four/weights/bp.npy")
let wv: Tensor[f32, (64, 2)]   = load("apps/games/connect_four/weights/wv.npy")
let bv: Tensor[f32, (2,)]      = load("apps/games/connect_four/weights/bv.npy")

let h1_pre = matmul(board, w1)
let h1_biased = add(h1_pre, b1)
let h1 = relu(h1_biased)
let h2_pre = matmul(h1, w2)
let h2_biased = add(h2_pre, b2)
let h2 = relu(h2_biased)

let p_pre = matmul(h2, wp)
let p_biased = add(p_pre, bp)
let policy = softmax(p_biased, axis=1)

let v_pre = matmul(h2, wv)
let v_biased = add(v_pre, bv)
let value = tanh(v_biased)

save(policy, "policy.npy")
save(value, "value.npy")
