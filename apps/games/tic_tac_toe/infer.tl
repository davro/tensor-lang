// apps/games/tic_tac_toe/infer.tl
//
// Single-board forward pass through the trained policy network. This is
// the "TensorLang player": tools/agent.py writes the current board to
// cache/apps/games/tic_tac_toe/infer.tl/board.npy before every AI turn,
// invokes this file as a fresh subprocess, then reads back move.npy —
// a (1, 9) probability distribution over the 9 cells — and picks the
// highest-probability LEGAL cell itself. Masking illegal (occupied)
// cells happens in Python, not here, since the driver already knows
// which cells are occupied and this keeps the network itself simple.
//
// Board encoding matches train.tl exactly: +1 = the mover's own mark,
// -1 = the opponent's mark, 0 = empty — always from the perspective of
// whoever is about to move (the AI, every time this file runs).
//
// Run directly, for manual testing:
//   python3 tensorlang.py apps/games/tic_tac_toe/infer.tl

let board: Tensor[f32, (1, 9)] = load("cache/apps/games/tic_tac_toe/infer.tl/board.npy")

let w1: Tensor[f32, (9, 64)]  = load("cache/apps/games/tic_tac_toe/train.tl/weights/w1.npy")
let b1: Tensor[f32, (64,)]    = load("cache/apps/games/tic_tac_toe/train.tl/weights/b1.npy")
let w2: Tensor[f32, (64, 9)]  = load("cache/apps/games/tic_tac_toe/train.tl/weights/w2.npy")
let b2: Tensor[f32, (9,)]     = load("cache/apps/games/tic_tac_toe/train.tl/weights/b2.npy")

let h_pre        = matmul(board, w1)
let h_pre_biased = add(h_pre, b1)
let h            = relu(h_pre_biased)
let y_pre        = matmul(h, w2)
let y_pre_biased = add(y_pre, b2)
let move_probs   = softmax(y_pre_biased, axis=1)

save(move_probs, "move.npy")
