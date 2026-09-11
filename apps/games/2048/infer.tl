// apps/games/2048/infer.tl
//
// Single-board forward pass through the trained policy network. This is
// the "TensorLang player": tools/agent.py's run_inference writes the
// current board to cache/apps/games/2048/infer.tl/board.npy (log2-one-hot
// encoded, via encode_board_onehot) before every AI turn, invokes this
// file as a fresh subprocess, then reads back move.npy — a (1, 4)
// probability distribution over the 4 directions — and picks the
// highest-probability LEGAL direction itself. Masking illegal (no-op)
// directions happens in Python, not here, since the driver already knows
// which directions are legal and this keeps the network itself simple —
// exact same split as tic_tac_toe's infer.tl / choose_move.
//
// Weights load from apps/games/2048/weights/ — the tracked, in-repo
// "production" location (a load() with a relative path resolves against
// the repo root, not against cache/ — see tensorlang/compiler.py's
// _load_data_file — so this works with no extra plumbing). train.tl
// itself still writes into cache/apps/games/2048/train.tl/weights/, same
// as any other .tl file's scratch output; tools/promote_weights.py is
// the explicit step that copies a training run you're happy with into
// this tracked location (with an automatic backup of whatever was here
// before — see that script's docstring). If this directory is empty
// (nothing promoted yet), this load() fails with a clear "Data file not
// found" error, which agent.py's choose_ai_move already catches and
// falls back to the original heuristic for.
//
// Board encoding matches train.tl exactly: 16 cells x 16 log2-one-hot
// value categories = 256 inputs (see tools/agent.py's
// encode_board_onehot). Output columns are in agent.DIRECTIONS order:
// left, right, up, down.
//
// Run directly, for manual testing:
//   python3 tensorlang.py apps/games/2048/infer.tl

let board: Tensor[f32, (1, 256)] = load("cache/apps/games/2048/infer.tl/board.npy")

let w1: Tensor[f32, (256, 128)] = load("apps/games/2048/weights/w1.npy")
let b1: Tensor[f32, (128,)]     = load("apps/games/2048/weights/b1.npy")
let w2: Tensor[f32, (128, 64)]  = load("apps/games/2048/weights/w2.npy")
let b2: Tensor[f32, (64,)]      = load("apps/games/2048/weights/b2.npy")
let w3: Tensor[f32, (64, 4)]    = load("apps/games/2048/weights/w3.npy")
let b3: Tensor[f32, (4,)]       = load("apps/games/2048/weights/b3.npy")

let h1_pre        = matmul(board, w1)
let h1_pre_biased = add(h1_pre, b1)
let h1            = relu(h1_pre_biased)
let h2_pre        = matmul(h1, w2)
let h2_pre_biased = add(h2_pre, b2)
let h2            = relu(h2_pre_biased)
let y_pre         = matmul(h2, w3)
let y_pre_biased  = add(y_pre, b3)
let move_probs    = softmax(y_pre_biased, axis=1)

save(move_probs, "move.npy")
