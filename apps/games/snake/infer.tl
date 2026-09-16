// apps/games/snake/infer.tl
//
// Single-board forward pass through the trained policy network. This is
// the "TensorLang player": tools/agent.py's run_inference writes the
// current board to cache/apps/games/snake/infer.tl/board.npy (one-hot
// encoded, via encode_state_onehot) before every AI turn, invokes this
// file as a fresh subprocess, then reads back move.npy — a (1, 4)
// probability distribution over the 4 directions — and picks the
// highest-probability SAFE direction itself (masking illegal/colliding
// directions happens in Python, not here — same split as every other
// app's infer.tl / choose_move).
//
// Weights load from apps/games/snake/weights/ — the tracked, in-repo
// "production" location (a load() with a relative path resolves against
// the repo root — see tensorlang/compiler.py's _load_data_file). train.tl
// itself writes into cache/apps/games/snake/train.tl/weights/, same as
// any other .tl file's scratch output; tools/promote_weights.py is the
// explicit step that copies a training run you're happy with into this
// tracked location. If this directory is empty (nothing promoted yet),
// this load() fails with a clear "Data file not found" error, which
// agent.py's choose_ai_move already catches and falls back to the
// heuristic solver for.
//
// Board encoding matches train.tl exactly: 144 cells x 5 one-hot
// categories = 720 inputs (see tools/agent.py's encode_state_onehot).
// Output columns are in agent.DIRECTIONS order: up, down, left, right.
//
// NOTE on batch size: the value-head "(1,1) looks like a scalar" CUDA
// dispatch bug documented in apps/games/connect_four/NOTES.md (bug #3)
// only bites when an output layer has exactly 1 unit, so a (1,1) tensor
// at batch size 1 gets misclassified as a scalar. Every layer here
// (including the final 4-unit output) has width > 1, so `add(y_pre, b3)`
// at this file's batch size of 1 produces a genuine, unambiguous (1, 4)
// output through the ordinary binary_broadcast path — no exotic dispatch
// route needed, unlike connect_four's single-unit value head.
//
// Run directly, for manual testing:
//   python3 tensorlang.py apps/games/snake/infer.tl

let board: Tensor[f32, (1, 720)] = load("cache/apps/games/snake/infer.tl/board.npy")

let w1: Tensor[f32, (720, 256)] = load("apps/games/snake/weights/w1.npy")
let b1: Tensor[f32, (256,)]     = load("apps/games/snake/weights/b1.npy")
let w2: Tensor[f32, (256, 64)]  = load("apps/games/snake/weights/w2.npy")
let b2: Tensor[f32, (64,)]      = load("apps/games/snake/weights/b2.npy")
let w3: Tensor[f32, (64, 4)]    = load("apps/games/snake/weights/w3.npy")
let b3: Tensor[f32, (4,)]       = load("apps/games/snake/weights/b3.npy")

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
