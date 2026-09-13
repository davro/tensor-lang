// apps/games/connect_four/train.tl
//
// Trains a policy+value network on self-play Connect Four data:
//
//   input (42)  -- one cell per board square, always encoded from the
//                  perspective of the player about to move:
//                  +1.0 = mover's own disc, -1.0 = opponent's disc,
//                  0.0 = empty. Same convention as tic_tac_toe's main.tl,
//                  extended from a 3x3 to a 6x7 board (row-major, 42 cells).
//     |
//   shared trunk: matmul/add/relu (42->128), matmul/add/relu (128->64)
//     |                                              |
//   policy head (64->7, softmax)          value head (64->2, tanh)
//     |                                              |
//   policy_loss = mse_loss(probs, y_policy)   value_loss = mse_loss(value, y_value)
//                        \                    /
//                         loss = add(policy_loss, value_loss)
//
// The value head has 2 output units, not 1, even though it predicts a
// single scalar (position value in [-1, 1]) — see the "REAL BUG #3"
// note below train.tl's tensor declarations for why a width-1 head
// crashes the compiler at inference time. y_value's second column is
// just the negation of the first (see tools/generate_data.py); the
// network is trained to predict [value, -value], and only column 0 is
// actually used (tools/agent.py averages (col0 - col1)/2 for a little
// extra robustness against either column drifting).
//
// Unlike tic_tac_toe (4520 possible board states total — exhaustively
// enumerable, effectively a lookup table), Connect Four has on the
// order of 4.5 trillion legal positions. Exhaustive enumeration isn't
// possible, so the dataset here is a fixed-size SAMPLE of self-play
// positions (see tools/generate_data.py): N games are played using a
// depth-limited alpha-beta solver (with some randomness for game
// diversity), and every non-terminal position visited is recorded with:
//   - a policy target: a one-hot vector for the solver's best move
//     *recomputed cleanly* for that exact position (independent of
//     whatever exploration noise caused the actual game to continue),
//   - a value target: the ACTUAL final outcome of that game from this
//     position's mover's perspective (+1 win / -1 loss / 0 draw) —
//     classic self-play-style value labeling, not a static heuristic
//     score, so the network learns to predict "who actually wins from
//     here" rather than "what a hand-written evaluation function says".
//
// N below MUST match tools/generate_data.py's --num-positions and the
// shapes of data/boards.npy / data/policy.npy / data/value.npy exactly
// (the loader has no way to reshape a mismatched file; a mismatch
// surfaces as a load() shape error at compile/run time). Default N is
// picked in generate_data.py's DEFAULT_POSITIONS constant; keep the two
// in sync if you change it.
//
// Loss: mse_loss on the softmax policy output against a one-hot target,
// not cross_entropy — same reasoning as tic_tac_toe's main.tl: this
// compiler's cross_entropy backward pass assumes a same-shaped target
// which doesn't match its documented (batch,)-class-index forward
// signature (see HANDOVER.md). mse_loss is the well-tested choice here.
//
// SGD workaround: per HANDOVER.md, mult's CUDA dispatch only reliably
// handles a (1,1) scalar broadcasting against a 2D tensor, not against a
// WIDER 1D vector (i.e. a bias of size > 1). Every bias update below
// therefore multiplies by a same-shaped learning-rate vector (lr_b1,
// lr_b2, lr_bp), exactly like tic_tac_toe's lr_b1/lr_b2 and
// decision_boundary's lr_b1/lr_b2. The single-element value-head bias
// (bv, shape (1,)) and every 2D weight matrix use the (1,1) lr directly,
// since that combination is documented as working.
//
// Hyperparameters (hidden=128/64, lr=8.0, 8000 epochs) are a starting
// point, NOT tuned against real hardware — this file was written and
// type/shape-checked (via check.py, no GPU available) but never actually
// trained, since this environment has no CUDA toolkit or GPU. Treat loss
// curves from the first real run as the actual source of truth on
// whether these need adjusting; see NOTES.md's verification checklist.
//
// Run (after tools/generate_data.py and tools/init_weights.py have run
// once — see run.sh, which does both automatically):
//   python3 tensorlang.py apps/games/connect_four/train.tl

let x:        Tensor[f32, (3000, 42)] = load("apps/games/connect_four/data/boards.npy")
let y_policy: Tensor[f32, (3000, 7)]  = load("apps/games/connect_four/data/policy.npy")
let y_value:  Tensor[f32, (3000, 2)]  = load("apps/games/connect_four/data/value.npy")

let w1: Tensor[f32, (42, 128)] = load("cache/apps/games/connect_four/train.tl/weights/w1.npy") with grad
let b1: Tensor[f32, (128,)]    = load("cache/apps/games/connect_four/train.tl/weights/b1.npy") with grad
let w2: Tensor[f32, (128, 64)] = load("cache/apps/games/connect_four/train.tl/weights/w2.npy") with grad
let b2: Tensor[f32, (64,)]     = load("cache/apps/games/connect_four/train.tl/weights/b2.npy") with grad

let wp: Tensor[f32, (64, 7)] = load("cache/apps/games/connect_four/train.tl/weights/wp.npy") with grad
let bp: Tensor[f32, (7,)]    = load("cache/apps/games/connect_four/train.tl/weights/bp.npy") with grad
let wv: Tensor[f32, (64, 2)] = load("cache/apps/games/connect_four/train.tl/weights/wv.npy") with grad
let bv: Tensor[f32, (2,)]    = load("cache/apps/games/connect_four/train.tl/weights/bv.npy") with grad

let lr: Tensor[f32, (1, 1)] = [[8.0]]

// Same-shaped learning-rate vectors for the wider (>1 element) bias
// gradients — see the SGD workaround note above.
let lr_b1: Tensor[f32, (128,)] = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
let lr_b2: Tensor[f32, (64,)]  = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
let lr_bp: Tensor[f32, (7,)]   = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
// REAL BUG #3, found on real GPU hardware: the value head originally
// had 1 output unit (bv shape (1,)). At batch=1 (infer.tl, a single
// board), that makes v_pre/v_biased shape (1,1) — and
// type_checker.py's `_is_scalar_shape` treats ANY all-ones shape as "a
// scalar for broadcast purposes" (a heuristic meant for mse_loss's own
// (1,1) output), so `add(v_pre, bv)` was silently typed as producing a
// 1D (1,) output instead of the correct 2D (1,1) — which then crashed
// compiler.py's `binary_broadcast` (`rows, cols = output_shape` on a
// 1-tuple). This is NOT something a batch=3000 training run can ever
// surface (v_pre is (3000,1) there, never all-ones) — it only shows up
// at single-sample inference, which is exactly infer.tl's whole job.
// Rather than patch around one specific add() call again (as bv's
// original lr_bv workaround attempted, incompletely — that fixed the
// SGD update but not this add()), the value head now has 2 output
// units instead of 1, so it can never produce an all-ones shape at any
// batch size — the exact same "always width > 1" property every other
// layer in this file already has. y_value's second column is just
// -value (see tools/generate_data.py); tools/agent.py uses
// (col0 - col1)/2. Same-shaped lr vector, same pattern as lr_bp above:
let lr_bv: Tensor[f32, (2,)]   = [8.0, 8.0]

for epoch in range(8000) {
    // shared trunk
    let h1_pre = matmul(x, w1)
    let h1_biased = add(h1_pre, b1)
    let h1 = relu(h1_biased)
    let h2_pre = matmul(h1, w2)
    let h2_biased = add(h2_pre, b2)
    let h2 = relu(h2_biased)

    // policy head
    let p_pre = matmul(h2, wp)
    let p_biased = add(p_pre, bp)
    let probs = softmax(p_biased, axis=1)
    let policy_loss = mse_loss(probs, y_policy)

    // value head
    let v_pre = matmul(h2, wv)
    let v_biased = add(v_pre, bv)
    let value = tanh(v_biased)
    let value_loss = mse_loss(value, y_value)

    let loss = add(policy_loss, value_loss)
    backward(loss)

    let w1_step = mult(lr, w1_grad)
    let w2_step = mult(lr, w2_grad)
    let wp_step = mult(lr, wp_grad)
    let wv_step = mult(lr, wv_grad)
    let b1_step = mult(lr_b1, b1_grad)
    let b2_step = mult(lr_b2, b2_grad)
    let bp_step = mult(lr_bp, bp_grad)
    let bv_step = mult(lr_bv, bv_grad)

    let w1_updated = minus(w1, w1_step)
    let w2_updated = minus(w2, w2_step)
    let wp_updated = minus(wp, wp_step)
    let wv_updated = minus(wv, wv_step)
    let b1_updated = minus(b1, b1_step)
    let b2_updated = minus(b2, b2_step)
    let bp_updated = minus(bp, bp_step)
    let bv_updated = minus(bv, bv_step)

    w1 = w1_updated
    w2 = w2_updated
    wp = wp_updated
    wv = wv_updated
    b1 = b1_updated
    b2 = b2_updated
    bp = bp_updated
    bv = bv_updated
}

// Final evaluation (distinct names, per established convention, to
// avoid CUDA kernel symbol collisions with the loop body above).
let h1_pre_final = matmul(x, w1)
let h1_biased_final = add(h1_pre_final, b1)
let h1_final = relu(h1_biased_final)
let h2_pre_final = matmul(h1_final, w2)
let h2_biased_final = add(h2_pre_final, b2)
let h2_final = relu(h2_biased_final)

let p_pre_final = matmul(h2_final, wp)
let p_biased_final = add(p_pre_final, bp)
let probs_final = softmax(p_biased_final, axis=1)
let policy_loss_final = mse_loss(probs_final, y_policy)

let v_pre_final = matmul(h2_final, wv)
let v_biased_final = add(v_pre_final, bv)
let value_final = tanh(v_biased_final)
let value_loss_final = mse_loss(value_final, y_value)

let loss_final = add(policy_loss_final, value_loss_final)

save(w1, "weights/w1.npy")
save(b1, "weights/b1.npy")
save(w2, "weights/w2.npy")
save(b2, "weights/b2.npy")
save(wp, "weights/wp.npy")
save(bp, "weights/bp.npy")
save(wv, "weights/wv.npy")
save(bv, "weights/bv.npy")
save(loss_final, "loss.npy")
save(policy_loss_final, "policy_loss.npy")
save(value_loss_final, "value_loss.npy")
