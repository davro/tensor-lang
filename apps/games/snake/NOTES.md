# apps/games/snake/ — status and how to run it

**Status: engine, BFS/flood-fill solver, data pipeline, and a pygame
frontend (human / single-AI / multi-agent arena, with animations) are
built and tested end-to-end in a sandbox with no GPU. `train.tl` and
`infer.tl` parse, type-check (`check.py`), and are math-verified against
a NumPy reference (`tools/verify_math.py`) — but, like every other app
in this repo, have never actually been run on real CUDA hardware.** See
"What was and wasn't verified" below before trusting a promoted network.

## What this is

A single-head move-picking MLP for Snake, following the same overall
shape as `2048` (TensorLang only ever runs the neural net; every game
rule — movement, collision, growth, food — lives in plain Python) and
reusing `2048`'s promote/rollback pattern for taking a trained network
into production:

```
input (720) -- one-hot per cell, 144 cells x 5 categories:
               empty / food / own head / own body / enemy
  |
shared trunk: matmul/add/relu (720->256), matmul/add/relu (256->64)
  |
output: matmul/add/softmax (64->4)   -- one logit per direction
  |
loss = mse_loss(softmax(logits), one-hot target)
```

Unlike connect_four (two output heads, an adversarial alpha-beta
solver) or 2048 (a slow expectimax oracle layered over a cheap
self-play driver), Snake uses ONE solver for both self-play driving AND
labeling: `tools/agent.py`'s `heuristic_choose_move` — BFS shortest path
to food, with a flood-fill safety check (don't walk into a dead end),
falling back to "maximize reachable open space" when no safe path to
food exists. Every state it's asked about is evaluated fresh as the
root of its own search, so there's no separate "driver vs. oracle"
distinction the way there is for 2048. See `tools/agent.py`'s and
`tools/generate_data.py`'s module docstrings for the full reasoning,
including why each of the 8 symmetry-augmented variants of a training
board is INDEPENDENTLY re-labeled rather than having its label derived
by rotating the original — the solver's tie-breaking (used only when no
safe path to food exists) isn't itself rotation-equivariant, even
though BFS shortest-path is, so re-running the solver per variant is
what keeps every label correct without having to reason about that.

## Cycle breaking (why a deterministic policy needs one)

**Update (post-training feedback):** once inference was fast, single-AI
mode was observed settling into a repeating loop after a while — moving
in a small cycle indefinitely without making progress. This isn't a
network-quality problem: both `choose_ai_move` and
`heuristic_choose_move` are pure, deterministic functions of the current
`(body, direction, food, other_blocked)`, and Snake's own dynamics are
equally deterministic. If an AI-controlled snake's `(resulting body,
food)` ever exactly repeats a combination it's already produced this
game, it is mathematically guaranteed to make the same decision again
from there and loop forever — this can happen to a perfectly-trained
network exactly as easily as an undertrained one; there's nothing about
training quality that prevents it.

Fixed in `tools/play.py`: `Player.seen_keys` tracks every `(resulting
body, food)` an AI-controlled snake has produced so far this game, and
`resolve_ai_direction` tries the network/heuristic's preferred move
first as always, but falls through to the next-best candidate (from
`non_trapping_moves`' own ranking, then any remaining legal move) if the
preferred one would reproduce a seen state — and only resorts to a
random tie-break in the (rare) case where every available move would
repeat something already seen. Verified against a hand-constructed
policy that's hard-coded to alternate between exactly two directions
forever (a guaranteed infinite 2-cycle if nothing intervened): the
cycle breaker still visited 20 distinct cells over 30 ticks rather than
oscillating between 2. Also stress-tested over 1000 ticks with the real
(fast NumPy) network path and freshly-initialized weights: 887 distinct
states visited, no hang, ~0.17ms/tick maintained.

## In-process NumPy inference (why gameplay doesn't touch infer.tl)

**Update (post-training feedback):** a real playtest of single-AI mode
took over an hour to reach a score of 30 — confirming a serious
performance bug, not just "network mode is a bit slower than the
heuristic." Root cause: `choose_ai_move` was spawning `infer.tl` as a
fresh subprocess for EVERY move — a whole new Python process that
re-parses and re-compiles the TensorLang program from scratch before
running a forward pass that itself takes microseconds. Almost all of
the cost was the repeated compile, none of it was the actual math.

Fixed by adding `agent.run_inference_fast`: it loads the promoted
weights once (`agent.load_promoted_weights`, cached per-process) and
runs the exact same forward pass — matmul/add/relu, matmul/add/relu,
matmul/add/softmax — directly in NumPy, in the same process that's
already running the game loop. `choose_ai_move` now calls this instead
of the old subprocess path. Measured: ~0.17ms/move (fresh random
weights, this sandbox), versus multiple SECONDS/move before.

The original subprocess path (`agent.run_inference`) is still there,
unused by gameplay — it's the AUTHORITATIVE implementation (the only
one that actually executes through the TensorLang compiler on a GPU),
and it's what `tools/verify_infer_matches_numpy.py` uses to confirm the
NumPy mirror hasn't drifted from it. **Run that verification script once
after any retrain/promote, on real GPU hardware** — neither this
sandbox nor the one that originally wrote train.tl/infer.tl has a GPU,
so this specific check has NOT been run yet (see "What was and wasn't
verified" below). `tools/verify_math.py`'s finite-difference check
already validates the underlying math is internally consistent; this
separate script validates that `forward_numpy`'s specific
transcription of that math (weight file names, matrix orientations,
which array is `w` vs `w.T`) actually matches what infer.tl computes,
which a from-scratch derivation can't catch on its own.

## Continuous agent/environment interaction

This app runs the environment continuously with 2-4 kinds of controller
sharing the same loop, per the "agent(s) interacting continuously with
an environment" framing:

- **Human**: arrow keys / WASD steer one snake; the environment
  (movement, collision, food, growth) advances on a fixed tick
  regardless of whether a key was pressed that tick — a real continuous
  loop, not a turn-based one.
- **Single AI**: one snake, driven every tick by `agent.choose_ai_move`
  (trained network first, heuristic fallback) — a continuous
  agent/environment loop with no human in it at all.
- **Arena (multiple agents, one shared environment)**: 2-4 AI snakes on
  the SAME grid, each independently calling `choose_ai_move` every
  tick, competing for the same food and dying if they run into a wall,
  themselves, OR any other snake's body. This is the "multiple agents"
  case: every snake perceives the others as live obstacles
  (`other_blocked`, threaded through `engine.step_snake`,
  `heuristic_choose_move`, and the network's "enemy" input channel) and
  the environment resolves all of them every tick.

## The engine (`tools/engine.py`)

Plain-Python, snake-count-agnostic: every function operates on one
snake's body plus a `blocked` set of cells it must not enter (food
excluded), so the same collision/pathfinding code serves 1 snake
(single-player/single-AI) or N snakes sharing a board (arena) without
knowing which situation it's in. `SnakeState` is a small dataclass
(body, direction, alive, score, just_ate); `step_snake` is the one
function that actually mutates a snake by a tick, and is also what
`tools/generate_data.py`'s self-play and `tools/play.py`'s live loop
both call — no separate "simulate" vs. "real" path the way 2048 needs
(there's no TensorLang tensor-ops engine for the game rules here, only
for the policy net, so there's nothing to keep in sync).

Tested: 30-seed heuristic self-play runs (avg score ~45 on the 12x12
board before the solver eventually traps itself — see "Improving..."
below), a 500-episode randomized check that the identity symmetry
variant reproduces the untransformed board+heading exactly, and a
2000-tick single-AI stability run with no crashes or invalid states.

## Board encoding (`tools/agent.py::encode_state_onehot`)

One-hot per cell, 5 categories x 144 cells (12x12) = 720 inputs:
`empty`, `food`, `own head`, `own body`, `enemy` (any OTHER snake's
body, head included). The SAME function is used to label training data
and to run live inference, so the two can never drift apart — same
discipline as every other app's board encoding.

**Important limitation**: self-play (`tools/generate_data.py`) is
single-snake only — the `enemy` channel is always all-zero in every
training row. In arena mode, that channel IS populated with real
obstacles at inference time, so the network is being asked to
generalize past its training distribution there. This doesn't make
arena mode unsafe: `choose_ai_move`'s legality masking and the
heuristic fallback's BFS/flood-fill both treat `other_blocked` as real
obstacles regardless of what the network saw during training — it just
means the network's own play in arena mode may be more conservative or
less sharp around other snakes than its single-player play, until it's
fine-tuned on genuine multi-snake self-play data (see "Improving..."
below for that as a next step).

## Data generation (`tools/generate_data.py`)

2500 self-play `(body, food)` states (BFS+flood-fill self-play, 15%
random-safe-move exploration for state diversity, always labeled by a
fresh solver call — DAgger-style, same as 2048's `EXPLORE_PROB`),
expanded into 8 dihedral-symmetry variants each (independently
re-labeled — see above) = 20000 total rows, matching `train.tl`'s
hardcoded `(20000, 720)` / `(20000, 4)` shapes exactly. Deterministic
(fixed seed) — re-running reproduces byte-identical `data/*.npy`.

```bash
python3 apps/games/snake/tools/generate_data.py
```

Takes well under a minute on a modern CPU (pure Python — this never
touches the TensorLang compiler or a GPU, exactly like every other
app's `generate_data.py`).

## What was and wasn't verified

**Verified, no GPU needed:**
- The engine: 30-seed self-play runs, a 500-trial randomized check that
  `d4_coord_variants`'s identity transform round-trips exactly, and a
  2000-tick stability run.
- `train.tl` and `infer.tl` parse and TYPE CHECK against the real
  TensorLang compiler (`check.py`) — every declared shape, every
  inferred `_grad` shape, resolves correctly.
- The architecture's MATH: `tools/verify_math.py` hand-derives the
  forward+backward pass in NumPy on a small synthetic case and
  finite-difference-checks every gradient (worst-case relative error
  ~3e-8) — this validates the DESIGN, independent of whether
  TensorLang's own CUDA kernels implement each op correctly.
- `tools/generate_data.py` was actually run end-to-end (pure Python):
  the shipped `data/*.npy` is real, not a placeholder — 20000 rows from
  2500 base states x 8 symmetries (see `data/meta.json`).
- `agent.py`'s fallback path was exercised for real: with no `nvcc`
  installed, `infer.tl` genuinely fails to run (confirmed via direct
  `tensorlang.py` invocation), and `choose_ai_move` correctly catches
  that and falls back to `heuristic_choose_move` — confirmed across
  single-AI and multi-agent-arena runs.
- `run_inference_fast`'s in-process forward pass: tested end-to-end with
  freshly-initialized (untrained) weights through `choose_ai_move` and
  timed at ~0.17ms/move — confirms the mechanism works and is fast, but
  see below for what this does NOT confirm.
- `tools/play.py`: a headless smoke test (SDL dummy driver) exercises
  human/AI/arena `new_game`, `step_all` over hundreds of ticks
  (including real eating/growth/game-over transitions), and every draw
  function, with no exceptions.
- `promote_weights.py`/`rollback_weights.py`: tested end-to-end with
  placeholder weight files — promote backs up + copies correctly,
  rollback swaps correctly, running rollback twice returns to start.

**NOT verified — needs real GPU hardware:**
- That TensorLang's actual CUDA kernels for `matmul`/`add`/`relu`/
  `softmax`/`mse_loss` numerically agree with the NumPy reference above.
  Individual ops are documented elsewhere in this repo as
  hardware-verified (HANDOVER.md, via tic_tac_toe/2048), but this
  specific combination (720-wide input, two hidden layers, this exact
  loss) has never actually been compiled or run.
- Whether 8000 epochs at lr=4.0 (borrowed from 2048's already-tuned
  setting for a similarly-shaped problem) is a good setting here — see
  train.tl's header for the reasoning behind starting there, and
  "Improving..." below for what to check once a real loss curve exists.
- **Whether `run_inference_fast`'s NumPy forward pass actually matches
  infer.tl's real TensorLang/CUDA output** for genuinely trained
  (not just freshly-initialized) weights. The math is validated
  (`verify_math.py`) and the mechanism works (above), but a
  transcription slip (wrong weight file, transposed matmul) between
  `forward_numpy` and `infer.tl` couldn't be ruled out without actually
  comparing the two side by side — that's exactly what
  `tools/verify_infer_matches_numpy.py` does. Run it once on real GPU
  hardware after your first `--train` + `--promote`, before trusting the
  fast gameplay path.
- How strong the resulting network actually plays relative to the
  heuristic it's imitating. The heuristic itself scores ~45 on average
  (30-seed sample, see engine testing above) before eventually trapping
  itself — a stronger label source (deeper lookahead, or genuine
  multi-step safety simulation instead of the single-step approximation
  `heuristic_choose_move` uses) would raise the ceiling on what the
  network can ever imitate.

## Suggested first real run, on a machine with a GPU + CUDA toolkit

```bash
cd tensor-lang
bash build.sh --install               # if not already done
source python-env/bin/activate
python3 apps/games/snake/tools/verify_math.py       # no GPU needed, run first
python3 check.py apps/games/snake/train.tl          # no GPU needed
python3 check.py apps/games/snake/infer.tl          # no GPU needed
./apps/games/snake/run.sh --train                   # the actual GPU run
# inspect cache/apps/games/snake/train.tl/loss.npy —
# if it's dropped to a sensible plateau, then:
./apps/games/snake/run.sh --promote
python3 apps/games/snake/tools/verify_infer_matches_numpy.py   # confirms the fast
                                                                # gameplay path matches
                                                                # infer.tl's real output
./apps/games/snake/run.sh --play
```

If `--train` fails or produces garbage, `run.sh --play` still works —
`agent.py` falls back to the heuristic solver, so every mode (human,
single-AI, arena) is fully playable from the very first run either way.

## Improving the trained network's actual play quality

**Update (post-training feedback):** after a real GPU training run, the
trained single-AI mode was observed getting stuck in corners — walking
into a position with no safe way out. Root cause: `choose_ai_move`
originally only masked out moves that were an IMMEDIATE wall/self/enemy
collision; it never checked whether a move walked the snake into a
pocket with no room to escape a few steps later, the way
`heuristic_choose_move`'s flood-fill safety check does. Fixed by adding
`agent.non_trapping_moves`: before consulting the network, the pool of
candidate directions is filtered down to those that leave at least
`len(body)` cells of reachable open space (the same flood-fill check
the heuristic already used) — the network still picks which of THOSE
directions to go, it just can't pick a self-trapping one when a
non-trapping option exists. Verified against 4000 sampled real self-play
decision points: 13 had a genuinely trapping option among the
otherwise-legal moves, and the filter correctly excluded it in every
case. This does not fix the network's underlying path-planning
sharpness (see below) — it's a Python-side safety net in front of
whatever direction the network or heuristic proposes.

Beyond that, in roughly the order of impact per unit effort:

1. **More self-play data.** 2500 base states x 8 symmetries = 20000
   rows is a reasonable starting point but Snake's true state space
   (every body-shape x food-position combination on a 12x12 grid) is
   vastly larger. Raise `BASE_STATES` in `generate_data.py`, update
   `train.tl`'s `Tensor[f32, (N, ...)]` declarations to match the new
   N exactly, and re-run `init_weights.py`.
2. **A stronger label oracle.** `heuristic_choose_move`'s safety check
   is a single-step flood-fill approximation (see its docstring's note
   on the tail-vacating assumption) — a deeper simulation (actually
   walk the full BFS path and re-check safety at each step, or a
   multi-ply lookahead) would produce better labels for the network to
   imitate, at the cost of slower data generation.
3. **Multi-snake self-play**, to close the "enemy channel always zero
   in training" gap documented above: run self-play with 2+ snakes on
   the same board (reusing `engine.step_snake`'s existing
   `other_blocked` parameter — no engine changes needed), so the
   network sees real obstacle patterns in the `enemy` channel during
   training, not just at arena-mode inference time.
4. **More training, or a tuned learning rate**, once a real loss curve
   exists — see `train.tl`'s header for the reasoning behind the
   current lr=4.0/8000-epoch starting point.

## `cuInit failed` / `nvcc` not found (no GPU / driver issues)

This only matters for `--train` and for `tools/verify_infer_matches_numpy.py`
now — actual gameplay never touches TensorLang or a GPU at all (see
"In-process NumPy inference" above). If `--train` fails or a GPU has
driver issues, the game itself is unaffected: `agent.py`'s
`choose_ai_move` falls back to the heuristic solver whenever no weights
have been promoted, and the fallback (and the in-game `H` toggle to
force it) has no GPU dependency either way. See connect_four/NOTES.md's
"`cuInit failed`" section for the general driver-troubleshooting
checklist for the `--train` step itself, which applies here unchanged.
