// apps/trading/gmx_charts/indicators.tl
//
// Composite trading signal: SMA(10) / SMA(30) crossover, gated by RSI(14)
// so a crossover is ignored while the market is already overbought/
// oversold in that direction. Reads the rolling (200, 8) bar window
// tools/tensor_store.py maintains and writes a single composite decision
// back for tools/signal_agent.py to read.
//
// Bar window column layout (documented convention, not a TensorLang
// type — see tools/tensor_store.py for the source of truth):
//   0 timestamp | 1 open | 2 high | 3 low | 4 close
//   5 volume_usd | 6 volume_confirmed | 7 trade_count
//
// Composite output encoding (signal.npy, a single float):
//   +1.0 = buy   -1.0 = sell   0.0 = hold / gated out
//
// Deliberately arithmetic rather than branching: TensorLang's grammar
// defines if/elif/else and comparison_expr, but ast_builder.py does not
// yet dispatch 'if_statement' — it parses but nothing executes. greater/
// less/equal ARE fully wired to real CUDA kernels (kernel_generator.py),
// so this composes the decision from tensor arithmetic instead of
// waiting on that gap to be closed. Revisit once if/elif lands — this
// will read more clearly as explicit branches.
//
// Run directly, for manual testing:
//   python3 tensorlang.py apps/trading/gmx_charts/indicators.tl

let window: Tensor[f64, (200, 8)] = load("cache/apps/trading/gmx_charts/indicators.tl/window.npy")

// --- SMA(10) / SMA(30) on closing price (column 4) ---
let fast_window = window[190:200, 4:5]
let slow_window = window[170:200, 4:5]
let sma_fast = mean(fast_window, axis=0)
let sma_slow = mean(slow_window, axis=0)

// --- RSI(14) on closing price ---
let closes_prev = window[185:199, 4:5]
let closes_curr = window[186:200, 4:5]
let deltas       = minus(closes_curr, closes_prev)
let gains        = relu(deltas)
let inv_deltas   = minus(closes_prev, closes_curr)
let losses       = relu(inv_deltas)
let avg_gain     = mean(gains, axis=0)
let avg_loss     = mean(losses, axis=0)
// NOTE: if avg_loss is exactly 0 (an all-gain window) this divides by
// zero. Left as-is deliberately — see tools/signal_agent.py, which
// checks for this on the host side before trusting the result, rather
// than silently smoothing it over here with a made-up epsilon.
let rs           = div(avg_gain, avg_loss)
let one          = fill(1.0, (1,))
let hundred      = fill(100.0, (1,))
let rsi          = minus(hundred, div(hundred, add(one, rs)))

// --- Gate thresholds (materialised as tensors: greater/less require
//     bound NAMEs on both sides, not literal NUMBERs) ---
let overbought = fill(70.0, (1,))
let oversold   = fill(30.0, (1,))

let is_uptrend     = greater(sma_fast, sma_slow)
let is_downtrend   = less(sma_fast, sma_slow)
let not_overbought = less(rsi, overbought)
let not_oversold   = greater(rsi, oversold)

let gated_buy  = mult(is_uptrend, not_overbought)
let gated_sell = mult(is_downtrend, not_oversold)
let composite  = minus(gated_buy, gated_sell)

save(composite, "signal.npy")
