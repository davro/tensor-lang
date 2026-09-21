// apps/trading/gmx_charts/indicators_no_gate.tl
//
// A/B baseline for indicators.tl: identical SMA(10)/SMA(30) crossover,
// with the RSI(70/30) gate removed entirely. Exists to answer one
// question directly — is the RSI gate helping or hurting — by giving
// backtest.py something to compare indicators.tl against on the exact
// same historical window (see tools/backtest.py's --entry flag and
// --save-snapshot/--load-snapshot).
//
// Composite output encoding, same as indicators.tl:
//   +1.0 = buy   -1.0 = sell   0.0 = hold

let window: Tensor[f64, (200, 8)] = load("cache/apps/trading/gmx_charts/indicators_no_gate.tl/window.npy")

let fast_window = window[190:200, 4:5]
let slow_window = window[170:200, 4:5]
let sma_fast = mean(fast_window, axis=0)
let sma_slow = mean(slow_window, axis=0)

let is_uptrend = greater(sma_fast, sma_slow)
let is_downtrend = less(sma_fast, sma_slow)
let composite = minus(is_uptrend, is_downtrend)

save(composite, "signal.npy")
