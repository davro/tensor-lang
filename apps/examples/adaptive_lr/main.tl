// Branching trainer: two models, two losses, two optimiser steps, and a branch inside a loop.
// Demonstrates: for + if/else with an optimiser step in each branch, two backward() passes per
// iteration, and a numeric literal compared against a loss (`if loss1 > 1.0`).
//
// Model 1 starts far from its target, so its loss is above the threshold in iterations 0 and 1
// (large learning rate) and below it from iteration 2 on (small learning rate). Model 2 always
// takes the same step. Per-iteration loss1: 30, 1.875, 0.117, 0.085, 0.061, 0.044.

let x:  Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y1: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let y2: Tensor[f32, (4, 1)] = [[3.0], [6.0], [9.0], [12.0]]
let w1: Tensor[f32, (1, 1)] = [[0.0]] with grad
let w2: Tensor[f32, (1, 1)] = [[0.5]] with grad

for epoch in range(6) {
    // Model 1 fits y1 = 2x. Large steps while its loss is above the threshold, small steps below it.
    let p1    = matmul(x, w1)
    let loss1 = mse_loss(p1, y1)
    backward(loss1)
    if loss1 > 1.0 {
        sgd(w1, w1_grad, 0.05)
    } else {
        sgd(w1, w1_grad, 0.01)
    }

    // Model 2 fits y2 = 3x with its own loss and its own optimiser step, every iteration.
    let p2    = matmul(x, w2)
    let loss2 = mse_loss(p2, y2)
    backward(loss2)
    sgd(w2, w2_grad, 0.02)
}

// Post-loop evaluation (distinct names to avoid kernel symbol clash)
let p1_final    = matmul(x, w1)
let loss1_final = mse_loss(p1_final, y1)
let p2_final    = matmul(x, w2)
let loss2_final = mse_loss(p2_final, y2)
