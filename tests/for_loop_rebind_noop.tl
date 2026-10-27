let w: Tensor[f32, (1, 1)] = fill(0.0, (1, 1))

for i in range(3) {
    w = w   // no-op rebind (same storage)
}

save(w, "w.npy")

// @EXPECTED
// { "w": [[0.0]] }

