// Test cross entropy loss for classification
// Predictions: softmax probabilities for 3 samples, 3 classes each
// Targets: class indices [1, 0, 2]
// Expected: -log(prob of correct class) averaged over batch

let predictions: Tensor[f32, (3, 3)] = [[0.2, 0.7, 0.1],    // Sample 1: class 1 (0.7 prob)
                                        [0.6, 0.3, 0.1],    // Sample 2: class 0 (0.6 prob)
                                        [0.1, 0.2, 0.7]]    // Sample 3: class 2 (0.7 prob)

let targets: Tensor[f32, (3,)] = [1.0, 0.0, 2.0]           // True class labels

let loss = cross_entropy(predictions, targets)
loss
