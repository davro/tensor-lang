// ============================================================================
// TensorLang: Slice in ML Pipeline
// ============================================================================
// Description: Real-world ML use case - batch processing with slicing
//
// Architecture Pattern:
//   Input data (4x3) -> Slice batches -> Process -> Combine
//
// Test Cases:
//   1. Split data into batches using slicing
//   2. Process each batch independently
//   3. Extract features from different data segments
//   4. Slice for train/validation splits
//
// Expected Output Variables:
//   batch1_processed, batch2_processed, feature_subset, train_split
//
// @EXPECTED
// {
//   "batch1_processed": [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]],
//   "batch2_processed": [[49.0, 64.0, 81.0], [100.0, 121.0, 144.0]],
//   "feature_subset": [[2.0, 3.0], [5.0, 6.0], [8.0, 9.0], [11.0, 12.0]],
//   "train_split": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
// }

// Input data matrix (4x3) - represents 4 samples with 3 features each
let data : Tensor[f32, (4, 3)] = [
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    10.0, 11.0, 12.0
]

// Test Case 1: Extract first batch (rows 0-1)
let batch1 = data[0:2, 0:3]
let batch1_processed = mult(batch1, batch1)  // Element-wise square

// Test Case 2: Extract second batch (rows 2-3)
let batch2 = data[2:4, 0:3]
let batch2_processed = mult(batch2, batch2)

// Test Case 3: Feature extraction - select specific columns
let feature_subset = data[0:4, 1:3]  // Select features 1 and 2 from all samples

// Test Case 4: Train/validation split (first 3 rows for training)
let train_split = data[0:3, 0:3]

// Additional: Validation split (last row)
// let val_split = data[3:4, 0:3]