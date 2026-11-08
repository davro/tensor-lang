// ============================================================================
// TensorLang: a numeric literal on either side of an infix comparison
// ============================================================================
// x = [1,2,3]
//   x > 1.5   -> [0,1,1]
//   x != 1.0  -> [0,1,1]
//   2.5 > x   -> [1,1,0]   (literal on the left)
// The literal is lowered to a (1,) tensor that broadcasts. Expected values derived by hand.
//
// @EXPECTED
// {"big": [0.0, 1.0, 1.0], "notone": [0.0, 1.0, 1.0], "small": [1.0, 1.0, 0.0]}
let x: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let big = x > 1.5
let notone = x != 1.0
let small = 2.5 > x
