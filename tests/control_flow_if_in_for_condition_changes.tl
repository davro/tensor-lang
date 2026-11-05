// ============================================================================
// TensorLang: if inside a for loop, condition flips between iterations (EXPLORATORY)
// ============================================================================
// x climbs 0 -> 1 -> 2, then `x < limit` (2 < 2) turns false and x stays 2
// for the remaining iterations: final x = [2.0]. The existing for/if test
// cannot tell, because both of its branches give the same answer. If the
// condition result is read from a stale host cache after the first iteration,
// the branch keeps firing and x ends at [4.0]. Expected value derived by hand.
//
// @EXPECTED
// {"x": [2.0]}
let x: Tensor[f32, (1,)] = [0.0]
let one: Tensor[f32, (1,)] = [1.0]
let limit: Tensor[f32, (1,)] = [2.0]
for i in range(4) {
    if x < limit {
        let x_new = add(x, one)
        x = x_new
    }
}
x
