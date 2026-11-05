// ============================================================================
// TensorLang: top-level alias tracks its source across a branch rebind
// ============================================================================
// compiler.py deliberately re-resolves every top-level `let a = z` alias after
// execution, so an alias is another NAME for the variable, not a snapshot.
// `a` is declared before the branch, the branch rebinds z to x, and `a`
// follows: a = [5.0], z = [5.0]. If snapshot semantics are ever wanted, they
// need an explicit copy, not an alias. Documented behaviour, pinned by this test.
// Expected value derived by hand and confirmed against the observed failure
// of the earlier snapshot-expecting version (a came back as 5.0).
//
// @EXPECTED
// {"a": [5.0], "z": [5.0]}
let z: Tensor[f32, (1,)] = [1.0]
let x: Tensor[f32, (1,)] = [5.0]
let a = z
if x > z {
    let z = x
} else {
    let z = z
}
a
z
