# TensorLang audit: updated

The audit’s critical correctness items are now resolved. The codebase is at **125/125 tests passing** (`122` original + `3` scoped-kernel regression tests) and `ruff check tensorlang/` is clean.

## Completed from the original audit

* Kernel library loaded by absolute path (fixes dlopen name collision across same-process compiles)
* Broadcast `equal` dispatch fixed
* `(1,1)` scalar broadcast dispatch fixed
* `transpose(..., axes=...)` implemented
* `layer_norm(..., axis=0)` fixed
* `linear` wired into autograd
* `layer_norm` wired into autograd
* Full `argmin` / `argmax` reductions implemented
* `gelu` and `swish` forward + autograd implemented
* `cross_entropy` wired into autograd
* `build.sh --lint` no longer exits the sourcing shell
* Lint cleanup completed
* README and HANDOVER updated to match implementation
* Scoped kernel names implemented for loop bodies

## Remaining limitations / future work

### Language and compiler

* `if` / `elif` / `else` is grammar-only and not executed
* `any()` / `all()` are grammar-only
* `f64` is parsed but effectively `f32`
* For-loop bodies support only a subset of operations
* `save()` and alias resolution are not supported inside loops
* Nested loops are not supported
* Most kernels are 1D/2D only
* Axis support remains limited for several ops
* `batch_norm` has no gamma/beta affine parameters
* `--transpile` (WGSL) is not functional

### Performance and architecture

* `matmul` is still a naive CUDA kernel, not cuBLAS/tiled
* No kernel fusion yet
* No zero-copy Python/PyTorch interop yet
* Tier-B multi-GPU single-program execution is not implemented
* Mixed precision is not implemented or validated

### Testing and tooling

* Hash-named .so per build, so recompiling edited source in one long-lived process can’t pick up a stale library
* Configurable cache directory (cache_dir argument or TENSORLANG_CACHE) so tests don’t depend on chdir
* `@EXPECT_FAILURE` checks exit code only
* No subprocess timeout
* Stale cache can still affect isolated runs
* Test discovery is non-recursive
* GPU round-robin is static by submission order

## Recommended next focus

1. Better shape-mismatch error messages
2. Expand for-loop body coverage
3. cuBLAS-backed `matmul`
4. Kernel fusion
5. Zero-copy Python/PyTorch interop

## Bottom line

TensorLang is now in a much more consistent state: grammar, type checker, kernel generator, compiler, autograd, tests, lint, documentation, and scoped kernel names are aligned for the supported feature set. The next milestone should be **usability and performance**, not breadth.
