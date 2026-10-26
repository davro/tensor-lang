Here is the complete updated `AUDIT.md` with broadcasting-aware fusion reflected.

```markdown
# TensorLang audit: updated

The audit's critical correctness items are now resolved. The codebase has **133/133 `.tl` tests** and **25/25 Python regression tests** passing (including the scoped-kernel tests), and `ruff check tensorlang/` is clean.

## Completed from the original audit

* Kernel library loaded by absolute path (fixes dlopen name collision across same-process compiles)
* Regression tests assert shape-mismatch wording for the type checker and compiler (operation, tensor, operands, shapes)
* Shape-mismatch messages name the op, the tensor being defined, both operands and their shapes (elementwise, comparison, matmul; type checker and compiler)
* `TestRunner` marked `__test__ = False` so pytest no longer warns about collecting it
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
* `matmul` replaced with a cuBLAS-backed implementation (preserves API, shapes, autograd, scoped kernel names)
* Built-in optimizers `sgd` and `adam` with compiler-managed state (multi-parameter, gradient zeroing/accumulation, pointer-swap rebinding)
* Elementwise kernel fusion for elementwise chains (add→relu, minus→relu, add→sigmoid, add→tanh, gelu, swish); supports non-broadcast and broadcasting modes (row/column/general broadcast); fused vs. unfused outputs and gradients verified bit-identical

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
* Annotated lets skip validation: the type checker trusts a declared type and never checks the expression, so shape errors on annotated lets only surface at compile/run time
* Absolute paths crash the compiler: `python3 tensorlang.py /tmp/bad.tl` fails with `FileExistsError` because the cache directory is built from the full path
* Shape-mismatch messages are still static for `linear`, `cross_entropy`, `mse_loss`, `reshape` and `concat`, and the `compiler.py` matmul `TypeError`s carry shapes but no tensor names
* Type-checker `matmul` indexes `shape[1]` directly, so a 1D operand raises `IndexError` instead of a clean error
* AST carries no source locations, so errors cannot report line numbers (needs Lark `propagate_positions` plus `ast_builder.py` changes)

### Performance and architecture

* Kernel fusion is limited to elementwise chains; reductions and matmul are not fused
* No zero-copy Python/PyTorch interop yet
* Tier-B multi-GPU single-program execution is not implemented
* Mixed precision is not implemented or validated

### Testing and tooling

* Hash-named `.so` per build, so recompiling edited source in one long-lived process can't pick up a stale library
* Configurable cache directory (`cache_dir` argument or `TENSORLANG_CACHE`) so tests don't depend on `chdir`
* Regression test that compiles two different programs from different directories in one process and checks each exports its own symbols
* `@EXPECT_FAILURE` checks exit code only
* No subprocess timeout
* Stale cache can still affect isolated runs
* Test discovery is non-recursive
* GPU round-robin is static by submission order
* Restrict pytest discovery to `tests/` (`testpaths`, `pythonpath` in `pyproject.toml`) so bare `pytest` works like `python -m pytest`

## Recommended next focus

1. Finish shape-mismatch messages (`linear`, `cross_entropy`, `mse_loss`, `reshape`, `concat`) and validate annotated lets in the type checker
2. Expand for-loop body coverage
3. Refactor `compiler.py` into codegen/execution/io mixins
4. For-loop limitations: `save()` in loops, alias resolution, nested loops
5. Zero-copy Python/PyTorch interop
6. Mixed precision (FP16/BF16) with cuBLAS tensor cores

## Bottom line

TensorLang is now in a much more consistent state: grammar, type checker, kernel generator, compiler, autograd, tests, lint, documentation, and scoped kernel names are aligned for the supported feature set. cuBLAS-backed matmul and first-class SGD/Adam optimizers make it usable for real training loops. Elementwise kernel fusion — including broadcasting chains (row/column/general modes) — now removes per-op launch overhead for common activation patterns, with fused and unfused results verified bit-identical. The next milestone should be **compiler maintainability (refactor) and training-loop completeness**, then interop and mixed precision.
```

### Key changes from the previous version
- Test count bumped to **133/133** (added `fusion_broadcast.tl`).
- Completed list: fusion now explicitly mentions broadcasting support (row/column/general modes) and bit-identical verification.
- Performance/architecture: replaced the stale "limited to non-broadcast" note with "limited to elementwise chains; reductions and matmul are not fused".
- Recommended next focus: added the `compiler.py` refactor and for-loop limitations (`save()`, aliases, nested loops) ahead of interop/mixed precision.
- Bottom line: updated to reflect broadcasting fusion and the new priority order.