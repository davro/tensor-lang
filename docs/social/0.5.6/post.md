**TensorLang v0.5.6 is out — closing the gaps left by v0.5.5.**

This release turns previously silent crashes and wrong answers into clear compile errors, finishes `batch_norm`, and fixes two subtle loop bugs discovered by a new acceptance app.

**Key changes:**

• Unsupported comparison shapes and f16/f64 dtypes are now hard compile errors (`[E-SHAPE-MISMATCH]`, `[E-COMPARISON-DTYPE]`, `[E-UNSUPPORTED-DTYPE]`)
• `batch_norm` now supports optional `gamma`/`beta`, correct N-D (NCHW) channel handling, and matching gradients
• Two loop bugs fixed: `if` after a loop’s first `backward()` now runs, and statements after a loop containing an `if` execute the correct kernels
• Undeclared float64 `.npy`/`.npz` loads are safely inferred and converted to f32
• New acceptance app (`adaptive_lr`) and sweep tools to catch these edge cases early

**Test suite:** 294 `.tl` tests + 31 Python tests — all passing on 2× GTX 1080 Ti.

TensorLang keeps getting more reliable as a native, CUDA-powered ML language for high-performance tensor programs.

Repo: https://github.com/davro/tensor-lang

Feedback from compiler, CUDA, or ML infrastructure people is always welcome — what should we prioritise next?

#TensorLang #MachineLearning #CUDA #GPU #Compilers #AIEngineering #OpenSource #DeepLearning #Autograd #HighPerformanceComputing #NVIDIA #SoftwareEngineering #MLInfrastructure
