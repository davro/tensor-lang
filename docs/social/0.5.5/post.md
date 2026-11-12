TensorLang v0.5.5 is out - Focused on robustness.
This release is about making what already exists reliable: clearer errors, well-defined edge-case semantics, and a set of bugs in branch and loop autograd that are now fixed.

Key improvements:
• Stable diagnostic codes ([E-SHAPE-MISMATCH], [E-NAN-IN-CONDITION], [E-SCALAR-LITERAL], etc.) with helpful messages
• Bare scalar literals can no longer be bound with let (now a clear compile error)
• NaN in if/elif conditions is a runtime error instead of silently taking the else branch
• Empty tensors in conditions are rejected at compile time
• Comparison operators (>=, <=, !=, etc.) now work as first-class expressions
• backward() correctly works in any branch, nested branches, and multiple times inside loops
• Fixed several silent gradient bugs where only the first backward() ran, or wrong branches contributed to the gradient

Test suite: 268 .tl tests + 31 Python tests — all passing on 2× GTX 1080 Ti.

TensorLang continues to mature as a native, CUDA-powered ML language for high-performance tensor programs.

Repo: https://github.com/davro/tensor-lang

If you’re working on compilers, CUDA kernels, or ML infrastructure, I’d love your feedback on what’s missing or what you’d prioritise next.

#TensorLang #MachineLearning #CUDA #GPU #Compilers #AIEngineering #OpenSource #DeepLearning #Autograd #HighPerformanceComputing #NVIDIA #SoftwareEngineering #MLInfrastructure
