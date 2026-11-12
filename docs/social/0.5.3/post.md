🚀 TensorLang v0.5.3 is out!
Native ML language that compiles tensor operations into GPU-accelerated CUDA kernels, a streamlined alternative to Python-heavy ML workflows for static-shape, high-performance tensor programs.

This release completes the v0.5.3 control-flow and reduction work:
✅ Tensor-valued `if` / `elif` / `else`
✅ `>=`, `<=`, and `!=` in conditions
✅ `backward()` inside selected branches
✅ `any()` and `all()` reductions
✅ 196/196 `.tl` tests passing
✅ Validated on 2× NVIDIA GTX 1080 Ti

TensorLang now supports a much more expressive programming model for real ML workflows, including:
• Reverse-mode autograd (FP32/FP16)
• Training loops with nested loops, rebinding, and `save()`
• `sgd` and `adam` optimizers
• FP16 master weights and loss scaling
• N-D tensor operations and batched matmul
• Elementwise kernel fusion
• NumPy zero-copy interop

Next up for v0.5.4: remaining loop-body parity, `batch_norm`/`instance_norm` autograd, branch consistency checks, and multiple backward passes.

Repo: https://github.com/davro/tensor-lang

#TensorLang #MachineLearning #CUDA #GPU #Compilers #AIEngineering #OpenSource #DeepLearning #Autograd #HighPerformanceComputing #NVIDIA #SoftwareEngineering
