🚀 TensorLang v0.5.4 is out!
This release closes the remaining parity work for control flow, loop bodies, autograd, and multi-backward compilation:

✅ if / elif / else branch shape and dtype consistency checks
✅ Scalar literals in branch conditions
✅ User-defined functions and load() / save() file I/O in for-loop bodies
✅ batch_norm and instance_norm autograd
✅ Multiple backward() statements per compilation
✅ 216/216 .tl tests + 31/31 Python tests passing

TensorLang continues to compile tensor operations to CUDA kernels with N-D support, mixed precision, loss scaling, and built-in optimizers.

Repo: https://github.com/davro/tensor-lang

#TensorLang #MachineLearning #AI #CUDA #GPU #Compilers #DeepLearning #OpenSource #Autograd #MixedPrecision #MLSystems
