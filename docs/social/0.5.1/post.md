TensorLang v0.5.1 ships today: Autograd Coverage Expansion.
In v0.5.0, mixed precision (FP16/FP32) and N-dimensional tensors were complete, but the backward pass only covered a subset of ops.

You could train through matmul, activations, and losses, but reshape, transpose, slice, concat, max, min, argmax, argmin, and comparisons all broke the gradient chain.

That's fixed now. Every one of those ops has a working backward pass, including N-D variants:

• reshape / transpose : gradient flows through shape changes
• slice: gradient scatters back into the source tensor
• concat: gradient splits along the concat axis
• max / min: gradient flows to all tied extrema
• argmax / argmin: one-hot gradient at the index
• greater / less / equal: zero gradient (non-differentiable)

Also fixed along the way:

• N-D correctness in matmul, linear, cross_entropy, and sum/mean/layer_norm backward paths (batched axes, tuple axis support)
• CUDA shared-memory naming conflict in argmax/argmin reduction kernels
• Compiler now records 12 previously untracked ops so the autograd graph sees them

Test count: 152 → 162 .tl tests, all passing. 31 Python tests unchanged.

What's still ahead for v0.5.2: loss scaling for FP16 training, FP16 master weights in Adam, for-loop body parity (normalisations, comparisons, slicing inside loops), and batched matmul gradients.

If you're working on compilers, CUDA, or ML infrastructure, I'd love to hear what's missing or what you'd prioritise next.

https://github.com/davro/tensor-lang

#MachineLearning #CUDA #Compilers #GPU #DeepLearning #OpenSource #Autograd #TensorOperations #MLInfrastructure
