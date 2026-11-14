
**TensorLang v0.5.7 is out — a transformer block.**

This release was driven by a single experiment: write a one-layer GPT-2-style forward pass using only existing TensorLang operations, then verify every number against NumPy.

It works.

In the process it exposed a silent broadcasting bug that has existed in **every previous version**, including v0.5.6.

**What’s new:**

• A complete one-layer pre-LN GPT-2-style block (embeddings, layer norm, causal attention, residuals, GELU FFN, output projection) at toy size, verified against float64 NumPy
• Causal mask implemented with pure broadcasting — no new ops required
• Attention gradients also match NumPy

**Critical fix:**

• Square matrix + vector broadcasting (`add`/`minus`/`mult`/`div`) was reading the vector per-row instead of per-column (NumPy behaviour). This produced plausible but wrong results with no error, and broke the agreement between forward values and gradients. Non-square cases were already correct, which is why it stayed hidden for so long.

**Test suite:** 303 `.tl` tests + 31 Python tests — all passing on 2× GTX 1080 Ti.

This is a meaningful milestone: TensorLang can now express a real transformer block and catch subtle numerical bugs that would otherwise go unnoticed.

Repo: https://github.com/davro/tensor-lang

Compiler / CUDA / ML infra people — what would you like to see next?

#TensorLang #MachineLearning #CUDA #GPU #Compilers #Transformers #GPT #AIEngineering #OpenSource #DeepLearning #Autograd #HighPerformanceComputing #NVIDIA #SoftwareEngineering #MLInfrastructure
