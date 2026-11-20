#!/usr/bin/env python3
"""Measure, on YOUR GPU, how accurate the Adam bias-correction terms are when computed the way the kernel does it:
    1.0f - powf(beta, (float)t)            (float32, on the device)
against the exact value from the same float32 beta in float64.   python3 tools/check_adam_bias.py
For beta2 = 0.999 the true value is ~1e-3 * t, so any absolute error in powf is magnified ~1000x in the subtraction.
"""
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void bc(float beta, float* out, int n) {
    int t = threadIdx.x + 1;
    if (t <= n) out[t - 1] = 1.0f - powf(beta, (float)t);
}
""")
bc = mod.get_function("bc")
N = 12
print(f"{'beta':>7} {'t':>3} {'kernel 1-powf':>16} {'exact (f64)':>16} {'relative error':>15}")
worst = {}
for beta in (0.9, 0.999):
    out = np.zeros(N, np.float32)
    bc(np.float32(beta), cuda.Out(out), np.int32(N), block=(N, 1, 1), grid=(1, 1))
    b32 = float(np.float32(beta))
    for t in range(1, N + 1):
        exact = 1.0 - b32 ** t
        rel = abs(float(out[t - 1]) - exact) / exact
        worst[beta] = max(worst.get(beta, 0), rel)
        if t <= 5: print(f"{beta:7} {t:3d} {float(out[t - 1]):16.10f} {exact:16.10f} {rel:15.2e}")
print(f"\nworst relative error over t=1..{N}:  beta1=0.9: {worst[0.9]:.2e}   beta2=0.999: {worst[0.999]:.2e}")
print("A relative error r here becomes about r/2 in every Adam step (it sits under a square root for beta2);\n"
      "at lr=0.02 that is ~lr*r/2 per step per weight, so 1e-4 -> ~1e-6 per step.")
