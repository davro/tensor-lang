import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
import os
import logging
from datetime import datetime

# Set up logging
cache_dir = "cache/tests/portfolio.tl"
os.makedirs(cache_dir, exist_ok=True)
log_file = os.path.join(cache_dir, "execution.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Log start of execution
logger.info("Starting execution of portfolio.tl CUDA kernels")

# Define input tensors
returns = np.array([
    [0.12, 0.08, 0.15, 0.06],
    [0.08, 0.10, 0.12, 0.05],
    [0.06, 0.07, 0.09, 0.04],
    [0.15, 0.12, 0.18, 0.08],
    [0.05, 0.06, 0.04, 0.03],
    [0.02, 0.03, 0.04, 0.05]
], dtype=np.float32)

weights = np.array([[0.3], [0.25], [0.25], [0.2]], dtype=np.float32)

risk_factor = np.array([[1.2], [1.1], [0.8], [1.4], [0.6], [0.5]], dtype=np.float32)

# Log input tensors
logger.info(f"Input returns: shape {returns.shape}, sample {returns[:2]}")
logger.info(f"Input weights: shape {weights.shape}, sample {weights[:2]}")
logger.info(f"Input risk_factor: shape {risk_factor.shape}, sample {risk_factor[:2]}")

# Load the compiled CUDA kernel
kernel_so_path = os.path.join(cache_dir, "kernel.so")
if not os.path.exists(kernel_so_path):
    logger.error(f"CUDA kernel file not found at {kernel_so_path}")
    raise FileNotFoundError(f"CUDA kernel file not found at {kernel_so_path}")

# Load shared library
try:
    lib = ctypes.cdll.LoadLibrary(kernel_so_path)
    logger.info(f"Loaded CUDA kernel from {kernel_so_path}")
except OSError as e:
    logger.error(f"Failed to load kernel.so: {e}")
    raise

# Allocate GPU memory and copy input tensors
try:
    returns_gpu = cuda.mem_alloc(returns.nbytes)
    cuda.memcpy_htod(returns_gpu, returns)
    logger.info(f"Copied returns to GPU, shape: {returns.shape}")
except cuda.MemoryError as e:
    logger.error(f"Failed to allocate/copy returns to GPU: {e}")
    raise

try:
    weights_gpu = cuda.mem_alloc(weights.nbytes)
    cuda.memcpy_htod(weights_gpu, weights)
    logger.info(f"Copied weights to GPU, shape: {weights.shape}")
except cuda.MemoryError as e:
    logger.error(f"Failed to allocate/copy weights to GPU: {e}")
    raise

try:
    risk_factor_gpu = cuda.mem_alloc(risk_factor.nbytes)
    cuda.memcpy_htod(risk_factor_gpu, risk_factor)
    logger.info(f"Copied risk_factor to GPU, shape: {risk_factor.shape}")
except cuda.MemoryError as e:
    logger.error(f"Failed to allocate/copy risk_factor to GPU: {e}")
    raise

# Allocate GPU memory for outputs
portfolio_returns_shape = (5, 1)
portfolio_returns = np.empty(portfolio_returns_shape, dtype=np.float32)

try:
    portfolio_returns_gpu = cuda.mem_alloc(portfolio_returns.nbytes)
    logger.info(f"Allocated GPU memory for portfolio_returns, shape: {portfolio_returns_shape}")
except cuda.MemoryError as e:
    logger.error(f"Failed to allocate portfolio_returns GPU memory: {e}")
    raise

risk_adjusted_shape = (5, 1)
risk_adjusted = np.empty(risk_adjusted_shape, dtype=np.float32)
try:
    risk_adjusted_gpu = cuda.mem_alloc(risk_adjusted.nbytes)
    logger.info(f"Allocated GPU memory for risk_adjusted, shape: {risk_adjusted_shape}")
except cuda.MemoryError as e:
    logger.error(f"Failed to allocate risk_adjusted GPU memory: {e}")
    raise

# Define kernel function signatures
launch_matmul = lib.launch_matmul_portfolio_returns
launch_matmul.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

launch_div = lib.launch_div_risk_adjusted
launch_div.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
]

# Execute matmul kernel for portfolio_returns
M, N = returns.shape
P = weights.shape[1]
try:
    launch_matmul(
        ctypes.c_void_p(int(returns_gpu)),
        ctypes.c_void_p(int(weights_gpu)),
        ctypes.c_void_p(int(portfolio_returns_gpu)),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(P)
    )
    logger.info(f"Executed matmul for portfolio_returns, shape: {portfolio_returns_shape}")
except Exception as e:
    logger.error(f"Failed to execute matmul kernel: {e}")
    raise

# Copy portfolio_returns back to host
try:
    cuda.memcpy_dtoh(portfolio_returns, portfolio_returns_gpu)
    logger.info(f"Result portfolio_returns:\n{portfolio_returns}")
except cuda.MemoryError as e:
    logger.error(f"Failed to copy portfolio_returns from GPU: {e}")
    raise

# Save portfolio_returns to cache
portfolio_returns_path = os.path.join(cache_dir, "portfolio_returns.npy")
np.save(portfolio_returns_path, portfolio_returns)
logger.info(f"Saved portfolio_returns to {portfolio_returns_path}")

# Execute div kernel for risk_adjusted
size = risk_adjusted.size
try:
    launch_div(
        ctypes.c_void_p(int(portfolio_returns_gpu)),
        ctypes.c_void_p(int(risk_factor_gpu)),
        ctypes.c_void_p(int(risk_adjusted_gpu)),
        ctypes.c_int(size)
    )
    logger.info(f"Executed div for risk_adjusted, shape: {risk_adjusted_shape}")
except Exception as e:
    logger.error(f"Failed to execute div kernel: {e}")
    raise

# Copy risk_adjusted back to host
try:
    cuda.memcpy_dtoh(risk_adjusted, risk_adjusted_gpu)
    logger.info(f"Result risk_adjusted:\n{risk_adjusted}")
except cuda.MemoryError as e:
    logger.error(f"Failed to copy risk_adjusted from GPU: {e}")
    raise

# Save risk_adjusted to cache
risk_adjusted_path = os.path.join(cache_dir, "risk_adjusted.npy")
np.save(risk_adjusted_path, risk_adjusted)
logger.info(f"Saved risk_adjusted to {risk_adjusted_path}")

# Free GPU memory
try:
    returns_gpu.free()
    weights_gpu.free()
    portfolio_returns_gpu.free()
    risk_factor_gpu.free()
    risk_adjusted_gpu.free()
    logger.info("Freed all GPU memory")

    print(f'Portfolio: log saved to {log_file}')

except Exception as e:
    logger.error(f"Failed to free GPU memory: {e}")
    raise

# Log completion
logger.info("Execution completed successfully")
