import numpy as np

from .base import BaseGenerator


class ComparisonsMixin(BaseGenerator):
    def greater_broadcast(self, op_type, name, arg1, arg2, shape1, shape2, output_shape, size):
        if shape1 == shape2:
            kernel = f"""
__global__ void greater_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = (A[idx] > B[idx]) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_greater_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    greater_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (op_type, name, arg1, arg2, size)
        else:
            rows, cols = output_shape
            kernel = f"""
__global__ void greater_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = (A[i * cols + j] > B[j]) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_greater_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    greater_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_broadcast", name, arg1, arg2, rows, cols)

    def less_broadcast(self, op_type, name, arg1, arg2, shape1, shape2, output_shape):
        if shape1 == shape2:
            size = int(np.prod([int(dim) for dim in output_shape]))
            kernel = f"""
__global__ void less_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = (A[idx] < B[idx]) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_less_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    less_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}", name, arg1, arg2, size)
        else:
            rows, cols = output_shape
            kernel = f"""
__global__ void less_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = (A[i * cols + j] < B[j]) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_less_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    less_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_broadcast", name, arg1, arg2, rows, cols)

    def equal(self, op_type, name, arg1, arg2, shape1, shape2, output_shape):
        if shape1 == shape2:
            size = int(np.prod([int(dim) for dim in output_shape]))
            kernel = f"""
__global__ void equal_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = (fabsf(A[idx] - B[idx]) < 1e-6f) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_equal_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    equal_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}", name, arg1, arg2, size)
        else:
            rows, cols = output_shape
            kernel = f"""
__global__ void equal_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = (fabsf(A[i * cols + j] - B[j]) < 1e-6f) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_equal_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    equal_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_broadcast", name, arg1, arg2, rows, cols)

