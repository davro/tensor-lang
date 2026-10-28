"""
TensorLang Generator ComparisonsMixin
Kernel ComparisonsMixin

File: tensorlang/generators/comparisons.py
"""

import numpy as np

from .base import BaseGenerator


class ComparisonsMixin(BaseGenerator):
    def greater_broadcast(self, op_type, name, arg1, arg2, shape1, shape2, output_shape, size, dtype="f32"):
        # Comparisons always produce f32 output (0.0/1.0)
        out_ctype = "float"
        inp_ctype = self._ctype(dtype)
        include = self._include(dtype)
        a_read = self._read(dtype, "A", "idx")
        b_read = self._read(dtype, "B", "idx")

        if shape1 == shape2:
            kernel = include + f"""
__global__ void greater_kernel_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = ({a_read} > {b_read}) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_greater_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int size) {{
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
            a_read_bc = self._read(dtype, "A", "i * cols + j")
            b_read_bc = self._read(dtype, "B", "j")
            kernel = include + f"""
__global__ void greater_broadcast_kernel_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = ({a_read_bc} > {b_read_bc}) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_greater_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    greater_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_broadcast", name, arg1, arg2, rows, cols)

    def less_broadcast(self, op_type, name, arg1, arg2, shape1, shape2, output_shape, dtype="f32"):
        out_ctype = "float"
        inp_ctype = self._ctype(dtype)
        include = self._include(dtype)
        a_read = self._read(dtype, "A", "idx")
        b_read = self._read(dtype, "B", "idx")

        if shape1 == shape2:
            size = int(np.prod([int(dim) for dim in output_shape]))
            kernel = include + f"""
__global__ void less_kernel_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = ({a_read} < {b_read}) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_less_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int size) {{
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
            a_read_bc = self._read(dtype, "A", "i * cols + j")
            b_read_bc = self._read(dtype, "B", "j")
            kernel = include + f"""
__global__ void less_broadcast_kernel_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = ({a_read_bc} < {b_read_bc}) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_less_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    less_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_broadcast", name, arg1, arg2, rows, cols)

    def equal(self, op_type, name, arg1, arg2, shape1, shape2, output_shape, dtype="f32"):
        out_ctype = "float"
        inp_ctype = self._ctype(dtype)
        include = self._include(dtype)
        a_read = self._read(dtype, "A", "idx")
        b_read = self._read(dtype, "B", "idx")

        if shape1 == shape2:
            size = int(np.prod([int(dim) for dim in output_shape]))
            kernel = include + f"""
__global__ void equal_kernel_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = (fabsf({a_read} - {b_read}) < 1e-6f) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_equal_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int size) {{
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
            a_read_bc = self._read(dtype, "A", "i * cols + j")
            b_read_bc = self._read(dtype, "B", "j")
            kernel = include + f"""
__global__ void equal_broadcast_kernel_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = (fabsf({a_read_bc} - {b_read_bc}) < 1e-6f) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_equal_{name}({inp_ctype}* A, {inp_ctype}* B, {out_ctype}* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    equal_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_broadcast", name, arg1, arg2, rows, cols)
