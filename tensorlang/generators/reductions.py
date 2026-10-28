"""
TensorLang Generator ReductionMixin
Kernel ReductionMixin

File: tensorlang/generators/reductions.py
"""

import numpy as np

from .base import BaseGenerator


class ReductionMixin(BaseGenerator):
    def softmax(self, op_type, name, tensor_name, input_shape, axis, dtype="f32"):
        if axis is None:
            axis = len(input_shape) - 1

        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)
        out_write = lambda expr: self._write(dtype, expr)

        if len(input_shape) == 2 and axis == 1:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void softmax_kernel_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float max_val = {inp_read("row * cols")};
        for (int col = 1; col < cols; col++) {{
            max_val = fmaxf(max_val, {inp_read("row * cols + col")});
        }}

        float sum_exp = 0.0f;
        for (int col = 0; col < cols; col++) {{
            float exp_val = expf({inp_read("row * cols + col")} - max_val);
            output[row * cols + col] = {out_write("exp_val")};
            sum_exp += exp_val;
        }}

        for (int col = 0; col < cols; col++) {{
            float val = {inp_read("row * cols + col")};
            // recompute from stored exp (already in output as float-cast)
            output[row * cols + col] = {out_write(f"output[row * cols + col] / sum_exp")};
        }}
    }}
}}
extern "C" void launch_softmax_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    softmax_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = ('softmax', name, tensor_name, None, rows, cols, axis)
            return kernel, kernel_info

        elif len(input_shape) == 1:
            size = int(input_shape[0])
            kernel = include + f"""
__global__ void softmax_kernel_{name}({ctype}* input, {ctype}* output, int size) {{
    if (threadIdx.x == 0 && blockIdx.x == 0) {{
        float max_val = {inp_read("0")};
        for (int i = 1; i < size; i++) {{
            max_val = fmaxf(max_val, {inp_read("i")});
        }}

        float sum_exp = 0.0f;
        for (int i = 0; i < size; i++) {{
            float exp_val = expf({inp_read("i")} - max_val);
            output[i] = {out_write("exp_val")};
            sum_exp += exp_val;
        }}

        for (int i = 0; i < size; i++) {{
            output[i] = {out_write(f"output[i] / sum_exp")};
        }}
    }}
}}
extern "C" void launch_softmax_{name}({ctype}* input, {ctype}* output, int size) {{
    dim3 block(1);
    dim3 grid(1);
    softmax_kernel_{name}<<<grid, block>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = ('softmax_1d', name, tensor_name, None, size)
            return kernel, kernel_info
        else:
            raise ValueError(f"Unsupported softmax shape {input_shape} or axis {axis}")

    def sum(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))

            if dtype == "f16":
                # Single-block reduction for f16 (accumulate in float, cast result)
                kernel = include + f"""
__global__ void sum_full_kernel_{name}({ctype}* input, {ctype}* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float partial = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {{
        partial += {inp_read("i")};
    }}
    sdata[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        output[0] = {self._write(dtype, "sdata[0]")};
    }}
}}
extern "C" void launch_sum_{name}({ctype}* input, {ctype}* output, int size) {{
    dim3 block(256);
    dim3 grid(1);
    int shared_size = block.x * sizeof(float);
    sum_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            else:
                kernel = include + f"""
__global__ void sum_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        float* addr = (float*)output;
        atomicAdd(addr, sdata[0]);
    }}
}}
extern "C" void launch_sum_{name}(float* input, float* output, int size) {{
    float zero = 0.0f;
    cudaMemcpy(output, &zero, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    sum_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_full", name, tensor_name, None, size)

        elif len(input_shape) == 2 and axis == 1:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void sum_axis_kernel_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            sum += {inp_read("row * cols + col")};
        }}
        output[row] = {self._write(dtype, "sum")};
    }}
}}
extern "C" void launch_sum_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    sum_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis", name, tensor_name, None, rows, cols, axis)

        elif len(input_shape) == 2 and axis == 0:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void sum_axis0_kernel_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            sum += {inp_read("row * cols + col")};
        }}
        output[col] = {self._write(dtype, "sum")};
    }}
}}
extern "C" void launch_sum_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    sum_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

    def mean(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))

            if dtype == "f16":
                kernel = include + f"""
__global__ void mean_full_kernel_{name}({ctype}* input, {ctype}* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float partial = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {{
        partial += {inp_read("i")};
    }}
    sdata[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        output[0] = {self._write(dtype, "sdata[0] / (float)size")};
    }}
}}
extern "C" void launch_mean_{name}({ctype}* input, {ctype}* output, int size) {{
    dim3 block(256);
    dim3 grid(1);
    int shared_size = block.x * sizeof(float);
    mean_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            else:
                kernel = include + f"""
__global__ void mean_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        float* addr = (float*)output;
        atomicAdd(addr, sdata[0]);
    }}
}}
extern "C" void launch_mean_{name}(float* input, float* output, int size) {{
    float zero = 0.0f;
    cudaMemcpy(output, &zero, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    mean_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}

    float mean_val;
    cudaMemcpy(&mean_val, output, sizeof(float), cudaMemcpyDeviceToHost);
    mean_val /= size;
    cudaMemcpy(output, &mean_val, sizeof(float), cudaMemcpyHostToDevice);
}}
"""
            return kernel, (f"{op_type}_full", name, tensor_name, None, size)

        elif len(input_shape) == 2 and axis == 1:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void mean_axis_kernel_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            sum += {inp_read("row * cols + col")};
        }}
        output[row] = {self._write(dtype, "sum / (float)cols")};
    }}
}}
extern "C" void launch_mean_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    mean_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis", name, tensor_name, None, rows, cols, axis)

        elif len(input_shape) == 2 and axis == 0:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void mean_axis0_kernel_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            sum += {inp_read("row * cols + col")};
        }}
        output[col] = {self._write(dtype, "sum / (float)rows")};
    }}
}}
extern "C" void launch_mean_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    mean_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

    def max(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))

            if dtype == "f16":
                kernel = include + f"""
__global__ void max_full_kernel_{name}({ctype}* input, {ctype}* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float partial = -FLT_MAX;
    for (int i = tid; i < size; i += blockDim.x) {{
        partial = fmaxf(partial, {inp_read("i")});
    }}
    sdata[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        output[0] = {self._write(dtype, "sdata[0]")};
    }}
}}
extern "C" void launch_max_{name}({ctype}* input, {ctype}* output, int size) {{
    dim3 block(256);
    dim3 grid(1);
    int shared_size = block.x * sizeof(float);
    max_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            else:
                kernel = include + f"""
__global__ void max_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : -FLT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        float* addr = (float*)output;
        int* int_addr = (int*)addr;
        int old_val, new_val;
        do {{
            old_val = *int_addr;
            new_val = __float_as_int(fmaxf(__int_as_float(old_val), sdata[0]));
        }} while (atomicCAS(int_addr, old_val, new_val) != old_val);
    }}
}}
extern "C" void launch_max_{name}(float* input, float* output, int size) {{
    float neg_max = -FLT_MAX;
    cudaMemcpy(output, &neg_max, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    max_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_full", name, tensor_name, None, size)

        elif len(input_shape) == 2 and axis == 1:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void max_axis_kernel_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float max_val = {inp_read("row * cols")};
        for (int col = 1; col < cols; col++) {{
            max_val = fmaxf(max_val, {inp_read("row * cols + col")});
        }}
        output[row] = {self._write(dtype, "max_val")};
    }}
}}
extern "C" void launch_max_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    max_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis", name, tensor_name, None, rows, cols, axis)

        elif len(input_shape) == 2 and axis == 0:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void max_axis0_kernel_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float max_val = {inp_read("col")};
        for (int row = 1; row < rows; row++) {{
            max_val = fmaxf(max_val, {inp_read("row * cols + col")});
        }}
        output[col] = {self._write(dtype, "max_val")};
    }}
}}
extern "C" void launch_max_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    max_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

    def min(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))

            if dtype == "f16":
                kernel = include + f"""
__global__ void min_full_kernel_{name}({ctype}* input, {ctype}* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float partial = FLT_MAX;
    for (int i = tid; i < size; i += blockDim.x) {{
        partial = fminf(partial, {inp_read("i")});
    }}
    sdata[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        output[0] = {self._write(dtype, "sdata[0]")};
    }}
}}
extern "C" void launch_min_{name}({ctype}* input, {ctype}* output, int size) {{
    dim3 block(256);
    dim3 grid(1);
    int shared_size = block.x * sizeof(float);
    min_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            else:
                kernel = include + f"""
__global__ void min_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : FLT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        float* addr = (float*)output;
        int* int_addr = (int*)addr;
        int old_val, new_val;
        do {{
            old_val = *int_addr;
            new_val = __float_as_int(fminf(__int_as_float(old_val), sdata[0]));
        }} while (atomicCAS(int_addr, old_val, new_val) != old_val);
    }}
}}
extern "C" void launch_min_{name}(float* input, float* output, int size) {{
    float pos_max = FLT_MAX;
    cudaMemcpy(output, &pos_max, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    min_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_full", name, tensor_name, None, size)

        elif len(input_shape) == 2 and axis == 1:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void min_axis_kernel_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float min_val = {inp_read("row * cols")};
        for (int col = 1; col < cols; col++) {{
            min_val = fminf(min_val, {inp_read("row * cols + col")});
        }}
        output[row] = {self._write(dtype, "min_val")};
    }}
}}
extern "C" void launch_min_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    min_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis", name, tensor_name, None, rows, cols, axis)

        elif len(input_shape) == 2 and axis == 0:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void min_axis0_kernel_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float min_val = {inp_read("col")};
        for (int row = 1; row < rows; row++) {{
            min_val = fminf(min_val, {inp_read("row * cols + col")});
        }}
        output[col] = {self._write(dtype, "min_val")};
    }}
}}
extern "C" void launch_min_{name}({ctype}* input, {ctype}* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    min_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

    def argmax(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))
            kernel = include + f"""
struct ArgMaxValIdx_{name} {{ float val; int idx; }};

__global__ void argmax_full_kernel_{name}({ctype}* input, float* output, int size) {{
    extern __shared__ ArgMaxValIdx_{name} sdata[];
    int tid = threadIdx.x;

    sdata[tid].val = -FLT_MAX;
    sdata[tid].idx = -1;

    for (int i = tid; i < size; i += blockDim.x) {{
        float v = {inp_read("i")};
        if (v > sdata[tid].val) {{
            sdata[tid].val = v;
            sdata[tid].idx = i;
        }}
    }}
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s && sdata[tid + s].val > sdata[tid].val) {{
            sdata[tid] = sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        output[0] = (float)sdata[0].idx;
    }}
}}
extern "C" void launch_argmax_{name}({ctype}* input, float* output, int size) {{
    dim3 block(256);
    dim3 grid(1);
    int shared_size = block.x * sizeof(ArgMaxValIdx_{name});
    argmax_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, ("argmax_full", name, tensor_name, None, size)

        elif len(input_shape) == 2 and axis == 1:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void argmax_axis_kernel_{name}({ctype}* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float max_val = {inp_read("row * cols")};
        int max_idx = 0;
        for (int col = 1; col < cols; col++) {{
            float v = {inp_read("row * cols + col")};
            if (v > max_val) {{
                max_val = v;
                max_idx = col;
            }}
        }}
        output[row] = (float)max_idx;
    }}
}}
extern "C" void launch_argmax_{name}({ctype}* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    argmax_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis", name, tensor_name, None, rows, cols, axis)

    def argmin(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))
            kernel = include + f"""
struct ArgMinValIdx_{name} {{ float val; int idx; }};

__global__ void argmin_full_kernel_{name}({ctype}* input, float* output, int size) {{
    extern __shared__ ArgMinValIdx_{name} sdata[];
    int tid = threadIdx.x;

    sdata[tid].val = FLT_MAX;
    sdata[tid].idx = -1;

    for (int i = tid; i < size; i += blockDim.x) {{
        float v = {inp_read("i")};
        if (v < sdata[tid].val) {{
            sdata[tid].val = v;
            sdata[tid].idx = i;
        }}
    }}
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s && sdata[tid + s].val < sdata[tid].val) {{
            sdata[tid] = sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        output[0] = (float)sdata[0].idx;
    }}
}}
extern "C" void launch_argmin_{name}({ctype}* input, float* output, int size) {{
    dim3 block(256);
    dim3 grid(1);
    int shared_size = block.x * sizeof(ArgMinValIdx_{name});
    argmin_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, ("argmin_full", name, tensor_name, None, size)

        if len(input_shape) == 2 and axis == 1:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void argmin_axis_kernel_{name}({ctype}* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float min_val = {inp_read("row * cols")};
        int min_idx = 0;
        for (int col = 1; col < cols; col++) {{
            float v = {inp_read("row * cols + col")};
            if (v < min_val) {{
                min_val = v;
                min_idx = col;
            }}
        }}
        output[row] = (float)min_idx;
    }}
}}
extern "C" void launch_argmin_{name}({ctype}* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    argmin_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis", name, tensor_name, None, rows, cols, axis)
