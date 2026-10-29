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
        ndim = len(input_shape)
        if axis < 0:
            axis += ndim

        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)
        out_write = lambda expr: self._write(dtype, expr)

        # --- 2D, last axis (axis == ndim-1) ---
        if ndim == 2 and axis == 1:
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

        # --- 1D ---
        elif ndim == 1:
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

        # --- N-D general (any axis) ---
        else:
            total_elements = int(np.prod([int(d) for d in input_shape]))
            axis_size = int(input_shape[axis])
            kernel = include + f"""
__global__ void softmax_nd_kernel_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis, int total_elements) {{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Compute input strides
    int in_strides[8];
    in_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {{
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }}

    int axis_stride = in_strides[axis];
    int axis_size = in_shape[axis];

    // Decompose idx into coords

    // Decompose idx into coords
    int coords[8];
    int temp = idx;
    for (int d = ndim - 1; d >= 0; d--) {{
        coords[d] = temp % in_shape[d];
        temp /= in_shape[d];
    }}
    int this_axis_coord = coords[axis];

    // Base index with axis coord = 0
    coords[axis] = 0;
    int base_idx = 0;
    for (int d = 0; d < ndim; d++) {{
        base_idx += coords[d] * in_strides[d];
    }}

    // Find max along axis
    float max_val = -FLT_MAX;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_idx + k * axis_stride;
        max_val = fmaxf(max_val, {inp_read("in_idx")});
    }}

    // Sum of exp
    float sum_exp = 0.0f;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_idx + k * axis_stride;
        sum_exp += expf({inp_read("in_idx")} - max_val);
    }}

    // Normalize this element
    int this_in_idx = base_idx + this_axis_coord * axis_stride;
    float this_val = {inp_read("this_in_idx")};
    float exp_val = expf(this_val - max_val);
    output[idx] = {out_write("exp_val / sum_exp")};
}}
extern "C" void launch_softmax_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis, int total_elements) {{

    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    softmax_nd_kernel_{name}<<<grid, block>>>(input, output, in_shape, ndim, axis, total_elements);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = ('softmax_nd', name, tensor_name, None, total_elements, ndim, axis, list(input_shape))
            return kernel, kernel_info

    def sum(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ndim = len(input_shape)
        if axis is not None:
            if axis < 0:
                axis += ndim
            if ndim == 1:
                axis = None  # 1D axis reduction = full reduction

        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        # --- Full reduction (axis is None) ---
        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))

            if dtype == "f16":
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

        # --- 2D axis=1 ---
        elif ndim == 2 and axis == 1:
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

        # --- 2D axis=0 ---
        elif ndim == 2 and axis == 0:
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

        # --- N-D axis reduction ---
        else:
            num_outputs = 1
            for i, d in enumerate(input_shape):
                if i != axis:
                    num_outputs *= int(d)
            axis_size = int(input_shape[axis])

            kernel = include + f"""
__global__ void sum_axis_nd_kernel_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= num_outputs) return;

    // Compute input strides
    int in_strides[8];
    in_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {{
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }}

    // Decompose out_idx into output coords, map to input base (axis coord = 0)
    // Output has ndim-1 dims; map output dim j to input dim (j if j < axis else j+1)
    int out_strides[8];
    // Compute output strides (ndim-1 dims)
    int ndim_out = ndim - 1;
    if (ndim_out > 0) {{
        out_strides[ndim_out - 1] = 1;
        // Need output shape: in_shape without axis
        // Compute on the fly: out_shape[j] = in_shape[j < axis ? j : j+1]
        for (int d = ndim_out - 2; d >= 0; d--) {{
            int out_dim_d = (d < axis) ? in_shape[d] : in_shape[d + 1];
            int out_dim_next = (d + 1 < axis) ? in_shape[d + 1] : in_shape[d + 2];
            out_strides[d] = out_strides[d + 1] * out_dim_next;
        }}
    }}

    int out_coords[8];
    int temp = out_idx;
    for (int d = 0; d < ndim_out; d++) {{
        int out_dim = (d < axis) ? in_shape[d] : in_shape[d + 1];
        out_coords[d] = temp / out_strides[d];
        temp %= out_strides[d];
    }}

    int base_in_idx = 0;
    int out_d = 0;
    for (int d = 0; d < ndim; d++) {{
        if (d == axis) continue;
        base_in_idx += out_coords[out_d] * in_strides[d];
        out_d++;
    }}

    int axis_stride = in_strides[axis];

    float acc = 0.0f;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_in_idx + k * axis_stride;
        acc += {inp_read("in_idx")};
    }}
    output[out_idx] = {self._write(dtype, "acc")};
}}
extern "C" void launch_sum_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    dim3 block(256);
    dim3 grid((num_outputs + block.x - 1) / block.x);
    sum_axis_nd_kernel_{name}<<<grid, block>>>(input, output, in_shape, ndim, axis, num_outputs, axis_size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = (f"{op_type}_axis_nd", name, tensor_name, None, num_outputs, axis_size, axis, ndim, list(input_shape))
            return kernel, kernel_info

    def mean(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ndim = len(input_shape)
        if axis is not None:
            if axis < 0:
                axis += ndim
            if ndim == 1:
                axis = None

        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        # --- Full reduction ---
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

        # --- 2D axis=1 ---
        elif ndim == 2 and axis == 1:
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

        # --- 2D axis=0 ---
        elif ndim == 2 and axis == 0:
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

        # --- N-D axis reduction ---
        else:
            num_outputs = 1
            for i, d in enumerate(input_shape):
                if i != axis:
                    num_outputs *= int(d)
            axis_size = int(input_shape[axis])

            kernel = include + f"""
__global__ void mean_axis_nd_kernel_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= num_outputs) return;

    int in_strides[8];
    in_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {{
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }}

    int ndim_out = ndim - 1;
    int out_strides[8];
    if (ndim_out > 0) {{
        out_strides[ndim_out - 1] = 1;
        for (int d = ndim_out - 2; d >= 0; d--) {{
            int out_dim_next = (d + 1 < axis) ? in_shape[d + 1] : in_shape[d + 2];
            out_strides[d] = out_strides[d + 1] * out_dim_next;
        }}
    }}

    int out_coords[8];
    int temp = out_idx;
    for (int d = 0; d < ndim_out; d++) {{
        int out_dim = (d < axis) ? in_shape[d] : in_shape[d + 1];
        out_coords[d] = (out_strides[d] > 0) ? (temp / out_strides[d]) : 0;
        if (out_strides[d] > 0) temp %= out_strides[d];
    }}

    int base_in_idx = 0;
    int out_d = 0;
    for (int d = 0; d < ndim; d++) {{
        if (d == axis) continue;
        base_in_idx += out_coords[out_d] * in_strides[d];
        out_d++;
    }}

    int axis_stride = in_strides[axis];

    float acc = 0.0f;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_in_idx + k * axis_stride;
        acc += {inp_read("in_idx")};
    }}
    output[out_idx] = {self._write(dtype, "acc / (float)axis_size")};
}}
extern "C" void launch_mean_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    dim3 block(256);
    dim3 grid((num_outputs + block.x - 1) / block.x);
    mean_axis_nd_kernel_{name}<<<grid, block>>>(input, output, in_shape, ndim, axis, num_outputs, axis_size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = (f"{op_type}_axis_nd", name, tensor_name, None, num_outputs, axis_size, axis, ndim, list(input_shape))
            return kernel, kernel_info

    def max(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ndim = len(input_shape)
        if axis is not None:
            if axis < 0:
                axis += ndim
            if ndim == 1:
                axis = None

        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        # --- Full reduction ---
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

        # --- 2D axis=1 ---
        elif ndim == 2 and axis == 1:
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

        # --- 2D axis=0 ---
        elif ndim == 2 and axis == 0:
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

        # --- N-D axis reduction ---
        else:
            num_outputs = 1
            for i, d in enumerate(input_shape):
                if i != axis:
                    num_outputs *= int(d)
            axis_size = int(input_shape[axis])

            kernel = include + f"""
__global__ void max_axis_nd_kernel_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= num_outputs) return;

    int in_strides[8];
    in_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {{
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }}

    int ndim_out = ndim - 1;
    int out_strides[8];
    if (ndim_out > 0) {{
        out_strides[ndim_out - 1] = 1;
        for (int d = ndim_out - 2; d >= 0; d--) {{
            int out_dim_next = (d + 1 < axis) ? in_shape[d + 1] : in_shape[d + 2];
            out_strides[d] = out_strides[d + 1] * out_dim_next;
        }}
    }}

    int out_coords[8];
    int temp = out_idx;
    for (int d = 0; d < ndim_out; d++) {{
        int out_dim = (d < axis) ? in_shape[d] : in_shape[d + 1];
        out_coords[d] = (out_strides[d] > 0) ? (temp / out_strides[d]) : 0;
        if (out_strides[d] > 0) temp %= out_strides[d];
    }}

    int base_in_idx = 0;
    int out_d = 0;
    for (int d = 0; d < ndim; d++) {{
        if (d == axis) continue;
        base_in_idx += out_coords[out_d] * in_strides[d];
        out_d++;
    }}

    int axis_stride = in_strides[axis];

    float max_val = -FLT_MAX;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_in_idx + k * axis_stride;
        max_val = fmaxf(max_val, {inp_read("in_idx")});
    }}
    output[out_idx] = {self._write(dtype, "max_val")};
}}
extern "C" void launch_max_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    dim3 block(256);
    dim3 grid((num_outputs + block.x - 1) / block.x);
    max_axis_nd_kernel_{name}<<<grid, block>>>(input, output, in_shape, ndim, axis, num_outputs, axis_size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = (f"{op_type}_axis_nd", name, tensor_name, None, num_outputs, axis_size, axis, ndim, list(input_shape))
            return kernel, kernel_info

    def min(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ndim = len(input_shape)
        if axis is not None:
            if axis < 0:
                axis += ndim
            if ndim == 1:
                axis = None

        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        # --- Full reduction ---
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

        # --- 2D axis=1 ---
        elif ndim == 2 and axis == 1:
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

        # --- 2D axis=0 ---
        elif ndim == 2 and axis == 0:
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

        # --- N-D axis reduction ---
        else:
            num_outputs = 1
            for i, d in enumerate(input_shape):
                if i != axis:
                    num_outputs *= int(d)
            axis_size = int(input_shape[axis])

            kernel = include + f"""
__global__ void min_axis_nd_kernel_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= num_outputs) return;

    int in_strides[8];
    in_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {{
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }}

    int ndim_out = ndim - 1;
    int out_strides[8];
    if (ndim_out > 0) {{
        out_strides[ndim_out - 1] = 1;
        for (int d = ndim_out - 2; d >= 0; d--) {{
            int out_dim_next = (d + 1 < axis) ? in_shape[d + 1] : in_shape[d + 2];
            out_strides[d] = out_strides[d + 1] * out_dim_next;
        }}
    }}

    int out_coords[8];
    int temp = out_idx;
    for (int d = 0; d < ndim_out; d++) {{
        int out_dim = (d < axis) ? in_shape[d] : in_shape[d + 1];
        out_coords[d] = (out_strides[d] > 0) ? (temp / out_strides[d]) : 0;
        if (out_strides[d] > 0) temp %= out_strides[d];
    }}

    int base_in_idx = 0;
    int out_d = 0;
    for (int d = 0; d < ndim; d++) {{
        if (d == axis) continue;
        base_in_idx += out_coords[out_d] * in_strides[d];
        out_d++;
    }}

    int axis_stride = in_strides[axis];

    float min_val = FLT_MAX;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_in_idx + k * axis_stride;
        min_val = fminf(min_val, {inp_read("in_idx")});
    }}
    output[out_idx] = {self._write(dtype, "min_val")};
}}
extern "C" void launch_min_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    dim3 block(256);
    dim3 grid((num_outputs + block.x - 1) / block.x);
    min_axis_nd_kernel_{name}<<<grid, block>>>(input, output, in_shape, ndim, axis, num_outputs, axis_size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = (f"{op_type}_axis_nd", name, tensor_name, None, num_outputs, axis_size, axis, ndim, list(input_shape))
            return kernel, kernel_info

    def argmax(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ndim = len(input_shape)
        if axis is not None:
            if axis < 0:
                axis += ndim
            if ndim == 1:
                axis = None

        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        # --- Full reduction ---
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

        # --- 2D axis=1 ---
        elif ndim == 2 and axis == 1:
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

        # --- 2D axis=0 ---
        elif ndim == 2 and axis == 0:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void argmax_axis0_kernel_{name}({ctype}* input, float* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float max_val = {inp_read("col")};
        int max_idx = 0;
        for (int row = 1; row < rows; row++) {{
            float v = {inp_read("row * cols + col")};
            if (v > max_val) {{
                max_val = v;
                max_idx = row;
            }}
        }}
        output[col] = (float)max_idx;
    }}
}}
extern "C" void launch_argmax_{name}({ctype}* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    argmax_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

        # --- N-D axis reduction ---
        else:
            num_outputs = 1
            for i, d in enumerate(input_shape):
                if i != axis:
                    num_outputs *= int(d)
            axis_size = int(input_shape[axis])

            kernel = include + f"""
__global__ void argmax_axis_nd_kernel_{name}(
    {ctype}* input, float* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= num_outputs) return;

    int in_strides[8];
    in_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {{
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }}

    int ndim_out = ndim - 1;
    int out_strides[8];
    if (ndim_out > 0) {{
        out_strides[ndim_out - 1] = 1;
        for (int d = ndim_out - 2; d >= 0; d--) {{
            int out_dim_next = (d + 1 < axis) ? in_shape[d + 1] : in_shape[d + 2];
            out_strides[d] = out_strides[d + 1] * out_dim_next;
        }}
    }}

    int out_coords[8];
    int temp = out_idx;
    for (int d = 0; d < ndim_out; d++) {{
        int out_dim = (d < axis) ? in_shape[d] : in_shape[d + 1];
        out_coords[d] = (out_strides[d] > 0) ? (temp / out_strides[d]) : 0;
        if (out_strides[d] > 0) temp %= out_strides[d];
    }}

    int base_in_idx = 0;
    int out_d = 0;
    for (int d = 0; d < ndim; d++) {{
        if (d == axis) continue;
        base_in_idx += out_coords[out_d] * in_strides[d];
        out_d++;
    }}

    int axis_stride = in_strides[axis];

    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_in_idx + k * axis_stride;
        float v = {inp_read("in_idx")};
        if (v > max_val) {{
            max_val = v;
            max_idx = k;
        }}
    }}
    output[out_idx] = (float)max_idx;
}}
extern "C" void launch_argmax_{name}(
    {ctype}* input, float* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    dim3 block(256);
    dim3 grid((num_outputs + block.x - 1) / block.x);
    argmax_axis_nd_kernel_{name}<<<grid, block>>>(input, output, in_shape, ndim, axis, num_outputs, axis_size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = (f"{op_type}_axis_nd", name, tensor_name, None, num_outputs, axis_size, axis, ndim, list(input_shape))
            return kernel, kernel_info

    def argmin(self, op_type, name, tensor_name, axis, input_shape, dtype="f32"):
        ndim = len(input_shape)
        if axis is not None:
            if axis < 0:
                axis += ndim
            if ndim == 1:
                axis = None

        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        # --- Full reduction ---
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

        # --- 2D axis=1 ---
        if ndim == 2 and axis == 1:
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

        # --- 2D axis=0 ---
        elif ndim == 2 and axis == 0:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void argmin_axis0_kernel_{name}({ctype}* input, float* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float min_val = {inp_read("col")};
        int min_idx = 0;
        for (int row = 1; row < rows; row++) {{
            float v = {inp_read("row * cols + col")};
            if (v < min_val) {{
                min_val = v;
                min_idx = row;
            }}
        }}
        output[col] = (float)min_idx;
    }}
}}
extern "C" void launch_argmin_{name}({ctype}* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    argmin_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

        # --- N-D axis reduction ---
        else:
            num_outputs = 1
            for i, d in enumerate(input_shape):
                if i != axis:
                    num_outputs *= int(d)
            axis_size = int(input_shape[axis])

            kernel = include + f"""
__global__ void argmin_axis_nd_kernel_{name}(
    {ctype}* input, float* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= num_outputs) return;

    int in_strides[8];
    in_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {{
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }}

    int ndim_out = ndim - 1;
    int out_strides[8];
    if (ndim_out > 0) {{
        out_strides[ndim_out - 1] = 1;
        for (int d = ndim_out - 2; d >= 0; d--) {{
            int out_dim_next = (d + 1 < axis) ? in_shape[d + 1] : in_shape[d + 2];
            out_strides[d] = out_strides[d + 1] * out_dim_next;
        }}
    }}

    int out_coords[8];
    int temp = out_idx;
    for (int d = 0; d < ndim_out; d++) {{
        int out_dim = (d < axis) ? in_shape[d] : in_shape[d + 1];
        out_coords[d] = (out_strides[d] > 0) ? (temp / out_strides[d]) : 0;
        if (out_strides[d] > 0) temp %= out_strides[d];
    }}

    int base_in_idx = 0;
    int out_d = 0;
    for (int d = 0; d < ndim; d++) {{
        if (d == axis) continue;
        base_in_idx += out_coords[out_d] * in_strides[d];
        out_d++;
    }}

    int axis_stride = in_strides[axis];

    float min_val = FLT_MAX;
    int min_idx = 0;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_in_idx + k * axis_stride;
        float v = {inp_read("in_idx")};
        if (v < min_val) {{
            min_val = v;
            min_idx = k;
        }}
    }}
    output[out_idx] = (float)min_idx;
}}
extern "C" void launch_argmin_{name}(
    {ctype}* input, float* output,
    const int* in_shape, int ndim, int axis,
    int num_outputs, int axis_size) {{

    dim3 block(256);
    dim3 grid((num_outputs + block.x - 1) / block.x);
    argmin_axis_nd_kernel_{name}<<<grid, block>>>(input, output, in_shape, ndim, axis, num_outputs, axis_size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = (f"{op_type}_axis_nd", name, tensor_name, None, num_outputs, axis_size, axis, ndim, list(input_shape))
            return kernel, kernel_info
