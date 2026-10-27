import numpy as np

from .base import BaseGenerator


class ReductionMixin(BaseGenerator):
    def softmax(self, op_type, name, tensor_name, input_shape, axis):
        if axis is None:
            axis = len(input_shape) - 1

        if len(input_shape) == 2 and axis == 1:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel_params = "int rows, int cols"
            index = "row * cols + col"
            initial_index = "row * cols"
            loop_var = "col"
            outer_loop = "int row = blockIdx.x * blockDim.x + threadIdx.x;\n    if (row < rows) {"
            block = "256"
            grid = "(rows + block.x - 1) / block.x"
            kernel_info = ('softmax', name, tensor_name, None, rows, cols, axis)
        elif len(input_shape) == 1:
            size = int(input_shape[0])
            kernel_params = "int size"
            index = "i"
            initial_index = "0"
            loop_var = "i"
            outer_loop = "if (threadIdx.x == 0 && blockIdx.x == 0) {"
            block = "1"
            grid = "1"
            kernel_info = ('softmax_1d', name, tensor_name, None, size)
        else:
            raise ValueError(f"Unsupported softmax shape {input_shape} or axis {axis}")

        kernel = f"""
__global__ void softmax_kernel_{name}(float* input, float* output, {kernel_params}) {{
    {outer_loop}
        float max_val = input[{initial_index}];
        for (int {loop_var} = 1; {loop_var} < {(size if len(input_shape) == 1 else 'cols')}; {loop_var}++) {{
            max_val = fmaxf(max_val, input[{index}]);
        }}

        float sum_exp = 0.0f;
        for (int {loop_var} = 0; {loop_var} < {(size if len(input_shape) == 1 else 'cols')}; {loop_var}++) {{
            float exp_val = expf(input[{index}] - max_val);
            output[{index}] = exp_val;
            sum_exp += exp_val;
        }}

        for (int {loop_var} = 0; {loop_var} < {(size if len(input_shape) == 1 else 'cols')}; {loop_var}++) {{
            output[{index}] /= sum_exp;
        }}
    }}
}}
extern "C" void launch_softmax_{name}(float* input, float* output, {kernel_params}) {{
    dim3 block({block});
    dim3 grid({grid});
    softmax_kernel_{name}<<<grid, block>>>(input, output, {('rows, cols' if len(input_shape) == 2 else 'size')});
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, kernel_info

    def sum(self, op_type, name, tensor_name, axis, input_shape):
        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))
            kernel = f"""
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
        atomicAdd(output, sdata[0]);
    }}
}}
extern "C" void launch_sum_{name}(float* input, float* output, int size) {{
    cudaMemset(output, 0, sizeof(float));

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    sum_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_full", name, tensor_name, None, size)
        else:
            if len(input_shape) == 2 and axis == 1:
                rows, cols = int(input_shape[0]), int(input_shape[1])
                kernel = f"""
__global__ void sum_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            sum += input[row * cols + col];
        }}
        output[row] = sum;
    }}
}}
extern "C" void launch_sum_{name}(float* input, float* output, int rows, int cols) {{
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
                kernel = f"""
__global__ void sum_axis0_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            sum += input[row * cols + col];
        }}
        output[col] = sum;
    }}
}}
extern "C" void launch_sum_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    sum_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
                return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

    def mean(self, op_type, name, tensor_name, axis, input_shape):
        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))
            kernel = f"""
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
        atomicAdd(output, sdata[0]);
    }}
}}
extern "C" void launch_mean_{name}(float* input, float* output, int size) {{
    cudaMemset(output, 0, sizeof(float));

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
        else:
            if len(input_shape) == 2 and axis == 1:
                rows, cols = int(input_shape[0]), int(input_shape[1])
                kernel = f"""
__global__ void mean_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            sum += input[row * cols + col];
        }}
        output[row] = sum / cols;
    }}
}}
extern "C" void launch_mean_{name}(float* input, float* output, int rows, int cols) {{
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
                kernel = f"""
__global__ void mean_axis0_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            sum += input[row * cols + col];
        }}
        output[col] = sum / rows;
    }}
}}
extern "C" void launch_mean_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    mean_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
                return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

    def max(self, op_type, name, tensor_name, axis, input_shape):
        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))
            kernel = f"""
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
        float* addr = output;
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
        else:
            if len(input_shape) == 2 and axis == 1:
                rows, cols = int(input_shape[0]), int(input_shape[1])
                kernel = f"""
__global__ void max_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float max_val = input[row * cols];
        for (int col = 1; col < cols; col++) {{
            max_val = fmaxf(max_val, input[row * cols + col]);
        }}
        output[row] = max_val;
    }}
}}
extern "C" void launch_max_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    max_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}}
"""
                return kernel, (f"{op_type}_axis", name, tensor_name, None, rows, cols, axis)
            elif len(input_shape) == 2 and axis == 0:
                rows, cols = int(input_shape[0]), int(input_shape[1])
                kernel = f"""
__global__ void max_axis0_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float max_val = input[col];
        for (int row = 1; row < rows; row++) {{
            max_val = fmaxf(max_val, input[row * cols + col]);
        }}
        output[col] = max_val;
    }}
}}
extern "C" void launch_max_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    max_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}}
"""
                return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

    def min(self, op_type, name, tensor_name, axis, input_shape):
        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))
            kernel = f"""
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
        float* addr = output;
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
        else:
            if len(input_shape) == 2 and axis == 1:
                rows, cols = int(input_shape[0]), int(input_shape[1])
                kernel = f"""
__global__ void min_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float min_val = input[row * cols];
        for (int col = 1; col < cols; col++) {{
            min_val = fminf(min_val, input[row * cols + col]);
        }}
        output[row] = min_val;
    }}
}}
extern "C" void launch_min_{name}(float* input, float* output, int rows, int cols) {{
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
                kernel = f"""
__global__ void min_axis0_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float min_val = input[col];
        for (int row = 1; row < rows; row++) {{
            min_val = fminf(min_val, input[row * cols + col]);
        }}
        output[col] = min_val;
    }}
}}
extern "C" void launch_min_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    min_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
                return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, axis)

    def argmax(self, op_type, name, tensor_name, axis, input_shape):
        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))
            kernel = f"""
struct ArgMaxValIdx_{name} {{ float val; int idx; }};

__global__ void argmax_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ ArgMaxValIdx_{name} sdata[];
    int tid = threadIdx.x;

    sdata[tid].val = -FLT_MAX;
    sdata[tid].idx = -1;

    for (int i = tid; i < size; i += blockDim.x) {{
        if (input[i] > sdata[tid].val) {{
            sdata[tid].val = input[i];
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
extern "C" void launch_argmax_{name}(float* input, float* output, int size) {{
    dim3 block(256);
    dim3 grid(1);
    int shared_size = block.x * sizeof(ArgMaxValIdx_{name});
    argmax_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, ("argmax_full", name, tensor_name, None, size)
        else:
            if len(input_shape) == 2 and axis == 1:
                rows, cols = int(input_shape[0]), int(input_shape[1])
                kernel = f"""
__global__ void argmax_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float max_val = input[row * cols];
        int max_idx = 0;
        for (int col = 1; col < cols; col++) {{
            if (input[row * cols + col] > max_val) {{
                max_val = input[row * cols + col];
                max_idx = col;
            }}
        }}
        output[row] = (float)max_idx;
    }}
}}
extern "C" void launch_argmax_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    argmax_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
                return kernel, (f"{op_type}_axis", name, tensor_name, None, rows, cols, axis)

    def argmin(self, op_type, name, tensor_name, axis, input_shape):
        if axis is None:
            size = int(np.prod([int(dim) for dim in input_shape]))
            kernel = f"""
struct ArgMinValIdx_{name} {{ float val; int idx; }};

__global__ void argmin_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ ArgMinValIdx_{name} sdata[];
    int tid = threadIdx.x;

    sdata[tid].val = FLT_MAX;
    sdata[tid].idx = -1;

    for (int i = tid; i < size; i += blockDim.x) {{
        if (input[i] < sdata[tid].val) {{
            sdata[tid].val = input[i];
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
extern "C" void launch_argmin_{name}(float* input, float* output, int size) {{
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
            kernel = f"""
__global__ void argmin_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float min_val = input[row * cols];
        int min_idx = 0;
        for (int col = 1; col < cols; col++) {{
            if (input[row * cols + col] < min_val) {{
                min_val = input[row * cols + col];
                min_idx = col;
            }}
        }}
        output[row] = (float)min_idx;
    }}
}}
extern "C" void launch_argmin_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    argmin_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis", name, tensor_name, None, rows, cols, axis)

