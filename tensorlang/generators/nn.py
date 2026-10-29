"""
TensorLang Generator NeuralNetworkMixin
Kernel NeuralNetworkMixin

File: tensorlang/generators/nn.py
"""

import numpy as np

from .base import BaseGenerator


class NeuralNetworkMixin(BaseGenerator):
    def matmul(self, op_type, name, arg1, arg2, m, n, p, batch_count=1, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)

        if batch_count <= 1:
            # --- 2D matmul (existing, cuBLAS) ---
            if dtype == "f16":
                kernel = include + f"""
#include <stdio.h>
#include <cublas_v2.h>


extern "C" void launch_matmul_{name}(const {ctype}* A, const {ctype}* B, {ctype}* C,
                                    int M, int N, int P) {{
    static cublasHandle_t handle = nullptr;
    static bool handle_created = false;

    if (!handle_created) {{
        cublasStatus_t create_status = cublasCreate(&handle);
        if (create_status != CUBLAS_STATUS_SUCCESS) {{
            fprintf(stderr, "TensorLang: cublasCreate failed: %d\\n", (int)create_status);
            return;
        }}
        handle_created = true;
    }}

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        P, M, N,
        &alpha,
        B, CUDA_R_16F, P,
        A, CUDA_R_16F, N,
        &beta,
        C, CUDA_R_16F, P,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (status != CUBLAS_STATUS_SUCCESS) {{
        fprintf(stderr, "TensorLang: cublasGemmEx failed: %d\\n", (int)status);
    }}

    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            else:
                kernel = f"""
#include <stdio.h>
#include <cublas_v2.h>

extern "C" void launch_matmul_{name}(const float* A, const float* B, float* C,
                                    int M, int N, int P) {{
    static cublasHandle_t handle = nullptr;
    static bool handle_created = false;

    if (!handle_created) {{
        cublasStatus_t create_status = cublasCreate(&handle);
        if (create_status != CUBLAS_STATUS_SUCCESS) {{
            fprintf(stderr, "TensorLang: cublasCreate failed: %d\\n", (int)create_status);
            return;
        }}
        handle_created = true;
    }}

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        P, M, N,
        &alpha,
        B, P,
        A, N,
        &beta,
        C, P
    );

    if (status != CUBLAS_STATUS_SUCCESS) {{
        fprintf(stderr, "TensorLang: cublasSgemm failed: %d\\n", (int)status);
    }}

    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (op_type, name, arg1, arg2, m, n, p)

        # --- N-D batched matmul (strided batched GEMM) ---
        stride_a = m * n
        stride_b = n * p
        stride_c = m * p

        if dtype == "f16":
            kernel = include + f"""
#include <stdio.h>
#include <cublas_v2.h>

extern "C" void launch_matmul_{name}(const {ctype}* A, const {ctype}* B, {ctype}* C,
                                    int M, int N, int P, int batch_count) {{
    static cublasHandle_t handle = nullptr;
    static bool handle_created = false;

    if (!handle_created) {{
        cublasStatus_t create_status = cublasCreate(&handle);
        if (create_status != CUBLAS_STATUS_SUCCESS) {{
            fprintf(stderr, "TensorLang: cublasCreate failed: %d\\n", (int)create_status);
            return;
        }}
        handle_created = true;
    }}

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        P, M, N,
        &alpha,
        B, CUDA_R_16F, P, {stride_b},
        A, CUDA_R_16F, N, {stride_a},
        &beta,
        C, CUDA_R_16F, P, {stride_c},
        batch_count,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (status != CUBLAS_STATUS_SUCCESS) {{
        fprintf(stderr, "TensorLang: cublasGemmStridedBatchedEx failed: %d\\n", (int)status);
    }}

    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        else:
            kernel = f"""
#include <stdio.h>
#include <cublas_v2.h>

extern "C" void launch_matmul_{name}(const float* A, const float* B, float* C,
                                    int M, int N, int P, int batch_count) {{
    static cublasHandle_t handle = nullptr;
    static bool handle_created = false;

    if (!handle_created) {{
        cublasStatus_t create_status = cublasCreate(&handle);
        if (create_status != CUBLAS_STATUS_SUCCESS) {{
            fprintf(stderr, "TensorLang: cublasCreate failed: %d\\n", (int)create_status);
            return;
        }}
        handle_created = true;
    }}

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        P, M, N,
        &alpha,
        B, P, {stride_b},
        A, N, {stride_a},
        &beta,
        C, P, {stride_c},
        batch_count
    );

    if (status != CUBLAS_STATUS_SUCCESS) {{
        fprintf(stderr, "TensorLang: cublasSgemmStridedBatched failed: %d\\n", (int)status);
    }}

    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, ("matmul_batched", name, arg1, arg2, m, n, p, batch_count)

    def linear(self, op_type, name, input_name, weight_name, bias_name, input_shape, weight_shape, output_shape, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)
        w_read = lambda idx: self._read(dtype, "weight", idx)
        b_read = lambda idx: self._read(dtype, "bias", idx)

        ndim = len(input_shape)

        in_features = int(input_shape[-1])
        out_features = int(output_shape[-1])
        batch_size = 1
        for d in input_shape[:-1]:
            batch_size *= int(d)

        if ndim == 1:
            # Single vector — same launch signature as N-D (batch_size=1)
            kernel = include + f"""
__global__ void linear_nd_kernel_{name}({ctype}* input, {ctype}* weight, {ctype}* bias,
                                        {ctype}* output, int in_features, int out_features) {{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < out_features) {{
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < in_features; in_idx++) {{
            sum += {inp_read("in_idx")} * {w_read("in_idx * out_features + out_idx")};
        }}
        output[out_idx] = {self._write(dtype, "sum + " + b_read("out_idx"))};
    }}
}}
extern "C" void launch_linear_{name}({ctype}* input, {ctype}* weight, {ctype}* bias,
                                    {ctype}* output, int batch_size,
                                    int in_features, int out_features) {{
    dim3 block(256);
    dim3 grid((out_features + block.x - 1) / block.x);
    linear_nd_kernel_{name}<<<grid, block>>>(input, weight, bias, output,
                                            in_features, out_features);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_nd", name, input_name, weight_name, bias_name, 1, in_features, out_features)

        else:
            # N-D: batch_size rows, each is a linear transform
            kernel = include + f"""
__global__ void linear_nd_kernel_{name}({ctype}* input, {ctype}* weight, {ctype}* bias,
                                        {ctype}* output, int batch_size,
                                        int in_features, int out_features) {{
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_idx < out_features) {{
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < in_features; in_idx++) {{
            sum += {inp_read("batch_idx * in_features + in_idx")} *
                {w_read("in_idx * out_features + out_idx")};
        }}
        output[batch_idx * out_features + out_idx] = {self._write(dtype, "sum + " + b_read("out_idx"))};
    }}
}}
extern "C" void launch_linear_{name}({ctype}* input, {ctype}* weight, {ctype}* bias,
                                    {ctype}* output, int batch_size,
                                    int in_features, int out_features) {{
    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x,
            (batch_size + block.y - 1) / block.y);
    linear_nd_kernel_{name}<<<grid, block>>>(input, weight, bias, output,
                                            batch_size, in_features, out_features);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_nd", name, input_name, weight_name, bias_name, batch_size, in_features, out_features)


    def layer_norm(self, op_type, name, tensor_name, axis, eps, input_shape, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        ndim = len(input_shape)
        if axis is None:
            axis = ndim - 1
        if axis < 0:
            axis += ndim

        total_elements = int(np.prod([int(d) for d in input_shape]))

        # --- 1D ---
        if ndim == 1:
            size = int(input_shape[0])
            kernel = include + f"""
__global__ void layer_norm_1d_kernel_{name}({ctype}* input, {ctype}* output,
                                            int size, float eps) {{
    if (threadIdx.x == 0 && blockIdx.x == 0) {{
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {{
            sum += {inp_read("i")};
        }}
        float mean = sum / size;

        float var_sum = 0.0f;
        for (int i = 0; i < size; i++) {{
            float diff = {inp_read("i")} - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / size;
        float std_dev = sqrtf(variance + eps);

        for (int i = 0; i < size; i++) {{
            output[i] = {self._write(dtype, f"({inp_read('i')} - mean) / std_dev")};
        }}
    }}
}}
extern "C" void launch_layer_norm_{name}({ctype}* input, {ctype}* output,
                                        int size, float eps) {{
    dim3 block(1);
    dim3 grid(1);
    layer_norm_1d_kernel_{name}<<<grid, block>>>(input, output, size, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_1d", name, tensor_name, None, size, eps)

        # --- 2D, axis=1 (existing optimization) ---
        if ndim == 2 and axis == 1:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void layer_norm_kernel_{name}({ctype}* input, {ctype}* output,
                                        int rows, int cols, float eps) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {{
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            sum += {inp_read("row * cols + col")};
        }}
        float mean = sum / cols;

        float var_sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            float diff = {inp_read("row * cols + col")} - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / cols;
        float std_dev = sqrtf(variance + eps);

        for (int col = 0; col < cols; col++) {{
            float normalized = ({inp_read("row * cols + col")} - mean) / std_dev;
            output[row * cols + col] = {self._write(dtype, "normalized")};
        }}
    }}
}}
extern "C" void launch_layer_norm_{name}({ctype}* input, {ctype}* output,
                                        int rows, int cols, float eps) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    layer_norm_kernel_{name}<<<grid, block>>>(input, output, rows, cols, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_2d", name, tensor_name, None, rows, cols, eps)

        # --- 2D, axis=0 (existing optimization) ---
        if ndim == 2 and axis == 0:
            rows, cols = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void layer_norm_axis0_kernel_{name}({ctype}* input, {ctype}* output,
                                            int rows, int cols, float eps) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {{
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            sum += {inp_read("row * cols + col")};
        }}
        float mean = sum / rows;

        float var_sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            float diff = {inp_read("row * cols + col")} - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / rows;
        float std_dev = sqrtf(variance + eps);

        for (int row = 0; row < rows; row++) {{
            float normalized = ({inp_read("row * cols + col")} - mean) / std_dev;
            output[row * cols + col] = {self._write(dtype, "normalized")};
        }}
    }}
}}
extern "C" void launch_layer_norm_{name}({ctype}* input, {ctype}* output,
                                        int rows, int cols, float eps) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    layer_norm_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, eps)

        # --- N-D general (any axis) ---
        axis_size = int(input_shape[axis])
        num_lines = total_elements // axis_size

        kernel = include + f"""
__global__ void layer_norm_nd_kernel_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis, int axis_size,
    int num_lines, float eps) {{

    int line_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (line_idx >= num_lines) return;

    // Compute input strides
    int in_strides[8];
    in_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {{
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }}
    int axis_stride = in_strides[axis];

    // Decompose line_idx into output coords (skip axis dim)
    // line_idx indexes into the flattened non-axis dimensions
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
    int temp = line_idx;
    for (int d = 0; d < ndim_out; d++) {{
        int out_dim = (d < axis) ? in_shape[d] : in_shape[d + 1];
        out_coords[d] = (out_strides[d] > 0) ? (temp / out_strides[d]) : 0;
        if (out_strides[d] > 0) temp %= out_strides[d];
    }}

    // Base input index (axis coord = 0)
    int base_in_idx = 0;
    int out_d = 0;
    for (int d = 0; d < ndim; d++) {{
        if (d == axis) continue;
        base_in_idx += out_coords[out_d] * in_strides[d];
        out_d++;
    }}

    // Compute mean
    float sum = 0.0f;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_in_idx + k * axis_stride;
        sum += {inp_read("in_idx")};
    }}
    float mean = sum / (float)axis_size;

    // Compute variance
    float var_sum = 0.0f;
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_in_idx + k * axis_stride;
        float diff = {inp_read("in_idx")} - mean;
        var_sum += diff * diff;
    }}
    float variance = var_sum / (float)axis_size;
    float std_dev = sqrtf(variance + eps);

    // Normalize
    for (int k = 0; k < axis_size; k++) {{
        int in_idx = base_in_idx + k * axis_stride;
        float normalized = ({inp_read("in_idx")} - mean) / std_dev;
        output[in_idx] = {self._write(dtype, "normalized")};
    }}
}}
extern "C" void launch_layer_norm_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, int ndim, int axis, int axis_size,
    int num_lines, float eps) {{

    dim3 block(256);
    dim3 grid((num_lines + block.x - 1) / block.x);
    layer_norm_nd_kernel_{name}<<<grid, block>>>(
        input, output, in_shape, ndim, axis, axis_size, num_lines, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        kernel_info = (f"{op_type}_nd", name, tensor_name, None, total_elements, ndim, axis, axis_size, num_lines, eps, list(input_shape))
        return kernel, kernel_info

    def cross_entropy(self, op_type, name, pred_name, pred_shape, target_shape, target_name, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        pred_read = lambda idx: self._read(dtype, "predictions", idx)
        tgt_read = lambda idx: self._read(dtype, "targets", idx)

        ndim = len(pred_shape)
        num_classes = int(pred_shape[-1])

        # Flatten batch dims
        batch_size = 1
        for d in pred_shape[:-1]:
            batch_size *= int(d)

        # Handle target shape: (batch,) class indices or (batch, num_classes) one-hot
        if len(target_shape) == ndim:
            # One-hot or same shape: treat as class indices by taking argmax? 
            # Actually, if target has same shape as pred, it's one-hot. 
            # For simplicity, assume class indices (target_shape has ndim-1 dims)
            pass

        # Determine if targets are class indices (ndim-1) or one-hot (ndim)
        target_is_indices = (len(target_shape) == ndim - 1) or (len(target_shape) == 1 and ndim == 2)

        if dtype == "f16":
            kernel = include + f"""
__global__ void cross_entropy_kernel_{name}({ctype}* predictions, {ctype}* targets,
                                        {ctype}* output, int batch_size, int num_classes) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float partial = 0.0f;
    for (int b = tid; b < batch_size; b += blockDim.x) {{
        int target_class;
        if ({'1' if target_is_indices else '0'}) {{
            target_class = (int){tgt_read("b")};
        }} else {{
            // one-hot: find argmax
            target_class = 0;
            float max_val = {tgt_read("b * num_classes")};
            for (int c = 1; c < num_classes; c++) {{
                float v = {tgt_read("b * num_classes + c")};
                if (v > max_val) {{ max_val = v; target_class = c; }}
            }}
        }}
        target_class = max(0, min(target_class, num_classes - 1));

        float pred_prob = {pred_read("b * num_classes + target_class")};
        pred_prob = fmaxf(pred_prob, 1e-7f);
        pred_prob = fminf(pred_prob, 1.0f - 1e-7f);

        partial += -logf(pred_prob);
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
        float mean_loss = sdata[0] / (float)batch_size;
        output[0] = {self._write(dtype, "mean_loss")};
    }}
}}
extern "C" void launch_cross_entropy_{name}({ctype}* predictions, {ctype}* targets,
                                        {ctype}* output, int batch_size, int num_classes) {{
    dim3 block(256);
    dim3 grid(1);
    int shared_size = block.x * sizeof(float);
    cross_entropy_kernel_{name}<<<grid, block, shared_size>>>(predictions, targets, output,
                                                            batch_size, num_classes);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        else:
            kernel = include + f"""
__global__ void cross_entropy_kernel_{name}({ctype}* predictions, {ctype}* targets,
                                        float* output, int batch_size, int num_classes) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    float loss = 0.0f;
    if (batch_idx < batch_size) {{
        int target_class;
        {'target_class = (int)' + tgt_read('batch_idx') + ';' if target_is_indices else '''
        target_class = 0;
        float max_val = ''' + tgt_read('batch_idx * num_classes') + ''';
        for (int c = 1; c < num_classes; c++) {
            float v = ''' + tgt_read('batch_idx * num_classes + c') + ''';
            if (v > max_val) { max_val = v; target_class = c; }
        }'''}
        target_class = max(0, min(target_class, num_classes - 1));

        float pred_prob = {pred_read("batch_idx * num_classes + target_class")};
        pred_prob = fmaxf(pred_prob, 1e-7f);
        pred_prob = fminf(pred_prob, 1.0f - 1e-7f);

        loss = -logf(pred_prob);
    }}

    sdata[tid] = (batch_idx < batch_size) ? loss : 0.0f;
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
extern "C" void launch_cross_entropy_{name}({ctype}* predictions, {ctype}* targets,
                                        float* output, int batch_size, int num_classes) {{
    float zero = 0.0f;
    cudaMemcpy(output, &zero, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    cross_entropy_kernel_{name}<<<grid, block, shared_size>>>(predictions, targets, output,
                                                            batch_size, num_classes);
    cudaDeviceSynchronize();
    {self.cuda_debug()}

    float mean_loss;
    cudaMemcpy(&mean_loss, output, sizeof(float), cudaMemcpyDeviceToHost);
    mean_loss /= batch_size;
    cudaMemcpy(output, &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
}}
"""
        return kernel, (f"{op_type}", name, pred_name, target_name, batch_size, num_classes)

    def mse_loss(self, op_type, name, pred_name, target_name, total_elements, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        pred_read = lambda idx: self._read(dtype, "predictions", idx)
        tgt_read = lambda idx: self._read(dtype, "targets", idx)

        # Already N-D (flat reduction over total_elements)
        if dtype == "f16":
            kernel = include + f"""
__global__ void mse_loss_kernel_{name}({ctype}* predictions, {ctype}* targets,
                                    {ctype}* output, int total_elements) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float partial = 0.0f;
    for (int i = tid; i < total_elements; i += blockDim.x) {{
        float diff = {pred_read("i")} - {tgt_read("i")};
        partial += diff * diff;
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
        float mean_loss = sdata[0] / (float)total_elements;
        output[0] = {self._write(dtype, "mean_loss")};
    }}
}}
extern "C" void launch_mse_loss_{name}({ctype}* predictions, {ctype}* targets,
                                    {ctype}* output, int total_elements) {{
    dim3 block(256);
    dim3 grid(1);
    int shared_size = block.x * sizeof(float);
    mse_loss_kernel_{name}<<<grid, block, shared_size>>>(predictions, targets, output, total_elements);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        else:
            kernel = include + f"""
__global__ void mse_loss_kernel_{name}({ctype}* predictions, {ctype}* targets,
                                    float* output, int total_elements) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float squared_error = 0.0f;
    if (idx < total_elements) {{
        float diff = {pred_read("idx")} - {tgt_read("idx")};
        squared_error = diff * diff;
    }}

    sdata[tid] = (idx < total_elements) ? squared_error : 0.0f;
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
extern "C" void launch_mse_loss_{name}({ctype}* predictions, {ctype}* targets,
                                    float* output, int total_elements) {{
    float zero = 0.0f;
    cudaMemcpy(output, &zero, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    mse_loss_kernel_{name}<<<grid, block, shared_size>>>(predictions, targets, output, total_elements);
    cudaDeviceSynchronize();
    {self.cuda_debug()}

    float mean_loss;
    cudaMemcpy(&mean_loss, output, sizeof(float), cudaMemcpyDeviceToHost);
    mean_loss /= total_elements;
    cudaMemcpy(output, &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
}}
"""
        return kernel, (f"{op_type}", name, pred_name, target_name, total_elements)

    def batch_norm(self, op_type, name, input_shape, tensor_name, running_mean_name, eps, running_var_name, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        ndim = len(input_shape)

        # --- 2D (existing optimization) ---
        if ndim == 2:
            batch_size, num_features = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void batch_norm_2d_kernel_{name}({ctype}* input, float* running_mean, float* running_var,
                                            {ctype}* output, int batch_size, int num_features, float eps) {{
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (feature_idx < num_features && batch_idx < batch_size) {{
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float std_dev = sqrtf(var + eps);

        int idx = batch_idx * num_features + feature_idx;
        output[idx] = {self._write(dtype, f"({inp_read('idx')} - mean) / std_dev")};
    }}
}}
extern "C" void launch_batch_norm_{name}({ctype}* input, float* running_mean, float* running_var,
                                        {ctype}* output, int batch_size, int num_features, float eps) {{
    dim3 block(16, 16);
    dim3 grid((num_features + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    batch_norm_2d_kernel_{name}<<<grid, block>>>(input, running_mean, running_var, output,
                                                batch_size, num_features, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_2d", name, tensor_name, running_mean_name, batch_size, num_features, eps, running_var_name)

        # --- N-D general (channel dim = 1, NCHW convention) ---
        # Flatten: batch = product of dims except channel (dim 1) and spatial
        # For N-D, channel is at axis 1; normalize over batch + all spatial dims
        if ndim >= 3:
            num_features = int(input_shape[1])
            # Total elements per feature = product of all dims except channel
            total_per_feature = 1
            for i, d in enumerate(input_shape):
                if i != 1:
                    total_per_feature *= int(d)
            # Number of "spatial positions" = total_per_feature / batch_size
            # We'll just use the 2D-style kernel with batch = product of dims except channel
            batch_size = total_per_feature  # all non-channel elements treated as batch

            kernel = include + f"""
__global__ void batch_norm_nd_kernel_{name}({ctype}* input, float* running_mean, float* running_var,
                                            {ctype}* output, int batch_size, int num_features, float eps) {{
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (feature_idx < num_features && batch_idx < batch_size) {{
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float std_dev = sqrtf(var + eps);

        int idx = batch_idx * num_features + feature_idx;
        // Note: this assumes NCHW-like layout where channel is stride-1 in inner loop
        // For true N-D, the indexing needs stride computation. This is a simplified version.
        output[idx] = {self._write(dtype, f"({inp_read('idx')} - mean) / std_dev")};
    }}
}}
extern "C" void launch_batch_norm_{name}({ctype}* input, float* running_mean, float* running_var,
                                        {ctype}* output, int batch_size, int num_features, float eps) {{
    dim3 block(16, 16);
    dim3 grid((num_features + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    batch_norm_nd_kernel_{name}<<<grid, block>>>(input, running_mean, running_var, output,
                                                batch_size, num_features, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_nd", name, tensor_name, running_mean_name, batch_size, num_features, eps, running_var_name)

    def instance_norm(self, op_type, name, tensor_name, input_shape, eps, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        inp_read = lambda idx: self._read(dtype, "input", idx)

        ndim = len(input_shape)

        # --- 2D (existing optimization) ---
        if ndim == 2:
            batch_size, num_features = int(input_shape[0]), int(input_shape[1])
            kernel = include + f"""
__global__ void instance_norm_2d_kernel_{name}({ctype}* input, {ctype}* output,
                                            int batch_size, int num_features, float eps) {{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {{
        float sum = 0.0f;
        for (int f = 0; f < num_features; f++) {{
            sum += {inp_read("batch_idx * num_features + f")};
        }}
        float mean = sum / num_features;

        float var_sum = 0.0f;
        for (int f = 0; f < num_features; f++) {{
            float diff = {inp_read("batch_idx * num_features + f")} - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / num_features;
        float std_dev = sqrtf(variance + eps);

        for (int f = 0; f < num_features; f++) {{
            int idx = batch_idx * num_features + f;
            output[idx] = {self._write(dtype, f"({inp_read('idx')} - mean) / std_dev")};
        }}
    }}
}}
extern "C" void launch_instance_norm_{name}({ctype}* input, {ctype}* output,
                                        int batch_size, int num_features, float eps) {{
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    instance_norm_2d_kernel_{name}<<<grid, block>>>(input, output, batch_size, num_features, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_2d", name, tensor_name, None, None, batch_size, num_features, eps)

        # --- N-D general (normalize per-instance over all non-batch dims) ---
        if ndim >= 3:
            batch_size = int(input_shape[0])
            num_features = 1
            for d in input_shape[1:]:
                num_features *= int(d)

            kernel = include + f"""
__global__ void instance_norm_nd_kernel_{name}({ctype}* input, {ctype}* output,
                                            int batch_size, int num_features, float eps) {{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {{
        int base = batch_idx * num_features;
        float sum = 0.0f;
        for (int f = 0; f < num_features; f++) {{
            sum += {inp_read("base + f")};
        }}
        float mean = sum / num_features;

        float var_sum = 0.0f;
        for (int f = 0; f < num_features; f++) {{
            float diff = {inp_read("base + f")} - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / num_features;
        float std_dev = sqrtf(variance + eps);

        for (int f = 0; f < num_features; f++) {{
            int idx = base + f;
            output[idx] = {self._write(dtype, f"({inp_read('idx')} - mean) / std_dev")};
        }}
    }}
}}
extern "C" void launch_instance_norm_{name}({ctype}* input, {ctype}* output,
                                        int batch_size, int num_features, float eps) {{
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    instance_norm_nd_kernel_{name}<<<grid, block>>>(input, output, batch_size, num_features, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_nd", name, tensor_name, None, None, batch_size, num_features, eps)
