from .base import BaseGenerator


class NeuralNetworkMixin(BaseGenerator):
    def matmul(self, op_type, name, arg1, arg2, m, n, p):
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

    def linear(self, op_type, name, input_name, weight_name, bias_name, input_shape, weight_shape, output_shape):
        if len(input_shape) == 1:
            in_features = int(input_shape[0])
            out_features = int(output_shape[0])
            kernel = f"""
__global__ void linear_1d_kernel_{name}(float* input, float* weight, float* bias,
                                        float* output, int in_features, int out_features) {{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < out_features) {{
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < in_features; in_idx++) {{
            sum += input[in_idx] * weight[in_idx * out_features + out_idx];
        }}
        output[out_idx] = sum + bias[out_idx];
    }}
}}
extern "C" void launch_linear_{name}(float* input, float* weight, float* bias,
                                    float* output, int in_features, int out_features) {{
    dim3 block(256);
    dim3 grid((out_features + block.x - 1) / block.x);
    linear_1d_kernel_{name}<<<grid, block>>>(input, weight, bias, output,
                                            in_features, out_features);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_1d", name, input_name, weight_name, bias_name, in_features, out_features)
        elif len(input_shape) == 2:
            batch_size, in_features = int(input_shape[0]), int(input_shape[1])
            out_features = int(output_shape[1])
            kernel = f"""
__global__ void linear_2d_kernel_{name}(float* input, float* weight, float* bias,
                                        float* output, int batch_size,
                                        int in_features, int out_features) {{
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_idx < out_features) {{
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < in_features; in_idx++) {{
            sum += input[batch_idx * in_features + in_idx] *
                weight[in_idx * out_features + out_idx];
        }}
        output[batch_idx * out_features + out_idx] = sum + bias[out_idx];
    }}
}}
extern "C" void launch_linear_{name}(float* input, float* weight, float* bias,
                                    float* output, int batch_size,
                                    int in_features, int out_features) {{
    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x,
            (batch_size + block.y - 1) / block.y);
    linear_2d_kernel_{name}<<<grid, block>>>(input, weight, bias, output,
                                            batch_size, in_features, out_features);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_2d", name, input_name, weight_name, bias_name, batch_size, in_features, out_features)

    def layer_norm(self, op_type, name, tensor_name, axis, eps, input_shape):
        if axis is None:
            axis = len(input_shape) - 1

        if len(input_shape) == 2:
            if axis == 1:
                rows, cols = int(input_shape[0]), int(input_shape[1])
                kernel = f"""
__global__ void layer_norm_kernel_{name}(float* input, float* output,
                                        int rows, int cols, float eps) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {{
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            sum += input[row * cols + col];
        }}
        float mean = sum / cols;

        float var_sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            float diff = input[row * cols + col] - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / cols;
        float std_dev = sqrtf(variance + eps);

        for (int col = 0; col < cols; col++) {{
            float normalized = (input[row * cols + col] - mean) / std_dev;
            output[row * cols + col] = normalized;
        }}
    }}
}}
extern "C" void launch_layer_norm_{name}(float* input, float* output,
                                        int rows, int cols, float eps) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    layer_norm_kernel_{name}<<<grid, block>>>(input, output, rows, cols, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
                return kernel, (f"{op_type}_2d", name, tensor_name, None, rows, cols, eps)
            elif axis == 0:
                rows, cols = int(input_shape[0]), int(input_shape[1])
                kernel = f"""
__global__ void layer_norm_axis0_kernel_{name}(float* input, float* output,
                                            int rows, int cols, float eps) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {{
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            sum += input[row * cols + col];
        }}
        float mean = sum / rows;

        float var_sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            float diff = input[row * cols + col] - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / rows;
        float std_dev = sqrtf(variance + eps);

        for (int row = 0; row < rows; row++) {{
            float normalized = (input[row * cols + col] - mean) / std_dev;
            output[row * cols + col] = normalized;
        }}
    }}
}}
extern "C" void launch_layer_norm_{name}(float* input, float* output,
                                        int rows, int cols, float eps) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    layer_norm_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
                return kernel, (f"{op_type}_axis0", name, tensor_name, None, rows, cols, eps)
        elif len(input_shape) == 1:
            size = int(input_shape[0])
            kernel = f"""
__global__ void layer_norm_1d_kernel_{name}(float* input, float* output,
                                            int size, float eps) {{
    if (threadIdx.x == 0 && blockIdx.x == 0) {{
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {{
            sum += input[i];
        }}
        float mean = sum / size;

        float var_sum = 0.0f;
        for (int i = 0; i < size; i++) {{
            float diff = input[i] - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / size;
        float std_dev = sqrtf(variance + eps);

        for (int i = 0; i < size; i++) {{
            output[i] = (input[i] - mean) / std_dev;
        }}
    }}
}}
extern "C" void launch_layer_norm_{name}(float* input, float* output,
                                        int size, float eps) {{
    dim3 block(1);
    dim3 grid(1);
    layer_norm_1d_kernel_{name}<<<grid, block>>>(input, output, size, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_1d", name, tensor_name, None, size, eps)

    def cross_entropy(self, op_type, name, pred_name, pred_shape, target_shape, target_name):
        if len(pred_shape) == 2 and len(target_shape) == 1:
            batch_size, num_classes = int(pred_shape[0]), int(pred_shape[1])
            kernel = f"""
__global__ void cross_entropy_kernel_{name}(float* predictions, float* targets,
                                        float* output, int batch_size, int num_classes) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    float loss = 0.0f;
    if (batch_idx < batch_size) {{
        int target_class = (int)targets[batch_idx];
        target_class = max(0, min(target_class, num_classes - 1));

        float pred_prob = predictions[batch_idx * num_classes + target_class];
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
extern "C" void launch_cross_entropy_{name}(float* predictions, float* targets,
                                        float* output, int batch_size, int num_classes) {{
    cudaMemset(output, 0, sizeof(float));

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

    def mse_loss(self, op_type, name, pred_name, target_name, total_elements):
        kernel = f"""
__global__ void mse_loss_kernel_{name}(float* predictions, float* targets,
                                    float* output, int total_elements) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float squared_error = 0.0f;
    if (idx < total_elements) {{
        float diff = predictions[idx] - targets[idx];
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
extern "C" void launch_mse_loss_{name}(float* predictions, float* targets,
                                    float* output, int total_elements) {{
    cudaMemset(output, 0, sizeof(float));

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


    def batch_norm(self, op_type, name, input_shape, tensor_name, running_mean_name, eps, running_var_name):
        if len(input_shape) == 2:
            batch_size, num_features = int(input_shape[0]), int(input_shape[1])
            kernel = f"""
__global__ void batch_norm_2d_kernel_{name}(float* input, float* running_mean, float* running_var,
                                            float* output, int batch_size, int num_features, float eps) {{
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (feature_idx < num_features && batch_idx < batch_size) {{
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float std_dev = sqrtf(var + eps);

        int idx = batch_idx * num_features + feature_idx;
        output[idx] = (input[idx] - mean) / std_dev;
    }}
}}
extern "C" void launch_batch_norm_{name}(float* input, float* running_mean, float* running_var,
                                        float* output, int batch_size, int num_features, float eps) {{
    dim3 block(16, 16);
    dim3 grid((num_features + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    batch_norm_2d_kernel_{name}<<<grid, block>>>(input, running_mean, running_var, output,
                                                batch_size, num_features, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_2d", name, tensor_name, running_mean_name, batch_size, num_features, eps, running_var_name)

    def instance_norm(self, op_type, name, tensor_name, input_shape, eps):
        if len(input_shape) == 2:
            batch_size, num_features = int(input_shape[0]), int(input_shape[1])
            kernel = f"""
__global__ void instance_norm_2d_kernel_{name}(float* input, float* output,
                                            int batch_size, int num_features, float eps) {{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {{
        float sum = 0.0f;
        for (int f = 0; f < num_features; f++) {{
            sum += input[batch_idx * num_features + f];
        }}
        float mean = sum / num_features;

        float var_sum = 0.0f;
        for (int f = 0; f < num_features; f++) {{
            float diff = input[batch_idx * num_features + f] - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / num_features;
        float std_dev = sqrtf(variance + eps);

        for (int f = 0; f < num_features; f++) {{
            int idx = batch_idx * num_features + f;
            output[idx] = (input[idx] - mean) / std_dev;
        }}
    }}
}}
extern "C" void launch_instance_norm_{name}(float* input, float* output,
                                        int batch_size, int num_features, float eps) {{
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    instance_norm_2d_kernel_{name}<<<grid, block>>>(input, output, batch_size, num_features, eps);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_2d", name, tensor_name, None, None, batch_size, num_features, eps)
