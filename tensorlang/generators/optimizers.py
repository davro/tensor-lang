from .base import BaseGenerator


class OptimizersMixin(BaseGenerator):
    def sgd_update(self, op_type, name, param_name, grad_name, size):
        kernel = f"""
__global__ void sgd_update_kernel_{name}(float* param, const float* grad, float lr, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        param[idx] -= lr * grad[idx];
    }}
}}
extern "C" void launch_sgd_update_{name}(float* param, const float* grad, float lr, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    sgd_update_kernel_{name}<<<grid, block>>>(param, grad, lr, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, ('sgd_update', name, param_name, grad_name, size)

    def adam_update(self, op_type, name, param_name, grad_name, size):
        kernel = f"""
__global__ void adam_update_kernel_{name}(
    float* param,
    const float* grad,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t,
    int size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        float g = grad[idx];

        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

        float m_hat = m[idx] / (1.0f - powf(beta1, (float)t));
        float v_hat = v[idx] / (1.0f - powf(beta2, (float)t));

        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }}
}}
extern "C" void launch_adam_update_{name}(
    float* param,
    const float* grad,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t,
    int size
) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    adam_update_kernel_{name}<<<grid, block>>>(
        param, grad, m, v, lr, beta1, beta2, eps, t, size
    );
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, ('adam_update', name, param_name, grad_name, size)

