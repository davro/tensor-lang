"""
TensorLang Generator OptimizersMixin
Kernel OptimizersMixin

File: tensorlang/generators/optimizers.py
"""

from .base import BaseGenerator


class OptimizersMixin(BaseGenerator):
    def sgd_update(self, op_type, name, param_name, grad_name, size, dtype="f32", grad_dtype=None, master=False):
        if grad_dtype is None:
            grad_dtype = dtype

        if not master:
            ctype = self._ctype(dtype)
            include = self._include(dtype)
            param_read = self._read(dtype, "param", "idx")
            grad_read = self._read(grad_dtype, "grad", "idx")

            kernel = include + f"""
__global__ void sgd_update_kernel_{name}({ctype}* param, const {self._ctype(grad_dtype)}* grad, float lr, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        param[idx] = {self._write(dtype, f"{param_read} - lr * {grad_read}")};
    }}
}}
extern "C" void launch_sgd_update_{name}({ctype}* param, const {self._ctype(grad_dtype)}* grad, float lr, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    sgd_update_kernel_{name}<<<grid, block>>>(param, grad, lr, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, ('sgd_update', name, param_name, grad_name, size, None)

        # Master weights: FP16 param, FP32 master, grad in grad_dtype
        include = self._include("f16") + self._include("f32")
        grad_read = self._read(grad_dtype, "grad", "idx")
        master_read = "master[idx]"

        kernel = include + f"""
__global__ void sgd_update_kernel_{name}(__half* param, const {self._ctype(grad_dtype)}* grad, float* master, float lr, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        float g = {grad_read};
        float m = {master_read};
        m = m - lr * g;
        master[idx] = m;
        param[idx] = __float2half(m);
    }}
}}
extern "C" void launch_sgd_update_{name}(__half* param, const {self._ctype(grad_dtype)}* grad, float* master, float lr, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    sgd_update_kernel_{name}<<<grid, block>>>(param, grad, master, lr, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, ('sgd_update', name, param_name, grad_name, size, f"{param_name}_master")

    def adam_update(self, op_type, name, param_name, grad_name, size, dtype="f32", grad_dtype=None, master=False):
        if grad_dtype is None:
            grad_dtype = dtype

        if not master:
            ctype = self._ctype(dtype)
            include = self._include(dtype)
            param_read = self._read(dtype, "param", "idx")
            grad_read = self._read(grad_dtype, "grad", "idx")
            m_read = self._read(dtype, "m", "idx")
            v_read = self._read(dtype, "v", "idx")

            kernel = include + f"""
__global__ void adam_update_kernel_{name}(
    {ctype}* param,
    const {self._ctype(grad_dtype)}* grad,
    {ctype}* m,
    {ctype}* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t,
    int size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        float g = {grad_read};

        float m_val = {m_read};
        float v_val = {v_read};

        m_val = beta1 * m_val + (1.0f - beta1) * g;
        v_val = beta2 * v_val + (1.0f - beta2) * g * g;

        m[idx] = {self._write(dtype, "m_val")};
        v[idx] = {self._write(dtype, "v_val")};

        float m_hat = m_val / (1.0f - powf(beta1, (float)t));
        float v_hat = v_val / (1.0f - powf(beta2, (float)t));

        param[idx] = {self._write(dtype, f"{param_read} - lr * m_hat / (sqrtf(v_hat) + eps)")};
    }}
}}
extern "C" void launch_adam_update_{name}(
    {ctype}* param,
    const {self._ctype(grad_dtype)}* grad,
    {ctype}* m,
    {ctype}* v,
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
            return kernel, ('adam_update', name, param_name, grad_name, size, None)

        # Master weights: FP16 param, FP32 master/m/v, grad in grad_dtype
        include = self._include("f16") + self._include("f32")
        grad_read = self._read(grad_dtype, "grad", "idx")

        kernel = include + f"""
__global__ void adam_update_kernel_{name}(
    __half* param,
    const {self._ctype(grad_dtype)}* grad,
    float* master,
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
        float g = {grad_read};

        float master_val = master[idx];
        float m_val = m[idx];
        float v_val = v[idx];

        m_val = beta1 * m_val + (1.0f - beta1) * g;
        v_val = beta2 * v_val + (1.0f - beta2) * g * g;

        m[idx] = m_val;
        v[idx] = v_val;

        float m_hat = m_val / (1.0f - powf(beta1, (float)t));
        float v_hat = v_val / (1.0f - powf(beta2, (float)t));

        master_val = master_val - lr * m_hat / (sqrtf(v_hat) + eps);
        master[idx] = master_val;
        param[idx] = __float2half(master_val);
    }}
}}
extern "C" void launch_adam_update_{name}(
    __half* param,
    const {self._ctype(grad_dtype)}* grad,
    float* master,
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
        param, grad, master, m, v, lr, beta1, beta2, eps, t, size
    );
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, ('adam_update', name, param_name, grad_name, size, f"{param_name}_master")
