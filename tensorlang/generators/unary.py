from .base import BaseGenerator


class UnaryMixin(BaseGenerator):
    def generate_unary_op_kernel(self, op_type, name, arg1, size, operation):
        kernel_template = f"""
    __global__ void {op_type}_kernel_{name}(float* input, float* output, int size) {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {{
            output[idx] = {operation};
        }}
    }}
    extern "C" void launch_{op_type}_{name}(float* input, float* output, int size) {{
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        {op_type}_kernel_{name}<<<grid, block>>>(input, output, size);
        cudaDeviceSynchronize();
        {self.cuda_debug()}
    }}
    """
        return kernel_template, (op_type, name, arg1, None, size)

    def relu(self, op_type, name, arg1, size):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="fmaxf(0.0f, input[idx])")

    def gelu(self, op_type, name, arg1, size):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="0.5f * input[idx] * (1.0f + erff(input[idx] * 0.70710678f))")

    def swish(self, op_type, name, arg1, size):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="input[idx] / (1.0f + expf(-input[idx]))")

    def sigmoid(self, op_type, name, arg1, size):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="1.0f / (1.0f + expf(-input[idx]))")

    def tanh(self, op_type, name, arg1, size):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="tanhf(input[idx])")

