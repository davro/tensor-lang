"""
TensorLang Generator UnaryMixin
Kernel UnaryMixin

File: tensorlang/generators/unary.py
"""

from .base import BaseGenerator


class UnaryMixin(BaseGenerator):
    def generate_unary_op_kernel(self, op_type, name, arg1, size, operation, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)

        if dtype == "f16":
            op_casted = operation.replace("input[idx]", "__half2float(input[idx])")
            result_expr = f"__float2half({op_casted})"
        else:
            result_expr = operation

        kernel_template = include + f"""
__global__ void {op_type}_kernel_{name}({ctype}* input, {ctype}* output, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = {result_expr};
    }}
}}
extern "C" void launch_{op_type}_{name}({ctype}* input, {ctype}* output, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    {op_type}_kernel_{name}<<<grid, block>>>(input, output, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel_template, (op_type, name, arg1, None, size)

    def relu(self, op_type, name, arg1, size, dtype="f32"):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="fmaxf(0.0f, input[idx])", dtype=dtype)

    def gelu(self, op_type, name, arg1, size, dtype="f32"):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="0.5f * input[idx] * (1.0f + erff(input[idx] * 0.70710678f))", dtype=dtype)

    def swish(self, op_type, name, arg1, size, dtype="f32"):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="input[idx] / (1.0f + expf(-input[idx]))", dtype=dtype)

    def sigmoid(self, op_type, name, arg1, size, dtype="f32"):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="1.0f / (1.0f + expf(-input[idx]))", dtype=dtype)

    def tanh(self, op_type, name, arg1, size, dtype="f32"):
        return self.generate_unary_op_kernel(op_type, name, arg1, size, operation="tanhf(input[idx])", dtype=dtype)
