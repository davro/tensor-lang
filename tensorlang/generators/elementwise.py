"""
TensorLang Generator ElementwiseMixin
Kernel ElementwiseMixin

File: tensorlang/generators/elementwise.py
"""

import numpy as np

from .base import BaseGenerator


class ElementwiseMixin(BaseGenerator):
    def elementwise(self, op_type, name, arg1, arg2, size, dtype="f32"):
        """Simple elementwise op on flat arrays. Already N-D capable (operates on total element count)."""
        op_symbol = {'add': '+', 'minus': '-', 'mult': '*', 'div': '/'}[op_type]
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        a_read = self._read(dtype, "A")
        b_read = self._read(dtype, "B")
        result = self._write(dtype, f"{a_read} {op_symbol} {b_read}")

        kernel = include + f"""
// -----------------------------------------------------------------------------
// TensorLang -> Elementwise {op_type} ({op_symbol}) [{dtype}]
// -----------------------------------------------------------------------------
__global__ void {op_type}_kernel_{name}({ctype}* A, {ctype}* B, {ctype}* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = {result};
    }}
}}
extern "C" void launch_{op_type}_{name}({ctype}* A, {ctype}* B, {ctype}* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    {op_type}_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, (op_type, name, arg1, arg2, size)

    def binary_broadcast(self, op_name, op_type, name, arg1, arg2, shape1, shape2, output_shape, dtype="f32"):
        """2D broadcast (rows/cols). For N-D, callers should use binary_general_broadcast."""
        rows, cols = output_shape
        op_symbol = {'add': '+', 'minus': '-', 'mult': '*', 'div': '/'}[op_type]
        ctype = self._ctype(dtype)
        include = self._include(dtype)

        if shape1[0] == shape2[0] or (len(shape1) >= 1 and len(shape2) >= 1 and shape1[0] == shape2[0]):
            a_read = self._read(dtype, "A", "i * cols + j")
            b_read = self._read(dtype, "B", "i")
            result = self._write(dtype, f"{a_read} {op_symbol} {b_read}")
            kernel = include + f"""
__global__ void {op_type}_broadcast_rows_kernel_{name}({ctype}* A, {ctype}* B, {ctype}* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = {result};
    }}
}}
extern "C" void launch_{op_type}_{name}({ctype}* A, {ctype}* B, {ctype}* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    {op_type}_broadcast_rows_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f'{op_type}_broadcast_rows', name, arg1, arg2, rows, cols)
        else:
            a_read = self._read(dtype, "A", "i * cols + j")
            b_read = self._read(dtype, "B", "j")
            result = self._write(dtype, f"{a_read} {op_symbol} {b_read}")
            kernel = include + f"""
__global__ void {op_type}_broadcast_kernel_{name}({ctype}* A, {ctype}* B, {ctype}* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = {result};
    }}
}}
extern "C" void launch_{op_type}_{name}({ctype}* A, {ctype}* B, {ctype}* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    {op_type}_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f'{op_type}_broadcast', name, arg1, arg2, rows, cols)

    def binary_1d_broadcast(self, op_type, name, arg1, arg2, size, scalar_on_left=False, dtype="f32"):
        """1D scalar broadcast. For N-D scalar broadcast, callers should use binary_general_broadcast."""
        op_symbol = {'add': '+', 'minus': '-', 'mult': '*', 'div': '/'}[op_type]
        ctype = self._ctype(dtype)
        include = self._include(dtype)

        if scalar_on_left:
            a_read = self._read(dtype, "A", "0")
            b_read = self._read(dtype, "B_scalar", "idx")
            scalar_expr = f"{a_read} {op_symbol} {b_read}"
        else:
            a_read = self._read(dtype, "A", "idx")
            b_read = self._read(dtype, "B_scalar", "0")
            scalar_expr = f"{a_read} {op_symbol} {b_read}"

        result = self._write(dtype, scalar_expr)

        kernel = include + f"""
__global__ void {op_type}_scalar_kernel_{name}({ctype}* A, {ctype}* B_scalar, {ctype}* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = {result};
    }}
}}
extern "C" void launch_{op_type}_{name}({ctype}* A, {ctype}* B, {ctype}* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    {op_type}_scalar_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, (f'{op_type}_scalar', name, arg1, arg2, size)

    def binary_general_broadcast(self, op_name, op_type, name, arg1, arg2, shape1, shape2, output_shape, dtype="f32"):
        """General N-D broadcast. Primary path for N-D elementwise operations.

        Uses stride-based indexing via compute_broadcast_index to handle arbitrary
        N-D broadcasting (NumPy-style: align trailing dims, broadcast size-1 dims).
        """
        ndim_out = len(output_shape)
        total_elements = int(np.prod([int(dim) for dim in output_shape])) if output_shape else 1
        padded_shape1 = (1,) * (ndim_out - len(shape1)) + tuple(shape1)
        padded_shape2 = (1,) * (ndim_out - len(shape2)) + tuple(shape2)

        ctype = self._ctype(dtype)
        include = self._include(dtype)

        if not self._broadcast_helper_emitted:
            broadcast_helper = """
__device__ int compute_broadcast_index(int linear_idx, const int* out_shape, const int* in_shape, int ndim) {
    int in_idx = 0;
    int temp_idx = linear_idx;
    int out_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        int out_coord = temp_idx % out_shape[i];
        temp_idx /= out_shape[i];
        int in_coord = (in_shape[i] == 1) ? 0 : out_coord;
        in_idx += in_coord * out_stride;
        if (i > 0) {
            out_stride *= in_shape[i];
        }
    }
    return in_idx;
}
"""
            self._broadcast_helper_emitted = True
        else:
            broadcast_helper = ""

        op_symbol = {'add': '+', 'minus': '-', 'mult': '*', 'div': '/'}[op_type]

        a_read = self._read(dtype, "A", "idx1")
        b_read = self._read(dtype, "B", "idx2")
        result = self._write(dtype, f"{a_read} {op_symbol} {b_read}")

        kernel = broadcast_helper + include + f"""
__global__ void {op_type}_general_broadcast_kernel_{name}(
    {ctype}* A, {ctype}* B, {ctype}* C,
    int* shape1, int* shape2, int* out_shape,
    int ndim, int total_elements) {{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {{
        int idx1 = compute_broadcast_index(idx, out_shape, shape1, ndim);
        int idx2 = compute_broadcast_index(idx, out_shape, shape2, ndim);

        C[idx] = {result};
    }}
}}

extern "C" void launch_{op_type}_{name}(
    {ctype}* A, {ctype}* B, {ctype}* C,
    int* shape1, int* shape2, int* out_shape,
    int ndim, int total_elements) {{

    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);

    {op_type}_general_broadcast_kernel_{name}<<<grid, block>>>(
        A, B, C, shape1, shape2, out_shape, ndim, total_elements);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, ('general_broadcast', op_type, name, arg1, arg2,
                        list(padded_shape1), list(padded_shape2), list(output_shape), total_elements)

    def fill(self, op_type, name, arg1, arg2, size, op_value, dtype="f32"):
        """Fill kernel. Already N-D capable (operates on total element count)."""
        ctype = self._ctype(dtype)
        include = self._include(dtype)
        write = self._write(dtype, "value")

        kernel = include + f"""
__global__ void fill_kernel_{name}({ctype}* output, float value, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = {write};
    }}
}}
extern "C" void launch_fill_{name}({ctype}* output, float value, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    fill_kernel_{name}<<<grid, block>>>(output, value, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, (op_type, name, arg1, arg2, size, op_value)

    def fused_elementwise(self, op_type, name, chain, env, external_inputs, outputs, size,
                          broadcast=False, input_shapes=None, output_shape=None, dtype="f32"):
        """Fused elementwise kernel. N-D capable via general broadcast path.

        Broadcast mode detection:
        - identity: input shape == output shape → direct flat indexing
        - row/col: 2D-only optimizations (broadcast along rows or cols)
        - trailing: N-D input is a trailing suffix of output shape → modulo indexing
        - general: arbitrary N-D broadcast → compute_broadcast_index
        """
        ctype = self._ctype(dtype)
        include = self._include(dtype)

        value_map = {}
        for idx, a in enumerate(external_inputs):
            value_map[a] = ("in", idx)
        for idx, o in enumerate(outputs):
            value_map[o] = ("out", idx)

        def resolve_fast(arg):
            kind, i = value_map[arg]
            if kind == "in":
                return self._read(dtype, f"in{i}", "idx")
            return f"r{i}"

        def resolve_bc(arg):
            kind, i = value_map[arg]
            if kind == "in":
                return self._read(dtype, f"in{i}", f"in{i}_idx")
            return f"r{i}"

        _OP_EXPR = {
            "add": lambda a, b: f"{a} + {b}",
            "minus": lambda a, b: f"{a} - {b}",
            "mult": lambda a, b: f"{a} * {b}",
            "div": lambda a, b: f"{a} / {b}",
            "relu": lambda a, _b: f"fmaxf(0.0f, {a})",
            "sigmoid": lambda a, _b: f"1.0f / (1.0f + expf(-{a}))",
            "tanh": lambda a, _b: f"tanhf({a})",
            "gelu": lambda a, _b: f"0.5f * {a} * (1.0f + erff({a} * 0.70710678f))",
            "swish": lambda a, _b: f"{a} / (1.0f + expf(-{a}))",
        }

        if not broadcast:
            param_parts = [f"{ctype}* in{i}" for i in range(len(external_inputs))]
            param_parts += [f"{ctype}* out{i}" for i in range(len(outputs))]
            param_parts.append("int size")
            param_str = ", ".join(param_parts)

            body_lines = ["int idx = blockIdx.x * blockDim.x + threadIdx.x;", "if (idx < size) {"]
            for oi, (op, _out, args) in enumerate(chain):
                a0 = resolve_fast(args[0])
                a1 = resolve_fast(args[1]) if len(args) > 1 else a0
                expr = _OP_EXPR.get(op, lambda x, _y: x)(a0, a1)
                body_lines.append(f"    float r{oi} = {expr};")
            for oi in range(len(outputs)):
                body_lines.append(f"    out{oi}[idx] = {self._write(dtype, f'r{oi}')};" )
            body_lines.append("}")

            launch_args = ", ".join(
                [f"in{i}" for i in range(len(external_inputs))]
                + [f"out{i}" for i in range(len(outputs))]
                + ["size"]
            )
            kernel = include + f"""
__global__ void fused_elementwise_kernel_{name}({param_str}) {{
    {chr(10).join(body_lines)}
}}
extern "C" void launch_fused_elementwise_{name}({param_str}) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    fused_elementwise_kernel_{name}<<<grid, block>>>({launch_args});
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = ("fused_elementwise", name, chain, external_inputs, outputs, size, False, None, None)
            return kernel, kernel_info

        ndim = len(output_shape)
        padded_in_shapes = [(1,) * (ndim - len(s)) + tuple(s) for s in input_shapes]

        def _input_mode(inp_shape):
            if tuple(inp_shape) == tuple(output_shape):
                return "identity"
            # 2D-specific optimizations
            if len(output_shape) == 2 and len(inp_shape) == 1:
                if inp_shape[0] == output_shape[0]:
                    return "row"
                if inp_shape[0] == output_shape[1]:
                    return "col"
            # N-D: trailing broadcast (input is suffix of output shape)
            if len(inp_shape) < ndim and tuple(inp_shape) == tuple(output_shape[-len(inp_shape):]):
                return "trailing"
            # N-D: leading broadcast (input is prefix of output shape)
            if len(inp_shape) < ndim and tuple(inp_shape) == tuple(output_shape[:len(inp_shape)]):
                return "leading"
            return "general"

        modes = [_input_mode(s) for s in input_shapes]
        need_helper = any(m == "general" for m in modes)

        if need_helper and not self._broadcast_helper_emitted:
            broadcast_helper = """
__device__ int compute_broadcast_index(int linear_idx, const int* out_shape, const int* in_shape, int ndim) {
    int in_idx = 0;
    int temp_idx = linear_idx;
    int out_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        int out_coord = temp_idx % out_shape[i];
        temp_idx /= out_shape[i];
        int in_coord = (in_shape[i] == 1) ? 0 : out_coord;
        in_idx += in_coord * out_stride;
        if (i > 0) {
            out_stride *= in_shape[i];
        }
    }
    return in_idx;
}
"""
            self._broadcast_helper_emitted = True
        else:
            broadcast_helper = ""

        param_parts = [f"{ctype}* in{i}" for i in range(len(external_inputs))]
        param_parts += [f"{ctype}* out{i}" for i in range(len(outputs))]
        for i in range(len(external_inputs)):
            param_parts.append(f"int* in{i}_shape")
        param_parts.append("int* out_shape")
        param_parts.append("int ndim")
        param_parts.append("int size")
        param_str = ", ".join(param_parts)

        body_lines = ["int idx = blockIdx.x * blockDim.x + threadIdx.x;", "if (idx < size) {"]
        for i, mode in enumerate(modes):
            if mode == "identity":
                body_lines.append(f"    int in{i}_idx = idx;")
            elif mode == "row":
                body_lines.append(f"    int in{i}_idx = idx / out_shape[1];")
            elif mode == "col":
                body_lines.append(f"    int in{i}_idx = idx % out_shape[1];")
            elif mode == "trailing":
                # Input is a trailing suffix: index = idx % (product of input dims)
                in_size = 1
                for d in input_shapes[i]:
                    in_size *= int(d)
                body_lines.append(f"    int in{i}_idx = idx % {in_size};")
            elif mode == "leading":
                # Input is a leading prefix: index = idx / (product of trailing output dims)
                trailing_size = 1
                for d in output_shape[len(input_shapes[i]):]:
                    trailing_size *= int(d)
                body_lines.append(f"    int in{i}_idx = idx / {trailing_size};")
            else:
                body_lines.append(f"    int in{i}_idx = compute_broadcast_index(idx, out_shape, in{i}_shape, ndim);")

        for oi, (op, _out, args) in enumerate(chain):
            a0 = resolve_bc(args[0])
            a1 = resolve_bc(args[1]) if len(args) > 1 else a0
            expr = _OP_EXPR.get(op, lambda x, _y: x)(a0, a1)
            body_lines.append(f"    float r{oi} = {expr};")
        for oi in range(len(outputs)):
            body_lines.append(f"    out{oi}[idx] = {self._write(dtype, f'r{oi}')};" )
        body_lines.append("}")

        launch_args = ", ".join(
            [f"in{i}" for i in range(len(external_inputs))]
            + [f"out{i}" for i in range(len(outputs))]
            + [f"in{i}_shape" for i in range(len(external_inputs))]
            + ["out_shape", "ndim", "size"]
        )

        kernel = broadcast_helper + include + f"""
__global__ void fused_elementwise_kernel_{name}({param_str}) {{
    {chr(10).join(body_lines)}
}}
extern "C" void launch_fused_elementwise_{name}({param_str}) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    fused_elementwise_kernel_{name}<<<grid, block>>>({launch_args});
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        kernel_info = ("fused_elementwise", name, chain, external_inputs, outputs, size, True,
                       [list(s) for s in padded_in_shapes], list(output_shape))
        return kernel, kernel_info
