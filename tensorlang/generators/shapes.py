"""
TensorLang Generator ShapesMixin
Kernel ShapesMixin

File: tensorlang/generators/shapes.py
"""

import numpy as np

from .base import BaseGenerator


class ShapesMixin(BaseGenerator):
    def transpose(self, op_type, name, input_shape, tensor_name, axes=None, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)

        input_shape = tuple(int(d) for d in input_shape)
        ndim = len(input_shape)

        if axes is None:
            axes = list(range(ndim - 1, -1, -1))
        else:
            axes = [int(a) for a in axes]

        output_shape = tuple(input_shape[i] for i in axes)
        total_elements = int(np.prod(input_shape)) if input_shape else 1

        if ndim == 0:
            kernel = include + f"""
__global__ void transpose_kernel_{name}({ctype}* input, {ctype}* output, int total_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {{
        output[idx] = input[idx];
    }}
}}
extern "C" void launch_transpose_{name}({ctype}* input, {ctype}* output, int total_elements) {{
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    transpose_kernel_{name}<<<grid, block>>>(input, output, total_elements);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, ("transpose", name, tensor_name, None, total_elements)

        in_shape_str = ", ".join(str(d) for d in input_shape)
        out_shape_str = ", ".join(str(d) for d in output_shape)
        axes_str = ", ".join(str(a) for a in axes)

        kernel = include + f"""
__global__ void transpose_kernel_{name}({ctype}* input, {ctype}* output, int total_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {{
        const int in_shape[{ndim}] = {{{in_shape_str}}};
        const int out_shape[{ndim}] = {{{out_shape_str}}};
        const int axes[{ndim}] = {{{axes_str}}};

        int out_coords[{ndim}];
        int tmp = idx;
        for (int d = {ndim - 1}; d >= 0; --d) {{
            out_coords[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }}

        int in_coords[{ndim}];
        for (int d = 0; d < {ndim}; ++d) {{
            in_coords[axes[d]] = out_coords[d];
        }}

        int in_idx = 0;
        for (int d = 0; d < {ndim}; ++d) {{
            in_idx = in_idx * in_shape[d] + in_coords[d];
        }}

        output[idx] = input[in_idx];
    }}
}}
extern "C" void launch_transpose_{name}({ctype}* input, {ctype}* output, int total_elements) {{
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    transpose_kernel_{name}<<<grid, block>>>(input, output, total_elements);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, ("transpose", name, tensor_name, None, total_elements)

    def reshape(self, op_type, name, tensor_name, total_elements, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)

        kernel = include + f"""
__global__ void reshape_kernel_{name}({ctype}* input, {ctype}* output, int total_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {{
        output[idx] = input[idx];
    }}
}}
extern "C" void launch_reshape_{name}({ctype}* input, {ctype}* output, int total_elements) {{
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    reshape_kernel_{name}<<<grid, block>>>(input, output, total_elements);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, (f"{op_type}", name, tensor_name, None, total_elements)

    def slice(self, op_type, name, tensor_name, slice_specs, input_shape, output_shape, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)

        input_shape = tuple(int(d) for d in input_shape)
        output_shape = tuple(int(d) for d in output_shape)
        in_ndim = len(input_shape)
        out_ndim = len(output_shape)
        total_output = int(np.prod(output_shape)) if output_shape else 1

        # --- 2D slice (existing, kept as optimization) ---
        if in_ndim == 2 and len(slice_specs) <= 2:
            rows, cols = int(input_shape[0]), int(input_shape[1])

            row_spec = slice_specs[0] if len(slice_specs) > 0 else {'type': 'full_slice'}
            col_spec = slice_specs[1] if len(slice_specs) > 1 else {'type': 'full_slice'}

            def _normalize_spec(spec, dim_size):
                if spec['type'] == 'slice':
                    start = spec.get('start')
                    end = spec.get('end')
                    if start is None: start = 0
                    if end is None: end = dim_size
                    if start < 0: start = max(0, dim_size + start)
                    if end < 0: end = max(0, dim_size + end)
                    start = max(0, min(start, dim_size))
                    end = max(start, min(end, dim_size))
                    return start, end
                elif spec['type'] == 'index':
                    val = spec['value']
                    if val < 0: val = dim_size + val
                    val = max(0, min(val, dim_size - 1)) if dim_size > 0 else 0
                    return val, val + 1
                else:
                    return 0, dim_size

            row_start, row_end = _normalize_spec(row_spec, rows)
            col_start, col_end = _normalize_spec(col_spec, cols)

            out_rows = max(0, row_end - row_start)
            out_cols = max(0, col_end - col_start)

            kernel = include + f"""
__global__ void slice_kernel_{name}({ctype}* input, {ctype}* output,
                                int in_rows, int in_cols,
                                int row_start, int row_end,
                                int col_start, int col_end,
                                int out_rows, int out_cols) {{
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row < out_rows && out_col < out_cols) {{
        int in_row = row_start + out_row;
        int in_col = col_start + out_col;
        int in_idx = in_row * in_cols + in_col;
        int out_idx = out_row * out_cols + out_col;
        output[out_idx] = input[in_idx];
    }}
}}

extern "C" void launch_slice_{name}({ctype}* input, {ctype}* output,
                                int in_rows, int in_cols,
                                int row_start, int row_end,
                                int col_start, int col_end,
                                int out_rows, int out_cols) {{
    dim3 block(16, 16);
    dim3 grid((out_cols + block.x - 1) / block.x,
              (out_rows + block.y - 1) / block.y);
    slice_kernel_{name}<<<grid, block>>>(input, output, in_rows, in_cols,
                                        row_start, row_end,
                                        col_start, col_end,
                                        out_rows, out_cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_2d", name, tensor_name, None,
                            rows, cols, row_start, row_end,
                            col_start, col_end, out_rows, out_cols)

        # --- 1D slice (existing, kept as optimization) ---
        elif in_ndim == 1 and len(slice_specs) == 1:
            size = int(input_shape[0])
            spec = slice_specs[0]

            if spec['type'] == 'slice':
                start = spec.get('start')
                end = spec.get('end')
                if start is None: start = 0
                if end is None: end = size
                if start < 0: start = max(0, size + start)
                if end < 0: end = max(0, size + end)
            elif spec['type'] == 'index':
                val = spec['value']
                if val < 0: val = size + val
                start = max(0, min(val, size - 1)) if size > 0 else 0
                end = start + 1
            else:
                start, end = 0, size

            start = max(0, min(start, size))
            end = max(start, min(end, size))
            out_size = max(0, end - start)

            kernel = include + f"""
__global__ void slice_1d_kernel_{name}({ctype}* input, {ctype}* output, int start, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = input[start + idx];
    }}
}}

extern "C" void launch_slice_{name}({ctype}* input, {ctype}* output, int start, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    slice_1d_kernel_{name}<<<grid, block>>>(input, output, start, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_1d", name, tensor_name, None, start, out_size)

        # --- N-D slice (general) ---
        else:
            # Compute slice parameters for each input dimension
            offsets = []      # start offset for each input dim
            is_index = []     # 1 if dimension is indexed (removed), 0 otherwise
            fixed_vals = []   # fixed index value for indexed dims

            for i in range(in_ndim):
                spec = slice_specs[i] if i < len(slice_specs) else {'type': 'full_slice'}
                dim_size = int(input_shape[i])

                if spec['type'] == 'index':
                    val = spec['value']
                    if val < 0: val = dim_size + val
                    val = max(0, min(val, dim_size - 1)) if dim_size > 0 else 0
                    is_index.append(1)
                    fixed_vals.append(val)
                    offsets.append(0)
                elif spec['type'] == 'slice':
                    start = spec.get('start')
                    end = spec.get('end')
                    if start is None: start = 0
                    if end is None: end = dim_size
                    if start < 0: start = max(0, dim_size + start)
                    if end < 0: end = max(0, dim_size + end)
                    start = max(0, min(start, dim_size))
                    end = max(start, min(end, dim_size))
                    is_index.append(0)
                    fixed_vals.append(0)
                    offsets.append(start)
                else:  # full_slice
                    is_index.append(0)
                    fixed_vals.append(0)
                    offsets.append(0)

            offsets_str = ", ".join(str(v) for v in offsets)
            is_index_str = ", ".join(str(v) for v in is_index)
            fixed_vals_str = ", ".join(str(v) for v in fixed_vals)
            in_shape_str = ", ".join(str(d) for d in input_shape)
            out_shape_str = ", ".join(str(d) for d in output_shape)

            kernel = include + f"""
__global__ void slice_nd_kernel_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, const int* out_shape,
    const int* offsets, const int* is_index, const int* fixed_vals,
    int in_ndim, int out_ndim, int total_output) {{

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_output) return;

    // Decompose output flat index into output coords
    int out_coords[8];
    int out_strides[8];
    if (out_ndim > 0) {{
        out_strides[out_ndim - 1] = 1;
        for (int d = out_ndim - 2; d >= 0; d--) {{
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
        }}
        int temp = out_idx;
        for (int d = 0; d < out_ndim; d++) {{
            out_coords[d] = temp / out_strides[d];
            temp %= out_strides[d];
        }}
    }}

    // Map output coords to input coords
    int in_coords[8];
    int out_d = 0;
    for (int d = 0; d < in_ndim; d++) {{
        if (is_index[d]) {{
            in_coords[d] = fixed_vals[d];
        }} else {{
            in_coords[d] = out_coords[out_d] + offsets[d];
            out_d++;
        }}
    }}

    // Compute input flat index
    int in_strides[8];
    if (in_ndim > 0) {{
        in_strides[in_ndim - 1] = 1;
        for (int d = in_ndim - 2; d >= 0; d--) {{
            in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
        }}
    }}
    int in_idx = 0;
    for (int d = 0; d < in_ndim; d++) {{
        in_idx += in_coords[d] * in_strides[d];
    }}

    output[out_idx] = input[in_idx];
}}
extern "C" void launch_slice_{name}(
    {ctype}* input, {ctype}* output,
    const int* in_shape, const int* out_shape,
    const int* offsets, const int* is_index, const int* fixed_vals,
    int in_ndim, int out_ndim, int total_output) {{

    dim3 block(256);
    dim3 grid((total_output + block.x - 1) / block.x);
    slice_nd_kernel_{name}<<<grid, block>>>(
        input, output, in_shape, out_shape, offsets, is_index, fixed_vals,
        in_ndim, out_ndim, total_output);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = ("slice_nd", name, tensor_name, None, in_ndim, out_ndim, total_output,
                           list(input_shape), list(output_shape), offsets, is_index, fixed_vals)
            return kernel, kernel_info

    def concat(self, op_type, name, tensor_names, axis, env, dtype="f32"):
        ctype = self._ctype(dtype)
        include = self._include(dtype)

        # Normalize axis
        if not tensor_names:
            return None, None

        first_shape = tuple(int(d) for d in env[tensor_names[0]]['shape'])
        ndim = len(first_shape)
        if axis < 0:
            axis += ndim

        # --- 2 tensors, 2D, axis=0 (existing, kept as optimization) ---
        if len(tensor_names) == 2 and ndim == 2 and axis == 0:
            tensor1, tensor2 = tensor_names
            shape1 = env[tensor1]['shape']
            shape2 = env[tensor2]['shape']

            rows1, cols = int(shape1[0]), int(shape1[1])
            rows2 = int(shape2[0])

            kernel = include + f"""
__global__ void concat_axis0_kernel_{name}({ctype}* input1, {ctype}* input2, {ctype}* output,
                                        int rows1, int rows2, int cols) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {{
        if (row < rows1) {{
            output[row * cols + col] = input1[row * cols + col];
        }} else if (row < rows1 + rows2) {{
            int src_row = row - rows1;
            output[row * cols + col] = input2[src_row * cols + col];
        }}
    }}
}}
extern "C" void launch_concat_{name}({ctype}* input1, {ctype}* input2, {ctype}* output,
                                    int rows1, int rows2, int cols) {{
    dim3 block(16, 16);
    int total_rows = rows1 + rows2;
    dim3 grid((cols + block.x - 1) / block.x, (total_rows + block.y - 1) / block.y);
    concat_axis0_kernel_{name}<<<grid, block>>>(input1, input2, output, rows1, rows2, cols);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis0", name, tensor1, tensor2, rows1, rows2, cols)

        # --- 2 tensors, 2D, axis=1 (existing, kept as optimization) ---
        if len(tensor_names) == 2 and ndim == 2 and axis == 1:
            tensor1, tensor2 = tensor_names
            shape1 = env[tensor1]['shape']
            shape2 = env[tensor2]['shape']

            rows, cols1 = int(shape1[0]), int(shape1[1])
            cols2 = int(shape2[1])

            kernel = include + f"""
__global__ void concat_axis1_kernel_{name}({ctype}* input1, {ctype}* input2, {ctype}* output,
                                        int rows, int cols1, int cols2) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = cols1 + cols2;

    if (row < rows && col < total_cols) {{
        if (col < cols1) {{
            output[row * total_cols + col] = input1[row * cols1 + col];
        }} else {{
            int src_col = col - cols1;
            output[row * total_cols + col] = input2[row * cols2 + src_col];
        }}
    }}
}}
extern "C" void launch_concat_{name}({ctype}* input1, {ctype}* input2, {ctype}* output,
                                    int rows, int cols1, int cols2) {{
    dim3 block(16, 16);
    int total_cols = cols1 + cols2;
    dim3 grid((total_cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    concat_axis1_kernel_{name}<<<grid, block>>>(input1, input2, output, rows, cols1, cols2);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_axis1", name, tensor1, tensor2, rows, cols1, cols2)

        # --- 2 tensors, 1D (new) ---
        if len(tensor_names) == 2 and ndim == 1:
            tensor1, tensor2 = tensor_names
            shape1 = env[tensor1]['shape']
            shape2 = env[tensor2]['shape']

            size1 = int(shape1[0])
            size2 = int(shape2[0])
            total_size = size1 + size2

            kernel = include + f"""
__global__ void concat_1d_kernel_{name}({ctype}* input1, {ctype}* input2, {ctype}* output,
                                       int size1, int size2) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = size1 + size2;
    if (idx < total_size) {{
        if (idx < size1) {{
            output[idx] = input1[idx];
        }} else {{
            output[idx] = input2[idx - size1];
        }}
    }}
}}
extern "C" void launch_concat_{name}({ctype}* input1, {ctype}* input2, {ctype}* output,
                                    int size1, int size2) {{
    dim3 block(256);
    int total_size = size1 + size2;
    dim3 grid((total_size + block.x - 1) / block.x);
    concat_1d_kernel_{name}<<<grid, block>>>(input1, input2, output, size1, size2);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_1d", name, tensor1, tensor2, size1, size2)

        # --- N-D concat (general, 2 tensors) ---
        if len(tensor_names) == 2 and ndim >= 1:
            tensor1, tensor2 = tensor_names
            shape1 = tuple(int(d) for d in env[tensor1]['shape'])
            shape2 = tuple(int(d) for d in env[tensor2]['shape'])
            output_shape = tuple(int(d) for d in env[name]['shape']) if name in env else None

            # Compute output shape if not in env
            if output_shape is None:
                output_shape = list(shape1)
                output_shape[axis] = shape1[axis] + shape2[axis]
                output_shape = tuple(output_shape)

            total_output = int(np.prod(output_shape)) if output_shape else 1
            size1_axis = int(shape1[axis])

            shape1_str = ", ".join(str(d) for d in shape1)
            shape2_str = ", ".join(str(d) for d in shape2)
            out_shape_str = ", ".join(str(d) for d in output_shape)

            kernel = include + f"""
__global__ void concat_nd_kernel_{name}(
    {ctype}* input1, {ctype}* input2, {ctype}* output,
    const int* shape1, const int* shape2, const int* out_shape,
    int ndim, int axis, int size1_axis, int total_output) {{

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_output) return;

    // Decompose output flat index into output coords
    int out_coords[8];
    int out_strides[8];
    if (ndim > 0) {{
        out_strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) {{
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
        }}
        int temp = out_idx;
        for (int d = 0; d < ndim; d++) {{
            out_coords[d] = temp / out_strides[d];
            temp %= out_strides[d];
        }}
    }}

    // Compute strides for input1
    int in1_strides[8];
    if (ndim > 0) {{
        in1_strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) {{
            in1_strides[d] = in1_strides[d + 1] * shape1[d + 1];
        }}
    }}

    // Compute strides for input2
    int in2_strides[8];
    if (ndim > 0) {{
        in2_strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) {{
            in2_strides[d] = in2_strides[d + 1] * shape2[d + 1];
        }}
    }}

    int axis_coord = out_coords[axis];

    if (axis_coord < size1_axis) {{
        // Source is input1
        int src_idx = 0;
        for (int d = 0; d < ndim; d++) {{
            src_idx += out_coords[d] * in1_strides[d];
        }}
        output[out_idx] = input1[src_idx];
    }} else {{
        // Source is input2
        int src_coords[8];
        for (int d = 0; d < ndim; d++) {{
            src_coords[d] = out_coords[d];
        }}
        src_coords[axis] = axis_coord - size1_axis;

        int src_idx = 0;
        for (int d = 0; d < ndim; d++) {{
            src_idx += src_coords[d] * in2_strides[d];
        }}
        output[out_idx] = input2[src_idx];
    }}
}}
extern "C" void launch_concat_{name}(
    {ctype}* input1, {ctype}* input2, {ctype}* output,
    const int* shape1, const int* shape2, const int* out_shape,
    int ndim, int axis, int size1_axis, int total_output) {{

    dim3 block(256);
    dim3 grid((total_output + block.x - 1) / block.x);
    concat_nd_kernel_{name}<<<grid, block>>>(
        input1, input2, output, shape1, shape2, out_shape,
        ndim, axis, size1_axis, total_output);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            kernel_info = ("concat_nd", name, tensor1, tensor2, ndim, total_output,
                           list(shape1), list(shape2), list(output_shape), axis, size1_axis)
            return kernel, kernel_info

        return None, None
