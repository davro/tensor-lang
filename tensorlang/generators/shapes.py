import numpy as np

from .base import BaseGenerator


class ShapesMixin(BaseGenerator):
    def transpose(self, op_type, name, input_shape, tensor_name, axes=None):
        input_shape = tuple(int(d) for d in input_shape)
        ndim = len(input_shape)

        if axes is None:
            axes = list(range(ndim - 1, -1, -1))
        else:
            axes = [int(a) for a in axes]

        output_shape = tuple(input_shape[i] for i in axes)
        total_elements = int(np.prod(input_shape)) if input_shape else 1

        if ndim == 0:
            kernel = f"""
__global__ void transpose_kernel_{name}(float* input, float* output, int total_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {{
        output[idx] = input[idx];
    }}
}}
extern "C" void launch_transpose_{name}(float* input, float* output, int total_elements) {{
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

        kernel = f"""
__global__ void transpose_kernel_{name}(float* input, float* output, int total_elements) {{
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
extern "C" void launch_transpose_{name}(float* input, float* output, int total_elements) {{
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    transpose_kernel_{name}<<<grid, block>>>(input, output, total_elements);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, ("transpose", name, tensor_name, None, total_elements)

    def reshape(self, op_type, name, tensor_name, total_elements):
        kernel = f"""
__global__ void reshape_kernel_{name}(float* input, float* output, int total_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {{
        output[idx] = input[idx];
    }}
}}
extern "C" void launch_reshape_{name}(float* input, float* output, int total_elements) {{
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    reshape_kernel_{name}<<<grid, block>>>(input, output, total_elements);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
        return kernel, (f"{op_type}", name, tensor_name, None, total_elements)

    def slice(self, op_type, name, tensor_name, slice_specs, input_shape, output_shape):
        if len(input_shape) == 2 and len(slice_specs) <= 2:
            rows, cols = int(input_shape[0]), int(input_shape[1])

            row_spec = slice_specs[0] if len(slice_specs) > 0 else {'type': 'full_slice'}
            col_spec = slice_specs[1] if len(slice_specs) > 1 else {'type': 'full_slice'}

            if row_spec['type'] == 'slice':
                row_start = row_spec.get('start')
                row_end = row_spec.get('end')
                if row_start is None:
                    row_start = 0
                if row_end is None:
                    row_end = rows
            elif row_spec['type'] == 'index':
                row_start = row_spec['value']
                row_end = row_spec['value'] + 1
            else:
                row_start, row_end = 0, rows

            if col_spec['type'] == 'slice':
                col_start = col_spec.get('start')
                col_end = col_spec.get('end')
                if col_start is None:
                    col_start = 0
                if col_end is None:
                    col_end = cols
            elif col_spec['type'] == 'index':
                col_start = col_spec['value']
                col_end = col_spec['value'] + 1
            else:
                col_start, col_end = 0, cols

            row_start = max(0, min(row_start, rows))
            row_end = max(row_start, min(row_end, rows))
            col_start = max(0, min(col_start, cols))
            col_end = max(col_start, min(col_end, cols))

            out_rows = max(0, row_end - row_start)
            out_cols = max(0, col_end - col_start)

            kernel = f"""
__global__ void slice_kernel_{name}(float* input, float* output,
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

extern "C" void launch_slice_{name}(float* input, float* output,
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

        elif len(input_shape) == 1 and len(slice_specs) == 1:
            size = int(input_shape[0])
            spec = slice_specs[0]

            if spec['type'] == 'slice':
                start = spec.get('start')
                end = spec.get('end')
                if start is None:
                    start = 0
                if end is None:
                    end = size
            elif spec['type'] == 'index':
                start = spec['value']
                end = spec['value'] + 1
            else:
                start, end = 0, size

            start = max(0, min(start, size))
            end = max(start, min(end, size))
            out_size = max(0, end - start)

            kernel = f"""
__global__ void slice_1d_kernel_{name}(float* input, float* output, int start, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = input[start + idx];
    }}
}}

extern "C" void launch_slice_{name}(float* input, float* output, int start, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    slice_1d_kernel_{name}<<<grid, block>>>(input, output, start, size);
    cudaDeviceSynchronize();
    {self.cuda_debug()}
}}
"""
            return kernel, (f"{op_type}_1d", name, tensor_name, None, start, out_size)

    def concat(self, op_type, name, tensor_names, axis, env):
        if len(tensor_names) == 2 and axis == 0:
            tensor1, tensor2 = tensor_names
            shape1 = env[tensor1]['shape']
            shape2 = env[tensor2]['shape']

            if len(shape1) == 2:
                rows1, cols = int(shape1[0]), int(shape1[1])
                rows2 = int(shape2[0])

                kernel = f"""
__global__ void concat_axis0_kernel_{name}(float* input1, float* input2, float* output,
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
extern "C" void launch_concat_{name}(float* input1, float* input2, float* output,
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

        if len(tensor_names) == 2 and axis == 1:
            tensor1, tensor2 = tensor_names
            shape1 = env[tensor1]['shape']
            shape2 = env[tensor2]['shape']

            if len(shape1) == 2:
                rows, cols1 = int(shape1[0]), int(shape1[1])
                cols2 = int(shape2[1])

                kernel = f"""
__global__ void concat_axis1_kernel_{name}(float* input1, float* input2, float* output,
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
extern "C" void launch_concat_{name}(float* input1, float* input2, float* output,
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

