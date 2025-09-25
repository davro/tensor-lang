import sys
import os
import numpy as np
import subprocess
import traceback
from lark import Lark, Tree, Token, UnexpectedInput
from datetime import datetime
from ctypes import c_void_p, c_int, c_float
from functools import reduce

# ================================================================
#                      TensorLang version
# ================================================================
version = "0.2.1"

# ================================================================
#                 GRAMMAR lark + TensorLang file
# ================================================================
try:
    with open('tensorlang.lark', 'r') as f:
        grammar = f.read()
    parser = Lark(grammar, start='program', parser='lalr')
    print("TensorLang: Grammer tensorlang.lark opened.")

except FileNotFoundError:
    print("TensorLang: Grammer error tensorlang.lark not found.")
    sys.exit(1)

# Read tensorlang code from command line arguemnt default to a test
if len(sys.argv) < 2:
    tensorlang_file = 'tests/add.tl'  # Default
else:
    tensorlang_file = sys.argv[1]

print(f"TensorLang: Loaded {tensorlang_file}.")
try:
    with open(tensorlang_file, 'r') as f:
        code = f.read()
    print(f"TensorLang: CODE:\n{code}")

except FileNotFoundError:
    print(f"Error: {tensorlang_file} not found. Please provide a valid TensorLang file.")
    sys.exit(1)

# ================================================================
#                         PARSER + AST
# ================================================================
try:
    parse_tree = parser.parse(code)
    print(f"TensorLang: Parsed AST:\n{parse_tree.pretty()}")

    def build_ast(tree):
        ast = []
        output_tensor = None
        print(f"Building AST from {tree.data}")
        if tree.data == 'program':
            for child in tree.children:
                if isinstance(child, Tree) and child.data == 'statement':
                    if child.children[0].data == 'let_binding':
                        let_node = build_let_binding(child.children[0])
                        if let_node:
                            ast.append(let_node)
                            print(f"Added node to AST: {let_node}")
                    elif child.children[0].data == 'expr':
                        expr_node = build_expr(child.children[0])
                        if expr_node['type'] == 'name':
                            output_tensor = expr_node['name']
                            print(f"Set output tensor: {output_tensor}")
        # If no output_tensor is set, default to the last defined tensor
        if not output_tensor and ast:
            output_tensor = ast[-1]['name']
            print(f"No explicit output tensor; defaulting to last tensor: {output_tensor}")
        return ast, output_tensor

    def build_let_binding(tree):
        print(f"Building let_binding from {tree.data}")
        if tree.data != 'let_binding':
            print(f"Invalid let_binding node: {tree.data}")
            return None
        children = tree.children
        if len(children) == 3:
            name = children[0].value
            print(f"Processing let binding for {name}")
            if isinstance(children[1], Tree) and children[1].data == 'type':
                ty = build_type(children[1])
                expr = build_expr(children[2])
            else:
                ty = None
                expr = build_expr(children[2])
            return {'type': 'let', 'name': name, 'ty': ty, 'expr': expr, 'tree': tree}
        else:
            print(f"Unexpected number of children in let_binding: {len(children)}")
            return None

    def build_type(tree):
        print(f"Building type from {tree.data}")
        if tree.data != 'type':
            print(f"Invalid type node: {tree.data}")
            return None
        children = tree.children
        dtype_value = 'f32'  # Default
        if len(children) > 0 and isinstance(children[0], Token) and children[0].value in ['f32', 'f64']:
            dtype_value = children[0].value
        shape_tree = children[1] if len(children) > 1 and isinstance(children[1], Tree) else None
        shape = build_shape(shape_tree) if shape_tree else (0, 0)
        return {'dtype': dtype_value, 'shape': shape}

    def build_shape(tree):
        print(f"Building shape from {tree.data}")
        if tree.data != 'shape':
            print(f"Invalid shape node: {tree.data}")
            return (0, 0)
        nums = [int(float(child.value)) for child in tree.children if isinstance(child, Token) and child.type == 'NUMBER']
        print(f"Shape numbers: {nums}")
        return tuple(nums)

    def build_expr(tree):
        print(f"Building expr from {tree.data}")

        if tree.data == 'expr':
            tree = tree.children[0]  # Unwrap expr

        if isinstance(tree, Token) and tree.type == 'NAME':
            print(f"Expression is NAME: {tree.value}")
            return {'type': 'name', 'name': tree.value}

        if tree.data == 'matmul_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Matmul args: {args}")
            return {'type': 'matmul', 'args': args}

        elif tree.data == 'add_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Add args: {args}")
            return {'type': 'add', 'args': args}

        elif tree.data == 'minus_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Minus args: {args}")
            return {'type': 'minus', 'args': args}

        elif tree.data == 'mult_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Mult args: {args}")
            return {'type': 'mult', 'args': args}

        elif tree.data == 'div_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Div args: {args}")
            return {'type': 'div', 'args': args}

        elif tree.data == 'relu_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"ReLU args: {args}")
            return {'type': 'relu', 'args': args}


        elif tree.data == 'sigmoid_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Sigmoid args: {args}")
            return {'type': 'sigmoid', 'args': args}

        elif tree.data == 'tanh_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Tanh args: {args}")
            return {'type': 'tanh', 'args': args}

        elif tree.data == 'softmax_call':
            tensor_name = None
            axis = None
            
            for child in tree.children:
                if isinstance(child, Token) and child.type == 'NAME':
                    tensor_name = child.value
                elif isinstance(child, Token) and child.type == 'NUMBER':
                    axis = int(float(child.value))
            
            # Default to last axis if not specified
            print(f"Softmax args: tensor={tensor_name}, axis={axis}")
            return {'type': 'softmax', 'tensor': tensor_name, 'axis': axis}





        elif tree.data == 'tensor_literal':
            data = []
            is_1d = True
            for child in tree.children:
                if isinstance(child, Tree) and child.data == 'inner_array':
                    is_1d = False
                    inner_nums = [float(c.value) for c in child.children if isinstance(c, Token) and c.type == 'NUMBER']
                    data.extend(inner_nums)
                elif isinstance(child, Token) and child.type == 'NUMBER':
                    data.append(float(child.value))
            print(f"Tensor literal data: {data}, is_1d: {is_1d}")
            return {'type': 'tensor_literal', 'data': data, 'is_1d': is_1d, 'tree': tree}

        elif tree.data == 'fill_call':
            value = float(tree.children[0].value) if isinstance(tree.children[0], Token) and tree.children[0].type == 'NUMBER' else 0.0
            shape_tree = tree.children[1] if len(tree.children) > 1 and isinstance(tree.children[1], Tree) else None
            shape = build_shape(shape_tree) if shape_tree else (1,)
            print(f"Fill value: {value}, shape: {shape}")
            return {'type': 'fill', 'value': value, 'shape': shape}

        elif tree.data == 'sum_call':
            tensor_name = None
            axis = None
            for child in tree.children:
                if isinstance(child, Token) and child.type == 'NAME':
                    tensor_name = child.value
                elif isinstance(child, Token) and child.type == 'NUMBER':
                    axis = int(float(child.value))
            
            print(f"Sum args: tensor={tensor_name}, axis={axis}")
            return {'type': 'sum', 'tensor': tensor_name, 'axis': axis}

        elif tree.data == 'mean_call':
            tensor_name = None
            axis = None
            
            for child in tree.children:
                if isinstance(child, Token) and child.type == 'NAME':
                    tensor_name = child.value
                elif isinstance(child, Token) and child.type == 'NUMBER':
                    axis = int(float(child.value))
            
            print(f"Mean args: tensor={tensor_name}, axis={axis}")
            return {'type': 'mean', 'tensor': tensor_name, 'axis': axis}
            

        print(f"Unrecognized expr type: {tree.data}")
        return None

    # Type checker
    def type_checker(ast):
        print("TensorLang: Type checker")

        env = {}
        for node in ast:
            if node['type'] == 'let':
                name = node['name']
                expr = node['expr']
                ty = node['ty']
                print(f"Type checking {name}")
                if ty and isinstance(ty, dict) and 'dtype' in ty:
                    env[name] = {'dtype': ty['dtype'], 'shape': ty['shape']}
                    print(f"Assigned type from ty: {env[name]}")
                elif isinstance(expr, dict) and expr['type'] == 'tensor_literal':
                    data = expr['data']
                    num_elements = len(data)
                    if expr['is_1d']:
                        shape = (num_elements,)
                    else:
                        rows = sum(1 for child in expr['tree'].children if child.data == 'inner_array')
                        cols = num_elements // rows if rows > 0 else num_elements
                        shape = (rows, cols) if rows > 1 else (cols,)
                    env[name] = {'dtype': 'f32', 'shape': shape}
                    print(f"Inferred shape for {name}: {env[name]['shape']}")

                #elif isinstance(expr, dict) and expr['type'] in ['matmul', 'add', 'minus', 'mult', 'div', 'relu', 'fill', 'sum', 'mean']:
                elif isinstance(expr, dict) and expr['type'] in ['matmul', 'add', 'minus', 'mult', 'div', 'relu', 'sigmoid', 'tanh', 'softmax', 'fill', 'sum', 'mean']:

                    if expr['type'] == 'fill':
                        env[name] = {'dtype': 'f32', 'shape': expr['shape']}
                        print(f"Assigned type from fill: {env[name]}")
                    
                    elif expr['type'] in ['sum', 'mean']:
                        tensor_name = expr['tensor']
                        if tensor_name not in env:
                            print(f"Type error: Undefined tensor {tensor_name} for {expr['type']}")
                            return False, env
                        
                        input_shape = env[tensor_name]['shape']
                        axis = expr.get('axis')
                        
                        if axis is None:
                            # Full reduction - result is scalar (shape = ())
                            output_shape = ()
                        else:
                            # Reduction along specific axis
                            if axis < 0 or axis >= len(input_shape):
                                print(f"Type error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                                return False, env
                            # Remove the reduced dimension
                            output_shape = tuple(dim for i, dim in enumerate(input_shape) if i != axis)
                            if not output_shape:  # If all dimensions reduced, result is scalar
                                output_shape = ()
                        
                        env[name] = {'dtype': 'f32', 'shape': output_shape}
                        print(f"Assigned type for {name} ({expr['type']}): {env[name]}")

                    elif expr['type'] == 'softmax':
                        print(f"DEBUG: Processing softmax type checking for {name}")
                        tensor_name = expr['tensor']
                        print(f"DEBUG: Softmax tensor_name = {tensor_name}")
                        if tensor_name not in env:
                            print(f"Type error: Undefined tensor {tensor_name} for softmax")
                            return False, env
                        
                        # Softmax preserves input shape
                        env[name] = {'dtype': 'f32', 'shape': env[tensor_name]['shape']}
                        print(f"Assigned type for {name} (softmax): {env[name]}")

                    else:
                        # Handle other operations (matmul, add, minus, mult, div, relu, sigmoid, tanh etc.)
                        if expr['type'] in ['sum', 'mean']:
                            # This is handled above, but keeping this check for safety
                            pass
                        else:
                            arg_names = expr.get('args', [])
                            if arg_names:  # Only process if there are args
                                args = [env.get(arg_name) for arg_name in arg_names]
                                print(f"Checking args for {name}: {args}")
                                if not all(args):
                                    print(f"Type error: Undefined args for {name}")
                                    return False, env
                                    
                                if expr['type'] == 'matmul':
                                    if args[0]['shape'][1] != args[1]['shape'][0]:
                                        print(f"Type error: Matmul shape mismatch for {name}, {args[0]['shape']} x {args[1]['shape']}")
                                        return False, env
                                    env[name] = {'dtype': 'f32', 'shape': (args[0]['shape'][0], args[1]['shape'][1])}
                                    print(f"Assigned type for {name}: {env[name]}")

                                elif expr['type'] in ['add', 'minus', 'mult', 'div']:
                                    shape1, shape2 = args[0]['shape'], args[1]['shape']
                                    # Broadcasting rules: shapes are compatible if equal or one is 1
                                    if len(shape1) < len(shape2):
                                        shape1, shape2 = shape2, shape1  # Ensure shape1 is the larger shape
                                    if len(shape2) == 0:  # Scalar case
                                        output_shape = shape1
                                    else:
                                        output_shape = []
                                        for d1, d2 in zip(shape1[-len(shape2):], shape2):
                                            if d1 == d2 or d2 == 1:
                                                output_shape.append(d1)
                                            else:
                                                print(f"Type error: {expr['type']} shape mismatch for {name}, {shape1} != {shape2}")
                                                return False, env
                                        output_shape = shape1[:-len(shape2)] + tuple(output_shape)
                                    env[name] = {'dtype': 'f32', 'shape': output_shape}
                                    print(f"Assigned type for {name}: {env[name]}")

                                elif expr['type'] == 'relu':
                                    env[name] = {'dtype': 'f32', 'shape': args[0]['shape']}
                                    print(f"Assigned type for {name}: {env[name]}")

                                elif expr['type'] in ['sigmoid', 'tanh']:  # Move this here
                                    env[name] = {'dtype': 'f32', 'shape': args[0]['shape']}
                                    print(f"Assigned type for {name}: {env[name]}")

                else:
                    print(f"Type error: Unrecognized expr type for {name}: {expr['type']}")
                    return False, env
        return True, env

    def prod(lst):
        return reduce(lambda x, y: x * y, lst, 1)


    # ================================================================
    #                Parser / Compiler implementation
    # ================================================================

    # Build the AST 
    ast, output_tensor = build_ast(parse_tree)
    print(f"BUILT AST:\n{ast}")
    print(f"Output Tensor: {output_tensor}")

    # Run type checker
    success, env = type_checker(ast)
    print(f"TYPE CHECKER ENV:\n{env}")

    # CUDA generation and execution
    if success:
        # Store tensor data and GPU allocations
        tensors = {}
        gpu_allocs = {}
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        cuda_code = f"""
// -----------------------------------------------------------------------------
// This CUDA kernel was automatically generated by TensorLang
// TensorLang (c) 2025 David Stevens https://github.com/davro/tensor-lang
// Licensed under the GNU Lesser General Public License v3 (LGPL-3.0)
// -----------------------------------------------------------------------------
// NOTE: This file is machine-generated. Do not edit manually.
// -----------------------------------------------------------------------------
// Build date     : {date_str}
// TensorLang ver : {version}
// Source file    : {tensorlang_file}
// -----------------------------------------------------------------------------        
"""
        kernels = []

        # Generate kernels for operations
        for node in ast:
            if node['type'] == 'let' and isinstance(node['expr'], dict):
                name = node['name']
                expr = node['expr']
                print(f"Generating kernel for {name} ({expr['type']})")

                # ========================================
                # Tensor Literal
                # ========================================
                if expr['type'] == 'tensor_literal':
                    shape = tuple(int(dim) for dim in env[name]['shape'])
                    tensors[name] = np.array(expr['data'], dtype=np.float32).reshape(shape)
                    print(f"Initialized tensor {name} with shape {shape}")

                # ========================================
                # MATMUL
                # ========================================
                elif expr['type'] == 'matmul':
                    arg1, arg2 = expr['args']
                    m, n = int(env[arg1]['shape'][0]), int(env[arg1]['shape'][1])
                    p = int(env[arg2]['shape'][1])
                    kernel = f"""
__global__ void matmul_kernel_{name}(float* A, float* B, float* C, int M, int N, int P) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < P) {{
        float sum = 0.0;
        for (int k = 0; k < N; k++) {{
            sum += A[i * N + k] * B[k * P + j];
        }}
        C[i * P + j] = sum;
    }}
}}
extern "C" void launch_matmul_{name}(float* A, float* B, float* C, int M, int N, int P) {{
    dim3 block(16, 16);
    dim3 grid((P + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel_{name}<<<grid, block>>>(A, B, C, M, N, P);
    cudaDeviceSynchronize();
}}
"""
                    kernels.append(('matmul', name, arg1, arg2, m, n, p))
                    cuda_code += kernel

                # ========================================
                # ADD
                # ======================================== 
                elif expr['type'] == 'add':
                    arg1, arg2 = expr['args']
                    shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                    output_shape = env[name]['shape']
                    if shape1 == shape2:
                        # Element-wise add
                        size = int(np.prod([int(dim) for dim in output_shape]))
                        kernel = f"""
__global__ void add_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = A[idx] + B[idx];
    }}
}}
extern "C" void launch_add_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    add_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('add', name, arg1, arg2, size))
                    else:
                        # Broadcasting add (e.g., (4,5) + (5,) -> (4,5))
                        rows, cols = output_shape
                        kernel = f"""
__global__ void add_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = A[i * cols + j] + B[j];
    }}
}}
extern "C" void launch_add_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    add_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('add_broadcast', name, arg1, arg2, rows, cols))
                    cuda_code += kernel

                # ========================================
                # MINUS
                # ========================================
                elif expr['type'] == 'minus':
                    arg1, arg2 = expr['args']
                    shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                    output_shape = env[name]['shape']
                    if shape1 == shape2:
                        # Element-wise minus
                        size = int(np.prod([int(dim) for dim in output_shape]))
                        kernel = f"""
__global__ void minus_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = A[idx] - B[idx];
    }}
}}
extern "C" void launch_minus_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    minus_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('minus', name, arg1, arg2, size))
                    else:
                        # Broadcasting minus (e.g., (4,5) - (5,) -> (4,5))
                        rows, cols = output_shape
                        kernel = f"""
__global__ void minus_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = A[i * cols + j] - B[j];
    }}
}}
extern "C" void launch_minus_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    minus_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('minus_broadcast', name, arg1, arg2, rows, cols))
                    cuda_code += kernel

                # ========================================
                # MULT
                # ========================================
                elif expr['type'] == 'mult':
                    arg1, arg2 = expr['args']
                    shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                    output_shape = env[name]['shape']
                    if shape1 == shape2:
                        # Element-wise mult
                        size = int(np.prod([int(dim) for dim in output_shape]))
                        kernel = f"""
__global__ void mult_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = A[idx] * B[idx];
    }}
}}
extern "C" void launch_mult_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    mult_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('mult', name, arg1, arg2, size))
                    else:
                        # Broadcasting mult (e.g., (4,5) * (5,) -> (4,5))
                        rows, cols = output_shape
                        kernel = f"""
__global__ void mult_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = A[i * cols + j] * B[j];
    }}
}}
extern "C" void launch_mult_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    mult_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('mult_broadcast', name, arg1, arg2, rows, cols))
                    cuda_code += kernel

                # ========================================
                # DIV
                # ========================================
                elif expr['type'] == 'div':
                    arg1, arg2 = expr['args']
                    shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                    output_shape = env[name]['shape']
                    if shape1 == shape2:
                        # Element-wise div
                        size = int(np.prod([int(dim) for dim in output_shape]))
                        kernel = f"""
__global__ void div_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = A[idx] / B[idx];
    }}
}}
extern "C" void launch_div_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    div_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('div', name, arg1, arg2, size))
                    else:
                        # Broadcasting div (e.g., (4,5) / (5,) -> (4,5))
                        rows, cols = output_shape
                        kernel = f"""
__global__ void div_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = A[i * cols + j] / B[j];
    }}
}}
extern "C" void launch_div_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    div_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('div_broadcast', name, arg1, arg2, rows, cols))
                    cuda_code += kernel

                # ========================================
                # ReLU - Rectified linear unit
                # ========================================
                elif expr['type'] == 'relu':
                    arg1 = expr['args'][0]
                    size = int(np.prod([int(dim) for dim in env[arg1]['shape']]))
                    kernel = f"""
__global__ void relu_kernel_{name}(float* input, float* output, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = fmaxf(0.0f, input[idx]);
    }}
}}
extern "C" void launch_relu_{name}(float* input, float* output, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    relu_kernel_{name}<<<grid, block>>>(input, output, size);
    cudaDeviceSynchronize();
}}
"""
                    kernels.append(('relu', name, arg1, None, size))
                    cuda_code += kernel

                # ========================================
                # SIGMOID
                # ========================================
                elif expr['type'] == 'sigmoid':
                    arg1 = expr['args'][0]
                    size = int(np.prod([int(dim) for dim in env[arg1]['shape']]))
                    kernel = f"""
__global__ void sigmoid_kernel_{name}(float* input, float* output, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }}
}}
extern "C" void launch_sigmoid_{name}(float* input, float* output, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    sigmoid_kernel_{name}<<<grid, block>>>(input, output, size);
    cudaDeviceSynchronize();
}}
"""
                    kernels.append(('sigmoid', name, arg1, None, size))
                    cuda_code += kernel

                # ========================================
                # TANH
                # ========================================
                elif expr['type'] == 'tanh':
                    arg1 = expr['args'][0]
                    size = int(np.prod([int(dim) for dim in env[arg1]['shape']]))
                    kernel = f"""
__global__ void tanh_kernel_{name}(float* input, float* output, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = tanhf(input[idx]);
    }}
}}
extern "C" void launch_tanh_{name}(float* input, float* output, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    tanh_kernel_{name}<<<grid, block>>>(input, output, size);
    cudaDeviceSynchronize();
}}
"""
                    kernels.append(('tanh', name, arg1, None, size))
                    cuda_code += kernel

                # ========================================
                # SOFTMAX
                # ========================================
                elif expr['type'] == 'softmax':
                    tensor_name = expr['tensor']
                    axis = expr.get('axis')
                    input_shape = env[tensor_name]['shape']
                    
                    if axis is None:
                        # Default to last axis
                        axis = len(input_shape) - 1
                    
                    if len(input_shape) == 2 and axis == 1:
                        # Softmax along rows (most common case)
                        rows, cols = int(input_shape[0]), int(input_shape[1])
                        kernel = f"""
__global__ void softmax_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        // Find max for numerical stability
        float max_val = input[row * cols];
        for (int col = 1; col < cols; col++) {{
            max_val = fmaxf(max_val, input[row * cols + col]);
        }}
        
        // Compute exp(x - max) and sum
        float sum_exp = 0.0f;
        for (int col = 0; col < cols; col++) {{
            float exp_val = expf(input[row * cols + col] - max_val);
            output[row * cols + col] = exp_val;
            sum_exp += exp_val;
        }}
        
        // Normalize by sum
        for (int col = 0; col < cols; col++) {{
            output[row * cols + col] /= sum_exp;
        }}
    }}
}}
extern "C" void launch_softmax_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    softmax_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('softmax', name, tensor_name, None, rows, cols, axis))

                    elif len(input_shape) == 1:
                        # 1D softmax
                        size = int(input_shape[0])
                        kernel = f"""
__global__ void softmax_1d_kernel_{name}(float* input, float* output, int size) {{
    if (threadIdx.x == 0 && blockIdx.x == 0) {{
        // Find max for numerical stability
        float max_val = input[0];
        for (int i = 1; i < size; i++) {{
            max_val = fmaxf(max_val, input[i]);
        }}
        
        // Compute exp(x - max) and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < size; i++) {{
            float exp_val = expf(input[i] - max_val);
            output[i] = exp_val;
            sum_exp += exp_val;
        }}
        
        // Normalize by sum
        for (int i = 0; i < size; i++) {{
            output[i] /= sum_exp;
        }}
    }}
}}
extern "C" void launch_softmax_{name}(float* input, float* output, int size) {{
    dim3 block(1);
    dim3 grid(1);
    softmax_1d_kernel_{name}<<<grid, block>>>(input, output, size);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('softmax_1d', name, tensor_name, None, size))
                    cuda_code += kernel







                # ========================================
                # SUM
                # ========================================
                elif expr['type'] == 'sum':
                    tensor_name = expr['tensor']
                    axis = expr.get('axis')
                    input_shape = env[tensor_name]['shape']
                    print(f"DEBUG SUM: tensor={tensor_name}, axis={axis}, shape={input_shape}")
                    
                    if axis is None:
                        print("DEBUG: Taking sum_full path")
                        # Full reduction to scalar
                        size = int(np.prod([int(dim) for dim in input_shape]))
                        kernel = f"""
__global__ void sum_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}
    
    // Write result for this block to global memory
    if (tid == 0) {{
        atomicAdd(output, sdata[0]);
    }}
}}
extern "C" void launch_sum_{name}(float* input, float* output, int size) {{
    // Initialize output to zero
    cudaMemset(output, 0, sizeof(float));
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    sum_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
}}
"""
                        kernels.append(('sum_full', name, tensor_name, None, size))
                    else:
                        # Reduction along specific axis
                        print(f"DEBUG: Taking axis-specific path, axis={axis}")
                        if len(input_shape) == 2 and axis == 1:
                            print("DEBUG: Using sum_axis kernel (axis=1)")
                            # Sum along columns (each row sums to one value)
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
}}
"""
                            kernels.append(('sum_axis', name, tensor_name, None, rows, cols, axis))
                        elif len(input_shape) == 2 and axis == 0:
                            # Sum along rows (each column sums to one value)
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
}}
"""
                            kernels.append(('sum_axis0', name, tensor_name, None, rows, cols, axis))
                    cuda_code += kernel

                # ========================================
                # MEAN
                # ========================================
                elif expr['type'] == 'mean':
                    tensor_name = expr['tensor']
                    axis = expr.get('axis')
                    input_shape = env[tensor_name]['shape']
                    
                    if axis is None:
                        # Full mean to scalar
                        size = int(np.prod([int(dim) for dim in input_shape]))
                        kernel = f"""
__global__ void mean_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}
    
    // Write result for this block to global memory
    if (tid == 0) {{
        atomicAdd(output, sdata[0]);
    }}
}}
extern "C" void launch_mean_{name}(float* input, float* output, int size) {{
    // Initialize output to zero
    cudaMemset(output, 0, sizeof(float));
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    mean_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    
    // Divide by size to get mean
    float mean_val;
    cudaMemcpy(&mean_val, output, sizeof(float), cudaMemcpyDeviceToHost);
    mean_val /= size;
    cudaMemcpy(output, &mean_val, sizeof(float), cudaMemcpyHostToDevice);
}}
"""
                        kernels.append(('mean_full', name, tensor_name, None, size))
                    else:
                        # Mean along specific axis
                        if len(input_shape) == 2 and axis == 1:
                            # Mean along columns
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
}}
"""
                            kernels.append(('mean_axis', name, tensor_name, None, rows, cols, axis))
                        elif len(input_shape) == 2 and axis == 0:
                            # Mean along rows
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
}}
"""
                            kernels.append(('mean_axis0', name, tensor_name, None, rows, cols, axis))
                    cuda_code += kernel


                # ========================================
                # FILL
                # ========================================
                elif expr['type'] == 'fill':
                    size = int(np.prod([int(dim) for dim in expr['shape']]))
                    kernel = f"""
__global__ void fill_kernel_{name}(float* output, float value, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = value;
    }}
}}
extern "C" void launch_fill_{name}(float* output, float value, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    fill_kernel_{name}<<<grid, block>>>(output, value, size);
    cudaDeviceSynchronize();
}}
"""
                    kernels.append(('fill', name, None, None, size, expr['value']))
                    cuda_code += kernel



        if kernels:

            # Ensure cache output directory exists
            os.makedirs("cache", exist_ok=True)
            os.makedirs(f"cache/{tensorlang_file}", exist_ok=True)
            print("Created cache for tensor outputs directory")

            print("Generated CUDA kernels:")
            print(cuda_code.strip())

            # Compile CUDA code
            with open(f'cache/{tensorlang_file}/kernel.cu', 'w') as f:
                f.write(cuda_code)
            try:
                kernel_source_cuda = f'cache/{tensorlang_file}/kernel.cu'
                kernel_shared_object = f'cache/{tensorlang_file}/kernel.so'

                subprocess.run(['nvcc', '-o', kernel_shared_object, '--shared', '-Xcompiler', '-fPIC', '-lcudart', kernel_source_cuda], check=True)
                print("CUDA compiled to kernel.so!")

            except subprocess.CalledProcessError as e:
                print(f"CUDA compilation error: {e}")
                sys.exit(1)

            # Execute with PyCUDA
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit
                from ctypes import cdll

                lib = cdll.LoadLibrary(f'cache/{tensorlang_file}/kernel.so')
                print(f"CUDA Kernel loaded from cache/{tensorlang_file}/kernel.so")

                # Allocate GPU memory and copy inputs
                for name in env:
                    shape = tuple(int(dim) for dim in env[name]['shape'])
                    size_bytes = int(np.prod(shape) * np.float32().nbytes)
                    gpu_allocs[name] = cuda.mem_alloc(size_bytes)
                    if name in tensors:
                        cuda.memcpy_htod(gpu_allocs[name], tensors[name])
                        print(f"Copied {name} to GPU, shape: {tensors[name].shape}, sample: {tensors[name][:2] if tensors[name].ndim > 1 else tensors[name]}")
                    else:
                        print(f"Allocated GPU memory for {name} (uninitialized), shape: {shape}")

                # Execute operations
                for op_type, name, arg1, arg2, *dims in kernels:
                    shape = tuple(int(dim) for dim in env[name]['shape'])
                    print(f"Executing {op_type} for {name}, shape: {shape}")

                    if op_type == 'matmul':
                        m, n, p = dims
                        getattr(lib, f'launch_matmul_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])), c_int(m), c_int(n), c_int(p))
                    elif op_type == 'add':
                        size = dims[0]
                        getattr(lib, f'launch_add_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])), c_int(size))
                    elif op_type == 'add_broadcast':
                        rows, cols = dims
                        getattr(lib, f'launch_add_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])), c_int(rows), c_int(cols))
                    elif op_type == 'minus':
                        size = dims[0]
                        getattr(lib, f'launch_minus_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])), c_int(size))
                    elif op_type == 'minus_broadcast':
                        rows, cols = dims
                        getattr(lib, f'launch_minus_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])), c_int(rows), c_int(cols))
                    elif op_type == 'mult':
                        size = dims[0]
                        getattr(lib, f'launch_mult_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])), c_int(size))
                    elif op_type == 'mult_broadcast':
                        rows, cols = dims
                        getattr(lib, f'launch_mult_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])), c_int(rows), c_int(cols))
                    elif op_type == 'div':
                        size = dims[0]
                        getattr(lib, f'launch_div_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])), c_int(size))
                    elif op_type == 'div_broadcast':
                        rows, cols = dims
                        getattr(lib, f'launch_div_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])), c_int(rows), c_int(cols))
                    elif op_type == 'relu':
                        size = dims[0]
                        getattr(lib, f'launch_relu_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(size))


                    elif op_type == 'sigmoid':
                        size = dims[0]
                        getattr(lib, f'launch_sigmoid_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(size))
                    elif op_type == 'tanh':
                        size = dims[0]
                        getattr(lib, f'launch_tanh_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(size))
                    elif op_type == 'softmax':
                        rows, cols, axis = dims
                        getattr(lib, f'launch_softmax_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(rows), c_int(cols))
                    elif op_type == 'softmax_1d':
                        size = dims[0]
                        getattr(lib, f'launch_softmax_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(size))


                    elif op_type == 'sum_full':
                        size = dims[0]
                        getattr(lib, f'launch_sum_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(size))
                    elif op_type == 'sum_axis':
                        rows, cols, axis = dims
                        getattr(lib, f'launch_sum_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(rows), c_int(cols))
                    elif op_type == 'sum_axis0':
                        rows, cols, axis = dims
                        getattr(lib, f'launch_sum_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(rows), c_int(cols))
                    elif op_type == 'mean_full':
                        size = dims[0]
                        getattr(lib, f'launch_mean_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(size))
                    elif op_type == 'mean_axis':
                        rows, cols, axis = dims
                        getattr(lib, f'launch_mean_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(rows), c_int(cols))
                    elif op_type == 'mean_axis0':
                        rows, cols, axis = dims
                        getattr(lib, f'launch_mean_{name}')(c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])), c_int(rows), c_int(cols))


                    elif op_type == 'fill':
                        size, value = dims
                        getattr(lib, f'launch_fill_{name}')(c_void_p(int(gpu_allocs[name])), c_float(value), c_int(size))

                    # Save result for all computed tensors
                    output = np.zeros(shape, dtype=np.float32)
                    cuda.memcpy_dtoh(output, gpu_allocs[name])
                    tensors[name] = output

                    # Specific output used by tests runner for extracting the result of each tensor output logs
                    print(f"Result {name} ({op_type}):\n{output}")

                    np.save(f"cache/{tensorlang_file}/{name}.npy", output)
                    print(f"TensorLang Cache: saved {name} to cache/{tensorlang_file}/{name}.npy")

                # Free GPU memory
                for name, alloc in gpu_allocs.items():
                    alloc.free()
                    print(f"Freed GPU memory for {name}")

            except ImportError as e:
                print(f"PyCUDA error: {e}. Run 'pip install pycuda' and ensure CUDA toolkit is installed.")
                sys.exit(1)
            except Exception as e:
                print(f"Error executing CUDA kernel: {e}")
                traceback.print_exc()
                sys.exit(1)

except UnexpectedInput as e:
    print(f"Parse error: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during parsing or execution: {e}")
    traceback.print_exc()
    sys.exit(1)