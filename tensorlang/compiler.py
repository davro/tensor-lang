import sys
import os
import time
import errno
import textwrap  
import traceback 
import subprocess

from pathlib import Path
import numpy as np
from lark import Lark, Tree, Token, UnexpectedInput
from datetime import datetime
from ctypes import c_void_p, c_int, c_float
from functools import reduce
from typing import Optional
from pathlib import Path

from tensorlang.tensor_lang import TensorLang
from tensorlang.kernel_generator import KernelGenerator
from tensorlang.type_checker import type_checker
from tensorlang.ast_builder import build_ast

class TensorCompiler:

    def __init__(self, debug_mode=False, debug_info=False, debug_ast=False, cache_layers=False):
        self.tensorlang   = TensorLang()
        self.version      = self.tensorlang.version
        self.debug_mode   = debug_mode
        self.debug_info   = debug_info
        self.debug_ast    = debug_ast
        self.cache_layers = cache_layers
        self.tensorlang   = TensorLang()
        

    def prod(self, lst):
        """Compute product of list elements."""
        return reduce(lambda x, y: x * y, lst, 1)


    def can_broadcast(self, shape1, shape2):
        """Check if two shapes can broadcast together (NumPy rules)"""
        ndim = max(len(shape1), len(shape2))
        s1 = (1,) * (ndim - len(shape1)) + shape1
        s2 = (1,) * (ndim - len(shape2)) + shape2
        
        for d1, d2 in zip(s1, s2):
            if d1 != d2 and d1 != 1 and d2 != 1:
                return False
        return True


    def compile_and_execute(self, tensorlang_file):
        """Compile and execute a single .tl file."""
        
        # Resolve the file path (handle both absolute and relative)
        if tensorlang_file:
            file_path = Path(tensorlang_file)
        else:
            self.tensorlang.print(type="[TensorLang]", message=f"Missing file")
            return

        # if not file_path.is_absolute():
        #     file_path = Path.cwd() / file_path
        #print (f"FILE_PATH: {file_path}")

        # Check if file exists
        if not file_path.exists():
            print(f"Error: {tensorlang_file} not found at {file_path}")
            sys.exit(1)

        self.tensorlang.print_header(self.version)
        

        # Gather file details
        if file_path.suffix == '.tl':
            file_details = {
                "| Path     " : str(file_path),
                "| Name     " : file_path.name,
                "| Suffix   " : file_path.suffix or "None",
                "| Size     " : f"{file_path.stat().st_size} bytes",
                "| Modified " : time.ctime(file_path.stat().st_mtime),
            }
            details_str = "\n".join(f"{key}: {value}" for key, value in file_details.items())
            self.tensorlang.print(type=details_str)
        else:
            self.tensorlang.print(type=f"TensorLang file not found, suffix is not .tl") 
            self.tensorlang.seperator()
            sys.exit(1)

        with open(file_path, 'r') as f:
            code = f.read()

        self.tensorlang.seperator()
        self.tensorlang.print(type=f"{code}")
        self.tensorlang.seperator()

        try:
            with open('tensorlang.lark', 'r') as f:
                grammar = f.read()
            parser = Lark(grammar, start='program', parser='lalr')
            if self.debug_mode:
                self.tensorlang.print(message=f"Grammer tensorlang.lark opened.")
        except FileNotFoundError:
            if self.debug_mode:
                self.tensorlang.print(message=f"Grammer error tensorlang.lark not found.")
            sys.exit(1)

        try:
            self.tensorlang.print(type=f"TensorLang > Lark > Parser > Compiler > CUDA Kernel")
            self.tensorlang.seperator()

            parse_tree = parser.parse(code)
            if self.debug_ast:
                self.tensorlang.print(message=f"Parsed AST:\n{parse_tree.pretty()}")

            ast, output_tensor, functions = build_ast(parse_tree, self.debug_mode, self.debug_info)

            if self.debug_mode:
                print(f"BUILT AST:\n{ast}")
                print(f"Functions: {list(functions.keys())}")
                print(f"Output Tensor: {output_tensor}")

            success, env = type_checker(ast, {}, self.debug_info, self.debug_mode)

            if self.debug_mode:
                print(f"TYPE CHECKER ENV:\n{env}")

            # CUDA generation and execution
            if success:
                # Store tensor data and GPU allocations
                tensors = {}
                gpu_allocs = {}

                # Generate kernels for operations
                generator = KernelGenerator(self.debug_mode)
                kernels = []
                cuda_code = generator.cuda_header()
                for node in ast:
                    if node['type'] == 'let' and isinstance(node['expr'], dict):
                        name = node['name']
                        expr = node['expr']

                        # Skip alias assignments - no kernel needed
                        if expr['type'] == 'name':
                            continue

                        if self.debug_mode:
                            self.tensorlang.print(message=f"Generating kernel for {name} ({expr['type']})")

                        # ========================================
                        # Tensor Literal
                        # ========================================
                        if expr['type'] == 'tensor_literal':
                            shape = tuple(int(dim) for dim in env[name]['shape'])
                            tensors[name] = np.array(expr['data'], dtype=np.float32).reshape(shape)
                            if self.debug_info:
                                self.tensorlang.print(type="[INFO]", message=f"Kernel Tensor Initialized {name} with shape {shape}")

                        elif expr['type'] in ['add', 'minus', 'mult', 'div']:
                            arg1, arg2     = expr['args']
                            shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                            output_shape   = env[name]['shape']

                            self.tensorlang.print(message=f"Tensor {expr['type']}")
                            self.tensorlang.print(message=f"Tensor Shape1: {len(shape1)} Shape2: {len(shape2)}")

                            if shape1 == shape2:
                                size = int(np.prod([int(dim) for dim in output_shape]))
                                kernel, kernel_info = generator.elementwise(
                                    expr['type'], name, arg1, arg2, size
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            
                            elif len(shape1) == 2 and len(shape2) == 1:
                                kernel, kernel_info = generator.binary_broadcast(
                                    expr['type'], expr['type'], name, arg1, arg2, 
                                    shape1, shape2, output_shape
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            
                            elif len(shape1) == 1 and len(shape2) == 1 and (shape1[0] == 1 or shape2[0] == 1):
                                size = int(np.prod([int(dim) for dim in output_shape]))
                                kernel, kernel_info = generator.binary_1d_broadcast(
                                    expr['type'], name, arg1, arg2, size
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel

                            elif self.can_broadcast(shape1, shape2):
                                kernel, kernel_info = generator.binary_general_broadcast(
                                    expr['type'], name, arg1, arg2, 
                                    shape1, shape2, output_shape
                                )
                                kernels.append(kernel_info)
                                cuda_code += kernel
                            
                            else:
                                print(f"Error: Cannot {expr['type']} tensors with incompatible shapes {shape1} and {shape2}. "
                                    f"Broadcasting requires dimensions to match or be 1.")
                                return False, env

                        # ========================================
                        # FILL
                        # ========================================
                        elif expr['type'] == 'fill':
                            size = int(np.prod([int(dim) for dim in expr['shape']]))
                            kernel, kernel_info = generator.fill(
                                expr['type'], name, None, None, size, expr['value']
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                            
                        # ========================================
                        # MATMUL
                        # ========================================
                        elif expr['type'] == 'matmul':
                            arg1, arg2 = expr['args']
                            m, n       = int(env[arg1]['shape'][0]), int(env[arg1]['shape'][1])
                            p          = int(env[arg2]['shape'][1])
                            kernel, kernel_info = generator.matmul(
                                expr['type'], name, arg1, arg2, m, n, p
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # ReLU | SIGMOID | TANH
                        # ========================================
                        elif expr['type'] in ['relu', 'sigmoid', 'tanh']:
                            arg1 = expr['args'][0]
                            size = int(np.prod([int(dim) for dim in env[arg1]['shape']]))
                            # Dynamically call the method based on expr['type']
                            method = getattr(generator, f"{expr['type']}", None)
                            if method is None:
                                raise ValueError(f"No broadcast method found for operation: {expr['type']}")
                            kernel, kernel_info = method(
                                expr['type'], name, arg1, size
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # SOFTMAX
                        # ========================================
                        elif expr['type'] == 'softmax':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            
                            kernel, kernel_info = generator.softmax(
                                expr['type'], name, tensor_name, input_shape, axis
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # GREATER
                        # ========================================
                        elif expr['type'] == 'greater':
                            arg1, arg2     = expr['args']
                            shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                            output_shape   = env[name]['shape']
                            size           = int(np.prod([int(dim) for dim in output_shape]))

                            kernel, kernel_info = generator.greater_broadcast(
                                expr['type'], name, arg1, arg2, shape1, shape2, output_shape, size
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # LESS
                        # ========================================
                        elif expr['type'] == 'less':
                            arg1, arg2     = expr['args']
                            shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                            output_shape   = env[name]['shape']

                            kernel, kernel_info = generator.less_broadcast(
                                expr['type'], name, arg1, arg2, shape1, shape2, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # EQUAL
                        # ========================================
                        elif expr['type'] == 'equal':
                            arg1, arg2     = expr['args']
                            shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                            output_shape   = env[name]['shape']
                            
                            kernel, kernel_info = generator.equal(
                                expr['type'], name, arg1, arg2, shape1, shape2, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # SUM
                        # ========================================
                        elif expr['type'] == 'sum':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']

                            kernel, kernel_info = generator.sum(
                                expr['type'], name, tensor_name, axis, input_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # MEAN
                        # ========================================
                        elif expr['type'] == 'mean':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            
                            kernel, kernel_info = generator.mean(
                                expr['type'], name, tensor_name, axis, input_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # SLICE
                        # ========================================
                        elif expr['type'] == 'slice':
                            tensor_name  = expr['tensor']
                            slice_specs  = expr['specs']
                            input_shape  = env[tensor_name]['shape']
                            output_shape = env[name]['shape']
                            
                            kernel, kernel_info = generator.slice(
                                expr['type'], name, tensor_name, slice_specs, input_shape, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # MAX
                        # ========================================
                        elif expr['type'] == 'max':
                            tensor_name = expr['tensor']
                            axis = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            
                            kernel, kernel_info = generator.max(
                                expr['type'], name, tensor_name, axis, input_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # MIN (similar to MAX but with min operations)
                        # ========================================
                        elif expr['type'] == 'min':
                            tensor_name = expr['tensor']
                            axis = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            
                            kernel, kernel_info = generator.min(
                                expr['type'], name, tensor_name, axis, input_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # ARGMAX (returns indices of maximum values)
                        # ========================================
                        elif expr['type'] == 'argmax':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']
                            
                            kernel, kernel_info = generator.argmax(
                                expr['type'], name, tensor_name, axis, input_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # ARGMIN (similar to argmax but for minimum)
                        # ========================================
                        elif expr['type'] == 'argmin':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            input_shape = env[tensor_name]['shape']

                            kernel, kernel_info = generator.argmin(
                                expr['type'], name, tensor_name, axis, input_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # LINEAR LAYER
                        # ========================================
                        elif expr['type'] == 'linear':
                            input_name, weight_name, bias_name = expr['args']
                            input_shape  = env[input_name]['shape']
                            weight_shape = env[weight_name]['shape']
                            output_shape = env[name]['shape']
                            
                            kernel, kernel_info = generator.linear(
                                expr['type'], name, input_name, weight_name, bias_name, input_shape, weight_shape, output_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ========================================
                        # LAYER NORMALIZATION
                        # ========================================
                        elif expr['type'] == 'layer_norm':
                            tensor_name = expr['tensor']
                            axis        = expr.get('axis')
                            eps         = expr.get('eps', 1e-5)
                            input_shape = env[tensor_name]['shape']

                            kernel, kernel_info = generator.layer_norm(
                                expr['type'], name, tensor_name, axis, eps, input_shape
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ================================================================
                        # Generation: cross_entropy
                        # ================================================================
                        elif expr['type'] == 'cross_entropy':
                            pred_name, target_name = expr['args']
                            pred_shape = env[pred_name]['shape']
                            target_shape = env[target_name]['shape']
                            
                            kernel, kernel_info = generator.cross_entropy(
                                expr['type'], name, pred_name, pred_shape, target_shape, target_name
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ================================================================
                        # Generation: mse_loss
                        # ================================================================
                        elif expr['type'] == 'mse_loss':
                            pred_name, target_name = expr['args']
                            pred_shape     = env[pred_name]['shape']
                            total_elements = int(np.prod([int(dim) for dim in pred_shape]))
                            
                            kernel, kernel_info = generator.mse_loss(
                                expr['type'], name, pred_name, target_name, total_elements
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ================================================================
                        # Transpose, Reshape, Concat
                        # ================================================================
                        elif expr['type'] == 'transpose':
                            tensor_name  = expr['tensor']
                            axes         = expr.get('axes')
                            input_shape  = env[tensor_name]['shape']
                            output_shape = env[name]['shape']
                            
                            kernel, kernel_info = generator.transpose(
                                expr['type'], name, input_shape, tensor_name
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ================================================================
                        # Generation: batch_norm | batch normalization
                        # ================================================================
                        elif expr['type'] == 'batch_norm':
                            tensor_name       = expr['tensor']
                            running_mean_name = expr['running_mean']
                            running_var_name  = expr['running_var']
                            eps               = expr.get('eps', 1e-5)
                            input_shape       = env[tensor_name]['shape']
                            
                            kernel, kernel_info = generator.batch_norm(
                                expr['type'], name, input_shape, tensor_name, running_mean_name, eps, running_var_name
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ================================================================
                        # Generation: instance_norm | instance normalization
                        # ================================================================
                        elif expr['type'] == 'instance_norm':
                            tensor_name = expr['tensor']
                            eps         = expr.get('eps', 1e-5)
                            input_shape = env[tensor_name]['shape']
                            
                            kernel, kernel_info = generator.instance_norm(
                                expr['type'], name, tensor_name, input_shape, eps
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ================================================================
                        # Generation: reshape
                        # ================================================================
                        elif expr['type'] == 'reshape':
                            tensor_name    = expr['tensor']
                            input_shape    = env[tensor_name]['shape']
                            output_shape   = env[name]['shape']
                            total_elements = int(np.prod([int(dim) for dim in input_shape]))

                            kernel, kernel_info = generator.reshape(
                                expr['type'], name, tensor_name, total_elements
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # ================================================================
                        # Generation: concat
                        # ================================================================
                        elif expr['type'] == 'concat':
                            tensor_names = expr['tensors']
                            axis         = expr['axis']
                            kernel, kernel_info = generator.concat(
                                expr['type'], name, tensor_names, axis, env
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                # ================================================================
                # KERNEL COMPILATION, EXECUTION AND CACHING
                # ================================================================
                if kernels:

                    # Use relative path for cache based on file_path
                    cache_base = Path("cache")
                    print (f"CACHE: {cache_base}")
                    print (f"CACHE FILE_PATH: {file_path}")
                    print (f"CACHE FILE_PATH STEM: {file_path.stem}")
                    print (f"CACHE TENSORLANG_FILE: {tensorlang_file}")

                    #cache_file_dir = cache_base / file_path.stem
                    cache_file_dir = cache_base / file_path
                    cache_file_dir.mkdir(parents=True, exist_ok=True)

                    if self.debug_info:
                        self.tensorlang.print(type="[INFO]", message=f"Created cache for tensor outputs directory")
                        self.tensorlang.print(type="[INFO]", message=f"Generated CUDA kernels:")

                    # Paths for Kernel cuda code and shared object files
                    kernel_cu_path = cache_file_dir / "kernel.cu"
                    kernel_so_path = cache_file_dir / "kernel.so"

                    with open(kernel_cu_path, 'w') as f:
                        f.write(cuda_code)
                    try:
                        subprocess.run([
                                'nvcc', '-o', str(kernel_so_path), 
                                '--shared', 
                                '-Xcompiler', 
                                '-fPIC', 
                                '-lcudart', 
                                str(kernel_cu_path)
                            ], 
                            check=True
                        )
                        if self.debug_mode:
                            self.tensorlang.print(message=f"CUDA compiled to {kernel_so_path}!")

                    except subprocess.CalledProcessError as e:
                        print(f"CUDA compilation error: {e}")
                        sys.exit(1)

                    # =========================================
                    # Execute with PyCUDA
                    # =========================================
                    try:
                        import pycuda.driver as cuda
                        import pycuda.autoinit
                        from ctypes import cdll

                        lib = cdll.LoadLibrary(str(kernel_so_path))

                        if self.debug_mode:
                            self.tensorlang.print(message=f"CUDA Kernel loaded from {kernel_so_path}")

                        # Allocate GPU memory and copy inputs
                        for name in env:
                            shape            = tuple(int(dim) for dim in env[name]['shape'])
                            size_bytes       = int(np.prod(shape) * np.float32().nbytes)
                            gpu_allocs[name] = cuda.mem_alloc(size_bytes)
                            if name in tensors:
                                cuda.memcpy_htod(gpu_allocs[name], tensors[name])
                                if self.debug_mode:
                                    tensorlang.print(message=f"Copied {name} to GPU, shape: {tensors[name].shape}, sample: {tensors[name][:2] if tensors[name].ndim > 1 else tensors[name]}")
                            else:
                                if self.debug_mode:
                                    tensorlang.print(message=f"Allocated GPU memory for {name} (uninitialized), shape: {shape}")

                        # Kernel execution
                        for kernel_info in kernels:
                            op_type = kernel_info[0]
                            
                            # Handle general_broadcast specially BEFORE standard unpacking
                            if op_type == 'general_broadcast':
                                _, actual_op, name, arg1, arg2, padded_shape1, padded_shape2, output_shape_tuple, total_elements = kernel_info
                                shape = tuple(int(dim) for dim in env[name]['shape'])
                                
                                if self.debug_mode:
                                    self.tensorlang.print(type=f"TensorLang Executing {op_type} ({actual_op}) for {name}, shape: {shape}")
                                
                                ndim = len(output_shape_tuple)
                                
                                # Allocate GPU memory for shape arrays
                                shape1_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
                                shape2_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
                                out_shape_gpu = cuda.mem_alloc(ndim * np.int32().nbytes)
                                
                                # Copy shape data to GPU
                                shape1_array = np.array(padded_shape1, dtype=np.int32)
                                shape2_array = np.array(padded_shape2, dtype=np.int32)
                                out_shape_array = np.array(output_shape_tuple, dtype=np.int32)
                                
                                cuda.memcpy_htod(shape1_gpu, shape1_array)
                                cuda.memcpy_htod(shape2_gpu, shape2_array)
                                cuda.memcpy_htod(out_shape_gpu, out_shape_array)
                                
                                # Launch kernel with actual operation type
                                getattr(lib, f'launch_{actual_op}_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])),
                                    c_void_p(int(gpu_allocs[arg2])),
                                    c_void_p(int(gpu_allocs[name])),
                                    c_void_p(int(shape1_gpu)),
                                    c_void_p(int(shape2_gpu)),
                                    c_void_p(int(out_shape_gpu)),
                                    c_int(ndim),
                                    c_int(total_elements)
                                )
                                continue  # skip to next iteration
                            

                            # Standard unpacking for all other operations (moved OUTSIDE the if block)
                            op_type, name, arg1, arg2, *dims = kernel_info
                            shape = tuple(int(dim) for dim in env[name]['shape'])
                            
                            if op_type == 'matmul':
                                m, n, p = dims
                                getattr(lib, f'launch_matmul_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[arg2])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(m), c_int(n), c_int(p)
                                )

                            elif op_type in ['add', 'minus', 'mult', 'div']:
                                size = dims[0]
                                getattr(lib, f'launch_{op_type}_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[arg2])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(size)
                                )

                            elif op_type in ['add_scalar', 'minus_scalar', 'mult_scalar', 'div_scalar']:
                                size = dims[0]
                                op_base = op_type.split('_')[0]
                                getattr(lib, f'launch_{op_base}_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[arg2])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(size)
                                )

                            elif op_type in [
                                    'add_broadcast', 'minus_broadcast', 'mult_broadcast', 'div_broadcast',
                                    'greater_broadcast', 'less_broadcast', 'equal_broadcast'
                                ]:
                                op_name = op_type.split("_")[0]
                                rows, cols = dims
                                getattr(lib, f'launch_{op_name}_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[arg2])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type in ['relu', 'sigmoid', 'tanh']:
                                op_name = op_type.split("_")[0]
                                size = dims[0]
                                getattr(lib, f'launch_{op_name}_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(size)
                                )

                            elif op_type in [
                                    'greater', 'less', 'equal'
                                ]:
                                size = dims[0]
                                getattr(lib, f'launch_{op_type}_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])),
                                    c_void_p(int(gpu_allocs[arg2])),
                                    c_void_p(int(gpu_allocs[name])),
                                    c_int(size)
                                )

                            elif op_type == 'softmax_1d':
                                size = dims[0]
                                getattr(lib, f'launch_softmax_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(size)
                                )

                            elif op_type == 'softmax':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_softmax_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'sum_axis':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_sum_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'sum_axis0':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_sum_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )


                            elif op_type == 'slice_2d':
                                rows, cols, row_start, row_end, col_start, col_end, out_rows, out_cols = dims
                                getattr(lib, f'launch_slice_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])),
                                    c_int(rows), c_int(cols),
                                    c_int(row_start), c_int(row_end), c_int(col_start), c_int(col_end),
                                    c_int(out_rows), c_int(out_cols)
                                )
                            elif op_type == 'slice_1d':
                                start, out_size = dims
                                getattr(lib, f'launch_slice_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])),
                                    c_int(start), c_int(out_size)
                                )

                            elif op_type == 'sum_full':
                                size = dims[0]
                                getattr(lib, f'launch_sum_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(size)
                                )

                            elif op_type == 'mean_full':
                                size = dims[0]
                                getattr(lib, f'launch_mean_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(size)
                                )

                            elif op_type == 'mean_axis':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_mean_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'mean_axis0':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_mean_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'max_full':
                                size = dims[0]
                                getattr(lib, f'launch_max_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(size)
                                )

                            elif op_type == 'max_axis':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_max_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'max_axis0':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_max_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'min_full':
                                size = dims[0]
                                getattr(lib, f'launch_min_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(size)
                                )

                            elif op_type == 'min_axis':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_min_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'min_axis0':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_min_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'argmax_full':
                                size = dims[0]
                                getattr(lib, f'launch_argmax_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(size)
                                )

                            elif op_type == 'argmax_axis':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_argmax_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'argmin_axis':
                                rows, cols, axis = dims
                                getattr(lib, f'launch_argmin_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'fill':
                                size, value = dims
                                getattr(lib, f'launch_fill_{name}')(
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_float(value), c_int(size)
                                )

                            elif op_type == 'linear_1d':
                                input_name, weight_name, bias_name, in_features, out_features = arg1, arg2, dims[0], dims[1], dims[2]
                                getattr(lib, f'launch_linear_{name}')(
                                    c_void_p(int(gpu_allocs[input_name])),
                                    c_void_p(int(gpu_allocs[weight_name])),
                                    c_void_p(int(gpu_allocs[bias_name])),
                                    c_void_p(int(gpu_allocs[name])),
                                    c_int(in_features), c_int(out_features)
                                )
                            
                            # ================================================================
                            # Execution: linear_2d, layer_norm_2d, layer_norm_axis0, layer_norm_1d
                            # ================================================================
                            elif op_type == 'linear_2d':
                                ###############################################################
                                # Save result for all computed tensors
                                # Debug: Check input state before kernel
                                #input_debug = np.zeros((batch_size, in_features), dtype=np.float32)
                                #cuda.memcpy_dtoh(input_debug, gpu_allocs[input_name])
                                #print(f"DEBUG LINEAR: Input before kernel = {input_debug}")
                                ###############################################################

                                input_name, weight_name, bias_name, batch_size, in_features, out_features = arg1, arg2, dims[0], dims[1], dims[2], dims[3]
                                getattr(lib, f'launch_linear_{name}')(
                                    c_void_p(int(gpu_allocs[input_name])),
                                    c_void_p(int(gpu_allocs[weight_name])),
                                    c_void_p(int(gpu_allocs[bias_name])),
                                    c_void_p(int(gpu_allocs[name])),
                                    c_int(batch_size), c_int(in_features), c_int(out_features)
                                )

                                ###############################################################
                                # Debug: Check output immediately after kernel
                                #output_debug = np.zeros(shape, dtype=np.float32)
                                #cuda.memcpy_dtoh(output_debug, gpu_allocs[name])
                                #print(f"DEBUG LINEAR: Output immediately after kernel = {output_debug}")
                                ###############################################################

                            elif op_type in ['layer_norm_2d', 'layer_norm_axis0']:
                                rows, cols, eps = dims
                                getattr(lib, f'launch_layer_norm_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])),
                                    c_int(rows), c_int(cols), c_float(eps)
                                )

                            elif op_type == 'layer_norm_1d':
                                size, eps = dims
                                getattr(lib, f'launch_layer_norm_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])),
                                    c_int(size), c_float(eps)
                                )

                            # ================================================================
                            # Execution: cross_entropy, mse_loss
                            # ================================================================
                            elif op_type == 'cross_entropy':
                                batch_size, num_classes = dims
                                getattr(lib, f'launch_cross_entropy_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])),
                                    c_void_p(int(gpu_allocs[name])),
                                    c_int(batch_size), c_int(num_classes)
                                )
                            elif op_type == 'mse_loss':
                                total_elements = dims[0]
                                getattr(lib, f'launch_mse_loss_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])),
                                    c_void_p(int(gpu_allocs[name])),
                                    c_int(total_elements)
                                )

                            # ================================================================
                            # Execution: transpose_2d, reshape, concat_axis0
                            # ================================================================
                            elif op_type == 'transpose_2d':
                                rows, cols = dims
                                getattr(lib, f'launch_transpose_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])),
                                    c_int(rows), c_int(cols)
                                )
                            elif op_type == 'reshape':
                                total_elements = dims[0]
                                getattr(lib, f'launch_reshape_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])),
                                    c_int(total_elements)
                                )
                            elif op_type == 'concat_axis0':
                                rows1, rows2, cols = dims
                                getattr(lib, f'launch_concat_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[arg2])), c_void_p(int(gpu_allocs[name])),
                                    c_int(rows1), c_int(rows2), c_int(cols)
                                )


                            # ================================================================
                            # Execution: batch_norm_2d, instance_norm_2d
                            # ================================================================
                            elif op_type == 'batch_norm_2d':
                                running_mean_name, batch_size, num_features, eps, running_var_name = arg2, dims[0], dims[1], dims[2], dims[3]
                                getattr(lib, f'launch_batch_norm_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])),  # input tensor
                                    c_void_p(int(gpu_allocs[running_mean_name])),
                                    c_void_p(int(gpu_allocs[running_var_name])), 
                                    c_void_p(int(gpu_allocs[name])),  # output tensor
                                    c_int(batch_size), c_int(num_features), c_float(eps)
                                )

                            elif op_type == 'instance_norm_2d':
                                batch_size, num_features, eps = dims[-3], dims[-2], dims[-1]  # Get last 3 values
                                getattr(lib, f'launch_instance_norm_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])),
                                    c_void_p(int(gpu_allocs[name])),
                                    c_int(batch_size), c_int(num_features), c_float(eps)
                                )

                            # ================================================================
                            # Execution: function broadcasting
                            # ================================================================
                            elif op_type == 'minus_broadcast_rows':
                                rows, cols = dims

                                # Debug: Check if inputs have data
                                #if DEBUG_MODE:
                                if self.debug_mode:
                                    test_data = np.zeros((rows,), dtype=np.float32)
                                    cuda.memcpy_dtoh(test_data, gpu_allocs[arg2])
                                    print(f"DEBUG: B tensor ({arg2}) before minus: {test_data}")

                                getattr(lib, f'launch_minus_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[arg2])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            elif op_type == 'add_broadcast_rows':
                                rows, cols = dims
                                getattr(lib, f'launch_add_{name}')(
                                    c_void_p(int(gpu_allocs[arg1])), 
                                    c_void_p(int(gpu_allocs[arg2])), 
                                    c_void_p(int(gpu_allocs[name])), 
                                    c_int(rows), c_int(cols)
                                )

                            # else:
                                # Normal unpacking for all other operations
                                # op_type, name, arg1, arg2, *dims = kernel_info



                            # Save result for all computed tensors
                            if len(kernel_info) > 2:
                                name = kernel_info[1] if op_type != 'general_broadcast' else kernel_info[2]
                                if name in env:
                                    shape = tuple(int(dim) for dim in env[name]['shape'])
                                    output = np.zeros(shape, dtype=np.float32)
                                    cuda.memcpy_dtoh(output, gpu_allocs[name])
                                    tensors[name] = output

                                    if self.cache_layers:
                                        if self.debug_info:
                                            tensorlang.print(message=f"Cache Layer {name} cache/{file_path.stem}/{name}.npy")
                                        cache_npy_path = cache_file_dir / f"{name}.npy"
                                        # cache_npy_path = f"{tensorlang_file}/{name}.npy"
                                        np.save(cache_npy_path, output)

                                    print(f"Result {name} ({op_type}):\n{output}")

                        # Handle alias assignments
                        for node in ast:
                            if node['type'] == 'let' and isinstance(node.get('expr'), dict) and node['expr']['type'] == 'name':
                                alias_name = node['name']
                                source_name = node['expr']['name']
                                if source_name in tensors and alias_name not in tensors:
                                    tensors[alias_name] = tensors[source_name].copy()
                                    if self.debug_mode:
                                        self.tensorlang.print(type=f"Created alias: {alias_name} -> {source_name}")
                                    
                                    print(f"Result {alias_name} (alias):\n{tensors[alias_name]}")
                                    cache_npy_path = cache_file_dir / f"{alias_name}.npy"
                                    # cache_npy_path = f"{tensorlang_file}/{alias_name}.npy"
                                    
                                    if self.debug_mode:
                                        print(f"CACHE LAYER NPY: {cache_npy_path}")
                                    np.save(cache_npy_path, tensors[alias_name])

                        # Free GPU memory
                        for name, alloc in gpu_allocs.items():
                            alloc.free()
                            if self.debug_mode:
                                self.tensorlang.print(type=f"Freed GPU memory for {name}")

                    except ImportError as e:
                        print(f"PyCUDA error: {e}. Run 'pip install pycuda' and ensure CUDA toolkit is installed.")
                        sys.exit(1)
                    except Exception as e:
                        print(f"Error executing CUDA kernel: {e}")
                        traceback.print_exc()
                        sys.exit(1)

        except ValueError as e:
            print(f"TensorLang: Value Error: failed: {e}")
            sys.exit(1)

        except UnexpectedInput as e:
            print(f"TensorLang: Parse error: {e}")
            traceback.print_exc()
            sys.exit(1)

        except Exception as e:
            print(f"TensorLang: Unexpected error during parsing or execution: {e}")
            traceback.print_exc()
            sys.exit(1)

        self.tensorlang.seperator()
