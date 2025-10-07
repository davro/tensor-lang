import sys
import os
import time
import textwrap  
import argparse
import subprocess
import traceback
import numpy as np
from lark import Lark, Tree, Token, UnexpectedInput
from datetime import datetime
from ctypes import c_void_p, c_int, c_float
from functools import reduce
from typing import Optional
from pathlib import Path


from tensorlang.TensorLang import TensorLang

# ================================================================
#                      TensorLang version
# ================================================================
version = "0.2.5"

# Functions
def str_to_bool(value):
    """Convert a string to a boolean value."""
    if value.lower() in ('true', 't', '1', 'yes', 'y'):
        return True
    if value.lower() in ('false', 'f', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def main():
    # ================================================================
    #                 GRAMMAR lark + TensorLang file
    # ================================================================
    try:
        tensorlang = TensorLang()

        # Replace your current sys.argv handling with:
        tensorlangparser = argparse.ArgumentParser(description='TensorLang Compiler')
        tensorlangparser.add_argument('file', nargs='?', default='tests/add.tl', 
            help='TensorLang source file')
        tensorlangparser.add_argument('--debug', action='store_true', 
            help='Enable Debug slower (shown in output))')
        tensorlangparser.add_argument('--debug-info', action='store_true', 
            help='Enable Info (shown in output)')
        tensorlangparser.add_argument('--debug-ast', type=str_to_bool, default=False, 
            help='AST Abstract Syntax Tree (show in debug)')
        tensorlangparser.add_argument('--cache-layers', action='store_true', 
            help='Enable Cache Layers (shown in output))')

        # Examples
        #tensorlangparser.add_argument('--example', type=str_to_bool, default=False, 
        #tensorlangparser.add_argument('--example', action='store_true', 

        # Collect the parsed args
        args = tensorlangparser.parse_args()

        # Global variables
        tensorlang_file = args.file
        DEBUG_MODE      = args.debug
        DEBUG_INFO      = args.debug_info
        DEBUG_AST       = args.debug_ast
        CACHE_LAYERS    = args.cache_layers

        # TensorLang path for file information
        file_path = Path(tensorlang_file)

        tensorlang.print_header(version)
        # Gather file details
        if file_path.suffix == '.tl':
            file_details = {
                #"| Path Absolute": str(file_path.absolute()),
                "| Path     " : str(file_path),
                "| Name     " : file_path.name,
                "| Suffix   " : file_path.suffix or "None",  # e.g., .tl
                "| Size     " : f"{file_path.stat().st_size} bytes",  # Size in bytes
                "| Created  " : time.ctime(file_path.stat().st_ctime) if hasattr(file_path.stat(), 'st_ctime') else "Not available",
                "| Modified " : time.ctime(file_path.stat().st_mtime),  # Last modified time
                #"Is File": file_path.is_file(),
                #"Is Directory": file_path.is_dir(),
            }
            # Format the details for printing
            details_str = "\n".join(f"{key}: {value}" for key, value in file_details.items())

            # Call the print method with the details
            tensorlang.print(type=details_str)
        else:
            tensorlang.print(type=f"TensorLang file not found, suffix is not .tl") 
            tensorlang.seperator()
            sys.exit(1)

        with open(tensorlang_file, 'r') as f:
            code = f.read()

        tensorlang.seperator()
        tensorlang.print(type=f"{code}")
        tensorlang.seperator()
    except FileNotFoundError:
        if DEBUG_MODE:
            tensorlang.print(message=f"Error: {tensorlang_file} not found. Please provide a valid TensorLang file.")
        sys.exit(1)

    try:
        with open('tensorlang.lark', 'r') as f:
            grammar = f.read()
        parser = Lark(grammar, start='program', parser='lalr')
        if DEBUG_MODE:
            tensorlang.print(message=f"Grammer tensorlang.lark opened.")
    except FileNotFoundError:
        if DEBUG_MODE:
            tensorlang.print(message=f"Grammer error tensorlang.lark not found.")
        sys.exit(1)

    # ================================================================
    #                         PARSER + AST
    # ================================================================
    try:
        tensorlang.print(type=f"TensorLang > Lark > Parser > Compiler > CUDA Kernel")
        tensorlang.seperator()

        parse_tree = parser.parse(code)
        if DEBUG_AST:
            tensorlang.print(message=f"Parsed AST:\n{parse_tree.pretty()}")

        def build_ast(tree):
            ast = []
            output_tensor = None
            if tree.data == 'program':
                for child in tree.children:
                    if isinstance(child, Tree) and child.data == 'statement':
                        if child.children[0].data == 'let_binding':
                            let_node = build_let_binding(child.children[0])
                            if let_node:
                                ast.append(let_node)
                                if DEBUG_MODE:
                                    tensorlang.print(message=f"Added node to AST: {let_node}")
                        elif child.children[0].data == 'expr':
                            expr_node = build_expression(child.children[0])
                            if expr_node['type'] == 'name':
                                output_tensor = expr_node['name']
                                if DEBUG_INFO:
                                    tensorlang.print(type="[INFO]", message=f"Tensor Output Assigned: {output_tensor}")

            # If no output_tensor is set, default to the last defined tensor
            if not output_tensor and ast:
                output_tensor = ast[-1]['name']
                if DEBUG_MODE:
                    tensorlang.print(message=f"No explicit output tensor; defaulting to last tensor: {output_tensor}")
            return ast, output_tensor


        def build_let_binding(tree):
            if DEBUG_MODE:
                tensorlang.print(message=f"Building let_binding from {tree.data}")
            if tree.data != 'let_binding':
                if DEBUG_MODE:
                    tensorlang.print(message=f"Invalid let_binding node: {tree.data}")
                return None
            children = tree.children
            if len(children) == 3:
                name = children[0].value
                if DEBUG_MODE:
                    tensorlang.print(message=f"Processing let binding for {name}")
                if isinstance(children[1], Tree) and children[1].data == 'type':
                    ty = build_type(children[1])
                    expr = build_expression(children[2])
                else:
                    ty = None
                    expr = build_expression(children[2])
                return {'type': 'let', 'name': name, 'ty': ty, 'expr': expr, 'tree': tree}
            else:
                if DEBUG_MODE:
                    tensorlang.print(message=f"Unexpected number of children in let_binding: {len(children)}")
                return None
        

        def build_type(tree):
            if DEBUG_MODE:
                tensorlang.print(message=f"Building type from {tree.data}")
            
            if tree.data == 'concrete_type':
                # Existing concrete type logic
                children = tree.children
                dtype_value = 'f32'
                if len(children) > 0 and isinstance(children[0], Token) and children[0].value in ['f32', 'f64']:
                    dtype_value = children[0].value
                shape_tree = children[1] if len(children) > 1 and isinstance(children[1], Tree) else None
                shape = build_shape(shape_tree) if shape_tree else (0, 0)
                return {'dtype': dtype_value, 'shape': shape}
            
            elif tree.data == 'generic_type':
                # New: Generic type with symbolic dimensions
                generic_shape_tree = tree.children[0]
                generic_dims = []
                for child in generic_shape_tree.children:
                    if isinstance(child, Token):
                        if child.type == 'NAME':
                            # Generic dimension like 'M', 'N'
                            generic_dims.append(child.value)
                        elif child.type == 'NUMBER':
                            # Concrete dimension in generic context
                            generic_dims.append(int(float(child.value)))
                return {'dtype': 'f32', 'shape': tuple(generic_dims)}
            
            # Fallback for old 'type' node
            if tree.data == 'type':
                return build_type(tree.children[0])


        def build_shape(tree):
            if DEBUG_MODE:
                tensorlang.print(message=f"Building shape from {tree.data}")
            if tree.data != 'shape':
                if DEBUG_MODE:
                    tensorlang.print(message=f"Invalid shape node: {tree.data}")
                return (0, 0)
            nums = [int(float(child.value)) for child in tree.children if isinstance(child, Token) and child.type == 'NUMBER']
            if DEBUG_MODE:
                tensorlang.print(message=f"Shape numbers: {nums}")
            return tuple(nums)


        def build_expression(tree):
            if tree.data == 'expr':
                tree = tree.children[0]  # Unwrap expr

            if isinstance(tree, Token) and tree.type == 'NAME':
                if DEBUG_MODE:
                    tensorlang.print(message=f"Expression: {tree.type}: {tree.value}")
                return {'type': 'name', 'name': tree.value}

            # Add this BEFORE the NAME token check
            elif tree.data == 'user_function_call':
                return build_user_function_call(tree)

            # Tree Tokens
            if isinstance(tree, Token) and tree.type == 'NAME':
                if DEBUG_MODE:
                    tensorlang.print(message=f"Expression: {tree.type}: {tree.value}")
                return {'type': 'name', 'name': tree.value}

            if tree.data == 'matmul_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'matmul', 'args': args}

            elif tree.data == 'add_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'add', 'args': args}

            elif tree.data == 'minus_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'minus', 'args': args}

            elif tree.data == 'mult_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'mult', 'args': args}

            elif tree.data == 'div_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'div', 'args': args}

            elif tree.data == 'relu_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'relu', 'args': args}

            elif tree.data == 'sigmoid_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'sigmoid', 'args': args}

            elif tree.data == 'tanh_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
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
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} tensor={tensor_name}, axis={axis}")
                return {'type': 'softmax', 'tensor': tensor_name, 'axis': axis}

            elif tree.data == 'greater_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'greater', 'args': args}

            elif tree.data == 'less_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'less', 'args': args}

            elif tree.data == 'equal_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'equal', 'args': args}

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
                if DEBUG_MODE:
                    tensorlang.print(message=f"Tensor literal data: {data}, is_1d: {is_1d}")

                return {'type': 'tensor_literal', 'data': data, 'is_1d': is_1d, 'tree': tree}

            elif tree.data == 'fill_call':
                value = float(tree.children[0].value) if isinstance(tree.children[0], Token) and tree.children[0].type == 'NUMBER' else 0.0
                shape_tree = tree.children[1] if len(tree.children) > 1 and isinstance(tree.children[1], Tree) else None
                shape = build_shape(shape_tree) if shape_tree else (1,)

                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"{tree.data} value: {value}, shape: {shape}")
                return {'type': 'fill', 'value': value, 'shape': shape}

            elif tree.data == 'sum_call':
                tensor_name = None
                axis = None
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        axis = int(float(child.value))
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, axis={axis}")
                return {'type': 'sum', 'tensor': tensor_name, 'axis': axis}

            elif tree.data == 'mean_call':
                tensor_name = None
                axis = None
                
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        axis = int(float(child.value))
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, axis={axis}")

                return {'type': 'mean', 'tensor': tensor_name, 'axis': axis}

            elif tree.data == 'max_call':
                tensor_name = None
                axis = None
                
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        axis = int(float(child.value))
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, axis={axis}")
                return {'type': 'max', 'tensor': tensor_name, 'axis': axis}

            elif tree.data == 'min_call':
                tensor_name = None
                axis = None
                
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        axis = int(float(child.value))
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, axis={axis}")
                return {'type': 'min', 'tensor': tensor_name, 'axis': axis}

            elif tree.data == 'argmax_call':
                tensor_name = None
                axis = None
                
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        axis = int(float(child.value))                
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, axis={axis}")
                return {'type': 'argmax', 'tensor': tensor_name, 'axis': axis}

            elif tree.data == 'argmin_call':
                tensor_name = None
                axis = None
                
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        axis = int(float(child.value))
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, axis={axis}")
                return {'type': 'argmin', 'tensor': tensor_name, 'axis': axis}

            elif tree.data == 'linear_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'linear', 'args': args}

            elif tree.data == 'layer_norm_call':
                tensor_name = None
                axis = None
                eps = 1e-5  # Default epsilon value
                
                # Parse arguments
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        # Could be axis or eps - need to track which we're parsing
                        value = float(child.value)
                        # Check if it's a small value (likely epsilon) or larger (likely axis)
                        if value < 0.1:  # Heuristic for epsilon
                            eps = value
                        else:
                            axis = int(value)
                
                # Default to last axis if not specified
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, axis={axis}, eps={eps}")
                return {'type': 'layer_norm', 'tensor': tensor_name, 'axis': axis, 'eps': eps}

            # ================================================================
            # Expression: cross-entropy loss and mean squared error (MSE) loss
            # ================================================================
            elif tree.data == 'cross_entropy_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'cross_entropy', 'args': args}

            elif tree.data == 'mse_loss_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: {args}")
                return {'type': 'mse_loss', 'args': args}


            # ================================================================
            # Expression: slice feature
            # ================================================================
            elif tree.data == 'slice_expr':
                tensor_name = None
                slice_specs = []
                
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Tree):
                        if child.data == 'slice_spec':
                            spec = build_slice_spec(child)
                            slice_specs.append(spec)
                
                #print(f"Slice args: tensor={tensor_name}, specs={slice_specs}")
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, specs={slice_specs}")
                return {'type': 'slice', 'tensor': tensor_name, 'specs': slice_specs}


            # ================================================================
            # Expression: transpose, reshape, concat
            # ================================================================
            elif tree.data == 'transpose_call':
                tensor_name = None
                axes = None
                
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Tree):
                        # Parse axes specification (optional)
                        axis_values = []
                        for grandchild in child.children:
                            if isinstance(grandchild, Token) and grandchild.type == 'NUMBER':
                                axis_values.append(int(float(grandchild.value)))
                        if axis_values:
                            axes = tuple(axis_values)
                
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, axes={axes}")
                return {'type': 'transpose', 'tensor': tensor_name, 'axes': axes}

            elif tree.data == 'reshape_call':
                tensor_name = None
                new_shape   = None
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Tree) and child.data == 'shape':
                        new_shape = build_shape(child)
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} args: tensor={tensor_name}, new_shape={new_shape}")
                return {'type': 'reshape', 'tensor': tensor_name, 'new_shape': new_shape}

            elif tree.data == 'concat_call':
                tensor_names = []
                axis = None
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_names.append(child.value)
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        axis = int(float(child.value))
                if DEBUG_MODE:
                    tensorlang.print(message=f"Expression: {tree.data} args: tensors={tensor_names}, axis={axis}")
                return {'type': 'concat', 'tensors': tensor_names, 'axis': axis}

            # ================================================================
            # Expression: batch_norm_call, instance_norm_call
            # ================================================================
            elif tree.data == 'batch_norm_call':
                tensor_name = None
                running_mean_name = None
                running_var_name = None
                eps = 1e-5  # Default epsilon
                arg_count = 0
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        if arg_count == 0:
                            tensor_name = child.value
                        elif arg_count == 1:
                            running_mean_name = child.value
                        elif arg_count == 2:
                            running_var_name = child.value
                        arg_count += 1
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        eps = float(child.value)
                
                if DEBUG_MODE:
                    tensorlang.print(message=f"Expression: {tree.data} args: tensor={tensor_name}, mean={running_mean_name}, var={running_var_name}, eps={eps}")

                return {'type': 'batch_norm', 'tensor': tensor_name, 'running_mean': running_mean_name, 
                        'running_var': running_var_name, 'eps': eps}

            elif tree.data == 'instance_norm_call':
                tensor_name = None
                eps = 1e-5  # Default epsilon
                
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        eps = float(child.value)

                if DEBUG_MODE:
                    # print(f"InstanceNorm args: tensor={tensor_name}, eps={eps}")
                    tensorlang.print(message=f"{tree.data} args: tensor={tensor_name}, eps={eps}")

                return {'type': 'instance_norm', 'tensor': tensor_name, 'eps': eps}

            elif tree.data == 'user_function_call':
                if DEBUG_MODE:
                    tensorlang.print(message=f"Expression: user_function_call")
                return build_user_function_call(tree)

            print(f"Unrecognized expr type: {tree.data}")
            return None


        def build_slice_spec(tree):
            """Parse slice specification like 0:2, :, 1:, etc."""
            if tree.data == 'slice_spec':
                # Handle the slice_spec wrapper
                if len(tree.children) == 1:
                    child = tree.children[0]
                    if isinstance(child, Tree) and child.data == 'slice_range':
                        return build_slice_spec(child)  # Recursive call for slice_range
                    elif isinstance(child, Token):
                        if child.type == 'NUMBER':
                            return {'type': 'index', 'value': int(float(child.value))}
                        elif child.value == ':':
                            return {'type': 'full_slice'}
            elif tree.data == 'slice_range':
                # Handle slice_range: start:end
                if len(tree.children) == 2:
                    left, right = tree.children
                    if isinstance(left, Token) and isinstance(right, Token):
                        if left.value == ':':
                            # :end
                            end = int(float(right.value))
                            return {'type': 'slice', 'start': None, 'end': end}
                        elif right.value == ':':
                            # start:
                            start = int(float(left.value))
                            return {'type': 'slice', 'start': start, 'end': None}
                        elif left.type == 'NUMBER' and right.type == 'NUMBER':
                            # start:end
                            start = int(float(left.value))
                            end = int(float(right.value))
                            return {'type': 'slice', 'start': start, 'end': end}
            
            # Default fallback
            return {'type': 'full_slice'}


        # ================================================================
        # Build functions for Feature "function definitions and calls."
        # ================================================================
        def build_function_def(tree):
            """Parse function definition: fn name(params) -> return_type { body }"""
            if tree.data != 'function_def':
                return None
            
            children = tree.children
            name = children[0].value  # Function name
            
            # Parse parameters
            params = []
            param_list_idx = 1
            if len(children) > 1 and isinstance(children[1], Tree) and children[1].data == 'param_list':
                for param_tree in children[1].children:
                    if param_tree.data == 'param':
                        param_name = param_tree.children[0].value
                        param_type = build_type(param_tree.children[1])
                        params.append({'name': param_name, 'type': param_type})
                param_list_idx = 2
            
            # Parse return type (optional)
            return_type = None
            body_idx = param_list_idx
            
            # Check if next child is a type node (could be 'type', 'concrete_type', or 'generic_type')
            if param_list_idx < len(children) and isinstance(children[param_list_idx], Tree):
                child_data = children[param_list_idx].data
                if child_data in ['type', 'concrete_type', 'generic_type']:
                    return_type = build_type(children[param_list_idx])
                    body_idx = param_list_idx + 1
            
            # Parse function body - must be function_body node
            if body_idx >= len(children):
                print(f"ERROR: No function body found for function {name}")
                return None
                
            body_tree = children[body_idx]
            if body_tree.data != 'function_body':
                print(f"ERROR: Expected function_body, got {body_tree.data}")
                return None
            
            body_statements = []
            return_expr = None
            
            for stmt in body_tree.children:
                if isinstance(stmt, Tree):
                    if stmt.data == 'function_statement':
                        # Unwrap function_statement
                        inner_stmt = stmt.children[0]
                        if inner_stmt.data == 'let_binding':
                            body_statements.append(build_let_binding(inner_stmt))
                        elif inner_stmt.data == 'expr_statement':
                            body_statements.append(build_expression(inner_stmt.children[0]))
                    elif stmt.data == 'return_statement':
                        return_expr = build_expression(stmt.children[0])
            
            return {
                'type': 'function_def',
                'name': name,
                'params': params,
                'return_type': return_type,
                'body': body_statements,
                'return_expr': return_expr
            }


        def build_user_function_call(tree):
            """Parse user function call: func_name(arg1, arg2, ...)"""
            if tree.data != 'user_function_call':
                return None
            
            func_name = tree.children[0].value
            args = []
            
            # Parse arguments
            if len(tree.children) > 1 and isinstance(tree.children[1], Tree):
                arg_list = tree.children[1]
                if arg_list.data == 'arg_list':
                    for arg in arg_list.children:
                        args.append(build_expression(arg))
            
            return {
                'type': 'user_function_call',
                'func_name': func_name,
                'args': args
            }


        def inline_function_call(func_def, args, env, unique_suffix):
            """
            Inline a function call by creating new let bindings with unique names.
            
            Returns: (inlined_statements, return_value_name)
            """
            # Map parameter names to argument expressions
            param_map = {}
            for param, arg in zip(func_def['params'], args):
                param_map[param['name']] = arg
            
            inlined_statements = []
            name_mapping = {}  # Old name -> new name
            
            # Process function body statements
            for stmt in func_def['body']:
                if stmt['type'] == 'let':
                    old_name = stmt['name']
                    new_name = f"{old_name}_{unique_suffix}"
                    name_mapping[old_name] = new_name
                    
                    # Substitute parameter references in expression
                    new_expr = substitute_names(stmt['expr'], param_map, name_mapping)
                    
                    inlined_statements.append({
                        'type': 'let',
                        'name': new_name,
                        'ty': stmt['ty'],
                        'expr': new_expr,
                        'tree': stmt.get('tree')
                    })
            
            # Handle return expression
            return_expr = func_def['return_expr']
            if return_expr:
                return_value_name = f"__return_{unique_suffix}"
                final_expr = substitute_names(return_expr, param_map, name_mapping)
                
                inlined_statements.append({
                    'type': 'let',
                    'name': return_value_name,
                    'ty': None,
                    'expr': final_expr,
                    'tree': None
                })
                
                return inlined_statements, return_value_name
            
            return inlined_statements, None


        def substitute_names(expr, param_map, name_mapping):
            """
            Recursively substitute parameter names and local variable names in an expression.
            """
            if not isinstance(expr, dict):
                return expr
            
            new_expr = expr.copy()
            
            # Handle direct name references
            if expr['type'] == 'name':
                name = expr['name']
                # First check if it's a parameter
                if name in param_map:
                    param_expr = param_map[name]
                    # If the parameter maps to a simple name, return that
                    if isinstance(param_expr, dict) and param_expr.get('type') == 'name':
                        return param_expr
                    return param_expr
                # Then check if it's a local variable
                if name in name_mapping:
                    new_expr['name'] = name_mapping[name]
                return new_expr
            
            # Handle user function calls - recursively substitute in arguments
            if expr['type'] == 'user_function_call':
                new_expr['args'] = [substitute_names(arg, param_map, name_mapping) 
                                for arg in expr.get('args', [])]
                return new_expr
            
            # Handle operations with 'args' (like add, matmul, etc)
            if 'args' in expr:
                new_args = []
                for arg in expr['args']:
                    # arg should be a string variable name
                    if isinstance(arg, str):  # <-- Add type check
                        if arg in param_map:
                            # Get the actual argument expression
                            param_expr = param_map[arg]
                            if isinstance(param_expr, dict) and param_expr.get('type') == 'name':
                                new_args.append(param_expr['name'])
                            else:
                                new_args.append(arg)
                        elif arg in name_mapping:
                            new_args.append(name_mapping[arg])
                        else:
                            new_args.append(arg)
                    else:
                        # arg is already an expression dict - shouldn't happen for built-in ops
                        new_args.append(arg)
                new_expr['args'] = new_args
                return new_expr
            
            # Handle operations with 'tensor' field
            if 'tensor' in expr:
                tensor_name = expr['tensor']
                if tensor_name in param_map:
                    param_expr = param_map[tensor_name]
                    if param_expr['type'] == 'name':
                        new_expr['tensor'] = param_expr['name']
                elif tensor_name in name_mapping:
                    new_expr['tensor'] = name_mapping[tensor_name]
            return new_expr


        def build_ast_with_functions(tree):
            """Enhanced build_ast that handles function definitions"""
            functions = {}  # Store function definitions
            ast = []
            output_tensor = None
            call_counter = 0  # For generating unique names
            
            if tree.data == 'program':
                for child in tree.children:
                    if isinstance(child, Tree) and child.data == 'statement':
                        stmt = child.children[0]
                        
                        # Function definition
                        if stmt.data == 'function_def':
                            func_def = build_function_def(stmt)
                            if func_def:
                                functions[func_def['name']] = func_def
                                if DEBUG_MODE:
                                    print(f"Registered function: {func_def['name']}")
                        
                        # Let binding
                        elif stmt.data == 'let_binding':
                            let_node = build_let_binding(stmt)
                            if let_node:
                                if not let_node.get('expr'):
                                    if DEBUG_MODE:
                                        print(f"Warning: let_node has no expr: {let_node}")
                                    continue
                                    
                                # Check if expression is a user function call
                                if let_node['expr']['type'] == 'user_function_call':
                                    func_call = let_node['expr']
                                    func_name = func_call['func_name']
                                    
                                    if func_name in functions:
                                        # Inline the function
                                        func_def = functions[func_name]
                                        call_counter += 1
                                        
                                        print(f"DEBUG: Inlining function {func_name}")
                                        print(f"DEBUG: Args: {func_call['args']}")
                                        print(f"DEBUG: Function params: {func_def['params']}")

                                        inlined_stmts, return_name = inline_function_call(
                                            func_def, func_call['args'], {}, 
                                            f"{func_name}_{call_counter}"
                                        )
                                        
                                        print(f"DEBUG: Inlined {len(inlined_stmts)} statements")
                                        print(f"DEBUG: Return name: {return_name}")

                                        # In build_ast_with_functions, after inlining
                                        inlined_stmts, return_name = inline_function_call(
                                            func_def, func_call['args'], {}, 
                                            f"{func_name}_{call_counter}"
                                        )

                                        # Recursively inline any nested function calls
                                        # After getting inlined_stmts
                                        expanded_stmts = []
                                        local_mappings = {}  # Track what local variables map to

                                        for stmt in inlined_stmts:
                                            if stmt['type'] == 'let' and stmt['expr']['type'] == 'user_function_call':
                                                nested_call = stmt['expr']
                                                nested_func_name = nested_call['func_name']
                                                
                                                if nested_func_name in functions:
                                                    call_counter += 1
                                                    nested_inlined, nested_return = inline_function_call(
                                                        functions[nested_func_name],  # Get function from dict
                                                        nested_call['args'], 
                                                        {},
                                                        f"{nested_func_name}_{call_counter}"
                                                    )
                                                    
                                                    expanded_stmts.extend(nested_inlined)
                                                    
                                                    if nested_return:
                                                        # Map local var to actual return
                                                        local_mappings[stmt['name']] = nested_return
                                                else:
                                                    expanded_stmts.append(stmt)
                                            else:
                                                # Substitute any local mappings in this statement
                                                if stmt['type'] == 'let' and 'args' in stmt['expr']:
                                                    new_args = []
                                                    for arg in stmt['expr']['args']:
                                                        new_args.append(local_mappings.get(arg, arg))
                                                    stmt['expr']['args'] = new_args
                                                expanded_stmts.append(stmt)

                                        ast.extend(expanded_stmts)
                                        
                                        # Create final assignment
                                        if return_name:
                                            ast.append({
                                                'type': 'let',
                                                'name': let_node['name'],
                                                'ty': let_node['ty'],
                                                'expr': {'type': 'name', 'name': return_name},
                                                'tree': None
                                            })
                                    else:
                                        print(f"Error: Undefined function '{func_name}'")
                                        return None, None
                                else:
                                    ast.append(let_node)
                        
                        # Expression (output tensor)
                        elif stmt.data == 'expr':
                            expr_node = build_expression(stmt)
                            if expr_node and expr_node['type'] == 'name':
                                output_tensor = expr_node['name']
            
            # Default to last tensor if no explicit output
            if not output_tensor and ast:
                output_tensor = ast[-1]['name']
            
            return ast, output_tensor, functions


        # ================================================================
        # /Build functions for Feature "function definitions and calls."
        # ================================================================
        def type_checker(ast):
            env = {}
            for node in ast:
                if node['type'] == 'let':
                    name = node['name']
                    expr = node['expr']
                    ty = node['ty']

                    if ty and isinstance(ty, dict) and 'dtype' in ty:
                        # Check if shape contains generic dimensions
                        shape = ty['shape']
                        if any(isinstance(dim, str) for dim in shape):
                            # Generic type - will be resolved during function inlining
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Generic type for {name}: {ty}")
                            # Skip registration, will be resolved later
                            continue
                        else:
                            # Concrete type
                            env[name] = {'dtype': ty['dtype'], 'shape': ty['shape']}
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Tensor {name} {env[name]}")

                    # BUG incorrectly calculating dimensions for 2D arrays.
                    # elif isinstance(expr, dict) and expr['type'] == 'tensor_literal':
                    #     data = expr['data']
                    #     num_elements = len(data)
                    #     if expr['is_1d']:
                    #         shape = (num_elements,)
                    #     else:
                    #         rows = sum(1 for child in expr['tree'].children if child.data == 'inner_array')
                    #         cols = num_elements // rows if rows > 0 else num_elements
                    #         shape = (rows, cols) if rows > 1 else (cols,)
                    #     env[name] = {'dtype': 'f32', 'shape': shape}
                    #     print(f"Inferred shape for {name}: {env[name]['shape']}")

                    # BUG FIX Change the condition to preserve 2D shape when there are inner_arrays:
                    elif isinstance(expr, dict) and expr['type'] == 'tensor_literal':
                        data = expr['data']
                        num_elements = len(data)
                        if expr['is_1d']:
                            shape = (num_elements,)
                        else:
                            rows = sum(1 for child in expr['tree'].children if child.data == 'inner_array')
                            cols = num_elements // rows if rows > 0 else num_elements
                            shape = (rows, cols)  # Always 2D if not is_1d
                        env[name] = {'dtype': 'f32', 'shape': shape}
                        
                        print(f"Inferred shape for {name}: {env[name]['shape']}")

                    elif isinstance(expr, dict) and expr['type'] == 'name':
                        # Simple name reference - copy type from referenced tensor
                        ref_name = expr['name']
                        if ref_name not in env:
                            print(f"Type error: Undefined tensor {ref_name} referenced by {name}")
                            return False, env
                        
                        env[name] = {'dtype': env[ref_name]['dtype'], 'shape': env[ref_name]['shape']}
                        if DEBUG_INFO:
                            tensorlang.print(type="[INFO]", message=f"Type {name} assigned from {ref_name}: {env[name]}")

                    elif isinstance(expr, dict) and expr['type'] in [
                            'matmul', 'add', 'minus', 'mult', 'div', 
                            'relu', 'sigmoid', 'tanh', 'softmax', 
                            'fill', 'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 
                            'greater', 'less', 'equal', 
                            'linear', 'layer_norm', 'batch_norm', 'instance_norm', 'cross_entropy', 'mse_loss', 
                            'transpose', 'reshape', 'concat', 'slice',
                            'slice'
                        ]:

                        if expr['type'] == 'fill':
                            env[name] = {'dtype': 'f32', 'shape': expr['shape']}
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned from fill: {env[name]}")
                        
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
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} ({expr['type']}): {env[name]}")

                        elif expr['type'] == 'softmax':
                            #print(f"DEBUG: Processing softmax type checking for {name}")
                            tensor_name = expr['tensor']
                            if tensor_name not in env:
                                print(f"Type error: Undefined tensor {tensor_name} for {expr['type']}")
                                return False, env
                            
                            # Softmax preserves input shape
                            env[name] = {'dtype': 'f32', 'shape': env[tensor_name]['shape']}
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Tensor {tensor_name} {name}:({expr['type']}):{env[name]}")

                        elif expr['type'] == 'slice':
                            tensor_name = expr['tensor']
                            if tensor_name not in env:
                                print(f"Type error: Undefined tensor {tensor_name} for {expr['type']}")
                                return False, env
                            
                            input_shape = env[tensor_name]['shape']
                            slice_specs = expr['specs']
                            
                            # Calculate output shape based on slice specifications
                            output_shape = []
                            for i, spec in enumerate(slice_specs):
                                if i >= len(input_shape):
                                    print(f"Type error: Slice dimension {i} exceeds tensor dimensions {len(input_shape)}")
                                    return False, env
                                    
                                dim_size = input_shape[i]
                                
                                if spec['type'] == 'index':
                                    # Single index removes dimension
                                    continue  # Don't add to output_shape
                                elif spec['type'] == 'full_slice':
                                    # Full slice preserves dimension
                                    output_shape.append(dim_size)
                                elif spec['type'] == 'slice':
                                    start = spec.get('start', 0)
                                    end = spec.get('end', dim_size)
                                    
                                    # Handle negative indices and bounds checking
                                    if start is None:
                                        start = 0
                                    if end is None:
                                        end = dim_size
                                    if start < 0:
                                        start = max(0, dim_size + start)
                                    if end < 0:
                                        end = max(0, dim_size + end)
                                        
                                    start = min(start, dim_size)
                                    end = min(end, dim_size)
                                    
                                    slice_length = max(0, end - start)
                                    output_shape.append(slice_length)
                            
                            # Handle remaining dimensions (implicit full slices)
                            for i in range(len(slice_specs), len(input_shape)):
                                output_shape.append(input_shape[i])
                            
                            # Convert to tuple, handle scalar case
                            if not output_shape:
                                output_shape = ()  # Scalar result
                            else:
                                output_shape = tuple(output_shape)
                            
                            env[name] = {'dtype': 'f32', 'shape': output_shape}
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} (slice): {env[name]}")

                        # ================================================================
                        # Type: layer_norm, batch_norm, instance_norm
                        # ================================================================
                        elif expr['type'] == 'layer_norm':
                            tensor_name = expr['tensor']
                            if tensor_name not in env:
                                print(f"Type error: Undefined tensor {tensor_name} for layer_norm")
                                return False, env
                            
                            input_shape = env[tensor_name]['shape']
                            axis = expr.get('axis')
                            
                            # Default to last axis if not specified
                            if axis is None:
                                axis = len(input_shape) - 1
                            
                            # Validate axis
                            if axis < 0 or axis >= len(input_shape):
                                print(f"Type error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                                return False, env
                            
                            # Layer norm preserves input shape
                            env[name] = {'dtype': 'f32', 'shape': input_shape}

                            if DEBUG_MODE:
                                tensorlang.print(message=f"Type {expr['type']} Assigned for {name} (layer_norm): {env[name]}")
                                tensorlang.print(message=f"Type {expr['type']} layer_norm axis={axis}, input_shape={input_shape}")

                        elif expr['type'] == 'batch_norm':
                            tensor_name       = expr['tensor']
                            running_mean_name = expr['running_mean']
                            running_var_name  = expr['running_var']
                            
                            if tensor_name not in env:
                                print(f"Type error: Undefined tensor {tensor_name} for batch_norm")
                                return False, env
                            if running_mean_name not in env:
                                print(f"Type error: Undefined running_mean {running_mean_name} for batch_norm")
                                return False, env
                            if running_var_name not in env:
                                print(f"Type error: Undefined running_var {running_var_name} for batch_norm")
                                return False, env
                            
                            input_shape = env[tensor_name]['shape']
                            mean_shape  = env[running_mean_name]['shape']
                            var_shape   = env[running_var_name]['shape']
                            
                            # Batch norm: input (N, C, ...), running_mean/var (C,)
                            if len(input_shape) < 2:
                                print(f"Type error: BatchNorm input must be at least 2D, got {len(input_shape)}D")
                                return False, env
                            
                            num_features = input_shape[1]  # Channel dimension
                            expected_stats_shape = (num_features,)
                            
                            if mean_shape != expected_stats_shape:
                                print(f"Type error: BatchNorm running_mean shape {mean_shape} != expected {expected_stats_shape}")
                                return False, env
                            if var_shape != expected_stats_shape:
                                print(f"Type error: BatchNorm running_var shape {var_shape} != expected {expected_stats_shape}")
                                return False, env
                            
                            # Output shape same as input
                            env[name] = {'dtype': 'f32', 'shape': input_shape}
                            #print(f"Type {expr['type']} assigned for {name} (batch_norm): {env[name]}")
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} (batch_norm): {env[name]}")

                        elif expr['type'] == 'instance_norm':
                            tensor_name = expr['tensor']
                            if tensor_name not in env:
                                print(f"Type error: Undefined tensor {tensor_name} for instance_norm")
                                return False, env
                            
                            input_shape = env[tensor_name]['shape']
                            
                            # Instance norm preserves input shape
                            env[name] = {'dtype': 'f32', 'shape': input_shape}
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} (instance_norm): {env[name]}")

                        # ================================================================
                        # Type: max, min, argmax, argmin
                        # ================================================================
                        elif expr['type'] in ['max', 'min', 'argmax', 'argmin']:
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
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} ({expr['type']}): {env[name]}")

                        # ================================================================
                        # Type: transpose, reshape, concat
                        # ================================================================
                        elif expr['type'] == 'transpose':
                            tensor_name = expr['tensor']
                            if tensor_name not in env:
                                print(f"Type error: Undefined tensor {tensor_name} for transpose")
                                return False, env
                            
                            input_shape = env[tensor_name]['shape']
                            axes = expr.get('axes')
                            
                            if axes is None:
                                # Default transpose: reverse all dimensions
                                output_shape = tuple(reversed(input_shape))
                            else:
                                # Custom axes permutation
                                if len(axes) != len(input_shape):
                                    print(f"Type error: Transpose axes {axes} length doesn't match tensor dimensions {len(input_shape)}")
                                    return False, env
                                if set(axes) != set(range(len(input_shape))):
                                    print(f"Type error: Transpose axes {axes} must be permutation of {list(range(len(input_shape)))}")
                                    return False, env
                                output_shape = tuple(input_shape[i] for i in axes)
                            
                            env[name] = {'dtype': 'f32', 'shape': output_shape}
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} (transpose): {env[name]}")

                        elif expr['type'] == 'reshape':
                            tensor_name = expr['tensor']
                            if tensor_name not in env:
                                print(f"Type error: Undefined tensor {tensor_name} for reshape")
                                return False, env
                            
                            input_shape = env[tensor_name]['shape']
                            new_shape = expr['new_shape']
                            
                            # Validate that total number of elements remains the same
                            input_elements = int(np.prod([int(dim) for dim in input_shape]))
                            new_elements = int(np.prod([int(dim) for dim in new_shape]))
                            
                            if input_elements != new_elements:
                                print(f"Type error: Reshape element count mismatch. Input {input_elements} != output {new_elements}")
                                return False, env
                            
                            env[name] = {'dtype': 'f32', 'shape': new_shape}
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} (reshape): {env[name]}")

                        elif expr['type'] == 'concat':
                            tensor_names = expr['tensors']
                            axis = expr['axis']
                            
                            # Get tensor types
                            tensor_types = []
                            for tensor_name in tensor_names:
                                if tensor_name not in env:
                                    print(f"Type error: Undefined tensor {tensor_name} for concat")
                                    return False, env
                                tensor_types.append(env[tensor_name])
                            
                            # Validate concatenation compatibility
                            first_shape = tensor_types[0]['shape']
                            if axis < 0 or axis >= len(first_shape):
                                print(f"Type error: Concat axis {axis} out of bounds for shape {first_shape}")
                                return False, env
                            
                            # All tensors must have same number of dimensions
                            concat_dim_size = 0
                            for i, tensor_type in enumerate(tensor_types):
                                shape = tensor_type['shape']
                                if len(shape) != len(first_shape):
                                    print(f"Type error: Concat tensor {i} dimension count mismatch")
                                    return False, env
                                
                                # All dimensions except concat axis must match
                                for dim_idx, (dim1, dim2) in enumerate(zip(first_shape, shape)):
                                    if dim_idx != axis and dim1 != dim2:
                                        print(f"Type error: Concat tensor {i} shape mismatch at dimension {dim_idx}")
                                        return False, env
                                
                                # Sum up the concat dimension
                                concat_dim_size += shape[axis]
                            
                            # Compute output shape
                            output_shape = list(first_shape)
                            output_shape[axis] = concat_dim_size
                            env[name] = {'dtype': 'f32', 'shape': tuple(output_shape)}
                            if DEBUG_INFO:
                                tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} (concat): {env[name]}")

                        else:
                            # Handle other operations (matmul, add, minus, mult, div, relu, sigmoid, tanh etc.)
                            if expr['type'] in ['sum', 'mean']:
                                # This is handled above, but keeping this check for safety
                                pass
                            else:
                                arg_names = expr.get('args', [])
                                if arg_names:  # Only process if there are args
                                    args = [env.get(arg_name) for arg_name in arg_names]
                                    if DEBUG_INFO:
                                        tensorlang.print(type="[INFO]", message=f"Checking args for {name}: {args}")
                                    if not all(args):
                                        print(f"Type error: Undefined args for {name}")
                                        return False, env
                                        
                                    if expr['type'] == 'matmul':
                                        if args[0]['shape'][1] != args[1]['shape'][0]:
                                            print(f"Type error: Matmul shape mismatch for {name}, {args[0]['shape']} x {args[1]['shape']}")
                                            return False, env
                                        env[name] = {'dtype': 'f32', 'shape': (args[0]['shape'][0], args[1]['shape'][1])}
                                        if DEBUG_INFO:
                                            tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name}: {env[name]}")

                                    elif expr['type'] in ['add', 'minus', 'mult', 'div']:
                                        shape1, shape2 = args[0]['shape'], args[1]['shape']
                                        
                                        # Check if shapes are identical
                                        if shape1 == shape2:
                                            output_shape = shape1
                                        # Special case: 2D + 1D broadcasting
                                        # elif len(shape1) == 2 and len(shape2) == 1:
                                        #     if shape1[0] == shape2[0] or shape1[1] == shape2[0]:
                                        #         output_shape = shape1
                                        #     else:
                                        #         print(f"Type error: {expr['type']} shape mismatch for {name}, {shape1} != {shape2}")
                                        #         return False, env
                                        # The fix is to recognize that (1,) is essentially a scalar and should broadcast to anything:
                                        elif len(shape1) == 2 and len(shape2) == 1:
                                            # If shape2 is (1,), treat as scalar broadcast
                                            if shape2[0] == 1:
                                                output_shape = shape1
                                            # Otherwise check dimension match
                                            elif shape1[0] == shape2[0] or shape1[1] == shape2[0]:
                                                output_shape = shape1
                                            else:
                                                print(f"Type error: {expr['type']} shape mismatch for {name}, {shape1} != {shape2}")
                                                return False, env

                                        # General NumPy-style broadcasting
                                        else:
                                            # Pad shapes to same length
                                            ndim = max(len(shape1), len(shape2))
                                            padded_shape1 = (1,) * (ndim - len(shape1)) + shape1
                                            padded_shape2 = (1,) * (ndim - len(shape2)) + shape2
                                            
                                            output_shape = []
                                            broadcast_compatible = True
                                            for d1, d2 in zip(padded_shape1, padded_shape2):
                                                if d1 == d2:
                                                    output_shape.append(d1)
                                                elif d1 == 1:
                                                    output_shape.append(d2)
                                                elif d2 == 1:
                                                    output_shape.append(d1)
                                                else:
                                                    broadcast_compatible = False
                                                    break
                                            
                                            if not broadcast_compatible:
                                                print(f"Type error: {expr['type']} incompatible broadcast shapes {shape1} and {shape2}")
                                                return False, env
                                            
                                            output_shape = tuple(output_shape)
                                        
                                        env[name] = {'dtype': 'f32', 'shape': output_shape}
                                        if DEBUG_INFO:
                                            tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name}: {env[name]}")

                                    elif expr['type'] == 'relu':
                                        env[name] = {'dtype': 'f32', 'shape': args[0]['shape']}
                                        if DEBUG_INFO:
                                            tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name}: {env[name]}")

                                    elif expr['type'] in ['sigmoid', 'tanh']:
                                        env[name] = {'dtype': 'f32', 'shape': args[0]['shape']}
                                        if DEBUG_INFO:
                                            tensorlang.print(type="[INFO]", message=f"Type {expr['type']} for {name}: {env[name]}")

                                    elif expr['type'] in ['greater', 'less', 'equal']:
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
                                        # Note: Comparisons return float tensors (0.0 or 1.0) for CUDA compatibility
                                        env[name] = {'dtype': 'f32', 'shape': output_shape}
                                        if DEBUG_INFO:
                                            tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} ({expr['type']}): {env[name]}")


                                    elif expr['type'] == 'linear':
                                        if len(args) != 3:
                                            print(f"Type error: Linear requires exactly 3 arguments, got {len(args)}")
                                            return False, env
                                        
                                        input_tensor, weight_tensor, bias_tensor = args
                                        input_shape = input_tensor['shape']
                                        weight_shape = weight_tensor['shape']
                                        bias_shape = bias_tensor['shape']
                                        
                                        # Linear layer: input @ weight + bias
                                        # Input: (batch_size, in_features) or (in_features,)
                                        # Weight: (in_features, out_features)  
                                        # Bias: (out_features,)
                                        # Output: (batch_size, out_features) or (out_features,)
                                        if len(input_shape) == 1:
                                            # 1D input case: (in_features,)
                                            in_features = input_shape[0]
                                            if len(weight_shape) != 2 or weight_shape[0] != in_features:
                                                print(f"Type error: Linear weight shape mismatch. Expected (in_features={in_features}, out_features), got {weight_shape}")
                                                return False, env
                                            out_features = weight_shape[1]
                                            if len(bias_shape) != 1 or bias_shape[0] != out_features:
                                                print(f"Type error: Linear bias shape mismatch. Expected ({out_features},), got {bias_shape}")
                                                return False, env
                                            output_shape = (out_features,)
                                            
                                        elif len(input_shape) == 2:
                                            # 2D input case: (batch_size, in_features)
                                            batch_size, in_features = input_shape
                                            if len(weight_shape) != 2 or weight_shape[0] != in_features:
                                                print(f"Type error: Linear weight shape mismatch. Expected (in_features={in_features}, out_features), got {weight_shape}")
                                                return False, env
                                            out_features = weight_shape[1]
                                            if len(bias_shape) != 1 or bias_shape[0] != out_features:
                                                print(f"Type error: Linear bias shape mismatch. Expected ({out_features},), got {bias_shape}")
                                                return False, env
                                            output_shape = (batch_size, out_features)
                                        else:
                                            print(f"Type error: Linear input must be 1D or 2D tensor, got {len(input_shape)}D")
                                            return False, env
                                        
                                        env[name] = {'dtype': 'f32', 'shape': output_shape}
                                        if DEBUG_INFO:
                                            tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} (linear): {env[name]}")

                                    elif expr['type'] in ['cross_entropy', 'mse_loss']:
                                        if len(args) != 2:
                                            print(f"Type error: {expr['type']} requires exactly 2 arguments, got {len(args)}")
                                            return False, env
                                        
                                        predictions_tensor, targets_tensor = args
                                        pred_shape = predictions_tensor['shape']
                                        target_shape = targets_tensor['shape']
                                        
                                        if expr['type'] == 'cross_entropy':
                                            # Cross entropy loss
                                            # Predictions: (batch_size, num_classes) - softmax probabilities or logits
                                            # Targets: (batch_size,) - class indices OR (batch_size, num_classes) - one-hot
                                            
                                            if len(pred_shape) == 2:
                                                batch_size, num_classes = pred_shape
                                                
                                                if len(target_shape) == 1 and target_shape[0] == batch_size:
                                                    # Class indices format: targets shape (batch_size,)
                                                    output_shape = ()  # Scalar loss (mean over batch)
                                                elif len(target_shape) == 2 and target_shape == pred_shape:
                                                    # One-hot format: targets shape (batch_size, num_classes)
                                                    output_shape = ()  # Scalar loss (mean over batch)
                                                else:
                                                    print(f"Type error: Cross entropy target shape {target_shape} incompatible with predictions {pred_shape}")
                                                    return False, env
                                            else:
                                                print(f"Type error: Cross entropy predictions must be 2D (batch, classes), got {len(pred_shape)}D")
                                                return False, env
                                                
                                        elif expr['type'] == 'mse_loss':
                                            # MSE loss: predictions and targets must have same shape
                                            if pred_shape != target_shape:
                                                print(f"Type error: MSE loss shape mismatch. Predictions {pred_shape} != targets {target_shape}")
                                                return False, env
                                            output_shape = ()  # Scalar loss (mean over all elements)
                                        
                                        env[name] = {'dtype': 'f32', 'shape': output_shape}
                                        if DEBUG_INFO:
                                            tensorlang.print(type="[INFO]", message=f"Type {expr['type']} assigned for {name} ({expr['type']}): {env[name]}")

                    else:
                        print(f"Type error: Unrecognized expr type for {name}: {expr['type']}")
                        return False, env

            #if DEBUG_MODE:
            print(f"Final environment keys: {list(env.keys())}")
            return True, env

        def prod(lst):
            return reduce(lambda x, y: x * y, lst, 1)

        def generate_elementwise_kernel(op_type, name, arg1, arg2, size, cuda_debug_code):
            """Generate element-wise kernel for binary operations where shapes match"""
            op_symbol = {
                'add': '+',
                'minus': '-',
                'mult': '*',
                'div': '/'
            }[op_type]
            
            kernel = f"""
// -----------------------------------------------------------------------------
// TensorLang -> Elementwise {op_type} ({op_symbol})
// -----------------------------------------------------------------------------
__global__ void {op_type}_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = A[idx] {op_symbol} B[idx];
    }}
}}
extern "C" void launch_{op_type}_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    {op_type}_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
            return kernel, (op_type, name, arg1, arg2, size)


        def generate_broadcast_kernel(op_name, op_type, name, arg1, arg2, shape1, shape2, output_shape, cuda_debug_code):
            """Generate broadcast kernel for binary operations (add, minus, mult, div)"""
            rows, cols = output_shape
            
            # Determine broadcast direction
            if shape1[0] == shape2[0]:
                # Row broadcast: (3, 4) op (3,) - use B[i]
                op_symbol = {
                    'add': '+', 'minus': '-', 'mult': '*', 'div': '/'
                }[op_type]
                
                kernel = f"""
__global__ void {op_type}_broadcast_rows_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = A[i * cols + j] {op_symbol} B[i];
    }}
}}
extern "C" void launch_{op_type}_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    {op_type}_broadcast_rows_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
        """
                return kernel, (f'{op_type}_broadcast_rows', name, arg1, arg2, rows, cols)
            else:
                # Column broadcast: (3, 4) op (4,) - use B[j]
                op_symbol = {
                    'add': '+', 'minus': '-', 'mult': '*', 'div': '/'
                }[op_type]
                
                kernel = f"""
__global__ void {op_type}_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = A[i * cols + j] {op_symbol} B[j];
    }}
}}
extern "C" void launch_{op_type}_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    {op_type}_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
        """
                return kernel, (f'{op_type}_broadcast', name, arg1, arg2, rows, cols)

        def generate_general_broadcast_kernel(op_type, name, arg1, arg2, shape1, shape2, output_shape, cuda_debug_code):
            """Generate kernel for general NumPy-style broadcasting"""
            op_symbol = {
                'add': '+', 'minus': '-', 'mult': '*', 'div': '/'
            }[op_type]
            
            # Calculate strides and dimensions
            ndim_out = len(output_shape)
            total_elements = int(np.prod([int(dim) for dim in output_shape]))
            
            # Pad shapes with 1s on the left to match output dimensionality
            padded_shape1 = (1,) * (ndim_out - len(shape1)) + shape1
            padded_shape2 = (1,) * (ndim_out - len(shape2)) + shape2
            
            # Convert to strings for kernel code
            shape1_str = ', '.join(map(str, [int(d) for d in padded_shape1]))
            shape2_str = ', '.join(map(str, [int(d) for d in padded_shape2]))
            output_shape_str = ', '.join(map(str, [int(d) for d in output_shape]))
            
            kernel = f"""
__device__ int compute_broadcast_index(int linear_idx, const int* out_shape, const int* in_shape, int ndim) {{
    int in_idx = 0;
    int temp_idx = linear_idx;
    int out_stride = 1;
    
    // Compute index by iterating dimensions from right to left
    for (int i = ndim - 1; i >= 0; i--) {{
        int out_coord = temp_idx % out_shape[i];
        temp_idx /= out_shape[i];
        
        // If input dimension is 1, use index 0 (broadcast), else use coordinate
        int in_coord = (in_shape[i] == 1) ? 0 : out_coord;
        in_idx += in_coord * out_stride;
        
        if (i > 0) {{
            out_stride *= in_shape[i];
        }}
    }}
    return in_idx;
}}

__global__ void {op_type}_general_broadcast_kernel_{name}(
    float* A, float* B, float* C, 
    int* shape1, int* shape2, int* out_shape, 
    int ndim, int total_elements) {{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {{
        int idx1 = compute_broadcast_index(idx, out_shape, shape1, ndim);
        int idx2 = compute_broadcast_index(idx, out_shape, shape2, ndim);
        
        C[idx] = A[idx1] {op_symbol} B[idx2];
    }}
}}

extern "C" void launch_{op_type}_{name}(
    float* A, float* B, float* C,
    int* shape1, int* shape2, int* out_shape,
    int ndim, int total_elements) {{
    
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    
    {op_type}_general_broadcast_kernel_{name}<<<grid, block>>>(
        A, B, C, shape1, shape2, out_shape, ndim, total_elements);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
            # return kernel, ('general_broadcast', name, arg1, arg2, 
            #                 padded_shape1, padded_shape2, output_shape, total_elements)

            # In generate_general_broadcast_kernel function, change the return to include op_type:
            return kernel, ('general_broadcast', op_type, name, arg1, arg2, 
                            padded_shape1, padded_shape2, output_shape, total_elements)

        def can_broadcast(shape1, shape2):
            """Check if two shapes can broadcast together (NumPy rules)"""
            # Pad shorter shape with 1s on the left
            ndim = max(len(shape1), len(shape2))
            s1 = (1,) * (ndim - len(shape1)) + shape1
            s2 = (1,) * (ndim - len(shape2)) + shape2
            
            # Check each dimension
            for d1, d2 in zip(s1, s2):
                if d1 != d2 and d1 != 1 and d2 != 1:
                    return False
            return True

        def generate_scalar_broadcast_kernel(op_type, name, arg1, arg2, size, cuda_debug_code):
            """Handle broadcasting with scalar (0-D tensor)"""
            op_symbol = {
                'add': '+', 'minus': '-', 'mult': '*', 'div': '/'
            }[op_type]
            
            kernel = f"""
__global__ void {op_type}_scalar_kernel_{name}(float* A, float* B_scalar, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = A[idx] {op_symbol} B_scalar[0];
    }}
}}
extern "C" void launch_{op_type}_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    {op_type}_scalar_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
        """
            return kernel, (f'{op_type}_scalar', name, arg1, arg2, size)


        # ================================================================
        #                Parser / Compiler implementation
        # ================================================================

        # Build the AST 
        # ORIGINAL
        # ast, output_tensor = build_ast(parse_tree)
        # if DEBUG_MODE:
        #     print(f"BUILT AST:\n{ast}")
        #     print(f"Output Tensor: {output_tensor}")

        # Update
        ast, output_tensor, functions = build_ast_with_functions(parse_tree)
        if DEBUG_MODE:
            print(f"BUILT AST:\n{ast}")
            print(f"Functions: {list(functions.keys())}")
            print(f"Output Tensor: {output_tensor}")




        # Run type checker
        success, env = type_checker(ast)
        if DEBUG_MODE:
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

// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif
// -----------------------------------------------------------------------------
"""
            # CUDA Debugging
            if DEBUG_MODE:
                cuda_debug_code = "CUDA_CHECK_ERROR();"            
                cuda_code+= f"""
// TensorLang: DEBUG
#include <stdio.h>

// CUDA Error Checking Macro
#define CUDA_CHECK_ERROR() do {{ \\
    cudaError_t err = cudaGetLastError(); \\
    if (err != cudaSuccess) {{ \\
        printf("CUDA Error: %s\\n", cudaGetErrorString(err)); \\
    }} \\
}} while(0)
// -----------------------------------------------------------------------------
"""
            else:
                cuda_debug_code = "// DEBUG DISABLED"
                cuda_code+= f"""
    // TensorLang: PRODUCTION

"""

            # Generate kernels for operations
            kernels = []
            for node in ast:
                if node['type'] == 'let' and isinstance(node['expr'], dict):
                    name = node['name']
                    expr = node['expr']

                    # Skip alias assignments - no kernel needed
                    if expr['type'] == 'name':
                        continue

                    if DEBUG_MODE:
                        tensorlang.print(message=f"Generating kernel for {name} ({expr['type']})")

                    # ========================================
                    # Tensor Literal
                    # ========================================
                    if expr['type'] == 'tensor_literal':
                        shape = tuple(int(dim) for dim in env[name]['shape'])
                        tensors[name] = np.array(expr['data'], dtype=np.float32).reshape(shape)
                        if DEBUG_INFO:
                            tensorlang.print(type="[INFO]", message=f"Kernel Tensor Initialized {name} with shape {shape}")


                    elif expr['type'] in ['add', 'minus', 'mult', 'div']:
                        arg1, arg2 = expr['args']
                        shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                        output_shape = env[name]['shape']

                        tensorlang.print(message=f"Tensor {expr['type']}")
                        tensorlang.print(message=f"Tensor Shape1: {len(shape1)} Shape2: {len(shape2)}")

                        # Case 1: Identical shapes (element-wise)
                        if shape1 == shape2:
                            size = int(np.prod([int(dim) for dim in output_shape]))
                            kernel, kernel_info = generate_elementwise_kernel(
                                expr['type'], name, arg1, arg2, size, cuda_debug_code
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                        
                        # Case 2: 2D + 1D broadcasting
                        elif len(shape1) == 2 and len(shape2) == 1:
                            tensorlang.print(message=f"Tensor Case 2: 2D + 1D broadcasting")
                            kernel, kernel_info = generate_broadcast_kernel(
                                expr['type'], expr['type'], name, arg1, arg2, 
                                shape1, shape2, output_shape, cuda_debug_code
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                        
                        # Case 2.5: 1D tensors where one has size 1
                        elif len(shape1) == 1 and len(shape2) == 1 and (shape1[0] == 1 or shape2[0] == 1):
                            print ("Case 2.5: 1D tensors where one has size 1")
                            size = max(int(shape1[0]), int(shape2[0])) # potential issue
                            size = int(np.prod([int(dim) for dim in output_shape]))
                            kernel, kernel_info = generate_scalar_broadcast_kernel(
                                expr['type'], name, arg1, arg2, size, cuda_debug_code
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # Case 3: General broadcasting (NumPy-style)
                        elif can_broadcast(shape1, shape2):
                            print ("Case 4: General broadcasting (NumPy-style)")
                            kernel, kernel_info = generate_general_broadcast_kernel(
                                expr['type'], name, arg1, arg2, 
                                shape1, shape2, output_shape, cuda_debug_code
                            )
                            # Modify kernel_info to include expr['type'] at the front
                            kernels.append(kernel_info)  # Just append directly
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
    {cuda_debug_code}
}}
"""
                        kernels.append(('fill', name, None, None, size, expr['value']))
                        cuda_code += kernel
                        
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
    {cuda_debug_code}
}}
"""
                        kernels.append(('matmul', name, arg1, arg2, m, n, p))
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
    {cuda_debug_code}
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
    {cuda_debug_code}
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
    {cuda_debug_code}
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
    {cuda_debug_code}
}}
"""
                            kernels.append(('softmax', name, tensor_name, None, rows, cols, axis))
                            cuda_code += kernel

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
    {cuda_debug_code}
}}
"""
                            kernels.append(('softmax_1d', name, tensor_name, None, size))
                            cuda_code += kernel

                    # GREATER
                    elif expr['type'] == 'greater':
                        arg1, arg2 = expr['args']
                        shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                        output_shape = env[name]['shape']
                        
                        if shape1 == shape2:
                            # Element-wise greater
                            size = int(np.prod([int(dim) for dim in output_shape]))
                            kernel = f"""
__global__ void greater_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = (A[idx] > B[idx]) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_greater_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    greater_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('greater', name, arg1, arg2, size))
                            cuda_code += kernel
                        else:
                            # Broadcasting greater (e.g., (4,5) > (5,) -> (4,5))
                            rows, cols = output_shape
                            kernel = f"""
__global__ void greater_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = (A[i * cols + j] > B[j]) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_greater_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    greater_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('greater_broadcast', name, arg1, arg2, rows, cols))
                            cuda_code += kernel

                    # LESS
                    elif expr['type'] == 'less':
                        arg1, arg2 = expr['args']
                        shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                        output_shape = env[name]['shape']
                        
                        if shape1 == shape2:
                            # Element-wise less
                            size = int(np.prod([int(dim) for dim in output_shape]))
                            kernel = f"""
__global__ void less_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = (A[idx] < B[idx]) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_less_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    less_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('less', name, arg1, arg2, size))
                            cuda_code += kernel
                        else:
                            # Broadcasting less
                            rows, cols = output_shape
                            kernel = f"""
__global__ void less_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = (A[i * cols + j] < B[j]) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_less_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    less_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('less_broadcast', name, arg1, arg2, rows, cols))
                            cuda_code += kernel

                    # EQUAL
                    elif expr['type'] == 'equal':
                        arg1, arg2 = expr['args']
                        shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                        output_shape = env[name]['shape']
                        
                        if shape1 == shape2:
                            # Element-wise equal
                            size = int(np.prod([int(dim) for dim in output_shape]))
                            kernel = f"""
__global__ void equal_kernel_{name}(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        C[idx] = (fabsf(A[idx] - B[idx]) < 1e-6f) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_equal_{name}(float* A, float* B, float* C, int size) {{
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    equal_kernel_{name}<<<grid, block>>>(A, B, C, size);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('equal', name, arg1, arg2, size))
                            cuda_code += kernel
                        else:
                            # Broadcasting equal
                            rows, cols = output_shape
                            kernel = f"""
__global__ void equal_broadcast_kernel_{name}(float* A, float* B, float* C, int rows, int cols) {{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {{
        C[i * cols + j] = (fabsf(A[i * cols + j] - B[j]) < 1e-6f) ? 1.0f : 0.0f;
    }}
}}
extern "C" void launch_equal_{name}(float* A, float* B, float* C, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    equal_broadcast_kernel_{name}<<<grid, block>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('equal_broadcast', name, arg1, arg2, rows, cols))
                            cuda_code += kernel


                    # ========================================
                    # SUM
                    # ========================================
                    elif expr['type'] == 'sum':
                        tensor_name = expr['tensor']
                        axis = expr.get('axis')
                        input_shape = env[tensor_name]['shape']
                        #print(f"DEBUG SUM: tensor={tensor_name}, axis={axis}, shape={input_shape}")
                        
                        if axis is None:
                            #print("DEBUG: Taking sum_full path")
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
    {cuda_debug_code}
}}
"""
                            kernels.append(('sum_full', name, tensor_name, None, size))
                            cuda_code += kernel
                        else:
                            # Reduction along specific axis
                            #print(f"DEBUG: Taking axis-specific path, axis={axis}")
                            if len(input_shape) == 2 and axis == 1:
                                #print("DEBUG: Using sum_axis kernel (axis=1)")
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
    {cuda_debug_code}
}}
"""
                                kernels.append(('sum_axis', name, tensor_name, None, rows, cols, axis))
                                cuda_code += kernel
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
    {cuda_debug_code}
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
    {cuda_debug_code}

    // Divide by size to get mean
    float mean_val;
    cudaMemcpy(&mean_val, output, sizeof(float), cudaMemcpyDeviceToHost);
    mean_val /= size;
    cudaMemcpy(output, &mean_val, sizeof(float), cudaMemcpyHostToDevice);
}}
"""
                            kernels.append(('mean_full', name, tensor_name, None, size))
                            cuda_code += kernel
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
    {cuda_debug_code}
}}
"""
                                kernels.append(('mean_axis', name, tensor_name, None, rows, cols, axis))
                                cuda_code += kernel
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
    {cuda_debug_code}
}}
"""
                                kernels.append(('mean_axis0', name, tensor_name, None, rows, cols, axis))
                                cuda_code += kernel

                    # ========================================
                    # SLICE
                    # ========================================
                    elif expr['type'] == 'slice':
                        tensor_name = expr['tensor']
                        slice_specs = expr['specs']
                        input_shape = env[tensor_name]['shape']
                        output_shape = env[name]['shape']
                        
                        # For now, implement common 2D slicing cases
                        if len(input_shape) == 2 and len(slice_specs) <= 2:
                            rows, cols = int(input_shape[0]), int(input_shape[1])
                            
                            # Parse slice specifications
                            row_spec = slice_specs[0] if len(slice_specs) > 0 else {'type': 'full_slice'}
                            col_spec = slice_specs[1] if len(slice_specs) > 1 else {'type': 'full_slice'}
                            
                            # Calculate slice bounds
                            if row_spec['type'] == 'slice':
                                row_start = row_spec.get('start', 0) or 0
                                row_end = row_spec.get('end', rows) or rows
                            elif row_spec['type'] == 'index':
                                row_start = row_spec['value']
                                row_end = row_spec['value'] + 1
                            else:  # full_slice
                                row_start, row_end = 0, rows
                                
                            if col_spec['type'] == 'slice':
                                col_start = col_spec.get('start', 0) or 0
                                col_end = col_spec.get('end', cols) or cols
                            elif col_spec['type'] == 'index':
                                col_start = col_spec['value']
                                col_end = col_spec['value'] + 1
                            else:  # full_slice
                                col_start, col_end = 0, cols
                            
                            # Bounds checking
                            row_start = max(0, min(row_start, rows))
                            row_end = max(row_start, min(row_end, rows))
                            col_start = max(0, min(col_start, cols))
                            col_end = max(col_start, min(col_end, cols))
                            
                            out_rows = row_end - row_start
                            out_cols = col_end - col_start
                            
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
    dim3 grid((out_cols + block.x - 1) / block.x, (out_rows + block.y - 1) / block.y);
    slice_kernel_{name}<<<grid, block>>>(input, output, in_rows, in_cols,
                                        row_start, row_end, col_start, col_end,
                                        out_rows, out_cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('slice_2d', name, tensor_name, None, rows, cols, row_start, row_end, col_start, col_end, out_rows, out_cols))
                            cuda_code += kernel
                        
                        elif len(input_shape) == 1 and len(slice_specs) == 1:
                            # 1D slicing
                            size = int(input_shape[0])
                            spec = slice_specs[0]
                            
                            if spec['type'] == 'slice':
                                start = spec.get('start', 0) or 0
                                end = spec.get('end', size) or size
                            elif spec['type'] == 'index':
                                start = spec['value']
                                end = spec['value'] + 1
                            else:  # full_slice
                                start, end = 0, size
                                
                            start = max(0, min(start, size))
                            end = max(start, min(end, size))
                            out_size = end - start
                            
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
    {cuda_debug_code}
}}
"""
                            kernels.append(('slice_1d', name, tensor_name, None, start, out_size))
                            cuda_code += kernel

                    # ========================================
                    # MAX
                    # ========================================
                    elif expr['type'] == 'max':
                        tensor_name = expr['tensor']
                        axis = expr.get('axis')
                        input_shape = env[tensor_name]['shape']
                        
                        if axis is None:
                            # Full reduction to scalar
                            size = int(np.prod([int(dim) for dim in input_shape]))
                            kernel = f"""
__global__ void max_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? input[i] : -FLT_MAX;
    __syncthreads();
    
    // Reduction in shared memory using max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }}
        __syncthreads();
    }}
    
    // Use atomicMax for the final reduction (requires int conversion)
    if (tid == 0) {{
        float* addr = output;
        int* int_addr = (int*)addr;
        int old_val, new_val;
        do {{
            old_val = *int_addr;
            new_val = __float_as_int(fmaxf(__int_as_float(old_val), sdata[0]));
        }} while (atomicCAS(int_addr, old_val, new_val) != old_val);
    }}
}}
extern "C" void launch_max_{name}(float* input, float* output, int size) {{
    // Initialize output to -FLT_MAX
    float neg_max = -FLT_MAX;
    cudaMemcpy(output, &neg_max, sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    max_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('max_full', name, tensor_name, None, size))
                            cuda_code += kernel
                        else:
                            # Reduction along specific axis
                            if len(input_shape) == 2 and axis == 1:
                                # Max along columns (each row max to one value)
                                rows, cols = int(input_shape[0]), int(input_shape[1])
                                kernel = f"""
__global__ void max_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float max_val = input[row * cols];
        for (int col = 1; col < cols; col++) {{
            max_val = fmaxf(max_val, input[row * cols + col]);
        }}
        output[row] = max_val;
    }}
}}
extern "C" void launch_max_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    max_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}}
"""
                                kernels.append(('max_axis', name, tensor_name, None, rows, cols, axis))
                                cuda_code += kernel
                            elif len(input_shape) == 2 and axis == 0:
                                # Max along rows (each column max to one value)
                                rows, cols = int(input_shape[0]), int(input_shape[1])
                                kernel = f"""
__global__ void max_axis0_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float max_val = input[col];
        for (int row = 1; row < rows; row++) {{
            max_val = fmaxf(max_val, input[row * cols + col]);
        }}
        output[col] = max_val;
    }}
}}
extern "C" void launch_max_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    max_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}}
"""
                                kernels.append(('max_axis0', name, tensor_name, None, rows, cols, axis))
                                cuda_code += kernel

                    # ========================================
                    # MIN (similar to MAX but with min operations)
                    # ========================================
                    elif expr['type'] == 'min':
                        tensor_name = expr['tensor']
                        axis = expr.get('axis')
                        input_shape = env[tensor_name]['shape']
                        
                        if axis is None:
                            # Full reduction to scalar
                            size = int(np.prod([int(dim) for dim in input_shape]))
                            kernel = f"""
__global__ void min_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? input[i] : FLT_MAX;
    __syncthreads();
    
    // Reduction in shared memory using min
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }}
        __syncthreads();
    }}
    
    // Use atomicMin-like operation for final reduction
    if (tid == 0) {{
        float* addr = output;
        int* int_addr = (int*)addr;
        int old_val, new_val;
        do {{
            old_val = *int_addr;
            new_val = __float_as_int(fminf(__int_as_float(old_val), sdata[0]));
        }} while (atomicCAS(int_addr, old_val, new_val) != old_val);
    }}
}}
extern "C" void launch_min_{name}(float* input, float* output, int size) {{
    // Initialize output to FLT_MAX
    float pos_max = FLT_MAX;
    cudaMemcpy(output, &pos_max, sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    min_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('min_full', name, tensor_name, None, size))
                            cuda_code += kernel
                        else:
                            # Similar axis-specific implementations as max...
                            if len(input_shape) == 2 and axis == 1:
                                rows, cols = int(input_shape[0]), int(input_shape[1])
                                kernel = f"""
__global__ void min_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float min_val = input[row * cols];
        for (int col = 1; col < cols; col++) {{
            min_val = fminf(min_val, input[row * cols + col]);
        }}
        output[row] = min_val;
    }}
}}
extern "C" void launch_min_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    min_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                                kernels.append(('min_axis', name, tensor_name, None, rows, cols, axis))
                                cuda_code += kernel
                            elif len(input_shape) == 2 and axis == 0:
                                rows, cols = int(input_shape[0]), int(input_shape[1])
                                kernel = f"""
__global__ void min_axis0_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {{
        float min_val = input[col];
        for (int row = 1; row < rows; row++) {{
            min_val = fminf(min_val, input[row * cols + col]);
        }}
        output[col] = min_val;
    }}
}}
extern "C" void launch_min_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x);
    min_axis0_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                                kernels.append(('min_axis0', name, tensor_name, None, rows, cols, axis))
                                cuda_code += kernel

                    # ========================================
                    # ARGMAX (returns indices of maximum values)
                    # ========================================
                    elif expr['type'] == 'argmax':
                        tensor_name = expr['tensor']
                        axis = expr.get('axis')
                        input_shape = env[tensor_name]['shape']
                        
                        if axis is None:
                            # Full argmax to scalar index
                            size = int(np.prod([int(dim) for dim in input_shape]))
                            kernel = f"""
__global__ void argmax_full_kernel_{name}(float* input, float* output, int size) {{
    extern __shared__ float sdata[];
    extern __shared__ int indices[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data and indices into shared memory
    if (i < size) {{
        sdata[tid] = input[i];
        indices[tid] = i;
    }} else {{
        sdata[tid] = -FLT_MAX;
        indices[tid] = -1;
    }}
    __syncthreads();
    
    // Reduction in shared memory - keep track of both value and index
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            if (sdata[tid + s] > sdata[tid]) {{
                sdata[tid] = sdata[tid + s];
                indices[tid] = indices[tid + s];
            }}
        }}
        __syncthreads();
    }}
    
    // Write index result for this block
    if (tid == 0) {{
        atomicMax((int*)output, indices[0]);  // This is a simplification
        // In practice, need proper atomic argmax operation
        output[0] = (float)indices[0];
    }}
}}
extern "C" void launch_argmax_{name}(float* input, float* output, int size) {{
    dim3 block(256);
    dim3 grid(1);  // Single block for simplicity in full reduction
    int shared_size = block.x * (sizeof(float) + sizeof(int));
    argmax_full_kernel_{name}<<<grid, block, shared_size>>>(input, output, size);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('argmax_full', name, tensor_name, None, size))
                            cuda_code += kernel
                        else:
                            # Axis-specific argmax (simplified for 2D case)
                            if len(input_shape) == 2 and axis == 1:
                                rows, cols = int(input_shape[0]), int(input_shape[1])
                                kernel = f"""
__global__ void argmax_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float max_val = input[row * cols];
        int max_idx = 0;
        for (int col = 1; col < cols; col++) {{
            if (input[row * cols + col] > max_val) {{
                max_val = input[row * cols + col];
                max_idx = col;
            }}
        }}
        output[row] = (float)max_idx;
    }}
}}
extern "C" void launch_argmax_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    argmax_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                                kernels.append(('argmax_axis', name, tensor_name, None, rows, cols, axis))
                                cuda_code += kernel

                    # ========================================
                    # ARGMIN (similar to argmax but for minimum)
                    # ========================================
                    elif expr['type'] == 'argmin':
                        tensor_name = expr['tensor']
                        axis = expr.get('axis')
                        input_shape = env[tensor_name]['shape']
                        
                        if len(input_shape) == 2 and axis == 1:
                            rows, cols = int(input_shape[0]), int(input_shape[1])
                            kernel = f"""
__global__ void argmin_axis_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {{
        float min_val = input[row * cols];
        int min_idx = 0;
        for (int col = 1; col < cols; col++) {{
            if (input[row * cols + col] < min_val) {{
                min_val = input[row * cols + col];
                min_idx = col;
            }}
        }}
        output[row] = (float)min_idx;
    }}
}}
extern "C" void launch_argmin_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    argmin_axis_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('argmin_axis', name, tensor_name, None, rows, cols, axis))
                            cuda_code += kernel

                    # ========================================
                    # LINEAR LAYER
                    # ========================================
                    elif expr['type'] == 'linear':
                        input_name, weight_name, bias_name = expr['args']
                        input_shape = env[input_name]['shape']
                        weight_shape = env[weight_name]['shape']
                        output_shape = env[name]['shape']
                        
                        if len(input_shape) == 1:
                            # 1D case: vector @ matrix + bias
                            in_features = int(input_shape[0])
                            out_features = int(output_shape[0])
                            
                            kernel = f"""
__global__ void linear_1d_kernel_{name}(float* input, float* weight, float* bias, 
                                        float* output, int in_features, int out_features) {{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx < out_features) {{
        float sum = 0.0f;
        
        // Compute dot product: input @ weight[:, out_idx]
        for (int in_idx = 0; in_idx < in_features; in_idx++) {{
            sum += input[in_idx] * weight[in_idx * out_features + out_idx];
        }}
        
        // Add bias
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
    {cuda_debug_code}
}}
"""
                            kernels.append(('linear_1d', name, input_name, weight_name, bias_name, in_features, out_features))
                            cuda_code += kernel
                        elif len(input_shape) == 2:
                            # 2D case: batch_matrix @ matrix + bias (broadcasted)
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
        
        // Compute dot product for this batch item and output feature
        for (int in_idx = 0; in_idx < in_features; in_idx++) {{
            sum += input[batch_idx * in_features + in_idx] * 
                weight[in_idx * out_features + out_idx];
        }}
        
        // Add bias and store result
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
    {cuda_debug_code}
}}
"""
                            kernels.append(('linear_2d', name, input_name, weight_name, bias_name, batch_size, in_features, out_features))
                            cuda_code += kernel

                    # LAYER NORMALIZATION
                    elif expr['type'] == 'layer_norm':
                        tensor_name = expr['tensor']
                        axis = expr.get('axis')
                        eps = expr.get('eps', 1e-5)
                        input_shape = env[tensor_name]['shape']

                        # Default to last axis
                        if axis is None:
                            axis = len(input_shape) - 1
                        
                        #print(f"DEBUG CUDA: layer_norm axis={axis}, input_shape={input_shape}")

                        if len(input_shape) == 2:
                            # 2D Layer norm - most common case
                            if axis == 1:
                                #print("DEBUG: Using row-wise normalization kernel")
                                # Normalize along features (each row independently)
                                rows, cols = int(input_shape[0]), int(input_shape[1])
                                kernel = f"""
__global__ void layer_norm_kernel_{name}(float* input, float* output, 
                                        int rows, int cols, float eps) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {{
        // Compute mean for this row
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            sum += input[row * cols + col];
        }}
        float mean = sum / cols;
        
        // Compute variance for this row
        float var_sum = 0.0f;
        for (int col = 0; col < cols; col++) {{
            float diff = input[row * cols + col] - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / cols;
        float std_dev = sqrtf(variance + eps);
        
        // Normalize this row
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
    {cuda_debug_code}
}}
"""
                                kernels.append(('layer_norm_2d', name, tensor_name, None, rows, cols, eps))
                                cuda_code += kernel
                            elif axis == 0:
                                print("DEBUG: Using column-wise normalization kernel")
                                # Normalize along batch dimension (each column independently)
                                rows, cols = int(input_shape[0]), int(input_shape[1])
                                kernel = f"""
__global__ void layer_norm_axis0_kernel_{name}(float* input, float* output,
                                            int rows, int cols, float eps) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {{
        // Compute mean for this column
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            sum += input[row * cols + col];
        }}
        float mean = sum / rows;
        
        // Compute variance for this column
        float var_sum = 0.0f;
        for (int row = 0; row < rows; row++) {{
            float diff = input[row * cols + col] - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / rows;
        float std_dev = sqrtf(variance + eps);
        
        // Normalize this column
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
    {cuda_debug_code}
}}
"""
                                kernels.append(('layer_norm_axis0', name, tensor_name, None, rows, cols, eps))
                                cuda_code += kernel
                        elif len(input_shape) == 1:
                            # 1D Layer norm
                            size = int(input_shape[0])
                            kernel = f"""
__global__ void layer_norm_1d_kernel_{name}(float* input, float* output,
                                            int size, float eps) {{
    // Single block implementation for 1D case
    if (threadIdx.x == 0 && blockIdx.x == 0) {{
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {{
            sum += input[i];
        }}
        float mean = sum / size;
        
        // Compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < size; i++) {{
            float diff = input[i] - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / size;
        float std_dev = sqrtf(variance + eps);
        
        // Normalize
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
    {cuda_debug_code}
}}
"""
                            kernels.append(('layer_norm_1d', name, tensor_name, None, size, eps))
                            cuda_code += kernel


                    # CROSS ENTROPY LOSS
                    elif expr['type'] == 'cross_entropy':
                        pred_name, target_name = expr['args']
                        pred_shape = env[pred_name]['shape']
                        target_shape = env[target_name]['shape']
                        
                        if len(pred_shape) == 2 and len(target_shape) == 1:
                            # Class indices format: (batch, classes) vs (batch,)
                            batch_size, num_classes = int(pred_shape[0]), int(pred_shape[1])
                            
                            kernel = f"""
__global__ void cross_entropy_kernel_{name}(float* predictions, float* targets,
                                        float* output, int batch_size, int num_classes) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes loss for one sample
    float loss = 0.0f;
    if (batch_idx < batch_size) {{
        int target_class = (int)targets[batch_idx];
        
        // Clamp target class to valid range
        target_class = max(0, min(target_class, num_classes - 1));
        
        // Get predicted probability for true class
        float pred_prob = predictions[batch_idx * num_classes + target_class];
        
        // Clamp probability to prevent log(0)
        pred_prob = fmaxf(pred_prob, 1e-7f);
        pred_prob = fminf(pred_prob, 1.0f - 1e-7f);
        
        // Compute negative log likelihood
        loss = -logf(pred_prob);
    }}
    
    // Load into shared memory for reduction
    sdata[tid] = (batch_idx < batch_size) ? loss : 0.0f;
    __syncthreads();
    
    // Reduce to compute mean loss
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}
    
    // Write result
    if (tid == 0) {{
        atomicAdd(output, sdata[0]);
    }}
}}
extern "C" void launch_cross_entropy_{name}(float* predictions, float* targets,
                                        float* output, int batch_size, int num_classes) {{
    // Initialize output to zero
    cudaMemset(output, 0, sizeof(float));
    
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    cross_entropy_kernel_{name}<<<grid, block, shared_size>>>(predictions, targets, output,
                                                            batch_size, num_classes);
    cudaDeviceSynchronize();
    {cuda_debug_code}
    
    // Divide by batch size to get mean
    float mean_loss;
    cudaMemcpy(&mean_loss, output, sizeof(float), cudaMemcpyDeviceToHost);
    mean_loss /= batch_size;
    cudaMemcpy(output, &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
}}
"""
                            kernels.append(('cross_entropy', name, pred_name, target_name, batch_size, num_classes))
                            cuda_code += kernel

                    # MSE LOSS
                    elif expr['type'] == 'mse_loss':
                        pred_name, target_name = expr['args']
                        pred_shape = env[pred_name]['shape']
                        
                        total_elements = int(np.prod([int(dim) for dim in pred_shape]))
                        
                        kernel = f"""
__global__ void mse_loss_kernel_{name}(float* predictions, float* targets,
                                    float* output, int total_elements) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute squared error for this element
    float squared_error = 0.0f;
    if (idx < total_elements) {{
        float diff = predictions[idx] - targets[idx];
        squared_error = diff * diff;
    }}
    
    // Load into shared memory for reduction
    sdata[tid] = (idx < total_elements) ? squared_error : 0.0f;
    __syncthreads();
    
    // Reduce to compute sum of squared errors
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}
    
    // Write result for this block
    if (tid == 0) {{
        atomicAdd(output, sdata[0]);
    }}
}}
extern "C" void launch_mse_loss_{name}(float* predictions, float* targets,
                                    float* output, int total_elements) {{
    // Initialize output to zero
    cudaMemset(output, 0, sizeof(float));
    
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    int shared_size = block.x * sizeof(float);
    mse_loss_kernel_{name}<<<grid, block, shared_size>>>(predictions, targets, output, total_elements);
    cudaDeviceSynchronize();
    {cuda_debug_code}

    // Divide by total elements to get mean
    float mean_loss;
    cudaMemcpy(&mean_loss, output, sizeof(float), cudaMemcpyDeviceToHost);
    mean_loss /= total_elements;
    cudaMemcpy(output, &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
}}
"""
                        kernels.append(('mse_loss', name, pred_name, target_name, total_elements))
                        cuda_code += kernel

                    # ================================================================
                    # Transpose, Reshape, Concat
                    # ================================================================
                    elif expr['type'] == 'transpose':
                        tensor_name = expr['tensor']
                        axes = expr.get('axes')
                        input_shape = env[tensor_name]['shape']
                        output_shape = env[name]['shape']
                        
                        if len(input_shape) == 2:
                            # 2D transpose (matrix transpose)
                            rows, cols = int(input_shape[0]), int(input_shape[1])
                            
                            kernel = f"""
__global__ void transpose_2d_kernel_{name}(float* input, float* output, int rows, int cols) {{
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_row < cols && out_col < rows) {{
        // output[out_row, out_col] = input[out_col, out_row]
        output[out_row * rows + out_col] = input[out_col * cols + out_row];
    }}
}}
extern "C" void launch_transpose_{name}(float* input, float* output, int rows, int cols) {{
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
    transpose_2d_kernel_{name}<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                            kernels.append(('transpose_2d', name, tensor_name, None, rows, cols))
                            cuda_code += kernel

                    # ================================================================
                    # batch_norm
                    # ================================================================

                    # BATCH NORMALIZATION
                    elif expr['type'] == 'batch_norm':
                        tensor_name = expr['tensor']
                        running_mean_name = expr['running_mean']
                        running_var_name = expr['running_var']
                        eps = expr.get('eps', 1e-5)
                        input_shape = env[tensor_name]['shape']
                        
                        if len(input_shape) == 2:
                            # 2D batch norm: (N, C) format
                            batch_size, num_features = int(input_shape[0]), int(input_shape[1])
                            
                            kernel = f"""
__global__ void batch_norm_2d_kernel_{name}(float* input, float* running_mean, float* running_var,
                                            float* output, int batch_size, int num_features, float eps) {{
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (feature_idx < num_features && batch_idx < batch_size) {{
        // Get running statistics for this feature
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float std_dev = sqrtf(var + eps);
        
        // Normalize this element
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
    {cuda_debug_code}
}}
"""
                            #kernels.append(('batch_norm_2d', name, tensor_name, running_mean_name, running_var_name, batch_size, num_features, eps))
                            kernels.append(('batch_norm_2d', name, tensor_name, running_mean_name, batch_size, num_features, eps, running_var_name))
                            cuda_code += kernel

                    # INSTANCE NORMALIZATION
                    elif expr['type'] == 'instance_norm':
                        tensor_name = expr['tensor']
                        eps = expr.get('eps', 1e-5)
                        input_shape = env[tensor_name]['shape']
                        
                        if len(input_shape) == 2:
                            # 2D instance norm: normalize each sample independently
                            batch_size, num_features = int(input_shape[0]), int(input_shape[1])
                            
                            kernel = f"""
__global__ void instance_norm_2d_kernel_{name}(float* input, float* output,
                                            int batch_size, int num_features, float eps) {{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {{
        // Compute mean for this sample
        float sum = 0.0f;
        for (int f = 0; f < num_features; f++) {{
            sum += input[batch_idx * num_features + f];
        }}
        float mean = sum / num_features;
        
        // Compute variance for this sample
        float var_sum = 0.0f;
        for (int f = 0; f < num_features; f++) {{
            float diff = input[batch_idx * num_features + f] - mean;
            var_sum += diff * diff;
        }}
        float variance = var_sum / num_features;
        float std_dev = sqrtf(variance + eps);
        
        // Normalize this sample
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
    {cuda_debug_code}
}}
"""
                            kernels.append(('instance_norm_2d', name, tensor_name, None, None, batch_size, num_features, eps))
                            cuda_code += kernel

                    # ================================================================
                    # Generation: reshape
                    # ================================================================
                    elif expr['type'] == 'reshape':
                        tensor_name = expr['tensor']
                        input_shape = env[tensor_name]['shape']
                        output_shape = env[name]['shape']
                        
                        total_elements = int(np.prod([int(dim) for dim in input_shape]))
                        
                        kernel = f"""
__global__ void reshape_kernel_{name}(float* input, float* output, int total_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {{
        // Simple memory copy - reshape is just a view change
        output[idx] = input[idx];
    }}
}}
extern "C" void launch_reshape_{name}(float* input, float* output, int total_elements) {{
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    reshape_kernel_{name}<<<grid, block>>>(input, output, total_elements);
    cudaDeviceSynchronize();
    {cuda_debug_code}
}}
"""
                        kernels.append(('reshape', name, tensor_name, None, total_elements))
                        cuda_code += kernel

                    # ================================================================
                    # Generation: concat
                    # ================================================================
                    elif expr['type'] == 'concat':
                        tensor_names = expr['tensors']
                        axis = expr['axis']
                        
                        if len(tensor_names) == 2 and axis == 0:
                            # Simple 2-tensor concatenation along axis 0 (most common case)
                            tensor1, tensor2 = tensor_names
                            shape1 = env[tensor1]['shape']
                            shape2 = env[tensor2]['shape']
                            output_shape = env[name]['shape']
                            
                            if len(shape1) == 2:
                                rows1, cols = int(shape1[0]), int(shape1[1])
                                rows2 = int(shape2[0])
                                total_rows = int(output_shape[0])
                                
                                kernel = f"""
__global__ void concat_axis0_kernel_{name}(float* input1, float* input2, float* output,
                                        int rows1, int rows2, int cols) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {{
        if (row < rows1) {{
            // Copy from first tensor
            output[row * cols + col] = input1[row * cols + col];
        }} else if (row < rows1 + rows2) {{
            // Copy from second tensor
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
    {cuda_debug_code}
}}
"""
                                kernels.append(('concat_axis0', name, tensor1, tensor2, rows1, rows2, cols))
                                cuda_code += kernel


            # =========================================
            # KERNEL COMPILATION, EXECUTION AND CACHING
            # =========================================
            if kernels:

                # Ensure cache output directory exists
                os.makedirs("cache", exist_ok=True)
                os.makedirs(f"cache/{tensorlang_file}", exist_ok=True)

                #if DEBUG_MODE:
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Created cache for tensor outputs directory")
                    tensorlang.print(type="[INFO]", message=f"Generated CUDA kernels:")

                if DEBUG_MODE:
                    tensorlang.print(type=f"{cuda_code.strip()}")
                    #print(cuda_code.strip())

                # Compile CUDA code
                with open(f'cache/{tensorlang_file}/kernel.cu', 'w') as f:
                    f.write(cuda_code)
                try:
                    kernel_source_cuda = f'cache/{tensorlang_file}/kernel.cu'
                    kernel_shared_object = f'cache/{tensorlang_file}/kernel.so'

                    subprocess.run([
                            'nvcc', '-o', kernel_shared_object, 
                            '--shared', 
                            '-Xcompiler', 
                            '-fPIC', 
                            '-lcudart', 
                            kernel_source_cuda
                        ], 
                        check=True
                    )
                    if DEBUG_MODE:
                        tensorlang.print(message=f"CUDA compiled to kernel.so!")

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

                    # TODO need file existance checking
                    lib = cdll.LoadLibrary(f'cache/{tensorlang_file}/kernel.so')

                    if DEBUG_MODE:
                        tensorlang.print(message=f"CUDA Kernel loaded from cache/{tensorlang_file}/kernel.so")

                    # Allocate GPU memory and copy inputs
                    for name in env:
                        shape            = tuple(int(dim) for dim in env[name]['shape'])
                        size_bytes       = int(np.prod(shape) * np.float32().nbytes)
                        gpu_allocs[name] = cuda.mem_alloc(size_bytes)
                        if name in tensors:
                            cuda.memcpy_htod(gpu_allocs[name], tensors[name])
                            if DEBUG_MODE:
                                tensorlang.print(message=f"Copied {name} to GPU, shape: {tensors[name].shape}, sample: {tensors[name][:2] if tensors[name].ndim > 1 else tensors[name]}")
                        else:
                            if DEBUG_MODE:
                                tensorlang.print(message=f"Allocated GPU memory for {name} (uninitialized), shape: {shape}")


                    # Execute operations
                    # Everything below used by tests runner to capture output 
                    # Don't use [DEBUG] or [INFO] messes with result arrays
                    for kernel_info in kernels:
                        op_type = kernel_info[0]
                        
                        # Handle general_broadcast specially BEFORE standard unpacking
                        if op_type == 'general_broadcast':
                            _, actual_op, name, arg1, arg2, padded_shape1, padded_shape2, output_shape_tuple, total_elements = kernel_info
                            shape = tuple(int(dim) for dim in env[name]['shape'])
                            
                            if DEBUG_MODE:
                                tensorlang.print(type=f"TensorLang Executing {op_type} ({actual_op}) for {name}, shape: {shape}")
                            
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

                        elif op_type in ['add_broadcast', 'minus_broadcast', 'mult_broadcast', 'div_broadcast']:
                            op_type_parts = op_type.split("_")[0]
                            rows, cols = dims
                            getattr(lib, f'launch_{op_type_parts}_{name}')(
                                c_void_p(int(gpu_allocs[arg1])), 
                                c_void_p(int(gpu_allocs[arg2])), 
                                c_void_p(int(gpu_allocs[name])), 
                                c_int(rows), c_int(cols)
                            )

                        elif op_type in ['relu', 'sigmoid', 'tanh']:
                            op_type_parts = op_type.split("_")[0]
                            size = dims[0]
                            getattr(lib, f'launch_{op_type_parts}_{name}')(
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

                        elif op_type == 'softmax_1d':
                            size = dims[0]
                            getattr(lib, f'launch_softmax_{name}')(
                                c_void_p(int(gpu_allocs[arg1])), 
                                c_void_p(int(gpu_allocs[name])), 
                                c_int(size)
                            )

                        elif op_type == 'greater':
                            size = dims[0]
                            getattr(lib, f'launch_greater_{name}')(
                                c_void_p(int(gpu_allocs[arg1])),  # Should be 'data'
                                c_void_p(int(gpu_allocs[arg2])),  # Should be 'zeros' 
                                c_void_p(int(gpu_allocs[name])),  # Should be 'mask'
                                c_int(size)
                            )

                        elif op_type == 'greater_broadcast':
                            rows, cols = dims
                            getattr(lib, f'launch_greater_{name}')(
                                c_void_p(int(gpu_allocs[arg1])), 
                                c_void_p(int(gpu_allocs[arg2])), 
                                c_void_p(int(gpu_allocs[name])), 
                                c_int(rows), c_int(cols)
                            )

                        elif op_type == 'less':
                            size = dims[0]
                            getattr(lib, f'launch_less_{name}')(
                                c_void_p(int(gpu_allocs[arg1])), 
                                c_void_p(int(gpu_allocs[arg2])), 
                                c_void_p(int(gpu_allocs[name])), 
                                c_int(size)
                            )

                        elif op_type == 'less_broadcast':
                            rows, cols = dims
                            getattr(lib, f'launch_less_{name}')(
                                c_void_p(int(gpu_allocs[arg1])), 
                                c_void_p(int(gpu_allocs[arg2])), 
                                c_void_p(int(gpu_allocs[name])), 
                                c_int(rows), 
                                c_int(cols)
                            )

                        elif op_type == 'equal':
                            size = dims[0]
                            getattr(lib, f'launch_equal_{name}')(
                                c_void_p(int(gpu_allocs[arg1])), 
                                c_void_p(int(gpu_allocs[arg2])), 
                                c_void_p(int(gpu_allocs[name])), 
                                c_int(size)
                            )

                        elif op_type == 'equal_broadcast':
                            rows, cols = dims
                            getattr(lib, f'launch_equal_{name}')(
                                c_void_p(int(gpu_allocs[arg1])), 
                                c_void_p(int(gpu_allocs[arg2])), 
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

                        elif op_type == 'layer_norm_2d':
                            rows, cols, eps = dims
                            getattr(lib, f'launch_layer_norm_{name}')(
                                c_void_p(int(gpu_allocs[arg1])), c_void_p(int(gpu_allocs[name])),
                                c_int(rows), c_int(cols), c_float(eps)
                            )
                        elif op_type == 'layer_norm_axis0':
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
                            #print(f"DEBUG instance_norm dims: {dims}")
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
                            if DEBUG_MODE:
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

                        # ================================================================
                        # Save result for all computed tensors
                        output = np.zeros(shape, dtype=np.float32)
                        cuda.memcpy_dtoh(output, gpu_allocs[name])
                        tensors[name] = output

                        if CACHE_LAYERS:
                            if DEBUG_INFO:
                                tensorlang.print(message=f"Cache Layer {name} cache/{tensorlang_file}/{name}.npy")

                            # TensorLang Cache Layer for outputs npy
                            np.save(f"cache/{tensorlang_file}/{name}.npy", output)

                            if DEBUG_INFO:
                                # Result seperator
                                tensorlang.seperator()

                        # Specific output used by tests runner for extracting the result 
                        # of each tensor output stored in the cache file name .log
                        print(f"Result {name} ({op_type}):\n{output}")

                    # ================================================================
                    # After all kernels execute
                    # Handle alias assignments - copy tensor data for name references
                    # alias_found = False
                    for node in ast:
                        if node['type'] == 'let' and isinstance(node.get('expr'), dict) and node['expr']['type'] == 'name':
                            #print (f"AST: {node}\n")
                            alias_name = node['name']
                            source_name = node['expr']['name']
                            if source_name in tensors and alias_name not in tensors:
                                tensors[alias_name] = tensors[source_name].copy()
                                if DEBUG_MODE:
                                    tensorlang.print(type=f"Created alias: {alias_name} -> {source_name}")
                                
                                # Print result for the alias to match test format
                                shape = tensors[alias_name].shape
                                tensorlang.seperator()
                                #print(f"Result {alias_name} (alias):\n{tensors[alias_name]}")
                                print(f"Result {alias_name} ({op_type}):\n{tensors[alias_name]}")

                    # Free GPU memory
                    for name, alloc in gpu_allocs.items():
                        alloc.free()
                        if DEBUG_MODE:
                            tensorlang.print(type=f"Freed GPU memory for {name}")

                except ImportError as e:
                    print(f"PyCUDA error: {e}. Run 'pip install pycuda' and ensure CUDA toolkit is installed.")
                    sys.exit(1)
                except Exception as e:
                    print(f"Error executing CUDA kernel: {e}")
                    traceback.print_exc()
                    sys.exit(1)

    except KeyboardInterrupt as e:
        print(f"Keyboard Interrupt")
        sys.exit(1)

    except UnexpectedInput as e:
        print(f"Parse error: {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during parsing or execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    tensorlang.seperator()

if __name__ == "__main__":
    main()