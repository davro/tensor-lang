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
from tensorlang.kernel_generator import KernelGenerator

# ================================================================
#                      TensorLang version
# ================================================================
version = "0.2.6"

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

            # ================================================================
            # Expression for types with args
            # ================================================================
            if tree.data in [
                    'matmul_call', 
                    'equal_call', 'greater_call', 
                    #'less_call'
                    'add_call', 'minus_call', 'mult_call', 'div_call', 
                    'relu_call', 'sigmoid_call', 'tanh_call',
                    'linear_call'
                ]:

                #expr_name = tree.data.split('_')[0]
                expr_name = tree.data.replace("_call", "")

                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                return {'type': expr_name, 'args': args}

            # Breaking edge case less?
            elif tree.data == 'less_call':
                args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
                return {'type': 'less', 'args': args}

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
            # Expression for types with axis
            # ================================================================
            elif tree.data in [
                    'softmax_call', 
                    'sum_call', 'mean_call', 'max_call', 'min_call',
                    'argmax_call', 'argmin_call'
                ]:
                tensor_name = None
                axis = None
                expr_name = tree.data.replace("_call", "")
                for child in tree.children:
                    if isinstance(child, Token) and child.type == 'NAME':
                        tensor_name = child.value
                    elif isinstance(child, Token) and child.type == 'NUMBER':
                        axis = int(float(child.value))
                # Default to last axis if not specified
                if DEBUG_INFO:
                    tensorlang.print(type="[INFO]", message=f"Expression {tree.data} tensor={tensor_name}, axis={axis}")
                return {'type': expr_name, 'tensor': tensor_name, 'axis': axis}

            # ================================================================
            # Expression for types with axis, return multiple tensors
            # ================================================================
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

            # ================================================================
            # Expression: layer_norm_call
            # ================================================================
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

            # ================================================================
            # Expression: instance_norm_call
            # ================================================================
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


        # ================================================================
        # Build slice specification
        # ================================================================
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


        # ================================================================
        # Build user function call
        # ================================================================
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


        # ================================================================
        # Inline function call
        # ================================================================
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


        # ================================================================
        # Substitute names
        # ================================================================
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


        # ================================================================
        # Build Abstract Syntax Tree
        # ================================================================
        def build_ast(tree):
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

                                        # In build_ast, after inlining
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
        # Type Checker Build functions for Feature "function definitions and calls."
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


        # ================================================================
        # Prod
        # ================================================================
        def prod(lst):
            return reduce(lambda x, y: x * y, lst, 1)


        # ================================================================
        # Can Broadcast
        # ================================================================
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
        ast, output_tensor, functions = build_ast(parse_tree)
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
            generator = KernelGenerator()
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
                        arg1, arg2     = expr['args']
                        shape1, shape2 = env[arg1]['shape'], env[arg2]['shape']
                        output_shape   = env[name]['shape']

                        tensorlang.print(message=f"Tensor {expr['type']}")
                        tensorlang.print(message=f"Tensor Shape1: {len(shape1)} Shape2: {len(shape2)}")

                        # Case 1: Identical shapes (element-wise)
                        if shape1 == shape2:
                            size = int(np.prod([int(dim) for dim in output_shape]))

                            kernel, kernel_info = generator.elementwise(
                                expr['type'], name, arg1, arg2, size, cuda_debug_code
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                        
                        # Case 2: 2D + 1D broadcasting
                        elif len(shape1) == 2 and len(shape2) == 1:
                            kernel, kernel_info = generator.binary_broadcast(
                                expr['type'], expr['type'], name, arg1, arg2, 
                                shape1, shape2, output_shape, cuda_debug_code
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel
                        
                        # Case 2.5: 1D tensors where one has size 1
                        elif len(shape1) == 1 and len(shape2) == 1 and (shape1[0] == 1 or shape2[0] == 1):
                            size = max(int(shape1[0]), int(shape2[0])) # potential issue
                            size = int(np.prod([int(dim) for dim in output_shape]))
                            kernel, kernel_info = generator.binary_1d_broadcast(
                                expr['type'], name, arg1, arg2, size, cuda_debug_code
                            )
                            kernels.append(kernel_info)
                            cuda_code += kernel

                        # Case 3: General broadcasting (NumPy-style)
                        elif can_broadcast(shape1, shape2):
                            kernel, kernel_info = generator.binary_general_broadcast(
                                expr['type'], name, arg1, arg2, 
                                shape1, shape2, output_shape, cuda_debug_code
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
                            expr['type'], name, None, None, size, expr['value'], cuda_debug_code
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
                            expr['type'], name, arg1, arg2, m, n, p, cuda_debug_code
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
                            expr['type'], name, arg1, size, cuda_debug_code
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
                            expr['type'], name, tensor_name, input_shape, axis, cuda_debug_code
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
                            expr['type'], name, arg1, arg2, shape1, shape2, output_shape, size, cuda_debug_code
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
                            expr['type'], name, arg1, arg2, shape1, shape2, output_shape, cuda_debug_code
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
                            expr['type'], name, arg1, arg2, shape1, shape2, output_shape, cuda_debug_code
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
                            expr['type'], name, tensor_name, axis, input_shape, cuda_debug_code
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
                            expr['type'], name, tensor_name, axis, input_shape, cuda_debug_code
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
                            expr['type'], name, tensor_name, slice_specs, input_shape, output_shape, cuda_debug_code
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
                            expr['type'], name, tensor_name, axis, input_shape, cuda_debug_code
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
                            expr['type'], name, tensor_name, axis, input_shape, cuda_debug_code
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
                            expr['type'], name, tensor_name, axis, input_shape, cuda_debug_code
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
                            expr['type'], name, tensor_name, axis, input_shape, cuda_debug_code
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
                            expr['type'], name, input_name, weight_name, bias_name, input_shape, weight_shape, output_shape, cuda_debug_code
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
                            expr['type'], name, tensor_name, axis, eps, input_shape, cuda_debug_code
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
                            expr['type'], name, pred_name, pred_shape, target_shape, target_name, cuda_debug_code
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
                            expr['type'], name, pred_name, target_name, total_elements, cuda_debug_code
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
                            expr['type'], name, input_shape, tensor_name, cuda_debug_code
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
                            expr['type'], name, input_shape, tensor_name, running_mean_name, eps, running_var_name, cuda_debug_code
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
                            expr['type'], name, tensor_name, input_shape, eps, cuda_debug_code
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
                            expr['type'], name, tensor_name, total_elements, cuda_debug_code
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
                            expr['type'], name, tensor_names, axis, env, cuda_debug_code
                        )
                        kernels.append(kernel_info)
                        cuda_code += kernel

            # ================================================================
            # KERNEL COMPILATION, EXECUTION AND CACHING
            # ================================================================
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
                        # ================================================================
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

                                # PrintSave result for the alias
                                np.save(f"cache/{tensorlang_file}/{alias_name}.npy", tensors[alias_name])

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
        print(f"TensorLang: Keyboard Interrupt")
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

    tensorlang.seperator()

if __name__ == "__main__":
    main()