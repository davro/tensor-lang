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
version = "0.2.2"

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

        elif tree.data == 'greater_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Greater args: {args}")
            return {'type': 'greater', 'args': args}

        elif tree.data == 'less_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Less args: {args}")
            return {'type': 'less', 'args': args}

        elif tree.data == 'equal_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Equal args: {args}")
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

        elif tree.data == 'max_call':
            tensor_name = None
            axis = None
            
            for child in tree.children:
                if isinstance(child, Token) and child.type == 'NAME':
                    tensor_name = child.value
                elif isinstance(child, Token) and child.type == 'NUMBER':
                    axis = int(float(child.value))
            
            print(f"Max args: tensor={tensor_name}, axis={axis}")
            return {'type': 'max', 'tensor': tensor_name, 'axis': axis}

        elif tree.data == 'min_call':
            tensor_name = None
            axis = None
            
            for child in tree.children:
                if isinstance(child, Token) and child.type == 'NAME':
                    tensor_name = child.value
                elif isinstance(child, Token) and child.type == 'NUMBER':
                    axis = int(float(child.value))
            
            print(f"Min args: tensor={tensor_name}, axis={axis}")
            return {'type': 'min', 'tensor': tensor_name, 'axis': axis}

        elif tree.data == 'argmax_call':
            tensor_name = None
            axis = None
            
            for child in tree.children:
                if isinstance(child, Token) and child.type == 'NAME':
                    tensor_name = child.value
                elif isinstance(child, Token) and child.type == 'NUMBER':
                    axis = int(float(child.value))
            
            print(f"Argmax args: tensor={tensor_name}, axis={axis}")
            return {'type': 'argmax', 'tensor': tensor_name, 'axis': axis}

        elif tree.data == 'argmin_call':
            tensor_name = None
            axis = None
            
            for child in tree.children:
                if isinstance(child, Token) and child.type == 'NAME':
                    tensor_name = child.value
                elif isinstance(child, Token) and child.type == 'NUMBER':
                    axis = int(float(child.value))
            
            print(f"Argmin args: tensor={tensor_name}, axis={axis}")
            return {'type': 'argmin', 'tensor': tensor_name, 'axis': axis}

        elif tree.data == 'linear_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Linear args: {args}")
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
            print(f"LayerNorm args: tensor={tensor_name}, axis={axis}, eps={eps}")
            return {'type': 'layer_norm', 'tensor': tensor_name, 'axis': axis, 'eps': eps}

        # ================================================================
        # Cross-entropy loss and mean squared error (MSE) loss 
        # ================================================================
        elif tree.data == 'cross_entropy_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"Cross entropy args: {args}")
            return {'type': 'cross_entropy', 'args': args}

        elif tree.data == 'mse_loss_call':
            args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
            print(f"MSE loss args: {args}")
            return {'type': 'mse_loss', 'args': args}


        # ================================================================
        # Slice feature
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
            
            print(f"Slice args: tensor={tensor_name}, specs={slice_specs}")
            return {'type': 'slice', 'tensor': tensor_name, 'specs': slice_specs}


        # ================================================================
        # Transpose, Reshape, Concat
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
            
            print(f"Transpose args: tensor={tensor_name}, axes={axes}")
            return {'type': 'transpose', 'tensor': tensor_name, 'axes': axes}

        elif tree.data == 'reshape_call':
            tensor_name = None
            new_shape = None
            
            for child in tree.children:
                if isinstance(child, Token) and child.type == 'NAME':
                    tensor_name = child.value
                elif isinstance(child, Tree) and child.data == 'shape':
                    new_shape = build_shape(child)
            
            print(f"Reshape args: tensor={tensor_name}, new_shape={new_shape}")
            return {'type': 'reshape', 'tensor': tensor_name, 'new_shape': new_shape}

        elif tree.data == 'concat_call':
            tensor_names = []
            axis = None
            
            for child in tree.children:
                if isinstance(child, Token) and child.type == 'NAME':
                    tensor_names.append(child.value)
                elif isinstance(child, Token) and child.type == 'NUMBER':
                    axis = int(float(child.value))
            
            print(f"Concat args: tensors={tensor_names}, axis={axis}")
            return {'type': 'concat', 'tensors': tensor_names, 'axis': axis}




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

                #elif isinstance(expr, dict) and expr['type'] in ['matmul', 'add', 'minus', 'mult', 'div', 'relu', 'sigmoid', 'tanh', 'softmax', 'fill', 'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'greater', 'less', 'equal', 'linear', 'layer_norm', 'slice']:
                elif isinstance(expr, dict) and expr['type'] in [
                        'matmul', 'add', 'minus', 'mult', 'div', 
                        'relu', 'sigmoid', 'tanh', 'softmax', 
                        'fill', 'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 
                        'greater', 'less', 'equal', 
                        'linear', 'layer_norm', 'cross_entropy', 'mse_loss', 
                        'transpose', 'reshape', 'concat', 'slice',
                        'slice'
                    ]:

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
                            print(f"Type error: Undefined tensor {tensor_name} for {expr['type']}")
                            return False, env
                        
                        # Softmax preserves input shape
                        env[name] = {'dtype': 'f32', 'shape': env[tensor_name]['shape']}
                        print(f"Assigned type for {name} (softmax): {env[name]}")


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
                        print(f"Assigned type for {name} (slice): {env[name]}")



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
                        print(f"Assigned type for {name} (layer_norm): {env[name]}")
                        print(f"DEBUG: layer_norm axis={axis}, input_shape={input_shape}")



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
                        print(f"Assigned type for {name} ({expr['type']}): {env[name]}")


                    # ================================================================
                    # Transpose, Reshape, Concat
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
                        print(f"Assigned type for {name} (transpose): {env[name]}")

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
                        print(f"Assigned type for {name} (reshape): {env[name]}")

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
                        print(f"Assigned type for {name} (concat): {env[name]}")


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
                                    print(f"Assigned type for {name} ({expr['type']}): {env[name]}")


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
                                    print(f"Assigned type for {name} (linear): {env[name]}")


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
                                    print(f"Assigned type for {name} ({expr['type']}): {env[name]}")


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
}}
"""
                        kernels.append(('greater', name, arg1, arg2, size))
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
}}
"""
                        kernels.append(('less', name, arg1, arg2, size))
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
}}
"""
                        kernels.append(('equal', name, arg1, arg2, size))
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
}}
"""
                        kernels.append(('slice_2d', name, tensor_name, None, rows, cols, row_start, row_end, col_start, col_end, out_rows, out_cols))
                    
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
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
}}
"""
                        kernels.append(('max_full', name, tensor_name, None, size))
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
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
}}
"""
                        kernels.append(('min_full', name, tensor_name, None, size))
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
}}
"""
                            kernels.append(('min_axis', name, tensor_name, None, rows, cols, axis))
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
}}
"""
                        kernels.append(('argmax_full', name, tensor_name, None, size))
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
}}
"""
                        kernels.append(('argmin_axis', name, tensor_name, None, rows, cols, axis))
                    
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
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
}}
"""
                        kernels.append(('linear_1d', name, input_name, weight_name, bias_name, in_features, out_features))
                        
                    elif len(input_shape) == 2:
                        # 2D case: batch_matrix @ matrix + bias (broadcasted)
                        batch_size, in_features = int(input_shape[0]), int(input_shape[1])
                        out_features = int(output_shape[1])
                        
                        kernel = f"""
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
                    
                    print(f"DEBUG CUDA: layer_norm axis={axis}, input_shape={input_shape}")

                    if len(input_shape) == 2:
                        # 2D Layer norm - most common case
                        if axis == 1:
                            print("DEBUG: Using row-wise normalization kernel")
                            # Normalize along features (each row independently)
                            rows, cols = int(input_shape[0]), int(input_shape[1])
                            kernel = f"""
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
}}
"""
                            kernels.append(('layer_norm_2d', name, tensor_name, None, rows, cols, eps))
                        elif axis == 0:
                            print("DEBUG: Using column-wise normalization kernel")
                            # Normalize along batch dimension (each column independently)
                            rows, cols = int(input_shape[0]), int(input_shape[1])
                            kernel = f"""
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
}}
"""
                            kernels.append(('layer_norm_axis0', name, tensor_name, None, rows, cols, eps))
                    
                    elif len(input_shape) == 1:
                        # 1D Layer norm
                        size = int(input_shape[0])
                        kernel = f"""
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
}}
"""
                        kernels.append(('transpose_2d', name, tensor_name, None, rows, cols))
                    
                    cuda_code += kernel

                # RESHAPE
                elif expr['type'] == 'reshape':
                    tensor_name = expr['tensor']
                    input_shape = env[tensor_name]['shape']
                    output_shape = env[name]['shape']
                    
                    total_elements = int(np.prod([int(dim) for dim in input_shape]))
                    
                    kernel = f"""
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
}}
"""
                    kernels.append(('reshape', name, tensor_name, None, total_elements))
                    
                    cuda_code += kernel

                # CONCAT
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
// Define FLT_MAX for CUDA device code
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif

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
                        getattr(lib, f'launch_matmul_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[arg2])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(m), c_int(n), c_int(p)
                        )

                    elif op_type == 'add':
                        size = dims[0]
                        getattr(lib, f'launch_add_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[arg2])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(size)
                        )

                    elif op_type == 'add_broadcast':
                        rows, cols = dims
                        getattr(lib, f'launch_add_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[arg2])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(rows), c_int(cols)
                        )

                    elif op_type == 'minus':
                        size = dims[0]
                        getattr(lib, f'launch_minus_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[arg2])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(size)
                        )

                    elif op_type == 'minus_broadcast':
                        rows, cols = dims
                        getattr(lib, f'launch_minus_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[arg2])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(rows), c_int(cols)
                        )

                    elif op_type == 'mult':
                        size = dims[0]
                        getattr(lib, f'launch_mult_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[arg2])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(size)
                        )

                    elif op_type == 'mult_broadcast':
                        rows, cols = dims
                        getattr(lib, f'launch_mult_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[arg2])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(rows), c_int(cols)
                        )

                    elif op_type == 'div':
                        size = dims[0]
                        getattr(lib, f'launch_div_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[arg2])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(size)
                        )

                    elif op_type == 'div_broadcast':
                        rows, cols = dims
                        getattr(lib, f'launch_div_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[arg2])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(rows), c_int(cols)
                        )

                    elif op_type == 'relu':
                        size = dims[0]
                        getattr(lib, f'launch_relu_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(size)
                        )

                    elif op_type == 'sigmoid':
                        size = dims[0]
                        getattr(lib, f'launch_sigmoid_{name}')(
                            c_void_p(int(gpu_allocs[arg1])), 
                            c_void_p(int(gpu_allocs[name])), 
                            c_int(size)
                        )

                    elif op_type == 'tanh':
                        size = dims[0]
                        getattr(lib, f'launch_tanh_{name}')(
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


                    # Save result for all computed tensors
                    output = np.zeros(shape, dtype=np.float32)
                    cuda.memcpy_dtoh(output, gpu_allocs[name])
                    tensors[name] = output

                    # Specific output used by tests runner for extracting the result of each tensor output logs
                    print(f"Result {name} ({op_type}):\n{output}")

                    # TensorLang Cache for outputs npy
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