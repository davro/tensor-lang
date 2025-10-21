# File: tensorlang/type_checker.py
# NEW FILE - Complete extraction from tensorlang.py main()

import numpy as np

def type_checker(ast, env, DEBUG_INFO=False, DEBUG_MODE=False):
    """
    Type check AST and build environment of tensor types.
    
    Args:
        ast: List of AST nodes from build_ast()
        env: Empty dict to populate with tensor types
        DEBUG_INFO: Print info messages
        DEBUG_MODE: Print debug messages
    
    Returns:
        (success: bool, env: dict with tensor types)
    """
    for node in ast:
        if node['type'] == 'let':
            name = node['name']
            expr = node['expr']
            ty = node['ty']

            if ty and isinstance(ty, dict) and 'dtype' in ty:
                shape = ty['shape']
                if any(isinstance(dim, str) for dim in shape):
                    if DEBUG_INFO:
                        print(f"[INFO] Generic type for {name}: {ty}")
                    continue
                else:
                    env[name] = {'dtype': ty['dtype'], 'shape': ty['shape']}
                    if DEBUG_INFO:
                        print(f"[INFO] Tensor {name} {env[name]}")

            elif isinstance(expr, dict) and expr['type'] == 'tensor_literal':
                data = expr['data']
                num_elements = len(data)
                if expr['is_1d']:
                    shape = (num_elements,)
                else:
                    rows = sum(1 for child in expr['tree'].children if child.data == 'inner_array')
                    cols = num_elements // rows if rows > 0 else num_elements
                    shape = (rows, cols)
                env[name] = {'dtype': 'f32', 'shape': shape}
                
                if DEBUG_MODE:
                    print(f"Inferred shape for {name}: {env[name]['shape']}")

            elif isinstance(expr, dict) and expr['type'] == 'name':
                ref_name = expr['name']
                if ref_name not in env:
                    print(f"Type error: Undefined tensor {ref_name} referenced by {name}")
                    return False, env
                
                env[name] = {'dtype': env[ref_name]['dtype'], 'shape': env[ref_name]['shape']}
                if DEBUG_INFO:
                    print(f"[INFO] Type {name} assigned from {ref_name}: {env[name]}")

            elif isinstance(expr, dict) and expr['type'] in [
                    'matmul', 'add', 'minus', 'mult', 'div', 
                    'relu', 'sigmoid', 'tanh', 'gelu', 'swish', 'softmax', 
                    'fill', 'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 
                    'greater', 'less', 'equal', 
                    'linear', 'layer_norm', 'batch_norm', 'instance_norm', 'cross_entropy', 'mse_loss', 
                    'transpose', 'reshape', 'concat', 'slice'
                ]:

                if expr['type'] == 'fill':
                    env[name] = {'dtype': 'f32', 'shape': expr['shape']}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned from fill: {env[name]}")
                
                elif expr['type'] in ['sum', 'mean']:
                    tensor_name = expr['tensor']
                    if tensor_name not in env:
                        print(f"Type error: Undefined tensor {tensor_name} for {expr['type']}")
                        return False, env
                    
                    input_shape = env[tensor_name]['shape']
                    axis = expr.get('axis')
                    if axis is None:
                        output_shape = ()
                    else:
                        if axis < 0 or axis >= len(input_shape):
                            print(f"Type error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                            return False, env
                        output_shape = tuple(dim for i, dim in enumerate(input_shape) if i != axis)
                        if not output_shape:
                            output_shape = ()
                    
                    env[name] = {'dtype': 'f32', 'shape': output_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                elif expr['type'] == 'softmax':
                    tensor_name = expr['tensor']
                    if tensor_name not in env:
                        print(f"Type error: Undefined tensor {tensor_name} for {expr['type']}")
                        return False, env
                    
                    env[name] = {'dtype': 'f32', 'shape': env[tensor_name]['shape']}
                    if DEBUG_INFO:
                        print(f"[INFO] Tensor {tensor_name} {name}:({expr['type']}):{env[name]}")

                elif expr['type'] in ['relu', 'sigmoid', 'tanh', 'gelu', 'swish']:
                    arg_name = expr['args'][0]
                    if arg_name not in env:
                        print(f"Type error: Undefined tensor {arg_name} for {expr['type']}")
                        return False, env
                    
                    env[name] = {'dtype': 'f32', 'shape': env[arg_name]['shape']}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} for {name}: {env[name]}")

                elif expr['type'] == 'slice':
                    tensor_name = expr['tensor']
                    if tensor_name not in env:
                        print(f"Type error: Undefined tensor {tensor_name} for {expr['type']}")
                        return False, env
                    
                    input_shape = env[tensor_name]['shape']
                    slice_specs = expr['specs']
                    
                    output_shape = []
                    for i, spec in enumerate(slice_specs):
                        if i >= len(input_shape):
                            print(f"Type error: Slice dimension {i} exceeds tensor dimensions {len(input_shape)}")
                            return False, env
                            
                        dim_size = input_shape[i]
                        
                        if spec['type'] == 'index':
                            continue
                        elif spec['type'] == 'full_slice':
                            output_shape.append(dim_size)
                        elif spec['type'] == 'slice':
                            start = spec.get('start', 0)
                            end = spec.get('end', dim_size)
                            
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
                    
                    for i in range(len(slice_specs), len(input_shape)):
                        output_shape.append(input_shape[i])
                    
                    if not output_shape:
                        output_shape = ()
                    else:
                        output_shape = tuple(output_shape)
                    
                    env[name] = {'dtype': 'f32', 'shape': output_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                elif expr['type'] == 'layer_norm':
                    tensor_name = expr['tensor']
                    if tensor_name not in env:
                        print(f"Type error: Undefined tensor {tensor_name} for layer_norm")
                        return False, env
                    
                    input_shape = env[tensor_name]['shape']
                    axis = expr.get('axis')
                    
                    if axis is None:
                        axis = len(input_shape) - 1
                    
                    if axis < 0 or axis >= len(input_shape):
                        print(f"Type error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                        return False, env
                    
                    env[name] = {'dtype': 'f32', 'shape': input_shape}

                    if DEBUG_MODE:
                        print(f"Type {expr['type']} Assigned for {name}: {env[name]}")

                elif expr['type'] == 'batch_norm':
                    tensor_name = expr['tensor']
                    running_mean_name = expr['running_mean']
                    running_var_name = expr['running_var']
                    
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
                    mean_shape = env[running_mean_name]['shape']
                    var_shape = env[running_var_name]['shape']
                    
                    if len(input_shape) < 2:
                        print(f"Type error: BatchNorm input must be at least 2D")
                        return False, env
                    
                    num_features = input_shape[1]
                    expected_stats_shape = (num_features,)
                    
                    if mean_shape != expected_stats_shape:
                        print(f"Type error: BatchNorm running_mean shape {mean_shape} != expected {expected_stats_shape}")
                        return False, env
                    if var_shape != expected_stats_shape:
                        print(f"Type error: BatchNorm running_var shape {var_shape} != expected {expected_stats_shape}")
                        return False, env
                    
                    env[name] = {'dtype': 'f32', 'shape': input_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                elif expr['type'] == 'instance_norm':
                    tensor_name = expr['tensor']
                    if tensor_name not in env:
                        print(f"Type error: Undefined tensor {tensor_name} for instance_norm")
                        return False, env
                    
                    input_shape = env[tensor_name]['shape']
                    env[name] = {'dtype': 'f32', 'shape': input_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                elif expr['type'] in ['max', 'min', 'argmax', 'argmin']:
                    tensor_name = expr['tensor']
                    if tensor_name not in env:
                        print(f"Type error: Undefined tensor {tensor_name} for {expr['type']}")
                        return False, env
                    
                    input_shape = env[tensor_name]['shape']
                    axis = expr.get('axis')
                    
                    if axis is None:
                        output_shape = ()
                    else:
                        if axis < 0 or axis >= len(input_shape):
                            print(f"Type error: Axis {axis} out of bounds for tensor {tensor_name} with shape {input_shape}")
                            return False, env
                        output_shape = tuple(dim for i, dim in enumerate(input_shape) if i != axis)
                        if not output_shape:
                            output_shape = ()
                    
                    env[name] = {'dtype': 'f32', 'shape': output_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                elif expr['type'] == 'transpose':
                    tensor_name = expr['tensor']
                    if tensor_name not in env:
                        print(f"Type error: Undefined tensor {tensor_name} for transpose")
                        return False, env
                    
                    input_shape = env[tensor_name]['shape']
                    axes = expr.get('axes')
                    
                    if axes is None:
                        output_shape = tuple(reversed(input_shape))
                    else:
                        if len(axes) != len(input_shape):
                            print(f"Type error: Transpose axes length mismatch")
                            return False, env
                        if set(axes) != set(range(len(input_shape))):
                            print(f"Type error: Transpose axes must be permutation")
                            return False, env
                        output_shape = tuple(input_shape[i] for i in axes)
                    
                    env[name] = {'dtype': 'f32', 'shape': output_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                elif expr['type'] == 'reshape':
                    tensor_name = expr['tensor']
                    if tensor_name not in env:
                        print(f"Type error: Undefined tensor {tensor_name} for reshape")
                        return False, env
                    
                    input_shape = env[tensor_name]['shape']
                    new_shape = expr['new_shape']
                    
                    input_elements = int(np.prod([int(dim) for dim in input_shape]))
                    new_elements = int(np.prod([int(dim) for dim in new_shape]))
                    
                    if input_elements != new_elements:
                        print(f"Type error: Reshape element count mismatch")
                        return False, env
                    
                    env[name] = {'dtype': 'f32', 'shape': new_shape}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                elif expr['type'] == 'concat':
                    tensor_names = expr['tensors']
                    axis = expr['axis']
                    
                    tensor_types = []
                    for tensor_name in tensor_names:
                        if tensor_name not in env:
                            print(f"Type error: Undefined tensor {tensor_name} for concat")
                            return False, env
                        tensor_types.append(env[tensor_name])
                    
                    first_shape = tensor_types[0]['shape']
                    if axis < 0 or axis >= len(first_shape):
                        print(f"Type error: Concat axis out of bounds")
                        return False, env
                    
                    concat_dim_size = 0
                    for i, tensor_type in enumerate(tensor_types):
                        shape = tensor_type['shape']
                        if len(shape) != len(first_shape):
                            print(f"Type error: Concat tensor dimension count mismatch")
                            return False, env
                        
                        for dim_idx, (dim1, dim2) in enumerate(zip(first_shape, shape)):
                            if dim_idx != axis and dim1 != dim2:
                                print(f"Type error: Concat tensor shape mismatch")
                                return False, env
                        
                        concat_dim_size += shape[axis]
                    
                    output_shape = list(first_shape)
                    output_shape[axis] = concat_dim_size
                    env[name] = {'dtype': 'f32', 'shape': tuple(output_shape)}
                    if DEBUG_INFO:
                        print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                else:
                    arg_names = expr.get('args', [])
                    if arg_names:
                        args = [env.get(arg_name) for arg_name in arg_names]
                        if DEBUG_INFO:
                            print(f"[INFO] Checking args for {name}: {args}")
                        if not all(args):
                            print(f"Type error: Undefined args for {name}")
                            return False, env
                            
                        if expr['type'] == 'matmul':
                            if args[0]['shape'][1] != args[1]['shape'][0]:
                                print(f"Type error: Matmul shape mismatch")
                                return False, env
                            env[name] = {'dtype': 'f32', 'shape': (args[0]['shape'][0], args[1]['shape'][1])}
                            if DEBUG_INFO:
                                print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                        elif expr['type'] in ['add', 'minus', 'mult', 'div']:
                            shape1, shape2 = args[0]['shape'], args[1]['shape']
                            
                            if shape1 == shape2:
                                output_shape = shape1
                            elif len(shape1) == 2 and len(shape2) == 1:
                                if shape2[0] == 1:
                                    output_shape = shape1
                                elif shape1[0] == shape2[0] or shape1[1] == shape2[0]:
                                    output_shape = shape1
                                else:
                                    print(f"Type error: {expr['type']} shape mismatch")
                                    return False, env
                            else:
                                ndim = max(len(shape1), len(shape2))
                                padded_shape1 = (1,) * (ndim - len(shape1)) + shape1
                                padded_shape2 = (1,) * (ndim - len(shape2)) + shape2
                                
                                output_shape = []
                                for d1, d2 in zip(padded_shape1, padded_shape2):
                                    if d1 == d2:
                                        output_shape.append(d1)
                                    elif d1 == 1:
                                        output_shape.append(d2)
                                    elif d2 == 1:
                                        output_shape.append(d1)
                                    else:
                                        print(f"Type error: {expr['type']} incompatible broadcast shapes")
                                        return False, env
                                
                                output_shape = tuple(output_shape)
                            
                            env[name] = {'dtype': 'f32', 'shape': output_shape}
                            if DEBUG_INFO:
                                print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                        elif expr['type'] in ['greater', 'less', 'equal']:
                            shape1, shape2 = args[0]['shape'], args[1]['shape']
                            if len(shape1) < len(shape2):
                                shape1, shape2 = shape2, shape1
                            if len(shape2) == 0:
                                output_shape = shape1
                            else:
                                output_shape = []
                                for d1, d2 in zip(shape1[-len(shape2):], shape2):
                                    if d1 == d2 or d2 == 1:
                                        output_shape.append(d1)
                                    else:
                                        print(f"Type error: {expr['type']} shape mismatch")
                                        return False, env
                                output_shape = shape1[:-len(shape2)] + tuple(output_shape)
                            env[name] = {'dtype': 'f32', 'shape': output_shape}
                            if DEBUG_INFO:
                                print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                        elif expr['type'] == 'linear':
                            if len(args) != 3:
                                print(f"Type error: Linear requires exactly 3 arguments")
                                return False, env
                            
                            input_tensor, weight_tensor, bias_tensor = args
                            input_shape = input_tensor['shape']
                            weight_shape = weight_tensor['shape']
                            bias_shape = bias_tensor['shape']
                            
                            if len(input_shape) == 1:
                                in_features = input_shape[0]
                                if len(weight_shape) != 2 or weight_shape[0] != in_features:
                                    print(f"Type error: Linear weight shape mismatch")
                                    return False, env
                                out_features = weight_shape[1]
                                if len(bias_shape) != 1 or bias_shape[0] != out_features:
                                    print(f"Type error: Linear bias shape mismatch")
                                    return False, env
                                output_shape = (out_features,)
                                
                            elif len(input_shape) == 2:
                                batch_size, in_features = input_shape
                                if len(weight_shape) != 2 or weight_shape[0] != in_features:
                                    print(f"Type error: Linear weight shape mismatch")
                                    return False, env
                                out_features = weight_shape[1]
                                if len(bias_shape) != 1 or bias_shape[0] != out_features:
                                    print(f"Type error: Linear bias shape mismatch")
                                    return False, env
                                output_shape = (batch_size, out_features)
                            else:
                                print(f"Type error: Linear input must be 1D or 2D tensor")
                                return False, env
                            
                            env[name] = {'dtype': 'f32', 'shape': output_shape}
                            if DEBUG_INFO:
                                print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

                        elif expr['type'] in ['cross_entropy', 'mse_loss']:
                            if len(args) != 2:
                                print(f"Type error: {expr['type']} requires exactly 2 arguments")
                                return False, env
                            
                            predictions_tensor, targets_tensor = args
                            pred_shape = predictions_tensor['shape']
                            target_shape = targets_tensor['shape']
                            
                            if expr['type'] == 'cross_entropy':
                                if len(pred_shape) == 2:
                                    batch_size, num_classes = pred_shape
                                    
                                    if len(target_shape) == 1 and target_shape[0] == batch_size:
                                        output_shape = ()
                                    elif len(target_shape) == 2 and target_shape == pred_shape:
                                        output_shape = ()
                                    else:
                                        print(f"Type error: Cross entropy target shape incompatible")
                                        return False, env
                                else:
                                    print(f"Type error: Cross entropy predictions must be 2D")
                                    return False, env
                                    
                            elif expr['type'] == 'mse_loss':
                                if pred_shape != target_shape:
                                    print(f"Type error: MSE loss shape mismatch")
                                    return False, env
                                output_shape = ()
                            
                            env[name] = {'dtype': 'f32', 'shape': output_shape}
                            if DEBUG_INFO:
                                print(f"[INFO] Type {expr['type']} assigned for {name}: {env[name]}")

            else:
                print(f"Type error: Unrecognized expr type for {name}: {expr['type']}")
                return False, env

    print(f"Final environment keys: {list(env.keys())}")
    return True, env