# File: tensorlang/ast_builder.py
"""AST building - extracted from main()"""

from lark import Tree, Token
from typing import Dict, List, Optional, Tuple

def build_ast(tree: Tree, DEBUG_MODE=False, DEBUG_INFO=False) -> Tuple[List, Optional[str], Dict]:
    """Build AST from parse tree, handling function definitions and inlining"""
    functions = {}
    ast = []
    output_tensor = None
    call_counter = 0
    
    if tree.data == 'program':
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'statement':
                stmt = child.children[0]
                
                # Function definition
                if stmt.data == 'function_def':
                    func_def = build_function_def(stmt, DEBUG_MODE, DEBUG_INFO)
                    if func_def:
                        functions[func_def['name']] = func_def
                        if DEBUG_MODE:
                            print(f"Registered function: {func_def['name']}")
                
                # Let binding
                elif stmt.data == 'let_binding':
                    let_node = build_let_binding(stmt, DEBUG_MODE, DEBUG_INFO)
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
                                func_def = functions[func_name]
                                call_counter += 1
                                
                                inlined_stmts, return_name = inline_function_call(
                                    func_def, func_call['args'], {}, 
                                    f"{func_name}_{call_counter}"
                                )
                                
                                expanded_stmts = []
                                local_mappings = {}
                                
                                for stmt in inlined_stmts:
                                    if stmt['type'] == 'let' and stmt['expr']['type'] == 'user_function_call':
                                        nested_call = stmt['expr']
                                        nested_func_name = nested_call['func_name']
                                        
                                        if nested_func_name in functions:
                                            call_counter += 1
                                            nested_inlined, nested_return = inline_function_call(
                                                functions[nested_func_name],
                                                nested_call['args'], 
                                                {},
                                                f"{nested_func_name}_{call_counter}"
                                            )
                                            
                                            expanded_stmts.extend(nested_inlined)
                                            
                                            if nested_return:
                                                local_mappings[stmt['name']] = nested_return
                                        else:
                                            expanded_stmts.append(stmt)
                                    else:
                                        if stmt['type'] == 'let' and 'args' in stmt['expr']:
                                            new_args = []
                                            for arg in stmt['expr']['args']:
                                                new_args.append(local_mappings.get(arg, arg))
                                            stmt['expr']['args'] = new_args
                                        expanded_stmts.append(stmt)
                                
                                ast.extend(expanded_stmts)
                                
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
                                return None, None, None
                        else:
                            ast.append(let_node)
                
                # Expression (output tensor)
                elif stmt.data == 'expr':
                    expr_node = build_expression(stmt, DEBUG_MODE, DEBUG_INFO)
                    if expr_node and expr_node['type'] == 'name':
                        output_tensor = expr_node['name']
    
    # Default to last tensor if no explicit output
    if not output_tensor and ast:
        output_tensor = ast[-1]['name']
    
    return ast, output_tensor, functions


def build_let_binding(tree: Tree, DEBUG_MODE=False, DEBUG_INFO=False) -> Optional[Dict]:
    """Parse: let name: Type = expr"""
    if DEBUG_MODE:
        print(f"Building let_binding from {tree.data}")
    if tree.data != 'let_binding':
        if DEBUG_MODE:
            print(f"Invalid let_binding node: {tree.data}")
        return None
    
    children = tree.children
    if len(children) == 3:
        name = children[0].value
        if DEBUG_MODE:
            print(f"Processing let binding for {name}")
        if isinstance(children[1], Tree) and children[1].data == 'type':
            ty = build_type(children[1], DEBUG_MODE, DEBUG_INFO)
            expr = build_expression(children[2], DEBUG_MODE, DEBUG_INFO)
        else:
            ty = None
            expr = build_expression(children[2], DEBUG_MODE, DEBUG_INFO)
        return {'type': 'let', 'name': name, 'ty': ty, 'expr': expr, 'tree': tree}
    else:
        if DEBUG_MODE:
            print(f"Unexpected number of children in let_binding: {len(children)}")
        return None


def build_type(tree: Tree, DEBUG_MODE=False, DEBUG_INFO=False) -> Dict:
    """Parse type annotation"""
    if DEBUG_MODE:
        print(f"Building type from {tree.data}")
    
    if tree.data == 'concrete_type':
        children = tree.children
        dtype_value = 'f32'
        if len(children) > 0 and isinstance(children[0], Token) and children[0].value in ['f32', 'f64']:
            dtype_value = children[0].value
        shape_tree = children[1] if len(children) > 1 and isinstance(children[1], Tree) else None
        shape = build_shape(shape_tree, DEBUG_MODE, DEBUG_INFO) if shape_tree else (0, 0)
        return {'dtype': dtype_value, 'shape': shape}
    
    elif tree.data == 'generic_type':
        generic_shape_tree = tree.children[0]
        generic_dims = []
        for child in generic_shape_tree.children:
            if isinstance(child, Token):
                if child.type == 'NAME':
                    generic_dims.append(child.value)
                elif child.type == 'NUMBER':
                    generic_dims.append(int(float(child.value)))
        return {'dtype': 'f32', 'shape': tuple(generic_dims)}
    
    if tree.data == 'type':
        return build_type(tree.children[0], DEBUG_MODE, DEBUG_INFO)


def build_shape(tree: Tree, DEBUG_MODE=False, DEBUG_INFO=False) -> Tuple:
    """Parse shape specification"""
    if DEBUG_MODE:
        print(f"Building shape from {tree.data}")
    if tree.data != 'shape':
        if DEBUG_MODE:
            print(f"Invalid shape node: {tree.data}")
        return (0, 0)
    nums = [int(float(child.value)) for child in tree.children if isinstance(child, Token) and child.type == 'NUMBER']
    if DEBUG_MODE:
        print(f"Shape numbers: {nums}")
    return tuple(nums)


def build_expression(tree: Tree, DEBUG_MODE=False, DEBUG_INFO=False) -> Dict:
    """Parse any expression"""
    if tree.data == 'expr':
        tree = tree.children[0]
    
    if isinstance(tree, Token) and tree.type == 'NAME':
        if DEBUG_MODE:
            print(f"Expression: {tree.type}: {tree.value}")
        return {'type': 'name', 'name': tree.value}
    
    elif tree.data == 'user_function_call':
        return build_user_function_call(tree, DEBUG_MODE, DEBUG_INFO)
    
    # Unary activations
    if tree.data in ['relu_call', 'sigmoid_call', 'tanh_call', 'gelu_call', 'swish_call']:
        expr_name = tree.data.replace("_call", "")
        args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
        return {'type': expr_name, 'args': args}
    
    # Binary operations
    elif tree.data in ['matmul_call', 'equal_call', 'greater_call', 'add_call', 'minus_call', 'mult_call', 'div_call', 'linear_call']:
        expr_name = tree.data.replace("_call", "")
        args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
        return {'type': expr_name, 'args': args}
    
    elif tree.data == 'less_call':
        args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
        return {'type': 'less', 'args': args}
    
    # Loss functions
    elif tree.data == 'cross_entropy_call':
        args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
        if DEBUG_INFO:
            print(f"Expression {tree.data} args: {args}")
        return {'type': 'cross_entropy', 'args': args}
    
    elif tree.data == 'mse_loss_call':
        args = [child.value for child in tree.children if isinstance(child, Token) and child.type == 'NAME']
        if DEBUG_INFO:
            print(f"Expression {tree.data} args: {args}")
        return {'type': 'mse_loss', 'args': args}
    
    # Axis operations
    elif tree.data in ['softmax_call', 'sum_call', 'mean_call', 'max_call', 'min_call', 'argmax_call', 'argmin_call']:
        tensor_name = None
        axis = None
        expr_name = tree.data.replace("_call", "")
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                tensor_name = child.value
            elif isinstance(child, Token) and child.type == 'NUMBER':
                axis = int(float(child.value))
        if DEBUG_INFO:
            print(f"Expression {tree.data} tensor={tensor_name}, axis={axis}")
        return {'type': expr_name, 'tensor': tensor_name, 'axis': axis}
    
    # Concat
    elif tree.data == 'concat_call':
        tensor_names = []
        axis = None
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                tensor_names.append(child.value)
            elif isinstance(child, Token) and child.type == 'NUMBER':
                axis = int(float(child.value))
        if DEBUG_MODE:
            print(f"Expression: {tree.data} args: tensors={tensor_names}, axis={axis}")
        return {'type': 'concat', 'tensors': tensor_names, 'axis': axis}
    
    # Tensor literal
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
            print(f"Tensor literal data: {data}, is_1d: {is_1d}")
        return {'type': 'tensor_literal', 'data': data, 'is_1d': is_1d, 'tree': tree}
    
    # Fill
    elif tree.data == 'fill_call':
        value = float(tree.children[0].value) if isinstance(tree.children[0], Token) and tree.children[0].type == 'NUMBER' else 0.0
        shape_tree = tree.children[1] if len(tree.children) > 1 and isinstance(tree.children[1], Tree) else None
        shape = build_shape(shape_tree, DEBUG_MODE, DEBUG_INFO) if shape_tree else (1,)
        if DEBUG_INFO:
            print(f"{tree.data} value: {value}, shape: {shape}")
        return {'type': 'fill', 'value': value, 'shape': shape}
    
    # Layer norm
    elif tree.data == 'layer_norm_call':
        tensor_name = None
        axis = None
        eps = 1e-5
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                tensor_name = child.value
            elif isinstance(child, Token) and child.type == 'NUMBER':
                value = float(child.value)
                if value < 0.1:
                    eps = value
                else:
                    axis = int(value)
        if DEBUG_INFO:
            print(f"Expression {tree.data} args: tensor={tensor_name}, axis={axis}, eps={eps}")
        return {'type': 'layer_norm', 'tensor': tensor_name, 'axis': axis, 'eps': eps}
    
    # Slice
    elif tree.data == 'slice_expr':
        tensor_name = None
        slice_specs = []
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                tensor_name = child.value
            elif isinstance(child, Tree):
                if child.data == 'slice_spec':
                    spec = build_slice_spec(child, DEBUG_MODE, DEBUG_INFO)
                    slice_specs.append(spec)
        if DEBUG_INFO:
            print(f"Expression {tree.data} args: tensor={tensor_name}, specs={slice_specs}")
        return {'type': 'slice', 'tensor': tensor_name, 'specs': slice_specs}
    
    # Transpose
    elif tree.data == 'transpose_call':
        tensor_name = None
        axes = None
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                tensor_name = child.value
            elif isinstance(child, Tree):
                axis_values = []
                for grandchild in child.children:
                    if isinstance(grandchild, Token) and grandchild.type == 'NUMBER':
                        axis_values.append(int(float(grandchild.value)))
                if axis_values:
                    axes = tuple(axis_values)
        if DEBUG_INFO:
            print(f"Expression {tree.data} args: tensor={tensor_name}, axes={axes}")
        return {'type': 'transpose', 'tensor': tensor_name, 'axes': axes}
    
    # Reshape
    elif tree.data == 'reshape_call':
        tensor_name = None
        new_shape = None
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                tensor_name = child.value
            elif isinstance(child, Tree) and child.data == 'shape':
                new_shape = build_shape(child, DEBUG_MODE, DEBUG_INFO)
        if DEBUG_INFO:
            print(f"Expression {tree.data} args: tensor={tensor_name}, new_shape={new_shape}")
        return {'type': 'reshape', 'tensor': tensor_name, 'new_shape': new_shape}
    
    # Batch norm
    elif tree.data == 'batch_norm_call':
        tensor_name = None
        running_mean_name = None
        running_var_name = None
        eps = 1e-5
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
            print(f"Expression: {tree.data} args: tensor={tensor_name}, mean={running_mean_name}, var={running_var_name}, eps={eps}")
        return {'type': 'batch_norm', 'tensor': tensor_name, 'running_mean': running_mean_name, 
                'running_var': running_var_name, 'eps': eps}
    
    # Instance norm
    elif tree.data == 'instance_norm_call':
        tensor_name = None
        eps = 1e-5
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                tensor_name = child.value
            elif isinstance(child, Token) and child.type == 'NUMBER':
                eps = float(child.value)
        if DEBUG_MODE:
            print(f"{tree.data} args: tensor={tensor_name}, eps={eps}")
        return {'type': 'instance_norm', 'tensor': tensor_name, 'eps': eps}
    
    print(f"Unrecognized expr type: {tree.data}")
    return None


def build_slice_spec(tree: Tree, DEBUG_MODE=False, DEBUG_INFO=False) -> Dict:
    """Parse slice specification like 0:2, :, 1:, etc."""
    if tree.data == 'slice_spec':
        if len(tree.children) == 1:
            child = tree.children[0]
            if isinstance(child, Tree) and child.data == 'slice_range':
                return build_slice_spec(child, DEBUG_MODE, DEBUG_INFO)
            elif isinstance(child, Token):
                if child.type == 'NUMBER':
                    return {'type': 'index', 'value': int(float(child.value))}
                elif child.value == ':':
                    return {'type': 'full_slice'}
    elif tree.data == 'slice_range':
        if len(tree.children) == 2:
            left, right = tree.children
            if isinstance(left, Token) and isinstance(right, Token):
                if left.value == ':':
                    end = int(float(right.value))
                    return {'type': 'slice', 'start': None, 'end': end}
                elif right.value == ':':
                    start = int(float(left.value))
                    return {'type': 'slice', 'start': start, 'end': None}
                elif left.type == 'NUMBER' and right.type == 'NUMBER':
                    start = int(float(left.value))
                    end = int(float(right.value))
                    return {'type': 'slice', 'start': start, 'end': end}
    
    return {'type': 'full_slice'}


def build_function_def(tree: Tree, DEBUG_MODE=False, DEBUG_INFO=False) -> Optional[Dict]:
    """Parse function definition: fn name(params) -> return_type { body }"""
    if tree.data != 'function_def':
        return None
    
    children = tree.children
    name = children[0].value
    
    params = []
    param_list_idx = 1
    if len(children) > 1 and isinstance(children[1], Tree) and children[1].data == 'param_list':
        for param_tree in children[1].children:
            if param_tree.data == 'param':
                param_name = param_tree.children[0].value
                param_type = build_type(param_tree.children[1], DEBUG_MODE, DEBUG_INFO)
                params.append({'name': param_name, 'type': param_type})
        param_list_idx = 2
    
    return_type = None
    body_idx = param_list_idx
    
    if param_list_idx < len(children) and isinstance(children[param_list_idx], Tree):
        child_data = children[param_list_idx].data
        if child_data in ['type', 'concrete_type', 'generic_type']:
            return_type = build_type(children[param_list_idx], DEBUG_MODE, DEBUG_INFO)
            body_idx = param_list_idx + 1
    
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
                inner_stmt = stmt.children[0]
                if inner_stmt.data == 'let_binding':
                    body_statements.append(build_let_binding(inner_stmt, DEBUG_MODE, DEBUG_INFO))
                elif inner_stmt.data == 'expr_statement':
                    body_statements.append(build_expression(inner_stmt.children[0], DEBUG_MODE, DEBUG_INFO))
            elif stmt.data == 'return_statement':
                return_expr = build_expression(stmt.children[0], DEBUG_MODE, DEBUG_INFO)
    
    return {
        'type': 'function_def',
        'name': name,
        'params': params,
        'return_type': return_type,
        'body': body_statements,
        'return_expr': return_expr
    }


def build_user_function_call(tree: Tree, DEBUG_MODE=False, DEBUG_INFO=False) -> Dict:
    """Parse user function call: func_name(arg1, arg2, ...)"""
    if tree.data != 'user_function_call':
        return None
    
    func_name = tree.children[0].value
    args = []
    
    if len(tree.children) > 1 and isinstance(tree.children[1], Tree):
        arg_list = tree.children[1]
        if arg_list.data == 'arg_list':
            for arg in arg_list.children:
                args.append(build_expression(arg, DEBUG_MODE, DEBUG_INFO))
    
    return {
        'type': 'user_function_call',
        'func_name': func_name,
        'args': args
    }


def inline_function_call(func_def: Dict, args: List, env: Dict, unique_suffix: str) -> Tuple[List, Optional[str]]:
    """Inline a function call by creating new let bindings with unique names"""
    param_map = {}
    for param, arg in zip(func_def['params'], args):
        param_map[param['name']] = arg
    
    inlined_statements = []
    name_mapping = {}
    
    for stmt in func_def['body']:
        if stmt['type'] == 'let':
            old_name = stmt['name']
            new_name = f"{old_name}_{unique_suffix}"
            name_mapping[old_name] = new_name
            
            new_expr = substitute_names(stmt['expr'], param_map, name_mapping)
            
            inlined_statements.append({
                'type': 'let',
                'name': new_name,
                'ty': stmt['ty'],
                'expr': new_expr,
                'tree': stmt.get('tree')
            })
    
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


def substitute_names(expr: Dict, param_map: Dict, name_mapping: Dict) -> Dict:
    """Recursively substitute parameter names and local variable names in an expression"""
    if not isinstance(expr, dict):
        return expr
    
    new_expr = expr.copy()
    
    if expr['type'] == 'name':
        name = expr['name']
        if name in param_map:
            param_expr = param_map[name]
            if isinstance(param_expr, dict) and param_expr.get('type') == 'name':
                return param_expr
            return param_expr
        if name in name_mapping:
            new_expr['name'] = name_mapping[name]
        return new_expr
    
    if expr['type'] == 'user_function_call':
        new_expr['args'] = [substitute_names(arg, param_map, name_mapping) 
                            for arg in expr.get('args', [])]
        return new_expr
    
    if 'args' in expr:
        new_args = []
        for arg in expr['args']:
            if isinstance(arg, str):
                if arg in param_map:
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
                new_args.append(arg)
        new_expr['args'] = new_args
        return new_expr
    
    if 'tensor' in expr:
        tensor_name = expr['tensor']
        if tensor_name in param_map:
            param_expr = param_map[tensor_name]
            if param_expr['type'] == 'name':
                new_expr['tensor'] = param_expr['name']
        elif tensor_name in name_mapping:
            new_expr['tensor'] = name_mapping[tensor_name]
    
    return new_expr