"""
TensorLang Autograd System
Implements automatic differentiation for training neural networks.

File: tensorlang/autograd.py
"""

import numpy as np

from tensorlang.tensor_lang import TensorLang


class ComputationNode:
    """Represents a single operation in the computation graph."""

    def __init__(self, op_type: str, output_name: str, inputs: list[str],
                 metadata: dict | None = None):
        self.op_type = op_type
        self.output_name = output_name
        self.inputs = inputs
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Node({self.op_type}: {self.inputs} -> {self.output_name})"


class ComputationGraph:
    """
    Tracks operations and computes gradients via backpropagation.
    """

    def __init__(self, debug_mode: bool = False):
        self.nodes: list[ComputationNode] = []
        self.tensors: dict[str, np.ndarray] = {}
        self.gradients: dict[str, np.ndarray] = {}
        self.requires_grad: set[str] = set()
        self.tensorlang = TensorLang()
        self.debug_mode = debug_mode

    def register_tensor(self, name: str, value: np.ndarray, requires_grad: bool = False):
        self.tensors[name] = value
        if requires_grad:
            self.requires_grad.add(name)
            self.gradients[name] = np.zeros_like(value)

    def add_operation(self, op_type: str, output_name: str, inputs: list[str],
                     metadata: dict | None = None):
        node = ComputationNode(op_type, output_name, inputs, metadata)
        self.nodes.append(node)
        if self.debug_mode:
            self.tensorlang.print(message=f"[AUTOGRAD] Recorded: {node}")

    def zero_grad(self):
        for name in self.requires_grad:
            self.gradients[name] = np.zeros_like(self.tensors[name])

    def backward(self, loss_name: str):
        if loss_name not in self.tensors:
            raise ValueError(f"Loss tensor '{loss_name}' not found in graph")

        loss_shape = self.tensors[loss_name].shape
        if loss_shape not in [(), (1,)]:
            self.tensorlang.print(message=f"[AUTOGRAD] Warning: Loss should be scalar, got shape {loss_shape}")

        if self.debug_mode:
            self.tensorlang.print(message=f"[AUTOGRAD] Starting backward pass from '{loss_name}'")
            self.tensorlang.print(message=f"[AUTOGRAD] -> Graph has {len(self.nodes)} operations")

        tensors_needing_gradients = set()
        tensors_needing_gradients.update(self.requires_grad)
        for node in self.nodes:
            tensors_needing_gradients.add(node.output_name)
            tensors_needing_gradients.update(node.inputs)
        tensors_needing_gradients.add(loss_name)

        for tensor_name in tensors_needing_gradients:
            if tensor_name in self.tensors:
                self.gradients[tensor_name] = np.zeros_like(self.tensors[tensor_name])
                if self.debug_mode:
                    self.tensorlang.print(message=f"[AUTOGRAD] Initialized gradient for '{tensor_name}' to zero")

        self.gradients[loss_name] = np.ones_like(self.tensors[loss_name])

        if self.debug_mode:
            self.tensorlang.print(message="[AUTOGRAD] Set loss gradient to 1.0")

        for node in reversed(self.nodes):
            if self.debug_mode:
                self.tensorlang.print(message=f"[AUTOGRAD] Processing: {node}")

            if node.output_name not in self.gradients:
                if self.debug_mode:
                    self.tensorlang.print(message=f"[AUTOGRAD] -> Skipping {node.output_name} - no gradient")
                continue

            output_grad = self.gradients[node.output_name]

            if np.all(output_grad == 0):
                if self.debug_mode:
                    self.tensorlang.print(message=f"[AUTOGRAD] -> Skipping {node.output_name} - zero gradient")
                continue

            self._backward_node(node)

        if self.debug_mode:
            self.tensorlang.print(message="[AUTOGRAD] Backward pass complete")
            self.tensorlang.print(message=f"[AUTOGRAD] Gradients computed for {len(self.gradients)} tensors")
            for name in sorted(self.requires_grad):
                if name in self.gradients:
                    grad = self.gradients[name]
                    self.tensorlang.print(message=f"[AUTOGRAD] -> (requires_grad) {name}.grad: \n"
                        f"shape={grad.shape}, \n"
                        f"norm={np.linalg.norm(grad):.6f}, \n"
                        f"mean_abs={np.mean(np.abs(grad)):.6f}, \n"
                        f"max_abs={np.max(np.abs(grad)):.6f}")

    def _backward_node(self, node: ComputationNode):
        if node.output_name not in self.gradients:
            return
        output_grad = self.gradients[node.output_name]

        ot = node.op_type

        # Element-wise
        if ot in ('add',):
            self._backward_add(node, output_grad)
        elif ot in ('minus', 'sub', 'subtract'):
            self._backward_minus(node, output_grad)
        elif ot in ('mult', 'mul', 'multiply'):
            self._backward_mult(node, output_grad)
        elif ot in ('div', 'divide', 'truediv'):
            self._backward_div(node, output_grad)

        # Matrix
        elif ot in ('matmul', 'mm', 'bmm'):
            self._backward_matmul(node, output_grad)

        # Activations
        elif ot in ('relu',):
            self._backward_relu(node, output_grad)
        elif ot in ('sigmoid',):
            self._backward_sigmoid(node, output_grad)
        elif ot in ('tanh',):
            self._backward_tanh(node, output_grad)
        elif ot in ('gelu',):
            self._backward_gelu(node, output_grad)
        elif ot in ('swish', 'silu'):
            self._backward_swish(node, output_grad)
        elif ot in ('softmax',):
            self._backward_softmax(node, output_grad)

        # Reductions
        elif ot in ('sum',):
            self._backward_sum(node, output_grad)
        elif ot in ('mean', 'avg', 'average'):
            self._backward_mean(node, output_grad)
        elif ot in ('max', 'maximum', 'amax'):
            self._backward_max(node, output_grad)
        elif ot in ('min', 'minimum', 'amin'):
            self._backward_min(node, output_grad)
        elif ot in ('argmax',):
            self._backward_argmax(node, output_grad)
        elif ot in ('argmin',):
            self._backward_argmin(node, output_grad)

        # Shape ops
        elif ot in ('reshape', 'view', 'flatten', 'squeeze', 'unsqueeze'):
            self._backward_reshape(node, output_grad)
        elif ot in ('transpose', 'permute', 'swapaxes', 't'):
            self._backward_transpose(node, output_grad)
        elif ot in ('slice', 'index', 'getitem', 'narrow'):
            self._backward_slice(node, output_grad)
        elif ot in ('concat', 'concatenate', 'cat'):
            self._backward_concat(node, output_grad)

        # NN
        elif ot in ('linear', 'dense', 'fc'):
            self._backward_linear(node, output_grad)
        elif ot in ('layer_norm', 'layernorm', 'ln'):
            self._backward_layer_norm(node, output_grad)
        elif ot in ('batch_norm', 'batchnorm', 'bn'):
            self._backward_batch_norm(node, output_grad)
        elif ot in ('instance_norm', 'instancenorm', 'in'):
            self._backward_instance_norm(node, output_grad)

        # Losses
        elif ot in ('mse_loss', 'mse'):
            self._backward_mse_loss(node, output_grad)
        elif ot in ('cross_entropy', 'ce', 'crossentropy'):
            self._backward_cross_entropy(node, output_grad)

        # Comparisons (non-differentiable -> zero grad)
        elif ot in ('greater', 'less', 'equal', 'greater_equal', 'less_equal',
                    'not_equal', 'gt', 'lt', 'eq', 'ge', 'le', 'ne'):
            self._backward_comparison(node, output_grad)

        else:
            if self.debug_mode:
                self.tensorlang.print(message=f"[AUTOGRAD] Warning: No backward implemented for {node.op_type}")

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _accumulate_grad(self, tensor_name: str, grad: np.ndarray):
        if tensor_name not in self.gradients:
            if tensor_name not in self.tensors:
                return
            self.gradients[tensor_name] = np.zeros_like(self.tensors[tensor_name])

        input_shape = self.tensors[tensor_name].shape
        grad_shape = np.asarray(grad).shape

        if input_shape != grad_shape:
            grad = self._unbroadcast(np.asarray(grad), input_shape)

        self.gradients[tensor_name] += grad

    def _unbroadcast(self, grad: np.ndarray, target_shape: tuple) -> np.ndarray:
        if grad.shape == target_shape:
            return grad

        if target_shape == ():
            target_shape = (1,)
            need_scalar_conversion = True
        else:
            need_scalar_conversion = False

        ndim_diff = len(grad.shape) - len(target_shape)
        if ndim_diff > 0:
            for _ in range(ndim_diff):
                grad = grad.sum(axis=0)
        elif ndim_diff < 0:
            raise ValueError(
                f"Invalid unbroadcast: grad shape {grad.shape} has fewer "
                f"dimensions than target {target_shape}"
            )

        assert len(grad.shape) == len(target_shape), \
            f"After prepend sum, shapes should match in ndim: {grad.shape} vs {target_shape}"

        for i in reversed(range(len(target_shape))):
            grad_dim = grad.shape[i]
            target_dim = target_shape[i]
            if target_dim == 1 and grad_dim != 1:
                grad = grad.sum(axis=i, keepdims=True)
            elif target_dim != grad_dim:
                raise ValueError(
                    f"Unbroadcast error at axis {i}: grad has size {grad_dim}, "
                    f"target has size {target_dim}."
                )

        if need_scalar_conversion:
            grad = grad.squeeze()
            target_shape = ()

        if grad.shape != target_shape:
            raise ValueError(
                f"Unbroadcast failed: expected shape {target_shape}, got {grad.shape}."
            )
        return grad

    @staticmethod
    def _as_tuple_axis(axis, ndim):
        if axis is None:
            return None
        if isinstance(axis, (tuple, list)):
            return tuple(a % ndim for a in axis)
        return (axis % ndim,)

    # ----------------------------------------------------------------
    # Element-wise
    # ----------------------------------------------------------------
    def _backward_add(self, node, output_grad):
        a, b = node.inputs
        self._accumulate_grad(a, output_grad)
        self._accumulate_grad(b, output_grad)

    def _backward_minus(self, node, output_grad):
        a, b = node.inputs
        self._accumulate_grad(a, output_grad)
        self._accumulate_grad(b, -np.asarray(output_grad))

    def _backward_mult(self, node, output_grad):
        a, b = node.inputs
        self._accumulate_grad(a, np.asarray(output_grad) * self.tensors[b])
        self._accumulate_grad(b, np.asarray(output_grad) * self.tensors[a])

    def _backward_div(self, node, output_grad):
        a, b = node.inputs
        b_val = self.tensors[b]
        og = np.asarray(output_grad)
        self._accumulate_grad(a, og / b_val)
        self._accumulate_grad(b, -og * self.tensors[a] / (b_val ** 2))

    # ----------------------------------------------------------------
    # Matrix
    # ----------------------------------------------------------------
    def _backward_matmul(self, node, output_grad):
        a, b = node.inputs
        a_val = self.tensors[a]
        b_val = self.tensors[b]
        og = np.asarray(output_grad)

        grad_a = og @ b_val.swapaxes(-2, -1)
        grad_b = a_val.swapaxes(-2, -1) @ og

        self._accumulate_grad(a, grad_a)
        self._accumulate_grad(b, grad_b)

    # ----------------------------------------------------------------
    # Activations
    # ----------------------------------------------------------------
    def _backward_relu(self, node, output_grad):
        inp = node.inputs[0]
        mask = (self.tensors[inp] > 0).astype(np.float32)
        self._accumulate_grad(inp, np.asarray(output_grad) * mask)

    def _backward_sigmoid(self, node, output_grad):
        inp = node.inputs[0]
        sigmoid_out = self.tensors[node.output_name]
        grad = np.asarray(output_grad) * sigmoid_out * (1 - sigmoid_out)
        self._accumulate_grad(inp, grad)

    def _backward_tanh(self, node, output_grad):
        inp = node.inputs[0]
        tanh_out = self.tensors[node.output_name]
        grad = np.asarray(output_grad) * (1 - tanh_out ** 2)
        self._accumulate_grad(inp, grad)

    def _backward_gelu(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        import math
        erf_vec = np.vectorize(math.erf)
        cdf = (0.5 * (1.0 + erf_vec(x / np.sqrt(2.0)))).astype(np.float32)
        pdf = (np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)).astype(np.float32)
        grad = np.asarray(output_grad) * (cdf + x * pdf)
        self._accumulate_grad(inp, grad)

    def _backward_swish(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        sigmoid_x = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
        grad = np.asarray(output_grad) * (sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x))
        self._accumulate_grad(inp, grad)

    def _backward_softmax(self, node, output_grad):
        inp = node.inputs[0]
        softmax_out = self.tensors[node.output_name]
        axis = node.metadata.get('axis', -1)
        if axis is None:
            axis = -1
        og = np.asarray(output_grad)
        grad = og * softmax_out
        sum_grad = np.sum(grad, axis=axis, keepdims=True)
        grad = grad - softmax_out * sum_grad
        self._accumulate_grad(inp, grad)

    # ----------------------------------------------------------------
    # Reductions
    # ----------------------------------------------------------------
    def _backward_sum(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        axis = node.metadata.get('axis', node.metadata.get('axes'))
        keepdims = node.metadata.get('keepdims', False)
        og = np.asarray(output_grad)

        if axis is None:
            scalar = og.item() if og.size == 1 else og
            grad = np.full(x.shape, scalar, dtype=x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)
        else:
            axes = self._as_tuple_axis(axis, x.ndim)
            grad = og
            if not keepdims:
                for a in sorted(axes):
                    grad = np.expand_dims(grad, a)
            grad = np.broadcast_to(grad, x.shape).copy()

        self._accumulate_grad(inp, grad)

    def _backward_mean(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        axis = node.metadata.get('axis', node.metadata.get('axes'))
        keepdims = node.metadata.get('keepdims', False)
        og = np.asarray(output_grad)

        if axis is None:
            count = x.size if x.size > 0 else 1
            scalar = (og.item() if og.size == 1 else og) / count
            grad = np.full(x.shape, scalar, dtype=x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)
        else:
            axes = self._as_tuple_axis(axis, x.ndim)
            count = int(np.prod([x.shape[a] for a in axes])) or 1
            grad = og / count
            if not keepdims:
                for a in sorted(axes):
                    grad = np.expand_dims(grad, a)
            grad = np.broadcast_to(grad, x.shape).copy()

        self._accumulate_grad(inp, grad)

    def _backward_max(self, node, output_grad):
        if len(node.inputs) >= 2:
            a, b = node.inputs[0], node.inputs[1]
            a_val = self.tensors[a]
            b_val = self.tensors[b]
            og = np.asarray(output_grad)
            mask_a = (a_val >= b_val)
            self._accumulate_grad(a, og * mask_a)
            self._accumulate_grad(b, og * (~mask_a))
            return

        inp = node.inputs[0]
        x = self.tensors[inp]
        axis = node.metadata.get('axis', node.metadata.get('axes'))
        keepdims = node.metadata.get('keepdims', False)
        og = np.asarray(output_grad)
        axes = self._as_tuple_axis(axis, x.ndim) if axis is not None else None

        if axes is None:
            max_val = x.max()
            mask = (x == max_val)
        else:
            max_val = x.max(axis=axes, keepdims=True)
            mask = (x == max_val)

        if axes is not None and not keepdims:
            og_exp = og
            for a in sorted(axes):
                og_exp = np.expand_dims(og_exp, a)
        else:
            og_exp = og
        og_b = np.broadcast_to(og_exp, x.shape).copy() if np.asarray(og_exp).shape != x.shape else np.asarray(og_exp).copy()

        grad = og_b * mask
        self._accumulate_grad(inp, grad)

    def _backward_min(self, node, output_grad):
        if len(node.inputs) >= 2:
            a, b = node.inputs[0], node.inputs[1]
            a_val = self.tensors[a]
            b_val = self.tensors[b]
            og = np.asarray(output_grad)
            mask_a = (a_val <= b_val)
            self._accumulate_grad(a, og * mask_a)
            self._accumulate_grad(b, og * (~mask_a))
            return

        inp = node.inputs[0]
        x = self.tensors[inp]
        axis = node.metadata.get('axis', node.metadata.get('axes'))
        keepdims = node.metadata.get('keepdims', False)
        og = np.asarray(output_grad)
        axes = self._as_tuple_axis(axis, x.ndim) if axis is not None else None

        if axes is None:
            min_val = x.min()
            mask = (x == min_val)
        else:
            min_val = x.min(axis=axes, keepdims=True)
            mask = (x == min_val)

        if axes is not None and not keepdims:
            og_exp = og
            for a in sorted(axes):
                og_exp = np.expand_dims(og_exp, a)
        else:
            og_exp = og
        og_b = np.broadcast_to(og_exp, x.shape).copy() if np.asarray(og_exp).shape != x.shape else np.asarray(og_exp).copy()

        grad = og_b * mask
        self._accumulate_grad(inp, grad)

    def _backward_argmax(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        axis = node.metadata.get('axis', node.metadata.get('axes', -1))
        if axis is None:
            axis = -1
        ax = int(axis) % x.ndim
        keepdims = node.metadata.get('keepdims', False)
        og = np.asarray(output_grad)

        if not keepdims:
            og_exp = np.expand_dims(og, ax)
        else:
            og_exp = og

        indices = np.argmax(x, axis=ax)
        grad = np.zeros_like(x, dtype=np.float32)
        np.put_along_axis(grad, np.expand_dims(indices, ax), og_exp, axis=ax)
        self._accumulate_grad(inp, grad)

    def _backward_argmin(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        axis = node.metadata.get('axis', node.metadata.get('axes', -1))
        if axis is None:
            axis = -1
        ax = int(axis) % x.ndim
        keepdims = node.metadata.get('keepdims', False)
        og = np.asarray(output_grad)

        if not keepdims:
            og_exp = np.expand_dims(og, ax)
        else:
            og_exp = og

        indices = np.argmin(x, axis=ax)
        grad = np.zeros_like(x, dtype=np.float32)
        np.put_along_axis(grad, np.expand_dims(indices, ax), og_exp, axis=ax)
        self._accumulate_grad(inp, grad)

    # ----------------------------------------------------------------
    # Shape ops
    # ----------------------------------------------------------------
    def _backward_reshape(self, node, output_grad):
        inp = node.inputs[0]
        target_shape = self.tensors[inp].shape
        grad = np.asarray(output_grad).reshape(target_shape)
        self._accumulate_grad(inp, grad)

    def _backward_transpose(self, node, output_grad):
        inp = node.inputs[0]
        x_shape = self.tensors[inp].shape
        og = np.asarray(output_grad)

        axis1 = node.metadata.get('axis1')
        axis2 = node.metadata.get('axis2')
        axes = node.metadata.get('axes', node.metadata.get('perm', node.metadata.get('axis')))

        if axis1 is not None and axis2 is not None:
            grad = np.swapaxes(og, int(axis1), int(axis2))
        elif isinstance(axes, (list, tuple)):
            if len(axes) == 2 and all(isinstance(a, int) for a in axes):
                grad = np.swapaxes(og, axes[0], axes[1])
            else:
                inv = [0] * len(axes)
                for i, a in enumerate(axes):
                    inv[int(a)] = i
                grad = np.transpose(og, inv)
        elif isinstance(axes, int):
            grad = np.swapaxes(og, axes, axes - 1)
        else:
            grad = np.transpose(og)

        if grad.shape != x_shape:
            grad = grad.reshape(x_shape)
        self._accumulate_grad(inp, grad)

    def _spec_to_slice(self, spec):
        """Convert a slice spec (dict, int, slice, or None) to a slice or int."""
        if spec is None:
            return slice(None)
        if isinstance(spec, slice):
            return spec
        if isinstance(spec, int):
            return spec
        if isinstance(spec, dict):
            stype = spec.get('type', '')
            if stype == 'slice':
                start = spec.get('start', None)
                end = spec.get('end', None)
                step = spec.get('step', None)
                return slice(start, end, step)
            if stype in ('index', 'int'):
                return spec.get('value', spec.get('index', 0))
            # fallback: try start/end
            if 'start' in spec or 'end' in spec:
                return slice(spec.get('start'), spec.get('end'), spec.get('step'))
        return spec

    def _backward_slice(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        og = np.asarray(output_grad)

        raw_slices = node.metadata.get('specs', node.metadata.get('slice_specs',
                node.metadata.get('slices', node.metadata.get('slice',
                node.metadata.get('slice_spec', node.metadata.get('indices'))))))

        if raw_slices is None:
            axis = node.metadata.get('axis', 0)
            start = node.metadata.get('start', None)
            stop = node.metadata.get('stop', None)
            step = node.metadata.get('step', None)
            ndim = x.ndim
            ax = int(axis) % ndim
            sl = [slice(None)] * ndim
            sl[ax] = slice(start, stop, step)
            slices = tuple(sl)
        else:
            if not isinstance(raw_slices, (list, tuple)):
                raw_slices = (raw_slices,)
            slices = tuple(self._spec_to_slice(s) for s in raw_slices)

        if not isinstance(slices, tuple):
            slices = (slices,)

        grad = np.zeros_like(x, dtype=np.result_type(x.dtype, np.float32))
        try:
            grad[slices] = og
        except ValueError:
            grad[slices] = np.broadcast_to(og, np.asarray(grad[slices]).shape)
        self._accumulate_grad(inp, grad)

    def _backward_concat(self, node, output_grad):
        inputs = node.inputs
        axis = node.metadata.get('axis', node.metadata.get('dim', 0))
        if axis is None:
            axis = 0
        axis = int(axis)
        ndim = self.tensors[inputs[0]].ndim
        axis = axis % ndim

        sizes = node.metadata.get('sizes', node.metadata.get('split_sizes'))
        if sizes is None:
            sizes = [self.tensors[n].shape[axis] for n in inputs]

        og = np.asarray(output_grad)
        split_indices = np.cumsum(sizes)[:-1]
        parts = np.split(og, split_indices, axis=axis)
        for name, part in zip(inputs, parts):
            self._accumulate_grad(name, part)

    # ----------------------------------------------------------------
    # NN
    # ----------------------------------------------------------------
    def _backward_linear(self, node, output_grad):
        inp, weight, bias = node.inputs
        x = self.tensors[inp]
        w = self.tensors[weight]
        og = np.asarray(output_grad)

        if w.ndim == 2:
            grad_x = og @ w.T
        else:
            grad_x = og @ w.swapaxes(-2, -1)
        self._accumulate_grad(inp, grad_x)

        if x.ndim == 1:
            grad_w = np.outer(x, og.reshape(-1) if og.ndim > 1 else og)
        else:
            x2 = x.reshape(-1, x.shape[-1])
            og2 = og.reshape(-1, og.shape[-1])
            grad_w = x2.T @ og2
        self._accumulate_grad(weight, grad_w)

        if og.ndim == 1:
            grad_b = og
        else:
            grad_b = og.reshape(-1, og.shape[-1]).sum(axis=0)
        self._accumulate_grad(bias, grad_b)

    def _backward_layer_norm(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        axis = node.metadata.get('axis', node.metadata.get('axes', -1))
        if axis is None:
            axis = -1
        if isinstance(axis, int):
            axes = (axis % x.ndim,)
        else:
            axes = tuple(a % x.ndim for a in axis)
        eps = node.metadata.get('eps', 1e-5)
        og = np.asarray(output_grad)

        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        std = np.sqrt(var + eps)

        N = int(np.prod([x.shape[a] for a in axes])) or 1
        grad_x_norm = og
        grad_var = np.sum(grad_x_norm * (x - mean) * -0.5 * (var + eps) ** -1.5,
                         axis=axes, keepdims=True)
        grad_mean = np.sum(grad_x_norm * -1 / std, axis=axes, keepdims=True) + \
                   grad_var * np.sum(-2 * (x - mean), axis=axes, keepdims=True) / N

        grad = grad_x_norm / std + grad_var * 2 * (x - mean) / N + grad_mean / N
        self._accumulate_grad(inp, grad)

    def _backward_batch_norm(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        og = np.asarray(output_grad)
        if x.ndim >= 2:
            axes = (0,) + tuple(range(2, x.ndim))
        else:
            axes = (0,)
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        eps = node.metadata.get('eps', 1e-5)
        std = np.sqrt(var + eps)
        N = int(np.prod([x.shape[a] for a in axes])) or 1
        grad_var = np.sum(og * (x - mean) * -0.5 * (var + eps) ** -1.5,
                         axis=axes, keepdims=True)
        grad_mean = np.sum(og * -1.0 / std, axis=axes, keepdims=True) + \
                   grad_var * np.sum(-2.0 * (x - mean), axis=axes, keepdims=True) / N
        grad = og / std + grad_var * 2.0 * (x - mean) / N + grad_mean / N
        self._accumulate_grad(inp, grad)

    def _backward_instance_norm(self, node, output_grad):
        inp = node.inputs[0]
        x = self.tensors[inp]
        og = np.asarray(output_grad)
        if x.ndim >= 3:
            axes = tuple(range(2, x.ndim))
        elif x.ndim == 2:
            axes = (1,)
        else:
            axes = None
        if not axes:
            self._accumulate_grad(inp, og)
            return
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        eps = node.metadata.get('eps', 1e-5)
        std = np.sqrt(var + eps)
        N = int(np.prod([x.shape[a] for a in axes])) or 1
        grad_var = np.sum(og * (x - mean) * -0.5 * (var + eps) ** -1.5,
                         axis=axes, keepdims=True)
        grad_mean = np.sum(og * -1.0 / std, axis=axes, keepdims=True) + \
                   grad_var * np.sum(-2.0 * (x - mean), axis=axes, keepdims=True) / N
        grad = og / std + grad_var * 2.0 * (x - mean) / N + grad_mean / N
        self._accumulate_grad(inp, grad)

    # ----------------------------------------------------------------
    # Losses
    # ----------------------------------------------------------------
    def _backward_mse_loss(self, node, output_grad):
        pred, target = node.inputs
        pred_val = self.tensors[pred]
        target_val = self.tensors[target]
        n = pred_val.size if pred_val.size > 0 else 1
        og = np.asarray(output_grad)
        scalar_grad = og.item() if og.size == 1 else og
        grad = 2 * (pred_val - target_val) / n * scalar_grad
        self._accumulate_grad(pred, grad)

    def _backward_cross_entropy(self, node, output_grad):
        pred, target = node.inputs
        pred_val = self.tensors[pred]
        target_val = self.tensors[target]
        og = np.asarray(output_grad)

        if target_val.shape != pred_val.shape:
            num_classes = pred_val.shape[-1]
            one_hot = np.zeros_like(pred_val)
            flat_pred_shape = (-1, num_classes)
            flat_target = target_val.reshape(-1)
            flat_one_hot = one_hot.reshape(-1, num_classes)
            class_indices = np.clip(flat_target.astype(np.int64), 0, num_classes - 1)
            flat_one_hot[np.arange(flat_one_hot.shape[0]), class_indices] = 1.0
            target_val = flat_one_hot.reshape(pred_val.shape)

        n_samples = int(np.prod(pred_val.shape[:-1])) if pred_val.ndim > 1 else 1
        if n_samples < 1:
            n_samples = 1
        scalar_grad = og.item() if og.size == 1 else og
        grad = (pred_val - target_val) / n_samples * scalar_grad
        self._accumulate_grad(pred, grad)

    # ----------------------------------------------------------------
    # Comparisons (non-differentiable)
    # ----------------------------------------------------------------
    def _backward_comparison(self, node, output_grad):
        for name in node.inputs:
            if name in self.tensors:
                self._accumulate_grad(name, np.zeros(self.tensors[name].shape, dtype=np.float32))


class AutogradContext:
    """
    Context manager for enabling/disabling gradient tracking.

    Usage:
        with AutogradContext.no_grad():
            # Operations here won't be tracked
            y = matmul(x, w)
    """

    _enabled = True

    @classmethod
    def is_enabled(cls):
        return cls._enabled

    @classmethod
    def no_grad(cls):
        """Context manager to temporarily disable gradient tracking."""
        return cls._NoGradContext()

    class _NoGradContext:
        def __enter__(self):
            AutogradContext._enabled = False

        def __exit__(self, *args):
            AutogradContext._enabled = True
