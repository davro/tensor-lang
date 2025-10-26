"""
TensorLang Autograd System
Implements automatic differentiation for training neural networks.

File: tensorlang/autograd.py
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from tensorlang.tensor_lang import TensorLang

class ComputationNode:
    """Represents a single operation in the computation graph."""
    
    def __init__(self, op_type: str, output_name: str, inputs: List[str], 
                 metadata: Optional[Dict] = None):
        self.op_type = op_type
        self.output_name = output_name
        self.inputs = inputs
        self.metadata = metadata or {}
        
    def __repr__(self):
        return f"Node({self.op_type}: {self.inputs} -> {self.output_name})"


class ComputationGraph:
    """
    Tracks operations and computes gradients via backpropagation.
    
    This class maintains a directed acyclic graph (DAG) of tensor operations
    and can compute gradients by traversing the graph in reverse topological order.
    """
    
    def __init__(self, debug_mode: bool = False):
        self.nodes: List[ComputationNode] = []
        self.tensors: Dict[str, np.ndarray] = {}
        self.gradients: Dict[str, np.ndarray] = {}
        self.requires_grad: Set[str] = set()
        self.tensorlang = TensorLang()
        self.debug_mode = debug_mode
        
    def register_tensor(self, name: str, value: np.ndarray, requires_grad: bool = False):
        """Register a tensor in the graph."""
        self.tensors[name] = value
        if requires_grad:
            self.requires_grad.add(name)
            self.gradients[name] = np.zeros_like(value)
            
    def add_operation(self, op_type: str, output_name: str, inputs: List[str], 
                     metadata: Optional[Dict] = None):
        """Record an operation in the computation graph."""
        node = ComputationNode(op_type, output_name, inputs, metadata)
        self.nodes.append(node)
        
        if self.debug_mode:
            self.tensorlang.print(message=f"[AUTOGRAD] Recorded: {node}")
            
    def zero_grad(self):
        """Reset all gradients to zero."""
        for name in self.requires_grad:
            self.gradients[name] = np.zeros_like(self.tensors[name])
            
    # def backward(self, loss_name: str):
    #     """
    #     Compute gradients via backpropagation.
        
    #     Args:
    #         loss_name: Name of the scalar loss tensor to backpropagate from
    #     """
    #     if loss_name not in self.tensors:
    #         raise ValueError(f"Loss tensor '{loss_name}' not found in graph")
            
    #     # Initialize gradient of loss to 1.0
    #     loss_shape = self.tensors[loss_name].shape
    #     if loss_shape not in [(), (1,)]:
    #         print(f"Warning: Loss should be scalar, got shape {loss_shape}")
            
    #     self.gradients[loss_name] = np.ones_like(self.tensors[loss_name])
        
    #     if self.debug_mode:
    #         print(f"\n[AUTOGRAD] Starting backward pass from '{loss_name}'")
    #         print(f"[AUTOGRAD] Graph has {len(self.nodes)} operations")
        
    #     # Traverse graph in reverse order
    #     for node in reversed(self.nodes):
    #         if self.debug_mode:
    #             print(f"\n[AUTOGRAD] Processing: {node}")
                
    #         self._backward_node(node)
            
    #     if self.debug_mode:
    #         print("\n[AUTOGRAD] Backward pass complete")
    #         for name in self.requires_grad:
    #             if name in self.gradients:
    #                 grad_norm = np.linalg.norm(self.gradients[name])
    #                 print(f"  {name}.grad: shape={self.gradients[name].shape}, norm={grad_norm:.6f}")


    def backward(self, loss_name: str):
        """
        Compute gradients via backpropagation.
        
        Args:
            loss_name: Name of the scalar loss tensor to backpropagate from
            
        Algorithm:
            1. Validate loss tensor exists
            2. Initialize ALL gradients to zero (prevents accumulation bugs)
            3. Set loss gradient to 1.0
            4. Traverse computation graph in reverse topological order
            5. Accumulate gradients for each operation
        """
        # ================================================================
        # Step 1: Validation
        # ================================================================
        if loss_name not in self.tensors:
            raise ValueError(f"Loss tensor '{loss_name}' not found in graph")
        
        # Validate loss is scalar
        loss_shape = self.tensors[loss_name].shape
        if loss_shape not in [(), (1,)]:
            self.tensorlang.print(message=f"[AUTOGRAD] Warning: Loss should be scalar, got shape {loss_shape}")
        
        if self.debug_mode:
            self.tensorlang.print(message=f"[AUTOGRAD] Starting backward pass from '{loss_name}'")
            self.tensorlang.print(message=f"[AUTOGRAD] -> Graph has {len(self.nodes)} operations")
        
        # ================================================================
        # Step 2: Initialize ALL gradients to zero BEFORE backward pass
        # This is CRITICAL to avoid gradient accumulation bugs
        # ================================================================
        
        # First, collect all tensors that need gradients
        tensors_needing_gradients = set()
        
        # Add all leaf tensors marked with 'with grad'
        tensors_needing_gradients.update(self.requires_grad)
        
        # Add all intermediate tensors that appear in the computation graph
        for node in self.nodes:
            # Output of this operation might need gradient
            tensors_needing_gradients.add(node.output_name)
            # Inputs to this operation might need gradient
            tensors_needing_gradients.update(node.inputs)
        
        # Add the loss tensor itself
        tensors_needing_gradients.add(loss_name)
        
        # Initialize gradients to zero for all tensors in the graph
        for tensor_name in tensors_needing_gradients:
            if tensor_name in self.tensors:
                self.gradients[tensor_name] = np.zeros_like(self.tensors[tensor_name])
                if self.debug_mode:
                    self.tensorlang.print(message=f"[AUTOGRAD] Initialized gradient for '{tensor_name}' to zero")
        
        # ================================================================
        # Step 3: Set loss gradient to 1.0 (seed gradient)
        # ================================================================
        self.gradients[loss_name] = np.ones_like(self.tensors[loss_name])
        
        if self.debug_mode:
            self.tensorlang.print(message=f"[AUTOGRAD] Set loss gradient to 1.0")
        
        # ================================================================
        # Step 4: Backward pass - traverse graph in REVERSE topological order
        # ================================================================
        for node in reversed(self.nodes):
            if self.debug_mode:
                self.tensorlang.print(message=f"[AUTOGRAD] Processing: {node}")
            
            # Skip if output doesn't have gradient yet (dead branch in graph)
            if node.output_name not in self.gradients:
                if self.debug_mode:
                    self.tensorlang.print(message=f"[AUTOGRAD] -> Skipping {node.output_name} - no gradient")
                continue
            
            output_grad = self.gradients[node.output_name]
            
            # Check if gradient is zero (optimization - can skip computation)
            if np.all(output_grad == 0):
                if self.debug_mode:
                    self.tensorlang.print(message=f"[AUTOGRAD] -> Skipping {node.output_name} - zero gradient")
                continue
            
            # Compute gradients for this node
            self._backward_node(node)
        
        # ================================================================
        # Step 5: Optional cleanup - remove intermediate gradients
        # ================================================================
        # Keep only gradients for tensors marked with 'requires_grad'
        # This saves memory but you can comment this out for debugging
        
        # Uncomment to keep ALL gradients (useful for debugging):
        # pass
        
        # Uncomment to keep only leaf gradients (saves memory):
        # for name in list(self.gradients.keys()):
        #     if name not in self.requires_grad and name != loss_name:
        #         del self.gradients[name]
        #         if self.debug_mode:
        #             print(f"[AUTOGRAD] Cleared intermediate gradient: {name}")
        
        # ================================================================
        # Step 6: Debug output
        # ================================================================
        if self.debug_mode:
            self.tensorlang.print(message=f"[AUTOGRAD] Backward pass complete")
            self.tensorlang.print(message=f"[AUTOGRAD] Gradients computed for {len(self.gradients)} tensors")
            
            # Print gradients for leaf tensors (requires_grad)
            for name in sorted(self.requires_grad):
                if name in self.gradients:
                    grad = self.gradients[name]
                    grad_norm = np.linalg.norm(grad)
                    grad_mean = np.mean(np.abs(grad))
                    grad_max = np.max(np.abs(grad))
                    self.tensorlang.print(message=f"[AUTOGRAD] -> (requires_grad) {name}.grad: \n"
                        f"shape={grad.shape}, \n"
                        f"norm={grad_norm:.6f}, \n"
                        f"mean_abs={grad_mean:.6f}, \n"
                        f"max_abs={grad_max:.6f}")
            
            # Print any unexpected gradients (for debugging)
            unexpected_grads = set(self.gradients.keys()) - self.requires_grad - {loss_name}
            if unexpected_grads:
                self.tensorlang.print(message=f"[AUTOGRAD] Intermediate gradients: {sorted(unexpected_grads)}")
                    
    
    def _backward_node(self, node: ComputationNode):
        """Compute gradients for a single operation."""
        
        # Skip if output doesn't have gradient (dead branch)
        if node.output_name not in self.gradients:
            return
            
        output_grad = self.gradients[node.output_name]
        
        # ============================================================
        # Element-wise Operations
        # ============================================================
        
        if node.op_type == 'add':
            self._backward_add(node, output_grad)
            
        elif node.op_type == 'minus':
            self._backward_minus(node, output_grad)
            
        elif node.op_type == 'mult':
            self._backward_mult(node, output_grad)
            
        elif node.op_type == 'div':
            self._backward_div(node, output_grad)
            
        # ============================================================
        # Matrix Operations
        # ============================================================
        
        elif node.op_type == 'matmul':
            self._backward_matmul(node, output_grad)
            
        # ============================================================
        # Activation Functions
        # ============================================================
        
        elif node.op_type == 'relu':
            self._backward_relu(node, output_grad)
            
        elif node.op_type == 'sigmoid':
            self._backward_sigmoid(node, output_grad)
            
        elif node.op_type == 'tanh':
            self._backward_tanh(node, output_grad)
            
        elif node.op_type == 'softmax':
            self._backward_softmax(node, output_grad)
            
        # ============================================================
        # Reduction Operations
        # ============================================================
        
        elif node.op_type == 'sum':
            self._backward_sum(node, output_grad)
            
        elif node.op_type == 'mean':
            self._backward_mean(node, output_grad)
            
        # ============================================================
        # Neural Network Layers
        # ============================================================
        
        elif node.op_type == 'linear':
            self._backward_linear(node, output_grad)
            
        elif node.op_type == 'layer_norm':
            self._backward_layer_norm(node, output_grad)
            
        # ============================================================
        # Loss Functions
        # ============================================================
        
        elif node.op_type == 'mse_loss':
            self._backward_mse_loss(node, output_grad)
            
        elif node.op_type == 'cross_entropy':
            self._backward_cross_entropy(node, output_grad)
            
        else:
            if self.debug_mode:
                self.tensorlang.print(message=f"[AUTOGRAD] Warning: No backward implemented for {node.op_type}")
    
    # ================================================================
    # Backward Pass Implementations
    # ================================================================
    
    def _accumulate_grad(self, tensor_name: str, grad: np.ndarray):
        """Accumulate gradient for a tensor."""
        # Initialize gradient storage if it doesn't exist
        if tensor_name not in self.gradients:
            if tensor_name not in self.tensors:
                # Tensor not registered - skip
                return
            self.gradients[tensor_name] = np.zeros_like(self.tensors[tensor_name])
            
        # Handle broadcasting: sum out dimensions that were broadcast
        input_shape = self.tensors[tensor_name].shape
        grad_shape = grad.shape
        
        if input_shape != grad_shape:
            grad = self._unbroadcast(grad, input_shape)
            
        self.gradients[tensor_name] += grad
    
    # def _unbroadcast(self, grad: np.ndarray, target_shape: Tuple) -> np.ndarray:
    #     """Reverse broadcasting by summing over broadcast dimensions."""
    #     # Sum over dimensions that were added
    #     ndim_diff = len(grad.shape) - len(target_shape)
    #     for _ in range(ndim_diff):
    #         grad = grad.sum(axis=0)
        
    #     # Sum over dimensions that were broadcast from 1 to n
    #     for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
    #         if target_dim == 1 and grad_dim != 1:
    #             grad = grad.sum(axis=i, keepdims=True)
                
    #     return grad
    

    def _unbroadcast(self, grad: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """
        Reverse broadcasting by summing over broadcast dimensions.
        
        During forward pass, NumPy broadcasting expands smaller tensors to match
        larger ones. During backward pass, we must sum gradients over the dimensions
        that were broadcast to recover the original shape.
        
        Broadcasting rules (NumPy):
        1. Prepend dimensions of size 1 to the smaller tensor
        2. Stretch dimensions of size 1 to match the larger tensor
        
        Unbroadcasting rules (this function):
        1. Sum over prepended dimensions (axis 0, repeatedly)
        2. Sum over stretched dimensions (keeping dimension as size 1)
        
        Args:
            grad: Gradient tensor with broadcast (expanded) shape
            target_shape: Original shape before broadcasting
            
        Returns:
            Gradient tensor with target shape
            
        Example:
            Forward:  (3,) + (2, 3) -> (2, 3)  [prepend then stretch]
            Backward: (2, 3) -> (3,)           [sum axis 0, then squeeze]
            
            Forward:  (3, 1, 5) + (3, 4, 5) -> (3, 4, 5)  [stretch middle dim]
            Backward: (3, 4, 5) -> (3, 1, 5)              [sum axis 1, keep dim]
        """
        
        if self.debug_mode:
            self.tensorlang.print(message=f"[AUTOGRAD] Unbroadcasting: {grad.shape} -> {target_shape}")
        
        # ================================================================
        # Step 0: Handle edge cases
        # ================================================================
        
        # If shapes already match, nothing to do
        if grad.shape == target_shape:
            return grad
        
        # If target is scalar (), convert to (1,) for easier handling
        if target_shape == ():
            target_shape = (1,)
            need_scalar_conversion = True
        else:
            need_scalar_conversion = False
        
        # ================================================================
        # Step 1: Sum over PREPENDED dimensions
        # ================================================================
        # During broadcasting, smaller tensors get dimensions prepended.
        # We need to sum these out.
        #
        # Example: (3,) -> (2, 3)
        #   Forward: prepend dimension -> (1, 3) -> broadcast -> (2, 3)
        #   Backward: sum axis 0 -> (3,)
        
        ndim_diff = len(grad.shape) - len(target_shape)
        
        if ndim_diff > 0:
            if self.debug_mode:
                self.tensorlang.print(message=f"[AUTOGRAD] Summing {ndim_diff} prepended dimensions")
            
            # Sum over leading dimensions (always axis 0, repeatedly)
            for _ in range(ndim_diff):
                grad = grad.sum(axis=0)
                if self.debug_mode:
                    self.tensorlang.print(message=f"[AUTOGRAD] -> After sum: {grad.shape}")
        
        elif ndim_diff < 0:
            # This should never happen in valid broadcasting
            raise ValueError(
                f"Invalid unbroadcast: grad shape {grad.shape} has fewer "
                f"dimensions than target {target_shape}"
            )
        
        # ================================================================
        # Step 2: Sum over STRETCHED dimensions (broadcast from 1 to n)
        # ================================================================
        # During broadcasting, dimensions of size 1 get stretched to size n.
        # We need to sum these back to size 1.
        #
        # Example: (3, 1, 5) -> (3, 4, 5)
        #   Forward: stretch middle dimension from 1 to 4
        #   Backward: sum axis 1 and keep dimension -> (3, 1, 5)
        #
        # CRITICAL: Must iterate in REVERSE order to avoid index shifting bugs!
        
        # After step 1, grad.shape and target_shape should have same ndim
        assert len(grad.shape) == len(target_shape), \
            f"After prepend sum, shapes should match in ndim: {grad.shape} vs {target_shape}"
        
        # Iterate in REVERSE to avoid axis index shifting
        for i in reversed(range(len(target_shape))):
            grad_dim = grad.shape[i]
            target_dim = target_shape[i]
            
            if target_dim == 1 and grad_dim != 1:
                # This dimension was broadcast from 1 to grad_dim
                if self.debug_mode:
                    self.tensorlang.print(message=f"[AUTOGRAD] Summing axis {i}: {grad_dim} -> 1")
                
                grad = grad.sum(axis=i, keepdims=True)
                
                if self.debug_mode:
                    self.tensorlang.print(message=f"[AUTOGRAD] -> After sum: {grad.shape}")
            
            elif target_dim != grad_dim:
                # Shapes don't match and it's not a broadcast from 1
                # This indicates a bug in the computation graph
                raise ValueError(
                    f"Unbroadcast error at axis {i}: grad has size {grad_dim}, "
                    f"target has size {target_dim}. This is not a valid broadcast pattern."
                )
        
        # ================================================================
        # Step 3: Handle scalar conversion
        # ================================================================
        if need_scalar_conversion:
            # Convert (1,) back to ()
            grad = grad.squeeze()
            target_shape = ()
        
        # ================================================================
        # Step 4: Final validation
        # ================================================================
        if grad.shape != target_shape:
            raise ValueError(
                f"Unbroadcast failed: expected shape {target_shape}, "
                f"got {grad.shape}. This indicates a bug in the unbroadcast logic."
            )
        
        if self.debug_mode:
            grad_norm = np.linalg.norm(grad)
            self.tensorlang.print(message=f"[AUTOGRAD] -> Unbroadcast complete: {grad.shape}, norm={grad_norm:.6f}")
        
        return grad


    def _backward_add(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of addition: passes gradient to both inputs."""
        a, b = node.inputs
        self._accumulate_grad(a, output_grad)
        self._accumulate_grad(b, output_grad)
    

    def _backward_minus(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of subtraction: +grad to first input, -grad to second."""
        a, b = node.inputs
        self._accumulate_grad(a, output_grad)
        self._accumulate_grad(b, -output_grad)
    

    def _backward_mult(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of multiplication: d(a*b)/da = b, d(a*b)/db = a."""
        a, b = node.inputs
        self._accumulate_grad(a, output_grad * self.tensors[b])
        self._accumulate_grad(b, output_grad * self.tensors[a])
    

    def _backward_div(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of division: d(a/b)/da = 1/b, d(a/b)/db = -a/b²."""
        a, b = node.inputs
        b_val = self.tensors[b]
        self._accumulate_grad(a, output_grad / b_val)
        self._accumulate_grad(b, -output_grad * self.tensors[a] / (b_val ** 2))
    

    def _backward_matmul(self, node: ComputationNode, output_grad: np.ndarray):
        """
        Gradient of matrix multiplication.
        If C = A @ B, then:
          dL/dA = dL/dC @ B.T
          dL/dB = A.T @ dL/dC
        """
        a, b = node.inputs
        a_val = self.tensors[a]
        b_val = self.tensors[b]
        
        # Gradient w.r.t. A
        grad_a = output_grad @ b_val.T
        self._accumulate_grad(a, grad_a)
        
        # Gradient w.r.t. B
        grad_b = a_val.T @ output_grad
        self._accumulate_grad(b, grad_b)
    

    def _backward_relu(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of ReLU: 1 where input > 0, else 0."""
        inp = node.inputs[0]
        mask = (self.tensors[inp] > 0).astype(np.float32)
        self._accumulate_grad(inp, output_grad * mask)
    

    def _backward_sigmoid(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of sigmoid: σ(x) * (1 - σ(x))."""
        inp = node.inputs[0]
        sigmoid_out = self.tensors[node.output_name]
        grad = output_grad * sigmoid_out * (1 - sigmoid_out)
        self._accumulate_grad(inp, grad)
    

    def _backward_tanh(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of tanh: 1 - tanh²(x)."""
        inp = node.inputs[0]
        tanh_out = self.tensors[node.output_name]
        grad = output_grad * (1 - tanh_out ** 2)
        self._accumulate_grad(inp, grad)
    

    def _backward_softmax(self, node: ComputationNode, output_grad: np.ndarray):
        """
        Gradient of softmax (simplified for cross-entropy combination).
        Full Jacobian is complex; this assumes typical usage with cross-entropy.
        """
        inp = node.inputs[0]
        softmax_out = self.tensors[node.output_name]
        
        # Simplified: for numerical stability with cross-entropy
        # Full derivative: S(i,j) = s_i * (δ_ij - s_j) where s = softmax
        axis = node.metadata.get('axis', -1)
        
        # For cross-entropy loss, this simplifies significantly
        grad = output_grad * softmax_out
        sum_grad = np.sum(grad, axis=axis, keepdims=True)
        grad = grad - softmax_out * sum_grad
        
        self._accumulate_grad(inp, grad)
    

    def _backward_sum(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of sum: broadcast gradient back to input shape."""
        inp = node.inputs[0]
        axis = node.metadata.get('axis')
        
        if axis is None:
            # Sum over all dimensions
            grad = np.full_like(self.tensors[inp], output_grad.item() if output_grad.size == 1 else output_grad)
        else:
            # Sum over specific axis
            grad = np.expand_dims(output_grad, axis=axis)
            grad = np.broadcast_to(grad, self.tensors[inp].shape)
            
        self._accumulate_grad(inp, grad)
    

    def _backward_mean(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of mean: broadcast gradient / count back to input."""
        inp = node.inputs[0]
        axis = node.metadata.get('axis')
        input_shape = self.tensors[inp].shape
        
        if axis is None:
            # Mean over all dimensions
            count = np.prod(input_shape)
            scalar_val = output_grad.item() if output_grad.size == 1 else output_grad
            grad = np.full_like(self.tensors[inp], scalar_val / count)
        else:
            # Mean over specific axis
            count = input_shape[axis]
            grad = np.expand_dims(output_grad / count, axis=axis)
            grad = np.broadcast_to(grad, input_shape)
            
        self._accumulate_grad(inp, grad)
    

    def _backward_linear(self, node: ComputationNode, output_grad: np.ndarray):
        """
        Gradient of linear layer: y = x @ w + b
        dy/dx = w.T
        dy/dw = x.T
        dy/db = 1
        """
        inp, weight, bias = node.inputs
        x = self.tensors[inp]
        w = self.tensors[weight]
        
        # Gradient w.r.t. input
        grad_x = output_grad @ w.T
        self._accumulate_grad(inp, grad_x)
        
        # Gradient w.r.t. weight
        if x.ndim == 1:
            grad_w = np.outer(x, output_grad)
        else:  # Batch processing
            grad_w = x.T @ output_grad
        self._accumulate_grad(weight, grad_w)
        
        # Gradient w.r.t. bias
        if output_grad.ndim == 1:
            grad_b = output_grad
        else:
            grad_b = output_grad.sum(axis=0)
        self._accumulate_grad(bias, grad_b)
    

    def _backward_layer_norm(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of layer normalization (simplified)."""
        inp = node.inputs[0]
        x = self.tensors[inp]
        axis = node.metadata.get('axis', -1)
        eps = node.metadata.get('eps', 1e-5)
        
        # Compute statistics
        mean = x.mean(axis=axis, keepdims=True)
        var = x.var(axis=axis, keepdims=True)
        std = np.sqrt(var + eps)
        
        # Normalized input
        x_norm = (x - mean) / std
        
        # Gradient computation
        N = x.shape[axis]
        grad_x_norm = output_grad
        grad_var = np.sum(grad_x_norm * (x - mean) * -0.5 * (var + eps) ** -1.5, 
                         axis=axis, keepdims=True)
        grad_mean = np.sum(grad_x_norm * -1 / std, axis=axis, keepdims=True) + \
                   grad_var * np.sum(-2 * (x - mean), axis=axis, keepdims=True) / N
        
        grad = grad_x_norm / std + grad_var * 2 * (x - mean) / N + grad_mean / N
        
        self._accumulate_grad(inp, grad)
    

    def _backward_mse_loss(self, node: ComputationNode, output_grad: np.ndarray):
        """Gradient of MSE loss: 2 * (pred - target) / n."""
        pred, target = node.inputs
        pred_val = self.tensors[pred]
        # Target typically doesn't require gradients
        target_val = self.tensors[target]
        
        n = pred_val.size
        scalar_grad = output_grad.item() if output_grad.size == 1 else output_grad
        grad = 2 * (pred_val - target_val) / n * scalar_grad
        
        self._accumulate_grad(pred, grad)

    
    def _backward_cross_entropy(self, node: ComputationNode, output_grad: np.ndarray):
        """
        Gradient of cross-entropy loss.
        When combined with softmax: grad = (softmax_output - target) / batch_size
        """
        pred, target = node.inputs
        pred_val = self.tensors[pred]  # Should be softmax output
        target_val = self.tensors[target]
        
        batch_size = pred_val.shape[0]
        scalar_grad = output_grad.item() if output_grad.size == 1 else output_grad
        grad = (pred_val - target_val) / batch_size * scalar_grad
        
        self._accumulate_grad(pred, grad)


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





# def test_autograd():
#     """Simple test of autograd system."""
#     print("Testing Autograd System\n" + "="*50)
    
#     # Create computation graph
#     graph = ComputationGraph(debug_mode=True)
    
#     # Simple example: y = x @ w, loss = sum(y)
#     x = np.array([[1.0, 2.0], [3.0, 4.0]])
#     w = np.array([[0.5, 0.5], [0.5, 0.5]])
    
#     # Register input tensors
#     graph.register_tensor('x', x, requires_grad=False)
#     graph.register_tensor('w', w, requires_grad=True)
    
#     # Forward pass: compute y = x @ w
#     y = x @ w
#     graph.register_tensor('y', y)
#     graph.add_operation('matmul', 'y', ['x', 'w'])
    
#     # Forward pass: compute loss = sum(y)
#     loss = y.sum()
#     graph.register_tensor('loss', np.array([loss]))
#     graph.add_operation('sum', 'loss', ['y'], {'axis': None})
    
#     # Backward pass
#     print("\n" + "="*50)
#     print("Starting backward pass...")
#     print("="*50)
#     graph.backward('loss')
    
#     print(f"\nGradient of w:\n{graph.gradients['w']}")
#     print(f"\nExpected: [[4, 4], [6, 6]]")
    
#     # Verify
#     expected = np.array([[4.0, 4.0], [6.0, 6.0]])
#     if np.allclose(graph.gradients['w'], expected):
#         print("\n✅ Test PASSED!")
#         return True
#     else:
#         print("\n❌ Test FAILED!")
#         print(f"Expected:\n{expected}")
#         print(f"Got:\n{graph.gradients['w']}")
        
#         # Debug info
#         print("\nDebug Info:")
#         print(f"y gradient: {graph.gradients.get('y', 'NOT SET')}")
#         print(f"loss gradient: {graph.gradients.get('loss', 'NOT SET')}")
#         return False
    

# if __name__ == '__main__':
#     test_autograd()