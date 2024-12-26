"""Proof-of-concept: Defining the computation graph for autograd applications
Use numpy.ndarrays as the underlying data structure. 
Build my own computation graphs for automatic backprop"""

import numpy as np
from typing import List


def smart_np_multiply(A: np.ndarray, B: np.ndarray):
    """If both A and B are numpy matrices, use matrix multiplication
    Otherwise, use scalar multiplication"""
    if np.isscalar(A) or np.isscalar(B):
        return A * B
    else:
        return A @ B
    

class NodeBase:
    def __init__(self, val: np.ndarray, parents: List=None, requires_grad: bool=False):
        self.val = val
        self.requires_grad = requires_grad
        self.parents = parents if parents is not None else []
        if self.requires_grad:
            self._grad = 0

    def backward(self, upstream_grad: np.ndarray):
        raise NotImplementedError

    def __repr__(self):
        print(f"Node {self.val}, requires_grad={self.requires_grad}")

    def get_grad(self):
        return self._grad

    def requires_grad_(self, requires_grad: bool=True):
        self.requires_grad = requires_grad
        self._grad = 0
        return self


class NodeLeaf(NodeBase):
    def __init__(self, val: np.ndarray):
        # parents = None given leaf nodes (here paradoxically children is referred to as parents)
        super().__init__(val, None)

    def backward(self, upstream_grad: np.ndarray=1):
        if self.requires_grad:
            self._grad += upstream_grad


class NodeUnary(NodeBase):
    def __init__(self, val: np.ndarray, parents: List=None):
        """Base class for a unary operator
        Unary operators involves only one input. Examples include: transpose, ReLU, Sigmoid, etc."""
        assert len(parents) == 1, "Unary operator has # of parents other than 2!"
        parent_node = parents[0][0]
        requires_grad = parent_node.requires_grad
        super().__init__(val, parents, requires_grad)

    def backward(self, upstream_grad: np.ndarray=1):
        """backward() is a DFS search algo
        This is the base class backward definition: may need to override"""
        if self.requires_grad:
            self._grad += upstream_grad     # visit this node
            for parent, local_grad in self.parents:     # DFS
                if isinstance(local_grad, str) and local_grad == 'transpose':
                    parent.backward(upstream_grad.T)
                else:
                    parent.backward(upstream_grad * local_grad) # Elementwise operation        


class NodeBinary(NodeBase):
    """Base class for a binary operator
    Binary operators involves two inputs. Examples include: multiply, addition, etc."""
    def __init__(self, val: np.ndarray, parents: List=None):
        assert len(parents) == 2, "Binary operator has # of parents other than 2!"
        requires_grad = parents[0][0].requires_grad and parents[1][0].requires_grad
        super().__init__(val, parents, requires_grad)

    def backward(self, upstream_grad: np.ndarray=1):
        raise NotImplementedError("The backward method for binary nodes should be overridden")


class NodeReduction(NodeBase):
    """Base class for reduction operations incuding: 
    summation, average, etc."""
    def __init__(self, val, parents = None):
        assert len(parents) == 1
        requires_grad = parents[0][0].requires_grad
        super().__init__(val, parents, requires_grad)

    def backward(self, upstream_grad: np.ndarray):
        raise NotImplementedError
    

class Summation1D(NodeReduction):
    def __init__(self, x: NodeBase, parents=None):
        val = np.sum(x.val)
        parents = [(x, None)]
        super().__init__(val, parents)

    def backward(self, upstream_grad: float=1.):
        '''Assumes the gradient from upward stream is a float scalar
        d(sum(x))/dx_i = 1'''
        if self.requires_grad:
            self._grad = upstream_grad
            for parent, local_grad in self.parents:
                # B/c Summation1D is straightforward, it is possible local_grad has not been set
                if local_grad is None:
                    local_grad = np.ones_like(self.val)
                new_grad = upstream_grad * local_grad
                parent.backward(new_grad)


class Mean1D(NodeReduction):
    def __init__(self, x: NodeBase, parents=None):
        val = np.mean(x.val)
        parents = [(x, None)]
        super().__init__(val, parents)

    def backward(self, upstream_grad: float=1.):
        '''Assumes the gradient from upward stream is a float scalar
        d(sum(x))/dx_i = 1/N'''
        if self.requires_grad:
            n = self.parents[0][0].val.size     # Equivalent to np.prod(a.shape)
            for parent, local_grad in self.parents:
                # Mean1D has local_grad initialized to None:
                if local_grad is None:
                    local_grad = np.ones_like(parent.val) / n
                new_grad = upstream_grad * local_grad
                parent.backward(new_grad)


class Multiply(NodeBinary):
    def __init__(self, x: NodeBase, y: NodeBase):
        val = x.val @ y.val
        parents = [(x, y.val.T), (y, x.val.T)]    # Note the trans here
        super().__init__(val, parents)

    def backward(self, upstream_grad: np.ndarray=1):
        """Override base class method:
        The 1st and 2nd parent has to be dealt with differently!!!"""
        if self.requires_grad:
            self._grad += upstream_grad
            p1, g1 = self.parents[0]
            p2, g2 = self.parents[1]
            p1.backward(smart_np_multiply(upstream_grad, g1))
            p2.backward(smart_np_multiply(g2, upstream_grad))


class Transpose(NodeUnary):
    def __init__(self, x: NodeBase):
        val = x.val.transpose()
        parents = [(x, 'transpose')]
        # parents = [(x, np.identity(val.shape[0]))]
        super().__init__(val, parents)


class ReLU(NodeUnary):
    def __init__(self, x: NodeBase):
        val = x.val
        mask = (val > 0).astype(float)
        val = val * mask
        parents = [(x, mask)]
        super().__init__(val, parents)


class Sigmoid(NodeUnary):
    def __init__(self, x: NodeBase):
        x_val = x.val
        val = np.exp(x_val) / (1+np.exp(x_val))
        local_grad = val*(1-val)
        parents = [(x, local_grad)]
        super().__init__(val, parents)


class Power(NodeUnary):
    #TODO: to implement tomorrow on  12/24
    def __init__(self, x: NodeBase, power: float):
        x_val = x.val ** power
        local_grad = power * x.val ** (power-1)
        parents = [(x, local_grad)]
        super().__init__(x_val, parents)


class Add(NodeBinary):
    """May use backward function from base"""
    def __init__(self, x: NodeBase, y: NodeBase):
        val = x.val + y.val
        parents = [(x, 1), (y, 1)]
        super().__init__(val, parents)


class MultiplyElementwise(NodeBinary):
    def __init__(self, x: NodeBase, multiplier: NodeBase):
        val = multiplier.val * x.val
        parents = [(x, multiplier.val)]
        super().__init__(val, parents)
