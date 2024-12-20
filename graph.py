"""Proof-of-concept: Defining the computation graph for autograd applications
Use numpy.ndarrays as the underlying data structure. 
Build my own computation graphs for backprop"""

import numpy as np
from typing import List


class Node:
    def __init__(self, val: np.ndarray, parents: List=None):
        self.val = val
        self._grad = 0
        self.parents = parents if parents is not None else []

    def backward(self, upstream_grad: np.ndarray=1):
        """backward() is a DFS search algo"""
        self._grad += upstream_grad     # visit this node
        for parent, local_grad in self.parents:     # DFS
            if isinstance(local_grad, str) and local_grad == 'transpose':
                parent.backward(upstream_grad.T)
            else:
                if hasattr(upstream_grad, '__len__') and hasattr(local_grad, '__len__'):    # if both are matrices
                    parent.backward(upstream_grad @ local_grad)
                    # parent.backward(local_grad @ upstream_grad)
                else:   # If one is scalar, use ordinary multiplication
                    parent.backward(upstream_grad * local_grad)     

    def __repr__(self):
        print(f"Node {self.val}")


class Multiply(Node):
    def __init__(self, x: Node, y: Node):
        val = x.val @ y.val
        parents = [(x, y.val.T), (y, x.val.T)]    # Note the trans here
        super().__init__(val, parents)


class Transpose(Node):
    def __init__(self, x: Node):
        val = x.val.transpose()
        parents = [(x, 'transpose')]
        # parents = [(x, np.identity(val.shape[0]))]
        super().__init__(val, parents)


class ReLU(Node):
    def __init__(self, x: Node):
        val = x.val
        mask = (val > 0).astype(float)
        val = val * mask
        parents = [(x, mask)]
        super().__init__(val, parents)


class Add(Node):
    def __init__(self, x: Node, y: Node):
        val = x.val + y.val
        parents = [(x, 1), (y, 1)]
        super().__init__(val, parents)


class MultiplyElementwise(Node):
    def __init__(self, x: Node, multiplier: Node):
        val = multiplier.val * x.val
        parents = [(x, multiplier.val)]
        super().__init__(val, parents)


def test1():
    np.random.seed(42)
    X = np.arange(1,7).reshape(6,1)
    # A = np.random.randn(6,6)
    # A = np.ones((6,6))
    A = np.identity(6)
    
    loss = X.T@A@X   # X^TAX

    # Ground truth:
    plpx_gt = 2*A@X
    plpa_gt = X@X.T

    # Use autograd:
    X_node = Node(X, None)
    A_node = Node(A, None)
    loss_node = Multiply(Multiply(Transpose(X_node), A_node), X_node)
    loss_node.backward()
    

    print(f"loss ground truth is: {np.squeeze(loss):.3f}")
    print(f"loss computation graph is: {np.squeeze(loss_node.val):.3f}")
    
    print(f"plpx: ground truth is: ")
    print(plpx_gt)
    print(f"plpx: autograd is: ")
    print(X_node._grad)

    print(f"plpa: ground truth is: ")
    print(plpa_gt)
    print(f"plpa: autograd is: ")
    print(A_node._grad)


def test2():
    X = np.arange(1,7).reshape(6,1)
    Y = np.ones((6,1))
    loss = X.T@Y

    # Ground truth:
    plpx_gt = Y
    plpy_gt = X

    # Use autograd:
    X_node = Node(X, None)
    Y_node = Node(Y, None)
    loss_node = Multiply(Transpose(X_node), Y_node)
    loss_node.backward()
    

    print(f"loss ground truth is: {np.squeeze(loss):.3f}")
    print(f"loss computation graph is: {np.squeeze(loss_node.val):.3f}")
    
    print(f"plpx: ground truth is: ")
    print(plpx_gt)
    print(f"plpx: autograd is: ")
    print(X_node._grad)

    print(f"plpy: ground truth is: ")
    print(plpy_gt)
    print(f"plpy: autograd is: ")
    print(Y_node._grad)



def main():
    test1()
    # test2()




if __name__ == "__main__":
    main()