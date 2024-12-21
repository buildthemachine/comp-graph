import numpy as np
import torch
from graph import Node, Multiply, Transpose, MultiplyElementwise


def test1():
    X = np.arange(1,7).reshape(6,1).astype(float)
    A = np.identity(6).astype(float)

    X_tensor = torch.from_numpy(X)
    X_tensor.requires_grad = True
    A_tensor = torch.from_numpy(A)
    A_tensor.requires_grad = True
    loss_torch = X_tensor.t() @ A_tensor @ X_tensor
    loss_torch.backward()

    X_node = Node(X, None)
    A_node = Node(A, None)
    loss_node = Multiply(Multiply(Transpose(X_node), A_node), X_node)
    loss_node.backward()
    print(f"plpx: autograd is: ")
    print(X_node._grad)
    print(f"plpx: torch is: ")
    print(X_tensor.grad)

    print(f"plpa: autograd is: ")
    print(A_node._grad)
    print(f"plpx: torch is: ")
    print(A_tensor.grad)

def main():
    test1()

if __name__ == '__main__': 
    main()