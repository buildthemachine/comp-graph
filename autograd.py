"""
Testing code for autograd functions
"""

import numpy as np
import torch
from graph import NodeLeaf, Multiply, Transpose, Mean1D, Power


def test_xtax():
    print("Begin testing XTAX...")
    X = np.arange(1,7).reshape(6,1).astype(float)
    A = np.identity(6).astype(float)

    X_tensor = torch.from_numpy(X)
    X_tensor.requires_grad = True
    A_tensor = torch.from_numpy(A)
    A_tensor.requires_grad = True
    loss_torch = X_tensor.t() @ A_tensor @ X_tensor
    loss_torch.backward()

    X_node = NodeLeaf(X).requires_grad_()
    A_node = NodeLeaf(A).requires_grad_()
    loss_node = Multiply(Multiply(Transpose(X_node), A_node), X_node)
    loss_node.backward()
    print(f"plpx: autograd is: ")
    print(X_node.get_grad())
    print(f"plpx: torch is: ")
    print(X_tensor.grad)

    print(f"plpa: autograd is: ")
    print(A_node.get_grad())
    print(f"plpx: torch is: ")
    print(A_tensor.grad)

def test_Mean1D():
    X = np.arange(1,7).reshape(6,1).astype(float)
    # A = np.identity(6).astype(float)
    A = np.random.randn(6,6).astype(float)
    X_tensor = torch.from_numpy(X).requires_grad_()
    A_tensor = torch.from_numpy(A).requires_grad_()

    X_node = NodeLeaf(X).requires_grad_()
    A_node = NodeLeaf(A).requires_grad_()
    loss_node = Mean1D(Multiply(Transpose(X_node), A_node))
    loss_node.backward()

    loss_torch = torch.mean(X_tensor.t() @ A_tensor)
    loss_torch.backward()

    print("value: autograd is:")
    print(loss_node.val)
    print("value: torch is:")
    print(loss_torch.detach().item())

    print(f"plpx: autograd is: ")
    print(X_node.get_grad())
    print(f"plpx: torch is: ")
    print(X_tensor.grad)


def test_Power():
    power = 3
    X = np.arange(1,7).reshape(6,1).astype(float)
    # A = np.identity(6).astype(float)
    A = np.random.randn(6,6).astype(float)
    X_tensor = torch.from_numpy(X).requires_grad_()
    A_tensor = torch.from_numpy(A).requires_grad_()

    X_node = NodeLeaf(X).requires_grad_()
    A_node = NodeLeaf(A).requires_grad_()
    loss_node = Mean1D(Multiply(Transpose(X_node), A_node))
    loss_node = Power(loss_node, power)
    loss_node.backward()

    loss_torch = torch.mean(X_tensor.t() @ A_tensor) ** power
    loss_torch.backward()

    print("value: autograd is:")
    print(loss_node.val)
    print("value: torch is:")
    print(loss_torch.detach().item())

    print(f"plpx: autograd is: ")
    print(X_node.get_grad())
    print(f"plpx: torch is: ")
    print(X_tensor.grad)


def main():
    # test_xtax()
    test_Mean1D()
    # test_Power()

if __name__ == '__main__': 
    main()