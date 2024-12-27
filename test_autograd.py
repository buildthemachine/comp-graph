"""
Testing code for graph.py functions
Author:     Yufei Shen
Date:       12/25/2024
"""

import logging
import numpy as np
import torch
from graph import NodeLeaf, Multiply, Transpose, Mean1D, Power
import unittest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TestXTAX(unittest.TestCase):
    """
    Test the gradient calculations for X^\top AX is corect
    """
    def setUp(self):
        logger.info("Begin testing XTAX...")
        X = np.arange(1,7).reshape(6,1).astype(float)
        A = np.identity(6).astype(float)

        self.X_tensor = torch.from_numpy(X)
        self.X_tensor.requires_grad = True
        self.A_tensor = torch.from_numpy(A)
        self.A_tensor.requires_grad = True
        loss_torch = self.X_tensor.t() @ self.A_tensor @ self.X_tensor
        loss_torch.backward()

        self.X_node = NodeLeaf(X).requires_grad_()
        self.A_node = NodeLeaf(A).requires_grad_()
        loss_node = Multiply(Multiply(Transpose(self.X_node), self.A_node), self.X_node)
        loss_node.backward()

    def test_equal(self):
        X_diff = np.fabs(self.X_tensor.grad.numpy() - self.X_node.get_grad())
        self.assertAlmostEqual(np.sum(X_diff), 0)
        A_diff = np.fabs(self.A_tensor.grad.numpy() - self.A_node.get_grad())
        self.assertAlmostEqual(np.sum(A_diff), 0)


class TestMean1D(unittest.TestCase):
    """
    Test the backprop in 'mean' operations
    """
    def setUp(self):
        X = np.arange(1,7).reshape(6,1).astype(float)
        # A = np.identity(6).astype(float)
        A = np.random.randn(6,6).astype(float)
        self.X_tensor = torch.from_numpy(X).requires_grad_()
        self.A_tensor = torch.from_numpy(A).requires_grad_()

        self.X_node = NodeLeaf(X).requires_grad_()
        self.A_node = NodeLeaf(A).requires_grad_()
        self.loss_node = Mean1D(Multiply(Transpose(self.X_node), self.A_node))
        self.loss_node.backward()

        self.loss_torch = torch.mean(self.X_tensor.t() @ self.A_tensor)
        self.loss_torch.backward()

    def test_equal(self):
        self.assertAlmostEqual(self.loss_node.val, self.loss_torch.detach().item())
        X_diff = np.fabs(self.X_node.get_grad() - self.X_tensor.grad.detach().numpy())
        self.assertAlmostEqual(np.sum(X_diff), 0)


class TestPower(unittest.TestCase):
    def setUp(self):
        power = 3
        X = np.arange(1,7).reshape(6,1).astype(float)
        A = np.random.randn(6,6).astype(float)
        self.X_tensor = torch.from_numpy(X).requires_grad_()
        self.A_tensor = torch.from_numpy(A).requires_grad_()

        self.X_node = NodeLeaf(X).requires_grad_()
        self.A_node = NodeLeaf(A).requires_grad_()
        self.loss_node = Mean1D(Multiply(Transpose(self.X_node), self.A_node))
        self.loss_node = Power(self.loss_node, power)
        self.loss_node.backward()

        self.loss_torch = torch.mean(self.X_tensor.t() @ self.A_tensor) ** power
        self.loss_torch.backward()

    def test_equal(self):
        self.assertAlmostEqual(self.loss_node.val, self.loss_torch.detach().item())
        X_diff = np.fabs(self.X_node.get_grad() - self.X_tensor.grad.detach().numpy())
        self.assertAlmostEqual(np.sum(X_diff), 0)


if __name__ == '__main__': 
    unittest.main()