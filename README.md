# Computational graphs for autograd

This repository contains Python POC implementations of autograd with a computational graph. The key ideas are: 
- Tensors implementated as numpy arrays
- Node dependencies implemented as directed acyclic graphs (DAG), where each node is visited via depth first search to keep its topological ordering

## Backpropagation math

A good reference for backpro is CS231 lecture notes in Apr 2018. However even in there, the matrix formulation of many operations are not derived in sufficient details. Here I've included a few details that I found useful in the implementation. 
- Note here that Einstein summation convention is used, where $C_{ij}=A_{ik}B_{kj}$ implies $\sum_kA_{ik}B_{kj}$. 

### Multiplication

Assume $A$ and $B$ are matrices and the final loss function is: 

$loss=L(C)=L(A\cdot B)$

Our goal here is to derive the matrix derivatives $\frac{\partial L}{\partial A}$ and $\frac{\partial L}{\partial B}$, where each one of them is a matrix. Because $C_{ij}=A_{ik}B_{kj}$, use the chain rule we have: 

$\Big(\frac{\partial L}{\partial A}\Big)_{ik}=\Big(\frac{\partial L}{\partial C}\Big)_{ij}\cdot \Big(\frac{\partial L}{\partial B}\Big)_{kj}= \Big(\frac{\partial L}{\partial C}\Big)_{ij}\cdot \Big(\frac{\partial L}{\partial B}\Big)^\top_{jk}=\frac{\partial L}{\partial C}\cdot \Big(\frac{\partial L}{\partial B}\Big)^\top$

Here the product sign $\cdot$ in first two lines are scalar multiplication, while in the last line is matrix multiplication. 

In a similar fashion, one can derive the partial derivative with respect to $B$: 

$\Big(\frac{\partial L}{\partial B}\Big)_{kj}=\Big(\frac{\partial L}{\partial C}\Big)_{ij}\cdot \Big(\frac{\partial L}{\partial A}\Big)_{ik}= \Big(\frac{\partial L}{\partial A}\Big)^\top_{ki} \cdot \Big(\frac{\partial L}{\partial C}\Big)_{ij}=\Big(\frac{\partial L}{\partial A}\Big)^\top\cdot \frac{\partial L}{\partial C}$

**It is important to note that the partial derivative with respect ot $A$ and $B$ are different. With respect to $A$, the upstream gradient $\partial L/\partial C$ should go first with the transposed local gradient $\big(\partial L/\partial B\big)^\top $ go second. On the other hand, with respect to $B$, the transposed local gradient $\big(\partial L/\partial A\big)^\top$ should go first, while the upstream graident $\partial L/\partial C$ second.**

