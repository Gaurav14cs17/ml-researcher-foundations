<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Automatic%20Differentiation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/autodiff.svg" width="100%">

*Caption: Automatic differentiation computes exact gradients efficiently. Forward mode propagates derivatives input→output (good for few inputs). Reverse mode (backprop) propagates output→input (good for scalar loss, many params).*

---

## 📂 Overview

Automatic differentiation is the backbone of deep learning frameworks. It computes exact gradients (not numerical approximations) efficiently by applying the chain rule systematically.

---

## 📐 Three Types of Differentiation

### 1. Symbolic Differentiation

```
Input: f(x) = x² + sin(x)
Output: f'(x) = 2x + cos(x)

Pros: Exact formula
Cons: Expression swell (derivatives get complex)
      Can't handle control flow (if, loops)

```

### 2. Numerical Differentiation

```
f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

Pros: Easy to implement
Cons: O(n) function evaluations for n parameters
      Numerical instability (ε too large or small)

```

### 3. Automatic Differentiation ✓

```
Decompose function into elementary operations
Apply chain rule at each step

Pros: Exact (machine precision)
      Efficient: O(1) cost per parameter
      Handles control flow

```

---

## 📐 Forward Mode vs Reverse Mode

### Forward Mode (Tangent Mode)

```
Compute ∂output/∂input for ONE input at a time

Chain rule (forward):
    ∂f/∂x = ∂f/∂z₁ · ∂z₁/∂x + ∂f/∂z₂ · ∂z₂/∂x + ...
    
Propagate: input → output
Cost: O(n) for n inputs, O(1) for each

Good when: #inputs << #outputs (Jacobian row by row)

```

### Reverse Mode (Adjoint Mode) ✓ Used in DL

```
Compute ∂output/∂ALL_inputs simultaneously

Chain rule (backward):
    ∂L/∂x = Σᵢ ∂L/∂zᵢ · ∂zᵢ/∂x
    
Propagate: output → input
Cost: O(m) for m outputs, O(1) for each

Good when: #outputs << #inputs (Jacobian column by column)
          Neural networks: 1 scalar loss, millions of params!

```

### Comparison

| Aspect | Forward Mode | Reverse Mode |
|--------|--------------|--------------|
| Direction | Input → Output | Output → Input |
| Per-pass | One input derivative | All input derivatives |
| Memory | O(1) | O(depth) |
| Best for | Few inputs | Few outputs |
| DL use | Rare | **Standard (backprop)** |

---

## 📊 Backpropagation = Reverse Mode AD

```
Forward pass: Compute and store intermediate values
    z₁ = Wx + b
    z₂ = relu(z₁)
    z₃ = Vz₂ + c
    L = loss(z₃, y)

Backward pass: Propagate gradients
    ∂L/∂z₃ = ∂loss/∂z₃
    ∂L/∂V = ∂L/∂z₃ · z₂ᵀ
    ∂L/∂z₂ = Vᵀ · ∂L/∂z₃
    ∂L/∂z₁ = ∂L/∂z₂ ⊙ relu'(z₁)
    ∂L/∂W = ∂L/∂z₁ · xᵀ
    ∂L/∂x = Wᵀ · ∂L/∂z₁

```

---

## 💻 Code Examples

### Simple Autodiff Implementation

```python
import numpy as np

class Var:
    """Variable with automatic differentiation"""
    
    def __init__(self, value, children=(), backward_fn=None):
        self.value = value
        self.grad = 0.0
        self.children = children
        self.backward_fn = backward_fn
    
    def __add__(self, other):
        out = Var(self.value + other.value, (self, other))
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward_fn = backward
        return out
    
    def __mul__(self, other):
        out = Var(self.value * other.value, (self, other))
        def backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out.backward_fn = backward
        return out
    
    def __pow__(self, n):
        out = Var(self.value ** n, (self,))
        def backward():
            self.grad += n * (self.value ** (n-1)) * out.grad
        out.backward_fn = backward
        return out
    
    def backward(self):
        """Topological sort + reverse accumulation"""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for v in reversed(topo):
            if v.backward_fn:
                v.backward_fn()

# Example: f(x, y) = x²y + y³
x = Var(2.0)
y = Var(3.0)
z = x**2 * y + y**3

z.backward()
print(f"z = {z.value}")       # 2²×3 + 3³ = 31
print(f"∂z/∂x = {x.grad}")    # 2xy = 12
print(f"∂z/∂y = {y.grad}")    # x² + 3y² = 31

```

### PyTorch Autograd

```python
import torch

# Create tensors with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Forward pass
z = x**2 * y + y**3

# Backward pass
z.backward()

print(f"z = {z.item()}")
print(f"∂z/∂x = {x.grad.item()}")  # 2xy = 12
print(f"∂z/∂y = {y.grad.item()}")  # x² + 3y² = 31

```

### Custom Autograd Function

```python
import torch
from torch.autograd import Function

class CustomReLU(Function):
    """Custom ReLU with explicit forward and backward"""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage
custom_relu = CustomReLU.apply
x = torch.randn(10, requires_grad=True)
y = custom_relu(x)
y.sum().backward()

```

### JAX Functional Autodiff

```python
import jax
import jax.numpy as jnp

def f(x, y):
    return x**2 * y + y**3

# Get gradient function
grad_f = jax.grad(f, argnums=(0, 1))  # Gradients w.r.t. both args

x, y = 2.0, 3.0
dx, dy = grad_f(x, y)
print(f"∂f/∂x = {dx}, ∂f/∂y = {dy}")

# Jacobian
def g(x):
    return jnp.array([x[0]**2, x[0]*x[1], x[1]**3])

jacobian = jax.jacobian(g)
print(jacobian(jnp.array([2.0, 3.0])))

# Hessian
hessian = jax.hessian(f)
print(hessian(2.0, 3.0))

```

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **Tape** | Record of operations for backward pass |
| **Gradient accumulation** | Gradients add when variable used multiple times |
| **Detach** | Stop gradient propagation |
| **no_grad** | Context for inference (no tape recording) |
| **Checkpointing** | Trade memory for compute |

---

## 🔗 Connection to Deep Learning

```
Autodiff (reverse mode)
    |
    +-- Backpropagation (specific to neural nets)
    |   +-- Chain rule through layers
    |   +-- Efficient gradient computation
    |
    +-- Frameworks
    |   +-- PyTorch (dynamic graph)
    |   +-- TensorFlow (static/eager)
    |   +-- JAX (functional transforms)
    |
    +-- Extensions
        +-- Higher-order derivatives
        +-- Jacobian/Hessian computation
        +-- Differentiable programming

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Computational Graph | [../computational-graph/](../computational-graph/) |
| 📖 | Gradient Flow | [../gradient-flow/](../gradient-flow/) |
| 📄 | Automatic Differentiation Survey | [arXiv](https://arxiv.org/abs/1502.05767) |
| 🎥 | Karpathy: micrograd | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| 💻 | micrograd repo | [GitHub](https://github.com/karpathy/micrograd) |
| 🇨🇳 | 自动微分原理详解 | [知乎](https://zhuanlan.zhihu.com/p/61287482) |
| 🇨🇳 | PyTorch Autograd源码 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88661776) |
| 🇨🇳 | 手写自动微分 | [B站](https://www.bilibili.com/video/BV1Le4y1s7HH) |
| 🇨🇳 | JAX自动微分 | [机器之心](https://www.jiqizhixin.com/articles/2020-08-04-3)

## 🔗 Where This Topic Is Used

| Framework | Autodiff |
|-----------|---------|
| **PyTorch** | Autograd dynamic graph |
| **TensorFlow** | GradientTape |
| **JAX** | Functional transforms |
| **All Training** | Gradient computation |

---

⬅️ [Back: Backpropagation](../../README.md) | ➡️ [Next: Computational Graph](../02_computational_graph/README.md)

---

⬅️ [Back: Backpropagation](../../README.md)

---

➡️ [Next: Computational Graph](../02_computational_graph/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
