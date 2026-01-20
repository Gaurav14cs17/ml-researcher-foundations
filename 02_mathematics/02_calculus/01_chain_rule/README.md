<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Chain%20Rule%20%26%20Backpropagation&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=32&desc=The%20Mathematical%20Engine%20Behind%20Deep%20Learning&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-02.01_Chain_Rule-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-Chain_Rule_Backprop_Autograd-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **The chain rule is why deep learning works.** It's how we compute gradients through millions of operations in neural networks, enabling training via backpropagation.

- 🔗 **Single Variable**: $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$

- 📊 **Multivariate**: $\frac{\partial L}{\partial x} = \sum_i \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x}$ (sum over all paths)

- 🔄 **Backprop**: Reverse-mode autodiff = efficient chain rule application

- ⚡ **PyTorch**: `loss.backward()` applies chain rule millions of times

---

## 📑 Table of Contents

1. [Single Variable Chain Rule](#1-single-variable-chain-rule)

2. [Multivariate Chain Rule](#2-multivariate-chain-rule)

3. [Proof of Chain Rule](#3-proof-of-chain-rule)

4. [Backpropagation](#4-backpropagation)

5. [Forward vs Backward Mode](#5-forward-vs-backward-mode)

6. [Computational Graphs](#6-computational-graphs)

7. [Code Implementation](#7-code-implementation)

8. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/backprop-graph.svg" width="100%">

```
+-----------------------------------------------------------------------------+
|                    CHAIN RULE IN NEURAL NETWORKS                             |
+-----------------------------------------------------------------------------+
|                                                                              |
|   FORWARD PASS (compute output):                                            |
|   -----------------------------                                              |
|   x --→ [W₁] --→ h₁ --→ [σ] --→ a₁ --→ [W₂] --→ ŷ --→ [Loss] --→ L        |
|                                                                              |
|   BACKWARD PASS (compute gradients via chain rule):                         |
|   ------------------------------------------------                          |
|   ∂L    ∂L   ∂ŷ    ∂L   ∂ŷ   ∂a₁   ∂L   ∂ŷ   ∂a₁  ∂h₁                    |
|   -- ← -- · -- ← -- · -- · -- ← -- · -- · -- · --                          |
|   ∂W₂   ∂ŷ   ∂W₂   ∂ŷ   ∂a₁  ∂W₁   ∂ŷ   ∂a₁  ∂h₁  ∂W₁                    |
|                                                                              |
|   KEY INSIGHT:                                                               |
|   ------------                                                               |
|   Gradient flows BACKWARD, multiplying local derivatives at each step       |
|   Each layer receives ∂L/∂(its output) and passes back ∂L/∂(its input)     |
|                                                                              |
+-----------------------------------------------------------------------------+

```

---

## 1. Single Variable Chain Rule

### 📌 Theorem

If $y = f(u)$ and $u = g(x)$, then:

```math
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = f'(g(x)) \cdot g'(x)

```

### 💡 Examples

**Example 1**: $y = \sin(x^2)$

```
Let u = x², y = sin(u)

dy/dx = dy/du · du/dx
      = cos(u) · 2x
      = cos(x²) · 2x
      = 2x·cos(x²)

```

**Example 2**: $y = e^{3x^2 + 2x}$

```
Let u = 3x² + 2x, y = eᵘ

dy/dx = eᵘ · (6x + 2)
      = (6x + 2)·e^(3x²+2x)

```

**Example 3**: Triple Composition $y = \ln(\cos(\sqrt{x}))$

```
Let v = √x, u = cos(v), y = ln(u)

dy/dx = dy/du · du/dv · dv/dx
      = (1/u) · (-sin(v)) · (1/2√x)
      = -sin(√x) / (2√x · cos(√x))
      = -tan(√x) / (2√x)

```

---

## 2. Multivariate Chain Rule

### 📌 Theorem

If $z = f(u, v)$ where $u = u(x, y)$ and $v = v(x, y)$:

```math
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}

```

### 📐 General Form

For $L = L(y_1, y_2, \ldots, y_m)$ where each $y_i = y_i(x_1, \ldots, x_n)$:

```math
\frac{\partial L}{\partial x_j} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_j}

```

**Matrix Form** (using Jacobians):

```math
\frac{\partial L}{\partial \mathbf{x}} = J_\mathbf{y}^T \frac{\partial L}{\partial \mathbf{y}}

```

### 💡 Example

```
z = x²y + xy³
u = x + 2y
v = x - y

Find ∂z/∂u and ∂z/∂v when expressed through (u,v).

First, solve for x,y in terms of u,v:
  u + v = 2x  →  x = (u+v)/2
  u - 2v = 3y →  y = (u-v)/3 (adjusting...)

Actually, easier to use inverse:
∂z/∂u = ∂z/∂x · ∂x/∂u + ∂z/∂y · ∂y/∂u

where:
  ∂z/∂x = 2xy + y³
  ∂z/∂y = x² + 3xy²

```

---

## 3. Proof of Chain Rule

### 🔍 Rigorous Proof (Single Variable)

```
Theorem: If g is differentiable at a and f is differentiable at g(a),
         then (f ∘ g)'(a) = f'(g(a)) · g'(a)

Proof:

Step 1: Define the difference quotient error
        For function f differentiable at u₀:
        
        f(u) - f(u₀) = f'(u₀)(u - u₀) + ε(u)(u - u₀)
        
        where ε(u) → 0 as u → u₀

Step 2: Apply to our composition
        Let u₀ = g(a), u = g(x)
        
        f(g(x)) - f(g(a)) = f'(g(a))(g(x) - g(a)) + ε(g(x))(g(x) - g(a))

Step 3: Divide by (x - a)
        f(g(x)) - f(g(a))     g(x) - g(a)                 g(x) - g(a)
        ----------------- = f'(g(a)) · ----------- + ε(g(x)) · -----------
             x - a                       x - a                    x - a

Step 4: Take limit as x → a
        • g(x) → g(a) by continuity (g differentiable ⟹ g continuous)
        • ε(g(x)) → ε(g(a)) = 0 (by definition of ε)
        • (g(x) - g(a))/(x - a) → g'(a)
        
        Therefore:
        (f ∘ g)'(a) = f'(g(a)) · g'(a) + 0 · g'(a) = f'(g(a)) · g'(a)  ∎

```

---

## 4. Backpropagation

### 📌 Core Algorithm

Backpropagation is the **efficient application of chain rule** to compute gradients in computational graphs.

```
FORWARD PASS:
-------------
For each node in topological order:
    Compute output from inputs
    Store values for backward pass

BACKWARD PASS:
--------------
Initialize: ∂L/∂L = 1

For each node in REVERSE topological order:
    Receive: ∂L/∂(output)  [gradient from downstream]
    
    For each input:
        Compute: ∂(output)/∂(input)  [local gradient]
        Accumulate: ∂L/∂(input) += ∂L/∂(output) · ∂(output)/∂(input)

```

### 🔍 Detailed Example: 2-Layer Network

```
Network: x → [W₁] → h → [ReLU] → a → [W₂] → ŷ → [MSE] → L

Forward:
  h = W₁x           (linear)
  a = max(0, h)     (ReLU)
  ŷ = W₂a           (linear)
  L = ‖ŷ - y‖²      (MSE loss)

Backward:
  ∂L/∂ŷ = 2(ŷ - y)                          [loss gradient]
  
  ∂L/∂W₂ = ∂L/∂ŷ · ∂ŷ/∂W₂ = 2(ŷ - y) · aᵀ  [W₂ gradient]
  ∂L/∂a = ∂L/∂ŷ · ∂ŷ/∂a = W₂ᵀ · 2(ŷ - y)   [pass to ReLU]
  
  ∂L/∂h = ∂L/∂a · ∂a/∂h = ∂L/∂a ⊙ 1[h>0]   [ReLU gradient]
  
  ∂L/∂W₁ = ∂L/∂h · ∂h/∂W₁ = ∂L/∂h · xᵀ     [W₁ gradient]

```

### 💻 Implementation from Scratch

```python
import numpy as np

class ComputationalGraph:
    """Simple autograd implementation."""
    
    def __init__(self):
        self.tape = []  # Record operations for backward
    
    def add(self, a, b):
        """z = a + b"""
        z = a + b
        def backward(grad_z):
            return grad_z, grad_z  # ∂z/∂a = ∂z/∂b = 1
        self.tape.append((backward, [a, b]))
        return z
    
    def mul(self, a, b):
        """z = a * b"""
        z = a * b
        def backward(grad_z):
            return grad_z * b, grad_z * a  # ∂z/∂a = b, ∂z/∂b = a
        self.tape.append((backward, [a, b]))
        return z
    
    def exp(self, x):
        """z = e^x"""
        z = np.exp(x)
        def backward(grad_z):
            return (grad_z * z,)  # ∂z/∂x = e^x = z
        self.tape.append((backward, [x]))
        return z
    
    def backward(self, grad_output=1.0):
        """Compute gradients via reverse-mode autodiff."""
        grads = {id(self.tape[-1][1][0]): grad_output}
        
        for backward_fn, inputs in reversed(self.tape):
            # Get gradient of output
            grad_out = grads.get(id(inputs[0]), 0)
            
            # Compute input gradients
            input_grads = backward_fn(grad_out)
            
            # Accumulate
            for inp, grad in zip(inputs, input_grads):
                if id(inp) in grads:
                    grads[id(inp)] += grad
                else:
                    grads[id(inp)] = grad
        
        return grads

```

---

## 5. Forward vs Backward Mode

### 📐 Comparison

```
Function f: ℝⁿ → ℝᵐ

FORWARD MODE (Tangent):
-----------------------
• Propagate: How does output change when input changes?
• Computes: One column of Jacobian per forward pass
• Cost: O(n) forward passes for full Jacobian
• Best when: n << m (few inputs, many outputs)

BACKWARD MODE (Adjoint):
------------------------
• Propagate: How does loss change when each intermediate changes?
• Computes: One row of Jacobian per backward pass
• Cost: O(m) backward passes for full Jacobian
• Best when: m << n (many inputs, few outputs)

DEEP LEARNING:
--------------
L: ℝᵐⁱˡˡⁱᵒⁿˢ → ℝ¹  (millions of params, one scalar loss)

Backward mode needs: 1 backward pass
Forward mode needs: millions of forward passes

→ Backward mode (backprop) is the only practical choice!

```

### 📊 Visual Comparison

```
Forward Mode:                    Backward Mode:
                                 
dx = 1 -→ ∂h₁/∂x -→ ∂h₂/∂x -→ ∂L/∂x    1 = ∂L/∂L ←- ∂L/∂h₂ ←- ∂L/∂h₁ ←- ∂L/∂x
                                 
"Push forward infinitesimal    "Pull back sensitivity
 perturbations"                  to loss"

```

---

## 6. Computational Graphs

### 📐 Graph Representation

```python
# Expression: L = (a * b + c)²

# Graph structure:
#     a --+
#         +-- mul --+
#     b --+         |
#                   +-- add -- square -- L
#     c ------------+

# Forward pass:
t1 = a * b       # mul node
t2 = t1 + c      # add node  
L = t2 ** 2      # square node

# Backward pass:
dL/dL = 1
dL/dt2 = 2*t2                    # d(x²)/dx = 2x
dL/dt1 = dL/dt2 * 1 = 2*t2       # d(x+c)/dx = 1
dL/dc = dL/dt2 * 1 = 2*t2        # d(x+c)/dc = 1
dL/da = dL/dt1 * b = 2*t2*b      # d(ab)/da = b
dL/db = dL/dt1 * a = 2*t2*a      # d(ab)/db = a

```

### 🔍 Handling Multiple Paths

```
When a variable is used multiple times, sum the gradients:

      +--→ f --+
x ----+        +--→ L
      +--→ g --+

∂L/∂x = ∂L/∂f · ∂f/∂x + ∂L/∂g · ∂g/∂x

This is why PyTorch accumulates gradients with +=

```

---

## 7. Code Implementation

### Complete PyTorch Example

```python
import torch
import torch.nn as nn

# Example 1: Basic chain rule verification
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2           # y = x²
z = torch.sin(y)     # z = sin(x²)

z.backward()

# Manual calculation:
# dz/dx = dz/dy · dy/dx = cos(y) · 2x = cos(4) · 4
manual_grad = torch.cos(x**2) * 2 * x
print(f"PyTorch grad: {x.grad.item():.6f}")
print(f"Manual grad:  {manual_grad.item():.6f}")

# Example 2: Neural network backprop
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

model = SimpleNet()
x = torch.randn(32, 10)
y = torch.randn(32, 1)

# Forward
pred = model(x)
loss = ((pred - y) ** 2).mean()

# Backward (chain rule applied automatically)
loss.backward()

# Gradients are now computed
print(f"∂L/∂W1 shape: {model.fc1.weight.grad.shape}")
print(f"∂L/∂W2 shape: {model.fc2.weight.grad.shape}")

# Example 3: Jacobian computation
def f(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1], torch.sin(x[0])])

x = torch.tensor([1.0, 2.0], requires_grad=True)

# Compute full Jacobian
from torch.autograd.functional import jacobian
J = jacobian(f, x)
print(f"Jacobian:\n{J}")
# [[2x₀, 1  ],
#  [x₁,  x₀ ],
#  [cos(x₀), 0]] = [[2, 1], [2, 1], [0.54, 0]]

```

### Gradient Checking

```python
def numerical_gradient(f, x, eps=1e-5):
    """Compute gradient numerically for verification."""
    grad = torch.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.clone()
        x_plus[i] += eps
        x_minus = x.clone()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Verify chain rule implementation
def test_function(x):
    return torch.sin(x ** 2).sum()

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = test_function(x)
y.backward()

analytical = x.grad
numerical = numerical_gradient(test_function, x.detach())

print(f"Analytical gradient: {analytical}")
print(f"Numerical gradient:  {numerical}")
print(f"Difference: {(analytical - numerical).abs().max():.2e}")

```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 🎥 | [3Blue1Brown: Backprop](https://www.youtube.com/watch?v=Ilg3gGewQ5U) | Visual explanation |
| 📄 | [Automatic Differentiation Survey](https://arxiv.org/abs/1502.05767) | Comprehensive overview |
| 📖 | Deep Learning Book Ch. 6 | Goodfellow et al. |
| 📖 | [CS231n Notes](https://cs231n.github.io/optimization-2/) | Stanford course |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Calculus Overview](../README.md) | [Mathematics](../../README.md) | [Derivatives](../02_derivatives/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
