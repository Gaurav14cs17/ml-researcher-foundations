# Gradients & Partial Derivatives

> **The compass of optimization - which way to go?**

![Gradient](./images/gradient.svg)

---

## 📖 What is a Gradient?

The gradient is a vector of all partial derivatives. It points in the direction of **steepest ascent**.

```
+---------------------------------------------------------+
|                                                         |
|   For f(x₁, x₂, ..., xₙ):                              |
|                                                         |
|                + ∂f/∂x₁ +                               |
|                | ∂f/∂x₂ |                               |
|   ∇f(x) =      |   ⋮    |                               |
|                | ∂f/∂xₙ |                               |
|                +        +                               |
|                                                         |
|   Size: n × 1 vector                                    |
|                                                         |
+---------------------------------------------------------+
```

---

## 🎯 Visual Intuition

```
         Mountain Surface (Loss Landscape)
         
              ↗ ∇f (gradient points UP)
             /
            •  You are here
           /|\
          / | \
         /  |  \
        ----+----  Valley (minimum)
        
   To minimize: Go OPPOSITE to gradient!
   
   x_new = x_old - α∇f(x_old)
           -----------------
           Gradient Descent!
```

---

## 📐 Step-by-Step Example

### Function: f(x, y) = x² + 2y²

**Step 1: Partial Derivatives**
```
∂f/∂x = 2x    (derivative treating y as constant)
∂f/∂y = 4y    (derivative treating x as constant)
```

**Step 2: Gradient Vector**
```
∇f(x,y) = [2x, 4y]ᵀ
```

**Step 3: Evaluate at Point (3, 2)**
```
∇f(3,2) = [2(3), 4(2)]ᵀ = [6, 8]ᵀ
```

**Step 4: Gradient Descent Step**
```
α = 0.1  (learning rate)

[x_new]   [3]         [6]   [3 - 0.6]   [2.4]
[y_new] = [2] - 0.1 × [8] = [2 - 0.8] = [1.2]
```

---

## 🌍 Where Gradients Are Used

| Application | How | Example |
|-------------|-----|---------|
| **Neural Networks** | Backpropagation computes ∇L | Training GPT |
| **Diffusion Models** | Score ∇log p(x) | Stable Diffusion |
| **Physics** | Force = -∇V (potential) | Molecular dynamics |
| **Economics** | Marginal utility = ∂U/∂x | Optimization |
| **Computer Graphics** | Surface normals | Shading |

---

## 💻 Code Examples

### PyTorch (Autograd)
```python
import torch

# Define parameters
x = torch.tensor([3.0, 2.0], requires_grad=True)

# Define function
def f(x):
    return x[0]**2 + 2*x[1]**2

# Compute gradient
loss = f(x)
loss.backward()

print(f"∇f = {x.grad}")  # tensor([6., 8.])
```

### NumPy (Manual)
```python
import numpy as np

def gradient_f(x, y):
    """Gradient of f(x,y) = x² + 2y²"""
    df_dx = 2 * x
    df_dy = 4 * y
    return np.array([df_dx, df_dy])

grad = gradient_f(3, 2)
print(f"∇f(3,2) = {grad}")  # [6, 8]
```

### JAX (Automatic)
```python
import jax
import jax.numpy as jnp

def f(x):
    return x[0]**2 + 2*x[1]**2

# Auto-compute gradient function
grad_f = jax.grad(f)

x = jnp.array([3.0, 2.0])
print(f"∇f = {grad_f(x)}")  # [6., 8.]
```

---

## ⚠️ Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Confusing ∇f direction | Gradient is ASCENT, not descent | Use **-**∇f |
| Wrong partial derivative | Forgot to treat others as constant | Check each variable |
| Not normalizing | Gradient can be huge | Clip or normalize |
| Ignoring numerical issues | Gradient vanishing/exploding | Use techniques like BatchNorm |

---

## 📊 Gradient Properties

| Property | Formula | Meaning |
|----------|---------|---------|
| **Linearity** | ∇(af + bg) = a∇f + b∇g | Gradients add |
| **Product Rule** | ∇(fg) = f∇g + g∇f | Chain rule |
| **Chain Rule** | ∇(f∘g) = (∇f)·(∇g) | Backprop! |
| **Zero at extrema** | ∇f(x*) = 0 | How we find optima |

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 🎥 | 3Blue1Brown - Gradient | [YouTube](https://www.youtube.com/watch?v=tIpKfDc295M) |
| 📖 | Deep Learning Book Ch.4 | [Link](https://www.deeplearningbook.org/) |
| 🇨🇳 | 知乎 - 梯度下降 | [知乎](https://zhuanlan.zhihu.com/p/25202034) |
| 🇨🇳 | B站 - 梯度讲解 | [B站](https://www.bilibili.com/video/BV1Vx411j7kT) |

---

---

➡️ [Next: Hessian](./hessian.md)
