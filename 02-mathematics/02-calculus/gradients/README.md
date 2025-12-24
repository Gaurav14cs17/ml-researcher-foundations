# Gradients

> **The direction of steepest ascent - foundation of optimization**

---

## 📐 Gradient Definition

```
For f: ℝⁿ → ℝ (scalar-valued function):

∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ ∈ ℝⁿ

Properties:
• Points in direction of steepest INCREASE
• Magnitude ‖∇f‖ = rate of maximum increase
• Perpendicular to level sets {x : f(x) = c}
• ∇f = 0 at local extrema (critical points)
```

### Directional Derivative

```
Rate of change of f in direction u (unit vector):

∂f/∂u = ∇f · u = ‖∇f‖ cos(θ)

Maximum when u = ∇f/‖∇f‖ (gradient direction)
Minimum when u = -∇f/‖∇f‖ (negative gradient)
Zero when u ⊥ ∇f (tangent to level set)
```

---

## 📐 Jacobian Matrix

<img src="./images/jacobian-matrix.svg" width="100%">

*Caption: The Jacobian matrix J maps changes in inputs to changes in outputs. Each row represents the gradient of one output function, and each column shows sensitivity to one input variable. This is fundamental for understanding how neural network layers transform data.*

```
For f: ℝⁿ → ℝᵐ (vector-valued function):

        + ∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ +
J(x) =  | ∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ |  ∈ ℝᵐˣⁿ
        |    ⋮        ⋮      ⋱      ⋮     |
        + ∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ +

Each row i = gradient of fᵢ (∇fᵢ)ᵀ
Each column j = sensitivity to xⱼ

Linear approximation:
f(x + δ) ≈ f(x) + J(x) · δ
```

---

## 📐 Hessian Matrix

```
For f: ℝⁿ → ℝ (scalar function):

        + ∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ +
H(x) =  | ∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ |  ∈ ℝⁿˣⁿ
        |     ⋮           ⋮        ⋱       ⋮      |
        + ∂²f/∂xₙ∂x₁  ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²   +

Properties:
• Symmetric (for smooth f): Hᵢⱼ = Hⱼᵢ (Schwarz's theorem)
• Curvature information: eigenvalues = curvature along eigenvectors
• Positive definite H ⟹ strict local minimum
• Negative definite H ⟹ strict local maximum
• Indefinite H ⟹ saddle point

Quadratic approximation (Taylor):
f(x + δ) ≈ f(x) + ∇f(x)ᵀδ + ½δᵀH(x)δ
```

---

## 🌍 ML Applications

| Concept | Application |
|---------|-------------|
| Gradient | Gradient descent: θ ← θ - α∇L(θ) |
| Jacobian | Normalizing flows (det J), backprop |
| Hessian | Newton's method, curvature analysis |
| ∇f = 0 | Finding optimal parameters |

### Gradient in Deep Learning

```
Loss function: L(θ) = 1/N Σᵢ loss(f(xᵢ; θ), yᵢ)

Training = finding θ* = argmin L(θ)
          ← follow -∇L(θ) direction

∇L(θ) computed via backpropagation (chain rule)
```

---

## 💻 Code Examples

```python
import torch
import numpy as np

# 1. Gradient of scalar function
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()  # y = x₁² + x₂² + x₃²
y.backward()
print(f"∇y = {x.grad}")  # [2, 4, 6] = 2x

# 2. Jacobian computation
from torch.autograd.functional import jacobian

def f(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]])

x = torch.tensor([1.0, 2.0])
J = jacobian(f, x)
print(f"Jacobian:\n{J}")
# [[2*x₀, 1  ],
#  [x₁,   x₀ ]] = [[2, 1], [2, 1]]

# 3. Hessian computation
from torch.autograd.functional import hessian

def f(x):
    return x[0]**3 + x[1]**3 + x[0]*x[1]

x = torch.tensor([1.0, 2.0])
H = hessian(f, x)
print(f"Hessian:\n{H}")
# [[6x₀, 1  ],
#  [1,   6x₁]] = [[6, 1], [1, 12]]

# 4. Gradient descent example
def loss_fn(x):
    return (x - 3)**2  # Minimum at x=3

x = torch.tensor([0.0], requires_grad=True)
lr = 0.1
for step in range(20):
    loss = loss_fn(x)
    loss.backward()
    with torch.no_grad():
        x -= lr * x.grad  # Gradient descent step
        x.grad.zero_()
print(f"Final x: {x.item():.4f}")  # ≈ 3.0
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 🎥 | 3Blue1Brown: Gradient | [YouTube](https://www.youtube.com/watch?v=tIpKfDc295M) |
| 🎥 | Khan Academy Calculus | [Khan](https://www.khanacademy.org/math/multivariable-calculus) |
| 📖 | Matrix Calculus | [Wikipedia](https://en.wikipedia.org/wiki/Matrix_calculus) |
| 🇨🇳 | 梯度与Jacobian详解 | [知乎](https://zhuanlan.zhihu.com/p/25202034) |
| 🇨🇳 | 多元微积分可视化 | [B站](https://www.bilibili.com/video/BV1Hb411w7FN) |
| 🇨🇳 | Hessian矩阵与优化 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88692979)

---

## 🔗 Where Gradients & Calculus Are Used

| Application | How It's Used |
|-------------|---------------|
| **Gradient Descent** | θ ← θ - α∇L(θ) update rule |
| **Backpropagation** | Chain rule computes ∂L/∂θ for all parameters |
| **Newton's Method** | Uses Hessian for second-order optimization |
| **Adam/AdamW** | Adaptive gradient scaling with momentum |
| **Normalizing Flows** | Jacobian determinant for density estimation |
| **Neural ODEs** | Gradient through ODE solver |
| **Physics-Informed NN** | PDE residuals as loss terms |

---


⬅️ [Back: Calculus](../)

---

⬅️ [Back: Derivatives](../derivatives/) | ➡️ [Next: Integration](../integration/)
