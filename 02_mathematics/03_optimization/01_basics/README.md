<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Optimization%20Basics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/optimization-basics.svg" width="100%">

*Caption: Optimization finds θ* = argmin f(θ). Minima: local vs global. Gradients point uphill; follow -∇f to descend. Critical points have ∇f=0. Saddle points are critical but not minima.*

---

## 📂 Overview

Optimization is the engine of ML - every trained model is the result of optimization. Understanding the landscape of loss functions helps design better training procedures.

---

## 📐 Mathematical Definitions

### Optimization Problem
```
minimize f(θ)  subject to g(θ) ≤ 0, h(θ) = 0

Where:
• f(θ): objective function (loss)
• g(θ): inequality constraints
• h(θ): equality constraints
• θ*: optimal solution
```

### Critical Points
```
∇f(θ*) = 0  (necessary for local optimum)

Types:
• Local minimum: f(θ*) ≤ f(θ) in neighborhood
• Global minimum: f(θ*) ≤ f(θ) everywhere
• Saddle point: ∇f = 0, not min/max
```

### Second-Order Conditions
```
At critical point θ* where ∇f(θ*) = 0:

• Hessian H = ∇²f(θ*)
• H ≻ 0 (positive definite) ⟹ local minimum
• H ≺ 0 (negative definite) ⟹ local maximum
• H indefinite ⟹ saddle point
```

### Gradient Descent
```
θₜ₊₁ = θₜ - α ∇f(θₜ)

• α: learning rate (step size)
• Converges to local minimum (for smooth, convex f)
• Rate: O(1/t) for convex, O(e^(-t)) for strongly convex
```

---

## 💻 Code Examples

```python
import torch
import numpy as np

# Simple gradient descent
def gradient_descent(f, grad_f, x0, lr=0.01, n_iters=100):
    x = x0.copy()
    for _ in range(n_iters):
        x = x - lr * grad_f(x)
    return x

# Example: minimize f(x) = x² + 2y²
f = lambda x: x[0]**2 + 2*x[1]**2
grad_f = lambda x: np.array([2*x[0], 4*x[1]])
x_opt = gradient_descent(f, grad_f, np.array([5.0, 3.0]))
# x_opt ≈ [0, 0]

# PyTorch autograd
x = torch.tensor([5.0, 3.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)
for _ in range(100):
    loss = x[0]**2 + 2*x[1]**2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Boyd: Convex Optimization | [Book](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 | Nocedal: Numerical Opt | [Book](https://www.springer.com/gp/book/9780387303031) |
| 🎓 | Stanford EE364A | [Course](https://web.stanford.edu/class/ee364a/) |
| 🇨🇳 | 优化基础 | [知乎](https://zhuanlan.zhihu.com/p/25383715) |
| 🇨🇳 | 梯度下降详解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88318597) |
| 🇨🇳 | 凸优化课程 | [B站](https://www.bilibili.com/video/BV1Vt411G7nq) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Optimization Overview](../README.md) | [Mathematics](../../README.md) | [Constrained](../02_constrained/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
