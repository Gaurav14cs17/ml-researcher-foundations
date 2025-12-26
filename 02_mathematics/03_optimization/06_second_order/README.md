<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Second-Order%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/second-order.svg" width="100%">

*Caption: Newton: θ ← θ - H⁻¹∇f uses Hessian curvature. Converges quadratically near optimum. Cost: O(n³) for Hessian inverse. Quasi-Newton (BFGS, L-BFGS) approximates H⁻¹ efficiently.*

---

## 📂 Overview

Second-order methods use curvature information to take smarter steps. While expensive for deep learning, they're standard for smaller problems and inspire adaptive learning rate methods like Adam.

---

## 📐 Mathematical Definitions

### Newton's Method
```
θₜ₊₁ = θₜ - H⁻¹∇f(θₜ)

Where H = ∇²f(θ) is the Hessian matrix

Convergence: Quadratic near optimum
             ||θₜ₊₁ - θ*|| ≤ c||θₜ - θ*||²

Cost: O(n³) for matrix inverse (expensive!)
```

### Hessian Properties
```
H = ∇²f(θ) = [∂²f/∂θᵢ∂θⱼ]

At minimum θ*:
• H ≻ 0 (positive definite)
• Eigenvalues = curvatures along eigenvectors
• Large eigenvalue = steep direction
• Small eigenvalue = flat direction
```

### Quasi-Newton Methods
```
Instead of computing H⁻¹, approximate it:
B_{t+1} ≈ H⁻¹

BFGS Update:
Bₜ₊₁ = (I - ρₜsₜyₜᵀ)Bₜ(I - ρₜyₜsₜᵀ) + ρₜsₜsₜᵀ

Where:
• sₜ = θₜ₊₁ - θₜ
• yₜ = ∇f(θₜ₊₁) - ∇f(θₜ)
• ρₜ = 1/(yₜᵀsₜ)
```

### L-BFGS (Limited Memory BFGS)
```
Store only last m (s, y) pairs
Memory: O(mn) instead of O(n²)
Used for large-scale optimization
```

### Natural Gradient
```
θₜ₊₁ = θₜ - η F⁻¹∇L(θₜ)

F = Fisher Information Matrix
F = E[∇log p(x|θ) ∇log p(x|θ)ᵀ]

Natural gradient accounts for parameter space geometry
TRPO/PPO use approximations of natural gradient
```

---

## 💻 Code Examples

```python
import numpy as np
from scipy.optimize import minimize

# L-BFGS-B for optimization
result = minimize(
    fun=loss_fn,
    x0=initial_params,
    method='L-BFGS-B',
    jac=grad_fn,  # Gradient function
    options={'maxiter': 100}
)
optimal_params = result.x

# Newton's method (small scale)
def newton_step(theta, grad, hessian):
    return theta - np.linalg.solve(hessian, grad)

# Hessian computation in PyTorch
import torch
from torch.autograd.functional import hessian

def loss_fn(params):
    return (params ** 2).sum()

H = hessian(loss_fn, torch.randn(3))

# Approximate second-order: Adam adapts per-parameter learning rate
# v_t ≈ diagonal of Hessian estimate
```

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Nocedal: Numerical Opt | [Book](https://www.springer.com/gp/book/9780387303031) |
| 📄 | Natural Gradient | [Paper](https://www.jmlr.org/papers/v3/amari02a.html) |
| 📖 | Boyd: Convex Opt Ch. 9 | [Book](https://web.stanford.edu/~boyd/cvxbook/) |
| 🇨🇳 | 牛顿法详解 | [知乎](https://zhuanlan.zhihu.com/p/25383715) |
| 🇨🇳 | L-BFGS原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88318597) |
| 🇨🇳 | 优化算法对比 | [B站](https://www.bilibili.com/video/BV1Vt411G7nq) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [First Order](../05_first_order/README.md) | [Optimization](../README.md) | [Stochastic](../07_stochastic/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
