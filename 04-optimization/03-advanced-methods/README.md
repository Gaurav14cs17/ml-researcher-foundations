<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=03 Advanced Methods&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📂 Advanced Optimization Methods

> **Beyond gradient descent: Quasi-Newton and advanced techniques**

---

## 🎯 Visual Overview

<img src="./images/advanced-methods.svg" width="100%">

*Caption: Advanced optimization methods include Natural Gradient (invariant updates), Proximal Methods (non-smooth regularizers), and research frontiers like SAM, LAMB, Lion, and Sophia.*

---

## 📂 Overview

Advanced methods go beyond basic gradient descent by:
- Using curvature information (Quasi-Newton)
- Exploiting conjugate directions (CG)
- Handling non-smooth objectives (Proximal)
- Achieving scale invariance (Natural Gradient)

---

## 📁 Topics

| File | Topic | When to Use |
|------|-------|-------------|
| [quasi-newton.md](./quasi-newton.md) | BFGS, L-BFGS | Medium-scale problems |
| [conjugate-gradient.md](./conjugate-gradient.md) | CG Method | Large linear systems |

---

## 📐 Mathematical Overview

### Quasi-Newton (BFGS)

```
Newton's Method:
    θ_{t+1} = θ_t - H⁻¹ ∇f(θ_t)
    
    H = ∇²f(θ)  ← Expensive to compute!

Quasi-Newton Idea:
    Approximate H⁻¹ directly using gradient history

BFGS Update:
    B_{t+1} = B_t + (y_t y_t^T)/(y_t^T s_t) - (B_t s_t s_t^T B_t)/(s_t^T B_t s_t)
    
    where:
    s_t = θ_{t+1} - θ_t
    y_t = ∇f(θ_{t+1}) - ∇f(θ_t)
```

### L-BFGS (Limited Memory)

```
Problem: BFGS stores full n×n matrix

L-BFGS: Store only last m pairs (s_i, y_i)
        Compute H⁻¹∇f implicitly
        Memory: O(mn) instead of O(n²)
        
Typical: m = 10-20 pairs
```

### Conjugate Gradient

```
For quadratic: f(x) = ½x^T A x - b^T x

CG finds optimal direction conjugate to previous:
    d_t^T A d_{t-1} = 0  (A-conjugate)

Converges in at most n iterations (exactly for quadratic)
Much faster than steepest descent for ill-conditioned A
```

---

## 📊 Method Comparison

| Method | Convergence | Memory | Per-iteration Cost |
|--------|-------------|--------|-------------------|
| **GD** | O(1/t) | O(n) | O(n) |
| **Momentum** | O(1/t²) | O(n) | O(n) |
| **BFGS** | Superlinear | O(n²) | O(n²) |
| **L-BFGS** | Superlinear | O(mn) | O(mn) |
| **CG** | O(√κ) | O(n) | O(n) |
| **Newton** | Quadratic | O(n²) | O(n³) |

```
κ = condition number = λ_max / λ_min
Larger κ = harder optimization
```

---

## 💻 Code Examples

### L-BFGS in PyTorch

```python
import torch
import torch.optim as optim

# Model and loss
model = MyModel()
criterion = torch.nn.MSELoss()

# L-BFGS optimizer
optimizer = optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=10  # m = 10 pairs
)

# Training with L-BFGS (closure required!)
def closure():
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    return loss

for epoch in range(100):
    loss = optimizer.step(closure)
    print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
```

### Scipy L-BFGS-B

```python
from scipy.optimize import minimize
import numpy as np

def rosenbrock(x):
    """Rosenbrock function"""
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rosenbrock_grad(x):
    """Gradient of Rosenbrock"""
    grad = np.zeros_like(x)
    grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
    grad[1:] += 200 * (x[1:] - x[:-1]**2)
    return grad

# Optimize
x0 = np.array([-1.0, 1.0, 0.0, 2.0])
result = minimize(
    rosenbrock,
    x0,
    method='L-BFGS-B',
    jac=rosenbrock_grad,
    options={'disp': True, 'maxiter': 100}
)
print(f"Minimum: {result.x}")
```

### Conjugate Gradient

```python
import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iter=None):
    """
    Solve Ax = b using Conjugate Gradient
    A must be positive definite
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    if max_iter is None:
        max_iter = n
    
    r = b - A @ x  # Residual
    d = r.copy()   # Direction
    r_norm_sq = r @ r
    
    for i in range(max_iter):
        if np.sqrt(r_norm_sq) < tol:
            break
        
        Ad = A @ d
        alpha = r_norm_sq / (d @ Ad)
        
        x = x + alpha * d
        r = r - alpha * Ad
        
        r_norm_sq_new = r @ r
        beta = r_norm_sq_new / r_norm_sq
        r_norm_sq = r_norm_sq_new
        
        d = r + beta * d
    
    return x, i+1

# Example
A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
x, iters = conjugate_gradient(A, b)
print(f"Solution: {x} in {iters} iterations")
```

---

## 🚀 Modern Research Frontiers

### SAM (Sharpness-Aware Minimization)

```
Find flat minima for better generalization:

    min_θ max_{||ε||≤ρ} L(θ + ε)

Approximation:
    1. Compute ε = ρ ∇L(θ) / ||∇L(θ)||
    2. Update θ ← θ - α ∇L(θ + ε)
```

### Lion Optimizer

```
Memory-efficient alternative to Adam:
    Uses sign() instead of full gradient moments
    Much lower memory than Adam
    Strong performance on transformers
```

### Sophia (2nd-Order for LLMs)

```
Scalable 2nd-order optimizer for LLMs:
    Uses Hessian diagonal approximation
    Clip updates based on Hessian
    Faster convergence than Adam
```

---

## 🔗 Dependency Graph

```
[basic-methods]
      ↓
quasi-newton --- conjugate-gradient
      ↓                   ↓
      +-------+-----------+
              ↓
      [metaheuristics]
              ↓
      [ML optimizers]
        (Adam, SAM, Lion)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Basic Methods | [../basic-methods/](../basic-methods/) |
| 📖 | Convex Optimization | [../convex-optimization/](../convex-optimization/) |
| 📄 | L-BFGS Paper | [ACM](https://doi.org/10.1090/S0025-5718-1980-0572855-7) |
| 📄 | SAM Paper | [arXiv](https://arxiv.org/abs/2010.01412) |
| 📄 | Lion Paper | [arXiv](https://arxiv.org/abs/2302.06675) |
| 🇨🇳 | 拟牛顿法详解 | [知乎](https://zhuanlan.zhihu.com/p/29672873) |
| 🇨🇳 | 共轭梯度法 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88819089) |
| 🇨🇳 | 高级优化算法 | [B站](https://www.bilibili.com/video/BV1Y64y1Q7hi) |
| 🇨🇳 | SAM优化器解读 | [机器之心](https://www.jiqizhixin.com/articles/2021-04-22-3) |
| 🇨🇳 | Lion优化器 | [PaperWeekly](https://www.paperweekly.site/papers/notes/4867)


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Basic Methods](../02-basic-methods/) | ➡️ [Next: Convex Optimization](../04-convex-optimization/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
