# ⚡ Basic Optimization Methods

> **The fundamental algorithms that power machine learning**

## 🎯 Visual Overview

<img src="./images/basic-methods.svg" width="100%">

*Caption: Basic optimization methods include Gradient Descent (first-order, cheap per iteration) and Newton's Method (second-order, fast convergence). Choose based on problem size and precision needs.*

---

## 📂 Topics in This Folder

| Folder | Topic | Order | Used In |
|--------|-------|-------|---------|
| [gradient-descent/](./gradient-descent/) | Gradient Descent | First-order | All DL |
| [newton/](./newton/) | Newton's Method | Second-order | Fast optimization |

---

## 🎯 First-Order vs Second-Order

```
+---------------------------------------------------------+
|                                                         |
|   FIRST-ORDER (uses gradient)                           |
|   -------------------------                             |
|   x_{k+1} = x_k - α∇f(x_k)                              |
|                                                         |
|   Pros: Cheap, scalable to billions of params           |
|   Cons: Can be slow, needs LR tuning                    |
|   Used: Neural networks (SGD, Adam, etc.)               |
|                                                         |
+---------------------------------------------------------+
|                                                         |
|   SECOND-ORDER (uses Hessian)                           |
|   --------------------------                            |
|   x_{k+1} = x_k - H⁻¹∇f(x_k)                            |
|                                                         |
|   Pros: Very fast convergence                           |
|   Cons: O(n³) per step, memory O(n²)                    |
|   Used: Small problems, L-BFGS                          |
|                                                         |
+---------------------------------------------------------+
```

---

## 📐 Mathematical Foundations

### Gradient Descent Update
```
θₜ₊₁ = θₜ - α ∇f(θₜ)

Convergence for convex f:
f(θₜ) - f(θ*) ≤ ||θ₀ - θ*||² / (2αt)

With Lipschitz gradient (L-smooth):
α ≤ 1/L guarantees convergence
```

### Newton's Method Update
```
θₜ₊₁ = θₜ - [∇²f(θₜ)]⁻¹ ∇f(θₜ)

Quadratic convergence near optimum:
||θₜ₊₁ - θ*|| ≤ C ||θₜ - θ*||²

But O(n³) cost per iteration!
```

---

## 📊 Convergence Comparison

```
                    Gradient Descent        Newton's Method
                    -----------------       ---------------
                    
Iteration 1:            ●                        ●
                       ╱|╲                       |
Iteration 2:          ● |                        |
                     ╱  |                        |
Iteration 3:        ●   |                        ●  ← Already there!
                   ╱    |
Iteration 4:      ●     |
                 ╱      |
Iteration N:    ●-------●

Rate:           O(1/k)                      O(log log(1/ε))
                (linear)                    (quadratic)
```

---

## 🌍 Which to Use?

| Scenario | Method | Why |
|----------|--------|-----|
| **Neural Networks** | SGD/Adam | Scalable, noise helps |
| **Logistic Regression** | L-BFGS | Small params, convex |
| **Scientific Computing** | Newton | Need precision |
| **Online Learning** | SGD | Data streams in |
| **Hyperparameter Opt** | Gradient-free | No derivatives |

---

## 🔗 Dependencies

```
foundations/calculus
         |
         +-- gradients.md ----> gradient-descent/
         |                           |
         +-- hessian.md ------> newton/
                                     |
                                     v
                              quasi-newton/ (BFGS, L-BFGS)
                              (in advanced-methods/)
```

---

## 💻 Quick Comparison Code

```python
import numpy as np
from scipy.optimize import minimize

def f(x):
    """Rosenbrock function (classic test)"""
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

x0 = np.array([-1.0, 1.0])

# First-order (gradient descent equivalent)
result_bfgs = minimize(f, x0, method='L-BFGS-B')
print(f"L-BFGS: {result_bfgs.nit} iterations")

# Second-order (Newton)
result_newton = minimize(f, x0, method='Newton-CG')
print(f"Newton: {result_newton.nit} iterations")
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Nocedal & Wright | [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| 🎥 | GD Visualization | [YouTube](https://www.youtube.com/watch?v=IHZwWFHWa-w) |
| 🇨🇳 | 知乎 梯度下降 | [知乎](https://zhuanlan.zhihu.com/p/25202034) |
| 🇨🇳 | B站 优化算法 | [B站](https://www.bilibili.com/video/BV1Vx411j7kT) |

---

⬅️ [Back: Foundations](../01-foundations/) | ➡️ [Next: Advanced Methods](../03-advanced-methods/)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---
