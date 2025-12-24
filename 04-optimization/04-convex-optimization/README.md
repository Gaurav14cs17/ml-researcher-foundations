<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=04 Convex Optimization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📈 Convex Optimization

> **The golden standard - global optimum guaranteed!**

## 🎯 Visual Overview

<img src="./images/convex-optimization.svg" width="100%">

*Caption: Convex problems have convex objective and constraints. Key property: local minimum = global minimum. Linear regression, logistic regression, SVM are convex. Deep learning is NOT convex.*

---

## 📂 Topics in This Folder

| Folder | Topic | Why Important |
|--------|-------|---------------|
| [convex-functions/](./convex-functions/) | Convex Functions | Foundation |
| [elbo/](./elbo/) | ELBO & Variational Inference | VAE, Diffusion! |

---

## 🎯 What is Convex Optimization?

A convex optimization problem has:
1. **Convex objective** function (bowl-shaped)
2. **Convex constraint** set (no holes, connected)

```
+---------------------------------------------------------+
|                                                         |
|   CONVEX OPTIMIZATION PROBLEM:                          |
|                                                         |
|   minimize   f(x)      where f is convex               |
|   subject to gᵢ(x) ≤ 0  where gᵢ are convex            |
|              Ax = b                                     |
|                                                         |
|   KEY PROPERTY: Any local minimum = global minimum!    |
|                                                         |
+---------------------------------------------------------+
```

---

## 🎯 Why Convexity Matters

```
NON-CONVEX (typical DL):           CONVEX:

        •  local min                    
       ╱ ╲                              
      ╱   ╲     •  local min             ╲     ╱
     ╱     ╲   ╱ ╲                        ╲   ╱
    ╱       ╲-╱   ╲                        ╲ ╱
                   ╲____• global            •
                                        global = local!

• Many local minima              • Only one minimum
• SGD might get stuck            • Any method finds it
• Need good initialization       • Initialization irrelevant
```

---

## 📐 Definition of Convex Function

```
A function f is CONVEX if for all x, y and θ ∈ [0,1]:

f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)
---------------   ------------------
Function at        Linear interpolation
midpoint           of function values

Visually: The chord is ABOVE the function

            • f(x)
           ╱|
          ╱ |   chord
         ╱  |
        ╱   |
       ╱    • f(y)
      ╱------------
     ╱   function
    •  below chord ✓
```

---

## 🌍 Real-World Convex Problems

| Problem | Convex? | Why? |
|---------|---------|------|
| **Linear Regression** | ✅ Yes | Quadratic loss |
| **Logistic Regression** | ✅ Yes | Log-loss is convex |
| **SVM** | ✅ Yes | Hinge loss + L2 |
| **LASSO** | ✅ Yes | L1 regularized |
| **Portfolio Optimization** | ✅ Yes | Mean-variance |
| **Deep Learning** | ❌ No | Non-linear activations |
| **Matrix Factorization** | ❌ No | Product of unknowns |

---

## 📊 Convex Examples

| Function | Formula | Convex? |
|----------|---------|---------|
| Linear | f(x) = aᵀx + b | ✅ Yes (and concave!) |
| Quadratic | f(x) = xᵀQx (Q≻0) | ✅ Yes |
| Norm | f(x) = ‖x‖ | ✅ Yes |
| Log-sum-exp | f(x) = log(Σeˣⁱ) | ✅ Yes |
| Negative entropy | f(x) = Σxᵢlog(xᵢ) | ✅ Yes |
| Exponential | f(x) = eˣ | ✅ Yes |
| x³ | f(x) = x³ | ❌ No |

---

## 🔗 Connection to Machine Learning

```
+---------------------------------------------------------+
|                                                         |
|   Deep Learning Loss Landscape (Non-Convex)             |
|                                                         |
|   But! We can use convex theory for:                    |
|                                                         |
|   1. ELBO → Lower bound on log-likelihood               |
|      (Even if p(x) is non-convex, ELBO can be           |
|       made convex in variational parameters!)           |
|                                                         |
|   2. Last layer → Often convex in last layer weights    |
|                                                         |
|   3. Analysis → Convex relaxations for insights         |
|                                                         |
+---------------------------------------------------------+
```

---

## 💻 Code: Check Convexity

```python
import numpy as np
from scipy.linalg import eigh

def is_convex_quadratic(Q):
    """Check if f(x) = x'Qx is convex by checking eigenvalues"""
    eigenvalues = eigh(Q, eigvals_only=True)
    return all(eigenvalues >= 0)

# Example: f(x,y) = x² + y² 
Q = np.array([[1, 0], [0, 1]])
print(f"Is f(x,y)=x²+y² convex? {is_convex_quadratic(Q)}")  # True

# Example: f(x,y) = x² - y²
Q = np.array([[1, 0], [0, -1]])
print(f"Is f(x,y)=x²-y² convex? {is_convex_quadratic(Q)}")  # False
```

---

## 📐 Advanced Convexity Concepts

### Second-Order Conditions

```
For twice-differentiable f:

f is convex ⟺ ∇²f(x) ≽ 0 (Hessian is positive semidefinite)
f is strictly convex ⟺ ∇²f(x) ≻ 0 (Hessian is positive definite)

Example: f(x) = ½xᵀQx + bᵀx
∇f(x) = Qx + b
∇²f(x) = Q
Convex ⟺ Q ≽ 0
```

### First-Order Conditions

```
For differentiable convex f:

f(y) ≥ f(x) + ∇f(x)ᵀ(y - x)

The tangent line/plane is BELOW the function!
Used in gradient descent convergence proofs.
```

### Strong Convexity

```
f is μ-strongly convex if:

f(y) ≥ f(x) + ∇f(x)ᵀ(y - x) + (μ/2)‖y - x‖²

Equivalently: ∇²f(x) ≽ μI

Benefits:
• Unique global minimum
• Faster convergence: O((1-μ/L)^k) vs O(1/k)
• More stable optimization
```

### Lipschitz Smoothness

```
f is L-smooth if gradient is Lipschitz:

‖∇f(x) - ∇f(y)‖ ≤ L‖x - y‖

Equivalently: ∇²f(x) ≼ LI

Used for: Learning rate bounds (α ≤ 1/L)
```

### Convergence Rates

```
Gradient Descent: θₜ₊₁ = θₜ - α∇f(θₜ)

For L-smooth convex f with step α ≤ 1/L:
f(θₜ) - f(θ*) ≤ O(1/t)

For L-smooth μ-strongly convex:
f(θₜ) - f(θ*) ≤ O((1 - μ/L)^t)

Condition number κ = L/μ determines speed
```

---

## 📚 Resources

### 📖 Books & Papers

| Title | Author | Focus |
|-------|--------|-------|
| Convex Optimization | Boyd & Vandenberghe | [Free PDF](https://web.stanford.edu/~boyd/cvxbook/) |
| Introductory Lectures on Convex Optimization | Nesterov | Advanced |

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 🎥 | Stanford CVX101 | [YouTube](https://www.youtube.com/playlist?list=PL3940DD956CDF0622) |
| 📖 | CMU 10-725 | [Course](https://www.stat.cmu.edu/~ryantibs/convexopt/) |
| 📄 | Boyd & Vandenberghe | [Book](https://web.stanford.edu/~boyd/cvxbook/) |
| 💻 | CVXPY | [Docs](https://www.cvxpy.org/) |
| 💻 | CVX MATLAB | [Link](http://cvxr.com/cvx/) |
| 🇨🇳 | 凸优化入门 | [知乎](https://zhuanlan.zhihu.com/p/25383715) |
| 🇨🇳 | 凸优化完整课程 | [B站](https://www.bilibili.com/video/BV1Pg4y187Ed) |
| 🇨🇳 | 凸优化应用 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088)

---

## 🔗 Where This Topic Is Used

| Topic | How Convexity Is Used |
|-------|----------------------|
| **SVM** | Dual is convex QP |
| **Logistic Regression** | Convex log-loss |
| **LASSO** | Convex L1 regularization |
| **Convergence Proofs** | Strong convexity bounds |
| **Neural Network Theory** | Local convexity analysis |
| **ELBO/VAE** | Variational lower bound |

---

⬅️ [Back: Advanced Methods](../03-advanced-methods/) | ➡️ [Next: Constrained Optimization](../05-constrained-optimization/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
