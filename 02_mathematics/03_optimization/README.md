<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Optimality Conditions
```
First-order necessary condition:
∇f(x*) = 0

Second-order sufficient condition:
∇²f(x*) ≻ 0 (positive definite)
```

### Convexity
```
Function f is convex iff:
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)  ∀λ ∈ [0,1]

Equivalently:
∇²f(x) ⪰ 0 (positive semidefinite)

Strong convexity (μ > 0):
f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²
```

### Lipschitz Smoothness
```
f has L-Lipschitz gradient if:
||∇f(x) - ∇f(y)|| ≤ L||x - y||

Equivalently:
f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + (L/2)||y-x||²
```

### Convergence Rates
```
GD with step size α = 1/L:
f(xₖ) - f(x*) ≤ L||x₀ - x*||² / (2k)   [O(1/k)]

Strongly convex (κ = L/μ):
||xₖ - x*|| ≤ (1 - 1/κ)^k ||x₀ - x*||   [Linear]

Nesterov acceleration:
f(xₖ) - f(x*) ≤ O(L||x₀ - x*||² / k²)  [O(1/k²)]
```

---

## 📂 Topics in This Folder

| Folder | Topics | ML Application |
|--------|--------|----------------|
| [basics/](./basics/) | Objectives, constraints, optima | Problem formulation |
| [convex/](./convex/) | Convex sets, functions, strong convexity | 🔥 Theory foundation |
| [duality/](./duality/) | Lagrangian, KKT conditions | SVM, constrained problems |
| [first-order/](./first-order/) | Gradient descent, momentum | 🔥 Training NNs |
| [second-order/](./second-order/) | Newton, quasi-Newton | Natural gradient |
| [stochastic/](./stochastic/) | SGD, Adam, adaptive | 🔥 Deep learning |

---

## 🎯 The Optimization Problem

```
minimize    f(x)           ← Objective function
subject to  gᵢ(x) ≤ 0     ← Inequality constraints
            hⱼ(x) = 0     ← Equality constraints
            x ∈ X          ← Domain

In ML:
• f(x) = Loss function (cross-entropy, MSE, ...)
• x = Model parameters (millions/billions of them!)
• Constraints often implicit (weight decay = soft constraint)
```

---

## 🌍 ML Optimization Landscape

```
                    Optimization Methods
                           |
          +----------------+----------------+
          |                |                |
    First-order      Second-order      Stochastic
    (gradient)       (Hessian)         (mini-batch)
          |                |                |
    +-----+-----+    +----+----+    +------+------+
    |           |    |         |    |             |
   GD        Momentum Newton  BFGS  SGD         Adam
    |           |              |     |             |
    |     Nesterov        L-BFGS   Variance     AdamW
    |                              reduction
    |                                            |
    +--------------------+-----------------------+
                         |
                         v
              Deep Learning Training
```

---

## 🔑 Key Convergence Results

| Method | Rate | Assumption | Paper |
|--------|------|------------|-------|
| GD (convex) | O(1/k) | Convex, L-smooth | - |
| GD (strongly convex) | O((1-μ/L)^k) | μ-strongly convex | - |
| Nesterov | O(1/k²) | Convex | [Nesterov 1983] |
| Newton | O(log log(1/ε)) | Strongly convex | - |
| SGD (convex) | O(1/√k) | Convex | - |

```
Convergence rate comparison:

Steps k | O(1/k²) | O((1-1/κ)^k) | O(1/k) | O(1/√k)
--------+---------+--------------+--------+---------
     10 |  0.010  |     0.35     |  0.10  |   0.32
    100 |  0.0001 |     0.00     |  0.01  |   0.10
   1000 | 1e-06   |     0.00     |  0.001 |   0.03

(Lower is better)
```

---

## 💻 Core Algorithms

```python

# Gradient Descent
def gradient_descent(f, grad_f, x0, lr=0.01, n_steps=100):
    x = x0
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
    return x

# SGD with Momentum
def sgd_momentum(params, grads, velocities, lr=0.01, momentum=0.9):
    for p, g, v in zip(params, grads, velocities):
        v[:] = momentum * v + g
        p[:] = p - lr * v
    return params, velocities

# Adam (simplified)
def adam_step(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Convex Optimization | [Boyd (free)](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 | Numerical Optimization | Nocedal & Wright |
| 🎓 | Stanford Convex Optimization | [YouTube](https://www.youtube.com/playlist?list=PL3940DD956CDF0622) |

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: 02-Calculus](../02_calculus/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
