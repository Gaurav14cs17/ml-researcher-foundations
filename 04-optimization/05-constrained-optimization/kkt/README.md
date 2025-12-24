# KKT Conditions

> **The complete theory for constrained optimization**

## 🎯 Visual Overview

<img src="./images/kkt-conditions.svg" width="100%">

*Caption: KKT conditions: stationarity, primal/dual feasibility, complementary slackness. Necessary for optimality; sufficient for convex problems. Active constraints have λ>0.*

---

## 📂 Topics in This Folder

| File | Topic | Application |
|------|-------|-------------|

---

## 🎯 What are KKT Conditions?

KKT (Karush-Kuhn-Tucker) conditions are **necessary conditions** for optimality in constrained optimization.

```
+---------------------------------------------------------+
|                                                         |
|   The Problem:                                          |
|                                                         |
|   minimize   f(x)                                       |
|   subject to gᵢ(x) ≤ 0,  i = 1,...,m  (inequality)     |
|              hⱼ(x) = 0,  j = 1,...,p  (equality)       |
|                                                         |
|   The Lagrangian:                                       |
|                                                         |
|   L(x,μ,λ) = f(x) + Σᵢ μᵢgᵢ(x) + Σⱼ λⱼhⱼ(x)           |
|                                                         |
+---------------------------------------------------------+
```

---

## 📐 The 5 KKT Conditions

```
At the optimal point (x*, μ*, λ*):

+---------------------------------------------------------+
|                                                         |
|   1. STATIONARITY                                       |
|      ∇f(x*) + Σᵢ μᵢ*∇gᵢ(x*) + Σⱼ λⱼ*∇hⱼ(x*) = 0       |
|      (Gradients balance at optimum)                     |
|                                                         |
|   2. PRIMAL FEASIBILITY                                 |
|      gᵢ(x*) ≤ 0  for all i                             |
|      hⱼ(x*) = 0  for all j                             |
|      (Solution satisfies constraints)                   |
|                                                         |
|   3. DUAL FEASIBILITY                                   |
|      μᵢ* ≥ 0  for all i                                |
|      (Multipliers non-negative for inequalities)        |
|                                                         |
|   4. COMPLEMENTARY SLACKNESS                            |
|      μᵢ* · gᵢ(x*) = 0  for all i                       |
|      (Either constraint active OR multiplier = 0)       |
|                                                         |
|   5. (For convex) SUFFICIENCY                           |
|      If f, gᵢ convex and hⱼ affine → KKT sufficient    |
|                                                         |
+---------------------------------------------------------+
```

---

## 🎯 Visual Intuition: Complementary Slackness

```
Case 1: Constraint ACTIVE (gᵢ = 0)
+---------------------------------+
|                                 |
|   Optimal point ON boundary     |
|                                 |
|         ●--------------         |
|        ╱                        |
|       ╱  feasible               |
|      ╱   region                 |
|                                 |
|   μᵢ > 0 (constraint matters)  |
|                                 |
+---------------------------------+

Case 2: Constraint INACTIVE (gᵢ < 0)
+---------------------------------+
|                                 |
|   Optimal point INSIDE          |
|                                 |
|         --------------          |
|        ╱    ●                   |
|       ╱  feasible               |
|      ╱   region                 |
|                                 |
|   μᵢ = 0 (constraint irrelevant)|
|                                 |
+---------------------------------+
```

---

## 🌍 Where KKT Is Used

| Application | How KKT Appears | Details |
|-------------|-----------------|---------|
| **SVM** | Margin constraints | μᵢ = 0 for non-support vectors |
| **Portfolio Optimization** | No short-selling | KKT for box constraints |
| **RLHF (LLM)** | KL constraint | Trust region methods |
| **Optimal Control** | State constraints | Pontryagin's principle |
| **Energy Dispatch** | Capacity limits | Generator scheduling |

---

## 💻 Example: Quadratic with Inequality

### Problem
```
minimize   f(x,y) = x² + y²
subject to g(x,y) = x + y - 1 ≤ 0
```

### Step 1: Lagrangian
```
L(x,y,μ) = x² + y² + μ(x + y - 1)
```

### Step 2: KKT Conditions
```
∂L/∂x = 2x + μ = 0  →  x = -μ/2
∂L/∂y = 2y + μ = 0  →  y = -μ/2

Complementarity: μ(x + y - 1) = 0
```

### Step 3: Solve Cases

**Case A: μ = 0** (constraint inactive)
```
x = 0, y = 0
Check: g(0,0) = -1 ≤ 0 ✓
Solution: (0, 0), f* = 0
```

**Case B: g = 0** (constraint active)
```
x + y = 1
x = y = -μ/2
→ -μ = 1 → μ = -1 < 0 ✗ (violates dual feasibility)
```

**Answer: (0, 0) with f* = 0**

---

## 💻 Code: Checking KKT

```python
import numpy as np
from scipy.optimize import minimize

def f(x):
    return x[0]**2 + x[1]**2

def g(x):
    return x[0] + x[1] - 1  # ≤ 0

# Solve with SLSQP (uses KKT internally)
result = minimize(
    f,
    x0=[0.5, 0.5],
    constraints={'type': 'ineq', 'fun': lambda x: -g(x)}
)

print(f"Optimal x: {result.x}")
print(f"Optimal f(x): {result.fun}")
print(f"Constraint g(x): {g(result.x)}")

# Check KKT manually
grad_f = 2 * result.x
print(f"∇f at optimum: {grad_f}")
```

---

## 📊 KKT vs Lagrange Multipliers

| Aspect | Lagrange | KKT |
|--------|----------|-----|
| **Constraints** | Equality only | Equality + Inequality |
| **Multipliers** | λ ∈ ℝ | μ ≥ 0 for inequalities |
| **Extra condition** | None | Complementary slackness |
| **Applications** | Physics | ML, optimization |

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 | Original KKT Paper | [1951](https://doi.org/10.1525/9780520347663-014) |
| 📖 | Boyd CVX Book Ch.5 | [Free PDF](https://web.stanford.edu/~boyd/cvxbook/) |
| 🎥 | KKT Explained | [YouTube](https://www.youtube.com/watch?v=uh1Dk68cfWs) |
| 🇨🇳 | 知乎 KKT详解 | [知乎](https://zhuanlan.zhihu.com/p/38163970) |
| 🇨🇳 | B站 KKT条件 | [B站](https://www.bilibili.com/video/BV1aE411o7qd) |

---

⬅️ [Back: Constrained Optimization](../) | ➡️ [Next: Lagrange](../lagrange/)

