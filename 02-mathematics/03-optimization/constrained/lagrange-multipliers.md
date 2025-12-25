<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Lagrange%20Multipliers&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Lagrange Multipliers

> **Optimizing with equality constraints**

---

## 📐 Problem

```
minimize f(x)
subject to g(x) = 0
```

---

## 📐 Method

```
Lagrangian: L(x, λ) = f(x) + λᵀg(x)

Necessary conditions:
∇ₓL = ∇f(x) + λᵀ∇g(x) = 0
∇_λL = g(x) = 0
```

---

## 🎯 Geometric Intuition

```
At optimum:
∇f is parallel to ∇g

Why? If not parallel, could move along constraint
to decrease f.

∇f = -λ∇g
```

---

## 💻 Example

```python
import numpy as np
from scipy.optimize import minimize

# Minimize f(x,y) = x² + y²
# Subject to: x + y = 1

def objective(vars):
    x, y, lam = vars
    return x**2 + y**2

def lagrangian(vars):
    x, y, lam = vars
    return x**2 + y**2 + lam * (x + y - 1)

# Solve ∇L = 0
# ∂L/∂x = 2x + λ = 0
# ∂L/∂y = 2y + λ = 0
# ∂L/∂λ = x + y - 1 = 0

# Solution: x = y = 0.5, λ = -1
x_opt, y_opt = 0.5, 0.5
print(f"Optimal: ({x_opt}, {y_opt})")
print(f"Objective: {x_opt**2 + y_opt**2}")  # 0.5
```

---

## 🌍 ML Applications

| Application | Constraint |
|-------------|------------|
| Max entropy | Σpᵢ = 1 |
| Softmax derivation | Σexp(zᵢ) normalization |
| Constrained RL | Safety constraints |

---

---

⬅️ [Back: Kkt Conditions](./kkt-conditions.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
