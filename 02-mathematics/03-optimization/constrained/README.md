<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Constrained%20Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Core Concept

Constrained optimization extends unconstrained methods to handle problems where solutions must satisfy specific constraints - equalities, inequalities, or both.

---

## 📚 Topics Covered

### 1. **Lagrange Multipliers**
- Equality constraints: g(x) = 0
- Method of Lagrange multipliers
- Geometric interpretation
- [→ Learn more](./lagrange-multipliers.md)

### 2. **KKT Conditions**
- Inequality constraints: g(x) ≤ 0
- Karush-Kuhn-Tucker optimality conditions
- Complementary slackness
- [→ Learn more](./kkt-conditions.md)

---

## 📐 General Problem

```
minimize   f(x)
subject to gᵢ(x) ≤ 0, i = 1,...,m  (inequality)
          hⱼ(x) = 0, j = 1,...,p  (equality)
```

### Lagrangian

```
L(x, μ, λ) = f(x) + Σμᵢgᵢ(x) + Σλⱼhⱼ(x)
```

---

## 🔑 Key Insights

### Equality Constraints (Lagrange)
```
At optimum: ∇f parallel to ∇g
∇f(x*) + λ∇g(x*) = 0
```

### Inequality Constraints (KKT)
```
Four conditions:
1. Stationarity: ∇L = 0
2. Primal feasibility: constraints satisfied
3. Dual feasibility: μᵢ ≥ 0
4. Complementary slackness: μᵢgᵢ(x*) = 0
```

---

## 🌍 Applications in ML

| Application | Type | Constraint |
|-------------|------|------------|
| **SVM** | Inequality | Margin constraints |
| **Max Entropy** | Equality | Σpᵢ = 1 |
| **Trust Region** | Inequality | \|\|Δx\|\| ≤ δ |
| **Safe RL** | Inequality | Safety constraints |
| **Portfolio** | Both | Budget + risk limits |

---

## 💻 Quick Example

```python
import numpy as np
from scipy.optimize import minimize

# Minimize f(x,y) = x² + y²
# Subject to: x + y = 1

def objective(X):
    return X[0]**2 + X[1]**2

def constraint(X):
    return X[0] + X[1] - 1

result = minimize(
    objective,
    x0=[0, 0],
    constraints={'type': 'eq', 'fun': constraint}
)

print(f"Optimal: {result.x}")  # [0.5, 0.5]
print(f"Value: {result.fun}")   # 0.5
```

---

## 📖 Resources

### Books
- **Convex Optimization** - Boyd & Vandenberghe (Ch 5)
- **Numerical Optimization** - Nocedal & Wright (Ch 12)

### Papers
- Karush (1939) - Original KKT conditions
- Kuhn & Tucker (1951) - Generalization

---

⬅️ [Back: Optimization](../)

---

⬅️ [Back: Basics](../basics/) | ➡️ [Next: Convex](../convex/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
