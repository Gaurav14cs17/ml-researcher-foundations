<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=03 Convex Optimization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 📈 Convex Optimization

> **Optimization problems with guaranteed global solutions**

---

## 📐 Mathematical Foundation

```
Convex Function:
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)  for λ ∈ [0,1]

Convex Set:
λx + (1-λ)y ∈ C for all x,y ∈ C

Key Property:
Local minimum = Global minimum (for convex problems)
```

---

## 🎯 Why It Matters

| Property | Benefit |
|----------|---------|
| **Global optimum** | No local minima traps |
| **Efficient algorithms** | Polynomial time |
| **Duality** | Bounds and certificates |
| **Theory** | Well-understood |

---

## 💻 Code Example

```python
import cvxpy as cp
import numpy as np

# Variables
x = cp.Variable(10)

# Objective (convex)
objective = cp.Minimize(cp.sum_squares(x))

# Constraints
constraints = [x >= 0, cp.sum(x) == 1]

# Solve
problem = cp.Problem(objective, constraints)
problem.solve()
print(f"Optimal value: {problem.value}")
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Boyd Convex Optimization | [Book](https://web.stanford.edu/~boyd/cvxbook/) |
| 🇨🇳 | 凸优化基础 | [知乎](https://zhuanlan.zhihu.com/p/25385801) |

---

⬅️ [Back: Advanced Methods](../03-advanced-methods/) | ➡️ [Next: Convex Optimization](../04-convex-optimization/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

