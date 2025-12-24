# Duality in Optimization

> **Solving problems through their dual**

---

## 🎯 Visual Overview

<img src="./images/duality.svg" width="100%">

*Caption: Primal: min f(x) s.t. g(x)≤0. Lagrangian: L(x,λ)=f(x)+λg(x). Dual: max_λ min_x L(x,λ). Weak duality: d*≤p*. Strong duality (convex): d*=p*. KKT conditions characterize optimality.*

---

## 📂 Overview

Duality provides lower bounds on optimization problems and enables efficient algorithms. SVMs are typically solved via the dual. Understanding duality deepens insight into constrained optimization.

---

## 📐 Mathematical Definitions

### Primal Problem
```
minimize   f(x)
subject to gᵢ(x) ≤ 0,  i = 1,...,m
           hⱼ(x) = 0,  j = 1,...,p

Optimal value: p*
```

### Lagrangian
```
L(x, λ, ν) = f(x) + Σᵢ λᵢgᵢ(x) + Σⱼ νⱼhⱼ(x)

Where:
• λᵢ ≥ 0: dual variables for inequalities
• νⱼ: dual variables for equalities
```

### Dual Problem
```
maximize   g(λ, ν) = inf_x L(x, λ, ν)
subject to λ ≥ 0

Optimal value: d*
```

### Weak and Strong Duality
```
Weak duality (always):
d* ≤ p*

Strong duality (under Slater's condition for convex):
d* = p*

Duality gap: p* - d* = 0 under strong duality
```

### KKT Conditions
```
Necessary conditions for optimality (under constraint qualification):

1. Stationarity:    ∇f(x*) + Σᵢλᵢ*∇gᵢ(x*) + Σⱼνⱼ*∇hⱼ(x*) = 0
2. Primal feasibility: gᵢ(x*) ≤ 0, hⱼ(x*) = 0
3. Dual feasibility:   λᵢ* ≥ 0
4. Complementary slackness: λᵢ*gᵢ(x*) = 0

For convex problems: KKT ⟺ optimal
```

---

## 💻 Code Examples

```python
import numpy as np
from scipy.optimize import minimize

# Example: minimize ||x||² subject to Ax = b
# Dual: maximize -¼||Aᵀλ||² + bᵀλ

def primal_problem(x, A, b):
    """Primal: min ||x||² s.t. Ax = b"""
    return np.sum(x**2)

def dual_problem(lam, A, b):
    """Dual: max -¼||Aᵀλ||² + bᵀλ"""
    return -0.25 * np.sum((A.T @ lam)**2) + b @ lam

# Solve primal
A = np.array([[1, 1], [1, -1]])
b = np.array([2, 0])
res_primal = minimize(lambda x: primal_problem(x, A, b),
                      x0=np.zeros(2),
                      constraints={'type': 'eq', 'fun': lambda x: A @ x - b})

# Solve dual
res_dual = minimize(lambda l: -dual_problem(l, A, b),  # Maximize by negating
                    x0=np.zeros(2))

# SVM dual formulation
# max Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
# s.t. Σαᵢyᵢ = 0, 0 ≤ αᵢ ≤ C

from sklearn.svm import SVC
svm = SVC(kernel='rbf')  # Solved via dual!
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Boyd: Convex Opt Ch. 5 | [Book](https://web.stanford.edu/~boyd/cvxbook/) |
| 🎓 | Stanford EE364A | [Course](https://web.stanford.edu/class/ee364a/) |
| 📄 | SVM Dual | [Paper](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) |
| 🇨🇳 | 对偶问题详解 | [知乎](https://zhuanlan.zhihu.com/p/38182879) |
| 🇨🇳 | KKT条件 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88318597) |
| 🇨🇳 | SVM与对偶 | [B站](https://www.bilibili.com/video/BV1Vt411G7nq) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: duality](../)

---

⬅️ [Back: Convex](../convex/) | ➡️ [Next: First Order](../first-order/)
