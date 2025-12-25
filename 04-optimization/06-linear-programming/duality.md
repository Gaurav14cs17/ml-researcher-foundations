<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=LP%20Duality&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/linear-programming.svg" width="100%">

*Caption: LP duality connects primal minimization with dual maximization. At optimum, both have equal objective values (strong duality).*

---

## 📐 Mathematical Foundations

### Primal-Dual Pair

```
PRIMAL (P):                      DUAL (D):
─────────────                    ─────────────
min  cᵀx                         max  bᵀy
s.t. Ax ≥ b                      s.t. Aᵀy ≤ c
     x ≥ 0                            y ≥ 0

Relationship:
• n primal variables ↔ n dual constraints
• m primal constraints ↔ m dual variables
• Primal min ↔ Dual max
```

### Converting Between Forms

```
Standard conversions:
┌──────────────────┬──────────────────┐
│     PRIMAL       │      DUAL        │
├──────────────────┼──────────────────┤
│ aᵢᵀx ≥ bᵢ       │ yᵢ ≥ 0           │
│ aᵢᵀx ≤ bᵢ       │ yᵢ ≤ 0           │
│ aᵢᵀx = bᵢ       │ yᵢ free          │
├──────────────────┼──────────────────┤
│ xⱼ ≥ 0          │ aⱼᵀy ≤ cⱼ       │
│ xⱼ ≤ 0          │ aⱼᵀy ≥ cⱼ       │
│ xⱼ free         │ aⱼᵀy = cⱼ       │
└──────────────────┴──────────────────┘
```

### Weak Duality Theorem

```
For any feasible x (primal) and y (dual):

    bᵀy ≤ cᵀx

Proof:
y ≥ 0, Ax ≥ b → yᵀAx ≥ yᵀb = bᵀy
x ≥ 0, Aᵀy ≤ c → xᵀAᵀy ≤ xᵀc = cᵀx

Therefore: bᵀy ≤ yᵀAx ≤ cᵀx  ✓

Implication: Any dual feasible gives lower bound on primal optimal
```

### Strong Duality Theorem

```
If primal has optimal solution x*, then dual has optimal solution y* with:

    cᵀx* = bᵀy*

The duality gap is zero at optimum!

Conditions for strong duality:
• Both primal and dual are feasible
• (Slater's condition for convex programs)
```

### Complementary Slackness

```
At optimum (x*, y*):

Primal slack × Dual variable = 0:
    y*ⱼ · (aⱼᵀx* - bⱼ) = 0  for all j

Dual slack × Primal variable = 0:
    x*ᵢ · (cᵢ - aᵢᵀy*) = 0  for all i

Interpretation:
• If constraint is slack, dual variable = 0
• If dual variable > 0, constraint is tight
```

---

## 💻 Code Example

```python
import numpy as np
from scipy.optimize import linprog

def solve_primal_dual(c, A_ub, b_ub):
    """
    Solve primal and verify dual relationships
    
    Primal: min cᵀx s.t. Ax ≤ b, x ≥ 0
    Dual:   max bᵀy s.t. Aᵀy ≤ c, y ≥ 0
    """
    # Solve primal
    result_primal = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    x_opt = result_primal.x
    primal_obj = result_primal.fun
    
    # Solve dual: max bᵀy → min -bᵀy
    m, n = A_ub.shape
    result_dual = linprog(
        -b_ub,  # Negate for maximization
        A_ub=A_ub.T,  # Transpose
        b_ub=c,
        method='highs'
    )
    y_opt = result_dual.x
    dual_obj = -result_dual.fun  # Negate back
    
    print(f"Primal optimal: {primal_obj:.6f}")
    print(f"Dual optimal:   {dual_obj:.6f}")
    print(f"Duality gap:    {abs(primal_obj - dual_obj):.6f}")
    
    # Verify complementary slackness
    primal_slack = b_ub - A_ub @ x_opt
    print(f"\nComplementary slackness check:")
    for j in range(m):
        cs = y_opt[j] * primal_slack[j]
        print(f"  y[{j}] * slack[{j}] = {y_opt[j]:.4f} * {primal_slack[j]:.4f} = {cs:.6f}")
    
    return x_opt, y_opt

# Example: Simple production problem
c = np.array([4, 3])  # Profits (minimize negative)
A = np.array([
    [2, 1],   # Machine 1 hours
    [1, 2],   # Machine 2 hours
])
b = np.array([8, 7])  # Available hours

x_opt, y_opt = solve_primal_dual(-c, A, b)  # Note: negate c for max
```

---

## 🌍 Economic Interpretation

```
Shadow Prices (Dual Variables):

y*ⱼ = ∂(optimal objective) / ∂bⱼ

• Marginal value of relaxing constraint j
• How much would you pay for one more unit of resource j?

Example:
If y*₁ = 1.5 for machine-1 hours constraint
→ One additional hour of machine 1 improves profit by $1.50
```

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Bertsimas & Tsitsiklis Ch. 4 | Introduction to Linear Optimization |
| 📖 | Boyd & Vandenberghe Ch. 5 | [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) |
| 🎥 | MIT OCW 6.251J | [Duality Lecture](https://ocw.mit.edu/courses/6-251j-introduction-to-mathematical-programming-fall-2009/) |
| 🇨🇳 | 对偶理论详解 | [知乎](https://zhuanlan.zhihu.com/p/38182879) |
| 🇨🇳 | 影子价格解释 | [CSDN](https://blog.csdn.net/Robert_Q/article/details/78832862) |

---

---

➡️ [Next: Interior Point](./interior-point.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
