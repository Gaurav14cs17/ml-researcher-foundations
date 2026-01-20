<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Lagrange%20Multipliers%20%26%20KKT&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Overview

This section covers the two fundamental tools for constrained optimization:
- **Lagrange Multipliers**: For equality constraints
- **KKT Conditions**: For equality AND inequality constraints

---

## 📐 Part 1: Lagrangian Method

### Problem Formulation

```
minimize    f(x)
subject to  g(x) = 0

where g(x) = [g₁(x), g₂(x), ..., gₘ(x)]ᵀ
```

### The Lagrangian Function

```
L(x, λ) = f(x) + λᵀg(x)
        = f(x) + Σᵢ λᵢgᵢ(x)

where λ = [λ₁, λ₂, ..., λₘ]ᵀ are Lagrange multipliers
```

### First-Order Necessary Conditions

```
At optimum (x*, λ*):

∇ₓL = ∇f(x*) + Σᵢ λᵢ*∇gᵢ(x*) = 0   (Stationarity)
∇λL = g(x*) = 0                      (Feasibility)

Geometric interpretation:
∇f(x*) = -Σᵢ λᵢ*∇gᵢ(x*)

"The objective gradient is a linear combination of constraint gradients"
```

---

## 📐 Proof: Why Lagrange Multipliers Work

**Theorem:** Let x* be a local minimum subject to g(x) = 0. If ∇g(x*) has full rank, then there exists λ* such that ∇f(x*) + λ*ᵀ∇g(x*) = 0.

**Proof:**

```
Step 1: Define the tangent space
T = {d ∈ ℝⁿ : ∇g(x*)ᵀd = 0}

This is the set of all directions that stay on the constraint surface
(to first order).

Step 2: Optimality implies no descent in T
If x* is a local minimum, then for all d ∈ T:
∇f(x*)ᵀd ≥ 0

(Otherwise we could decrease f while staying feasible)

But also -d ∈ T, so:
∇f(x*)ᵀ(-d) ≥ 0  ⟹  ∇f(x*)ᵀd ≤ 0

Therefore: ∇f(x*)ᵀd = 0 for all d ∈ T

Step 3: Characterize the orthogonal complement
T = null(∇g(x*)ᵀ)
T⊥ = range(∇g(x*)) = span of columns of ∇g(x*)

Step 4: ∇f must be in T⊥
Since ∇f(x*)ᵀd = 0 for all d ∈ T:
∇f(x*) ⊥ T  ⟹  ∇f(x*) ∈ T⊥

Step 5: Express ∇f as combination
∇f(x*) ∈ T⊥ = range(∇g(x*))

Therefore: ∇f(x*) = -∇g(x*)λ* for some λ* ∈ ℝᵐ

Rearranging: ∇f(x*) + ∇g(x*)λ* = 0  ∎
```

---

## 📐 Part 2: KKT Conditions

### Problem with Inequalities

```
minimize    f(x)
subject to  gᵢ(x) ≤ 0,  i = 1,...,m   (inequalities)
            hⱼ(x) = 0,  j = 1,...,p   (equalities)
```

### The KKT Lagrangian

```
L(x, μ, λ) = f(x) + Σᵢ μᵢgᵢ(x) + Σⱼ λⱼhⱼ(x)

where:
• μᵢ ≥ 0: multipliers for inequalities
• λⱼ ∈ ℝ: multipliers for equalities
```

### The 5 KKT Conditions

```
At optimum (x*, μ*, λ*):

1. STATIONARITY
   ∇f(x*) + Σᵢ μᵢ*∇gᵢ(x*) + Σⱼ λⱼ*∇hⱼ(x*) = 0

2. PRIMAL FEASIBILITY
   gᵢ(x*) ≤ 0  for all i
   hⱼ(x*) = 0  for all j

3. DUAL FEASIBILITY
   μᵢ* ≥ 0  for all i

4. COMPLEMENTARY SLACKNESS
   μᵢ* · gᵢ(x*) = 0  for all i

5. (For convex problems) SUFFICIENCY
   If f, gᵢ convex and hⱼ affine → KKT sufficient for optimality
```

---

## 📐 Understanding Complementary Slackness

```
μᵢ* · gᵢ(x*) = 0 means:

Either μᵢ* = 0  (multiplier is zero)
Or     gᵢ(x*) = 0  (constraint is active/binding)

Visual Interpretation:

Case 1: INACTIVE constraint (gᵢ < 0)
+-----------------------------------+
|   Optimal point inside region     |
|                                   |
|         •  x*                     |
|        ╱ ╲                        |
|       ╱   ╲  boundary gᵢ = 0     |
|      ╱     ╲                      |
|                                   |
|   Constraint doesn't matter       |
|   ⟹ μᵢ* = 0                      |
+-----------------------------------+

Case 2: ACTIVE constraint (gᵢ = 0)
+-----------------------------------+
|   Optimal point on boundary       |
|                                   |
|      ----●---- boundary           |
|         x*                        |
|        ╱ ╲                        |
|       ╱   ╲                       |
|                                   |
|   Constraint is binding           |
|   ⟹ μᵢ* > 0 possible             |
+-----------------------------------+
```

---

## 📐 Worked Example: Quadratic with Inequality

### Problem

```
minimize   f(x,y) = x² + y²
subject to g(x,y) = 1 - x - y ≤ 0  (i.e., x + y ≥ 1)
```

### Step 1: Write Lagrangian

```
L(x, y, μ) = x² + y² + μ(1 - x - y)
```

### Step 2: KKT Conditions

```
Stationarity:
∂L/∂x = 2x - μ = 0  ⟹  x = μ/2
∂L/∂y = 2y - μ = 0  ⟹  y = μ/2

Primal feasibility:
1 - x - y ≤ 0  ⟹  x + y ≥ 1

Dual feasibility:
μ ≥ 0

Complementary slackness:
μ(1 - x - y) = 0
```

### Step 3: Solve by Cases

**Case A: μ = 0 (constraint inactive)**
```
x = 0, y = 0
Check: 1 - 0 - 0 = 1 > 0  ✗ (violates primal feasibility!)
```

**Case B: μ > 0 (constraint active)**
```
From complementary slackness: x + y = 1
From stationarity: x = y = μ/2
Therefore: μ/2 + μ/2 = 1 ⟹ μ = 1 > 0 ✓

Solution: x* = y* = 1/2, μ* = 1
Optimal value: f* = 1/4 + 1/4 = 1/2
```

---

## 💻 Code Implementation

```python
import numpy as np
from scipy.optimize import minimize

def solve_kkt_example():
    """
    Solve: min x² + y²
           s.t. x + y ≥ 1
    """
    def f(xy):
        return xy[0]**2 + xy[1]**2
    
    def g(xy):
        return xy[0] + xy[1] - 1  # x + y - 1 ≥ 0
    
    # Solve with scipy (SLSQP uses KKT internally)
    result = minimize(
        f,
        x0=[0.5, 0.5],
        constraints={'type': 'ineq', 'fun': g}  # g(x) ≥ 0
    )
    
    print(f"Optimal x: {result.x}")
    print(f"Optimal f(x): {result.fun}")
    print(f"Constraint g(x): {g(result.x)}")
    
    # Verify KKT manually
    x, y = result.x
    grad_f = np.array([2*x, 2*y])
    grad_g = np.array([1, 1])
    
    # From stationarity: grad_f = μ * grad_g
    mu = grad_f[0] / grad_g[0]
    print(f"\nKKT verification:")
    print(f"μ* = {mu}")
    print(f"μ ≥ 0: {mu >= 0}")
    print(f"Complementary slackness (μ*g = 0): {abs(mu * (1 - x - y)) < 1e-6}")
    
    return result

result = solve_kkt_example()
```

---

## 📐 KKT for SVM (Support Vector Machine)

### Primal Problem

```
minimize    (1/2)||w||²
subject to  yᵢ(wᵀxᵢ + b) ≥ 1,  i = 1,...,n

Or equivalently:
minimize    (1/2)||w||²
subject to  1 - yᵢ(wᵀxᵢ + b) ≤ 0
```

### Lagrangian

```
L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ + b) - 1]
           = (1/2)||w||² - Σᵢ αᵢyᵢwᵀxᵢ - bΣᵢ αᵢyᵢ + Σᵢ αᵢ
```

### KKT Conditions

```
Stationarity:
∂L/∂w = w - Σᵢ αᵢyᵢxᵢ = 0  ⟹  w = Σᵢ αᵢyᵢxᵢ
∂L/∂b = -Σᵢ αᵢyᵢ = 0        ⟹  Σᵢ αᵢyᵢ = 0

Dual feasibility:
αᵢ ≥ 0

Complementary slackness:
αᵢ[yᵢ(wᵀxᵢ + b) - 1] = 0

Interpretation:
• αᵢ = 0: Point is NOT a support vector
• αᵢ > 0: Point IS a support vector (on margin)
```

### Dual Problem

```
Substituting w = Σᵢ αᵢyᵢxᵢ into L:

maximize   Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
subject to αᵢ ≥ 0
           Σᵢ αᵢyᵢ = 0

This is a quadratic program in α!
```

---

## 📐 Economic Interpretation: Shadow Prices

```
The Lagrange multiplier λ* has economic meaning:

λ* = ∂f*/∂b  (sensitivity of optimal value to constraint)

Example:
• Minimize cost f(x) subject to production g(x) ≥ b
• λ* = marginal cost of producing one more unit
• This is the "shadow price" of production capacity

In ML:
• SVM: αᵢ = "importance" of data point i
• Large αᵢ ⟹ Important support vector
• Zero αᵢ ⟹ Point doesn't affect decision boundary
```

---

## 📊 Comparison

| Aspect | Lagrange | KKT |
|--------|----------|-----|
| **Constraints** | Equality only | Equality + Inequality |
| **Multipliers** | λ ∈ ℝ | μ ≥ 0 for inequalities |
| **Extra condition** | None | Complementary slackness |
| **Applications** | Physics, simple ML | SVM, RL, general optimization |

---

## 🔗 Applications

| Application | Constraint Type | Example |
|-------------|-----------------|---------|
| **SVM** | Inequality | Margin ≥ 1 for all points |
| **Portfolio** | Equality + Inequality | Weights sum to 1, non-negative |
| **Physics** | Equality | Conserve energy/momentum |
| **RL (TRPO, PPO)** | KL constraint | Trust region |
| **Optimal Control** | Dynamics as equality | Trajectory optimization |

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Boyd CVX Ch.5 | [Free PDF](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 | Nocedal Ch.12 | [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| 🎥 | KKT Conditions | [YouTube](https://www.youtube.com/watch?v=uh1Dk68cfWs) |
| 📄 | Original KKT Paper | [1951](https://doi.org/10.1525/9780520347663-014) |
| 🇨🇳 | 知乎 KKT条件 | [知乎](https://zhuanlan.zhihu.com/p/38163970) |

---

⬅️ [Back: Constrained Optimization](../) | ➡️ [Next: Main Constrained Optimization](../../05_constrained_optimization/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
