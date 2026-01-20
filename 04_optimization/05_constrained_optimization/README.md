<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Constrained%20Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Subtopics

| Folder | Topic | Key Concept |
|--------|-------|-------------|
| [01_kkt/](./01_kkt/) | KKT Conditions | Inequality constraints |
| [02_lagrange/](./02_lagrange/) | Lagrange Multipliers | Equality constraints |

---

## 🎯 The Problem

```
Unconstrained:              Constrained:
                            
   Find x that               Find x that
   minimizes f(x)            minimizes f(x)
                             SUBJECT TO:
   • Go anywhere             • g(x) ≤ 0 (inequality)
                             • h(x) = 0 (equality)
                             
   ↓                         ↓
   
   Just set ∇f = 0           Need Lagrange/KKT!

```

---

## 📐 The Two Main Tools

```
+---------------------------------------------------------+

|                                                         |
|   LAGRANGE MULTIPLIERS              KKT CONDITIONS      |
|   --------------------              --------------      |
|                                                         |
|   For: h(x) = 0 only                For: g(x) ≤ 0 AND  |
|   (equality)                         h(x) = 0           |
|                                                         |
|   L = f(x) - λh(x)                  + Complementarity:  |
|                                     μᵢgᵢ(x) = 0         |
|   Solve: ∇L = 0                     μᵢ ≥ 0              |
|                                                         |
+---------------------------------------------------------+

```

---

## 📊 Visual Comparison

```
Unconstrained:            Constrained:

    ∇f = 0                  ∇f = λ∇g
       |                        |
       ↓                        ↓
       •                        • ← on boundary!
      ╱ ╲                      -+-
     ╱   ╲                   ╱  |  ╲  feasible
    ╱     ╲                 ╱   |   ╲ region
   
   Interior optimum        Boundary optimum

```

---

# Part 1: Lagrange Multipliers

## 📐 Mathematical Formulation

### Problem Setup

```
minimize    f(x)
subject to  gᵢ(x) = 0,  i = 1, ..., m

where x ∈ ℝⁿ, m < n (fewer constraints than variables)

```

### The Lagrangian Function

```
L(x, λ) = f(x) - Σᵢ λᵢ gᵢ(x)

where λ = (λ₁, ..., λₘ) are Lagrange multipliers

Alternative form (with +):
L(x, λ) = f(x) + Σᵢ λᵢ gᵢ(x)
(sign convention varies by textbook)

```

### First-Order Necessary Conditions

```
At optimum (x*, λ*):

Stationarity (gradient condition):
∇ₓL = ∇f(x*) - Σᵢ λᵢ*∇gᵢ(x*) = 0

Feasibility:
gᵢ(x*) = 0  for all i

Geometric interpretation:
∇f(x*) = Σᵢ λᵢ*∇gᵢ(x*)

The objective gradient is a linear combination of constraint gradients!

```

---

## 📐 Why Lagrange Multipliers Work: Complete Proof

**Theorem (First-Order Necessary Conditions):** Let x* be a local minimum of f(x) subject to g(x) = 0, where f and g are continuously differentiable, and ∇g(x*) ≠ 0. Then there exists λ* such that:

```
∇f(x*) + λ*∇g(x*) = 0

```

**Proof:**

```
Step 1: Define feasible directions
A direction d is feasible if there exists α > 0 such that:
x* + αd satisfies g(x* + αd) ≈ 0 for small α

Taylor expansion:
g(x* + αd) ≈ g(x*) + α∇g(x*)ᵀd
           = α∇g(x*)ᵀd           (since g(x*) = 0)

For feasibility: ∇g(x*)ᵀd = 0

So: Feasible directions lie in tangent space T = {d : ∇g(x*)ᵀd = 0}

Step 2: Optimality implies no descent direction
Since x* is a local minimum, f cannot decrease along any feasible direction:
∇f(x*)ᵀd ≥ 0  for all d ∈ T

Step 3: Characterize T⊥ (orthogonal complement)
T = {d : ∇g(x*)ᵀd = 0}
T⊥ = span(∇g(x*))

Step 4: ∇f must be in T⊥
If ∇f(x*) ∉ T⊥, then ∇f has a component in T.
Let d = -projection of ∇f onto T
Then ∇f(x*)ᵀd < 0 (descent direction in T)
Contradiction! (x* wouldn't be optimal)

Therefore: ∇f(x*) ∈ T⊥ = span(∇g(x*))

Step 5: Conclusion
∇f(x*) = -λ*∇g(x*) for some λ* ∈ ℝ
⟹ ∇f(x*) + λ*∇g(x*) = 0  ∎

```

---

## 🎯 Geometric Intuition

```
At optimum on constraint surface:

       ∇f           ∇g
        ↑          ↗
         \        /
          \      /
           \    /
            \  /
        -----*----- constraint g(x) = 0

∇f must be perpendicular to constraint surface
(otherwise we could move along surface and improve)

This means ∇f is parallel to ∇g: ∇f = λ∇g

```

---

## 📐 Second-Order Conditions

```
Bordered Hessian:
H_L = ∇²ₓₓL = ∇²f - Σᵢ λᵢ∇²gᵢ

For minimum: H_L positive definite on tangent space of constraints

Tangent space: {v : ∇gᵢ(x*)ᵀv = 0 for all i}

Check: vᵀ H_L v > 0 for all v in tangent space

```

---

## 💻 Code Example

```python
import numpy as np
from scipy.optimize import minimize

def lagrange_example():
    """
    Minimize f(x,y) = x² + y²
    Subject to g(x,y) = x + y - 1 = 0
    
    Solution: x* = y* = 0.5, λ* = 1
    """
    
    # Objective
    def f(xy):
        x, y = xy
        return x**2 + y**2
    
    # Constraint
    def g(xy):
        x, y = xy
        return x + y - 1
    
    # Solve using scipy (SLSQP handles equality constraints)
    result = minimize(
        f, 
        x0=[0, 0],
        constraints={'type': 'eq', 'fun': g}
    )
    
    print(f"Optimal x: {result.x}")
    print(f"Optimal f(x): {result.fun}")
    
    # Verify Lagrange conditions analytically
    x, y = result.x
    
    # ∇f = (2x, 2y) = (1, 1)
    # ∇g = (1, 1)
    # λ = 2x = 2(0.5) = 1
    
    grad_f = np.array([2*x, 2*y])
    grad_g = np.array([1, 1])
    lambda_approx = grad_f[0] / grad_g[0]
    
    print(f"λ* ≈ {lambda_approx}")
    print(f"∇f = λ∇g? {np.allclose(grad_f, lambda_approx * grad_g)}")
    
    return result

result = lagrange_example()

```

---

## 💡 Shadow Prices Interpretation

```
The Lagrange multiplier λ has economic meaning:

λ* = ∂f*/∂b

"How much would the optimal value improve
 if we relaxed the constraint by 1 unit?"

Example:
• Constraint: budget ≤ $100
• λ* = 5
• Meaning: $1 more budget → $5 more profit
• This is the "shadow price" of money!

```

---

# Part 2: KKT Conditions

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

## 📐 Example: Quadratic with Inequality

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

## 📐 KKT Proof (for Convex Problems)

**Theorem:** For a convex optimization problem with Slater condition, KKT conditions are both necessary and sufficient for optimality.

**Proof Sketch:**

```
Step 1: Define Lagrangian
  L(x, μ, λ) = f(x) + μᵀg(x) + λᵀh(x)

Step 2: Strong duality (with Slater)
  min_x max_{μ≥0,λ} L = max_{μ≥0,λ} min_x L
  
  Optimal primal value = Optimal dual value (no gap)

Step 3: At optimum (x*, μ*, λ*)
  x* minimizes L(x, μ*, λ*)
  ⟹ ∇_x L(x*, μ*, λ*) = 0  (stationarity)

Step 4: Complementary slackness
  Strong duality ⟹ f(x*) = L(x*, μ*, λ*) + μ*ᵀg(x*) + λ*ᵀh(x*)
  
  Since h(x*) = 0 and the bound is tight:
  μ*ᵀg(x*) = 0
  
  With μ* ≥ 0 and g(x*) ≤ 0:
  μᵢ*gᵢ(x*) = 0 for each i ∎

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

## 🌍 Real-World Applications

| Application | Constraint Type | Example |
|-------------|-----------------|---------|
| **SVM** | Inequality | Margin ≥ 1 for all points |
| **Portfolio** | Equality + Inequality | Weights sum to 1, non-negative |
| **Physics** | Equality | Conserve energy/momentum |
| **RL (TRPO, PPO)** | KL constraint | Trust region |
| **Optimal Control** | Dynamics as equality | Trajectory optimization |

---

## 🔗 Dependencies

```
foundations/calculus
         |
         ↓
basic-methods/gradient-descent
         |
         ↓
+--------+--------+

| CONSTRAINED OPT |
+-----------------+
| • lagrange/     |--> Used in SVM!

| • kkt/          |--> Used in RLHF!
+--------+--------+
         |
         ↓
    Interior Point Methods
    (linear-programming/)

```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Boyd CVX Ch.5 | [Free PDF](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 | Nocedal Ch.12 | [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| 🎥 | KKT Conditions | [YouTube](https://www.youtube.com/watch?v=uh1Dk68cfWs) |
| 📄 | Original KKT Paper | [1951](https://doi.org/10.1525/9780520347663-014) |
| 🇨🇳 | 知乎 KKT条件 | [知乎](https://zhuanlan.zhihu.com/p/38163970) |
| 🇨🇳 | B站 拉格朗日 | [B站](https://www.bilibili.com/video/BV1HP4y1Y79e) |

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Convex Optimization](../04_convex_optimization/) | ➡️ [Next: Linear Programming](../06_linear_programming/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
