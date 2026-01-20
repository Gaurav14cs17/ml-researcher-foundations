<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Convex%20Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Why Convexity Matters

```
+---------------------------------------------------------+
|                                                         |
|   Non-convex:                 Convex:                   |
|                                                         |
|       •                           •                     |
|      / \                         / \                    |
|     /   \   •                   /   \                   |
|    •     \ / \                 /     \                  |
|           •   •               •-------•                 |
|                                                         |
|   Many local minima          ONE global minimum         |
|   Hard to solve              Efficiently solvable       |
|   No guarantees              Polynomial algorithms      |
|                                                         |
+---------------------------------------------------------+

```

---

## 📐 Mathematical Definitions

### Convex Set

```
A set C ⊆ ℝⁿ is convex if:

∀x, y ∈ C and ∀θ ∈ [0,1]:
θx + (1-θ)y ∈ C

Interpretation:
"Line segment between any two points lies entirely in C"

Examples:
✓ Hyperplanes: {x : aᵀx = b}
✓ Halfspaces: {x : aᵀx ≤ b}
✓ Balls: {x : ||x - c|| ≤ r}
✓ Polyhedra: {x : Ax ≤ b}
✗ Non-convex: donut shape, star shape

```

### Convex Function

```
A function f: ℝⁿ → ℝ is convex if:

f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)

for all x, y ∈ dom(f) and θ ∈ [0,1]

Interpretation:
"Chord lies above the graph"

       f(y)●
           ╲ chord
            ╲
       f(θx+(1-θ)y)  ≤  θf(x)+(1-θ)f(y)
              ●
             ╱ graph
            ╱
       f(x)●

```

### Strict Convexity

```
f is strictly convex if:

f(θx + (1-θ)y) < θf(x) + (1-θ)f(y)

for all x ≠ y and θ ∈ (0,1)

Implication: Unique global minimum (if exists)

```

### Strong Convexity

```
f is μ-strongly convex if:

f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²

Equivalent condition:
f(x) - (μ/2)||x||² is convex

Properties:
• Unique minimum exists
• Gradient descent converges linearly
• Condition number κ = L/μ bounds convergence

```

---

## 📐 First and Second Order Conditions

### First-Order Condition

```
If f is differentiable:

f convex  ⟺  f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)  for all x, y

Interpretation:
"Tangent plane lies below the graph"

Consequence:
∇f(x*) = 0  ⟹  x* is GLOBAL minimum

```

### Second-Order Condition

```
If f is twice differentiable:

f convex  ⟺  ∇²f(x) ⪰ 0 (positive semi-definite) for all x

f strictly convex  ⟺  ∇²f(x) ≻ 0 (positive definite)

How to check:
• All eigenvalues of Hessian ≥ 0
• Principal minors ≥ 0
• Cholesky decomposition exists

```

---

## 📐 Examples of Convex Functions

### Common Convex Functions

```
1. Linear: f(x) = aᵀx + b
   Hessian: ∇²f = 0 ⪰ 0  ✓

2. Affine: f(x) = Ax + b
   (both convex and concave)

3. Quadratic: f(x) = (1/2)xᵀPx + qᵀx + r
   Convex iff P ⪰ 0

4. Norms: f(x) = ||x||_p for p ≥ 1
   Always convex (triangle inequality)

5. Exponential: f(x) = eˣ
   f''(x) = eˣ > 0  ✓

6. Log-sum-exp: f(x) = log(Σᵢ eˣⁱ)
   Smooth approximation to max function

7. Negative entropy: f(x) = Σᵢ xᵢ log xᵢ
   Convex on x > 0

```

### Common Concave Functions

```
1. Logarithm: f(x) = log(x)
   f''(x) = -1/x² < 0

2. Square root: f(x) = √x
   f''(x) = -1/(4x^(3/2)) < 0

3. Geometric mean: f(x) = (Πᵢ xᵢ)^(1/n)
   Concave on x > 0

```

---

## 📐 Preserving Convexity

### Operations That Preserve Convexity

```
1. Non-negative weighted sum:
   f = Σᵢ αᵢfᵢ with αᵢ ≥ 0
   convex if all fᵢ convex

2. Composition with affine:
   g(x) = f(Ax + b)
   convex if f convex

3. Pointwise maximum:
   f(x) = max{f₁(x), f₂(x), ..., fₖ(x)}
   convex if all fᵢ convex

4. Perspective:
   g(x, t) = tf(x/t)
   convex if f convex, t > 0

5. Partial minimization:
   g(x) = inf_{y∈C} f(x, y)
   convex if f convex in (x,y) and C convex

```

---

## 📐 Convex Optimization Problem

### Standard Form

```
minimize    f₀(x)           (objective)
subject to  fᵢ(x) ≤ 0       i = 1,...,m  (inequalities)
            hⱼ(x) = 0       j = 1,...,p  (equalities)

Convex program if:
• f₀, f₁, ..., fₘ are convex
• h₁, ..., hₚ are affine

```

### Key Property

```
LOCAL minimum = GLOBAL minimum

Proof:
Suppose x* is local min but not global.
Then ∃y with f(y) < f(x*).
By convexity: f(θy + (1-θ)x*) ≤ θf(y) + (1-θ)f(x*) < f(x*)
for all θ ∈ (0,1].
But θy + (1-θ)x* can be arbitrarily close to x*.
Contradiction with x* being local min! ∎

```

---

## 📐 Duality Theory

### Lagrangian

```
L(x, λ, ν) = f₀(x) + Σᵢ λᵢfᵢ(x) + Σⱼ νⱼhⱼ(x)

where λᵢ ≥ 0 (for inequalities)
      νⱼ ∈ ℝ (for equalities)

```

### Dual Function

```
g(λ, ν) = inf_x L(x, λ, ν)

Properties:
• g is always concave (even if primal not convex)
• g(λ, ν) ≤ p* for any λ ≥ 0, ν  (weak duality)

```

### Dual Problem

```
maximize    g(λ, ν)
subject to  λ ≥ 0

• Always a convex problem!
• Optimal value d* ≤ p*

```

### Strong Duality

```
d* = p* when:
1. Slater's condition: ∃x strictly feasible
   (fᵢ(x) < 0 for all i, hⱼ(x) = 0)
2. Problem is convex

Applications:
• Dual provides lower bound
• Complementary slackness for KKT
• Economic interpretation (shadow prices)

```

---

## 📐 Optimality Conditions (KKT)

### KKT Conditions for Convex Problems

```
For convex problem with strong duality:
x*, λ*, ν* optimal iff:

1. Stationarity:
   ∇f₀(x*) + Σᵢ λᵢ*∇fᵢ(x*) + Σⱼ νⱼ*∇hⱼ(x*) = 0

2. Primal feasibility:
   fᵢ(x*) ≤ 0, hⱼ(x*) = 0

3. Dual feasibility:
   λᵢ* ≥ 0

4. Complementary slackness:
   λᵢ*fᵢ(x*) = 0  for all i

```

### Using KKT to Solve Problems

```
Strategy:
1. Write down KKT conditions
2. Consider cases (which constraints active?)
3. Solve resulting system of equations
4. Verify solution satisfies all conditions

```

---

## 📐 Convex Optimization Algorithms

### Gradient Descent

```
For unconstrained smooth convex f:

x_{k+1} = x_k - α∇f(x_k)

Convergence (L-smooth):
  f(x_k) - f* ≤ O(1/k)

Convergence (μ-strongly convex):
  f(x_k) - f* ≤ (1 - μ/L)^k (f(x_0) - f*)

```

### Projected Gradient Descent

```
For constrained problem min f(x) s.t. x ∈ C:

x_{k+1} = Π_C(x_k - α∇f(x_k))

where Π_C is projection onto C:
Π_C(y) = argmin_{x∈C} ||x - y||²

```

### Proximal Gradient

```
For f(x) = g(x) + h(x) where g smooth, h non-smooth:

x_{k+1} = prox_{αh}(x_k - α∇g(x_k))

where prox_{h}(y) = argmin_x {h(x) + (1/2)||x-y||²}

```

### Interior Point Methods

```
For constrained problems:

Replace fᵢ(x) ≤ 0 with barrier:
minimize f₀(x) - (1/t)Σᵢ log(-fᵢ(x))

As t → ∞, solution → original optimal
Complexity: O(√m log(1/ε)) Newton steps

```

---

## 💻 Code Examples

### Using CVXPY

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
print(f"Optimal x: {x.value}")

```

### Check Convexity

```python
import numpy as np
from scipy.linalg import eigvalsh

def is_convex_quadratic(P):
    """Check if f(x) = x'Px is convex"""
    eigenvalues = eigvalsh(P)
    return np.all(eigenvalues >= -1e-10)

def check_hessian_psd(hessian_fn, x, eps=1e-6):
    """Check if Hessian is PSD at point x"""
    H = hessian_fn(x)
    eigenvalues = eigvalsh(H)
    return np.all(eigenvalues >= -eps)

# Example
P = np.array([[2, 1], [1, 3]])
print(f"Quadratic is convex: {is_convex_quadratic(P)}")

```

### Gradient Descent for Convex Function

```python
import numpy as np

def gradient_descent_convex(f, grad_f, x0, alpha=0.01, 
                            max_iter=1000, tol=1e-6):
    """
    Gradient descent for convex function.
    Guaranteed to find global minimum!
    """
    x = x0.copy()
    
    for k in range(max_iter):
        g = grad_f(x)
        
        if np.linalg.norm(g) < tol:
            print(f"Converged in {k} iterations")
            break
            
        x = x - alpha * g
    
    return x

# Example: Minimize ||Ax - b||² (convex!)
A = np.random.randn(10, 5)
b = np.random.randn(10)

def f(x):
    return 0.5 * np.linalg.norm(A @ x - b)**2

def grad_f(x):
    return A.T @ (A @ x - b)

x0 = np.zeros(5)
x_opt = gradient_descent_convex(f, grad_f, x0)

# Compare with closed-form solution
x_closed = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"GD solution: {x_opt}")
print(f"Closed form: {x_closed}")

```

---

## 📊 Convex vs Non-Convex

| Aspect | Convex | Non-Convex |
|--------|--------|------------|
| **Local minima** | = Global | Many |
| **Algorithms** | Polynomial | NP-hard |
| **Guarantees** | Strong | Few |
| **Neural nets** | No | Yes |
| **Linear regression** | Yes | N/A |
| **SVMs** | Yes | No |
| **Logistic reg.** | Yes | No |

---

## 🌍 Applications

| Application | Problem Type | Method |
|-------------|--------------|--------|
| **Portfolio opt.** | Quadratic | Interior point |
| **Compressed sensing** | L1 minimization | ADMM |
| **SVM** | Quadratic | SMO |
| **Logistic regression** | Log-likelihood | Newton |
| **Optimal transport** | Linear program | Sinkhorn |
| **Control** | SDP, SOCP | Interior point |

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Boyd Convex Optimization | [Free PDF](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 | Nesterov Intro Lectures | [Springer](https://link.springer.com/book/10.1007/978-1-4419-8853-9) |
| 🛠️ | CVXPY | [cvxpy.org](https://www.cvxpy.org/) |
| 🎥 | Stanford EE364a | [YouTube](https://www.youtube.com/playlist?list=PL3940DD956CDF0622) |
| 🇨🇳 | 凸优化基础 | [知乎](https://zhuanlan.zhihu.com/p/25385801) |

---

⬅️ [Back: Advanced Methods](../03_advanced_methods/) | ➡️ [Next: Convex Optimization (Main)](../04_convex_optimization/)

> **Note:** This folder covers additional convex optimization topics. See [04_convex_optimization](../04_convex_optimization/) for the main convex optimization content.

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
