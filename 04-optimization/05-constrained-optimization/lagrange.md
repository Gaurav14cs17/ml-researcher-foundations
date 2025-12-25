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

> **Elegant method for constrained optimization with equality constraints**

---

## 🎯 Visual Overview

<img src="./lagrange/images/lagrange.svg" width="100%">

*Caption: At the optimum, the gradient of the objective is parallel to the constraint gradient. The Lagrange multiplier λ measures the sensitivity of the objective to the constraint.*

---

## 📐 Mathematical Foundations

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

### Geometric Intuition

```
At optimum on constraint surface:

       ∇f           ∇g
        ↑          ↗
         \        /
          \      /
           \    /
            \  /
        ─────*───── constraint g(x) = 0

∇f must be perpendicular to constraint surface
(otherwise we could move along surface and improve)

This means ∇f is parallel to ∇g: ∇f = λ∇g
```

### Second-Order Conditions

```
Bordered Hessian:
H_L = ∇²ₓₓL = ∇²f - Σᵢ λᵢ∇²gᵢ

For minimum: H_L positive definite on tangent space of constraints

Tangent space: {v : ∇gᵢ(x*)ᵀv = 0 for all i}

Check: vᵀ H_L v > 0 for all v in tangent space
```

### Sensitivity Analysis (Shadow Prices)

```
Consider: min f(x) s.t. g(x) = b

Let f*(b) = optimal value as function of RHS

Then:
∂f*/∂bᵢ = λᵢ*

Interpretation:
λᵢ* = marginal value of relaxing constraint i
"How much would objective improve if we had one more unit of resource i?"
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

# Augmented Lagrangian Method
def augmented_lagrangian(f, g, x0, lambda0=0, mu=1.0, 
                         max_outer=50, max_inner=100, tol=1e-6):
    """
    Augmented Lagrangian: L_A(x,λ,μ) = f(x) + λg(x) + (μ/2)g(x)²
    
    Alternates:
    1. Minimize L_A over x (unconstrained)
    2. Update λ = λ + μg(x)
    3. Optionally increase μ
    """
    x = np.array(x0, dtype=waving)
    lam = lambda0
    
    for outer in range(max_outer):
        # Augmented Lagrangian
        def L_A(x):
            gx = g(x)
            return f(x) + lam * gx + (mu/2) * gx**2
        
        # Minimize over x
        result = minimize(L_A, x, method='BFGS')
        x = result.x
        
        # Check constraint violation
        gx = g(x)
        if abs(gx) < tol:
            print(f"Converged in {outer+1} outer iterations")
            break
        
        # Update multiplier
        lam = lam + mu * gx
        
        # Optionally increase penalty
        mu = min(mu * 1.5, 1e6)
    
    return x, lam

result = lagrange_example()
```

---

## 📊 Method Comparison

| Method | Handles | Pros | Cons |
|--------|---------|------|------|
| **Lagrange** | Equality only | Elegant theory | Needs analytical solution |
| **Penalty** | Any constraints | Simple | Ill-conditioning as μ→∞ |
| **Augmented Lagrangian** | Any constraints | Bounded μ | More complex |
| **Interior Point** | Inequality | Polynomial time | Complex implementation |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Boyd & Vandenberghe Ch. 5 | [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 | Nocedal & Wright Ch. 12 | Numerical Optimization |
| 🎥 | MIT 18.02 | [Lagrange Multipliers](https://ocw.mit.edu/) |
| 📄 | Lagrange (1788) | Mécanique analytique (original) |
| 🇨🇳 | 拉格朗日乘数法 | [知乎](https://zhuanlan.zhihu.com/p/38182879) |
| 🇨🇳 | 约束优化详解 | [CSDN](https://blog.csdn.net/google19890102/article/details/45920107) |

---

---

⬅️ [Back: Kkt](./kkt.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
