# Conjugate Gradient Method

> **Efficient iterative solver for large symmetric positive-definite systems**

---

## 🎯 Visual Overview

<img src="./images/advanced-methods.svg" width="100%">

*Caption: Conjugate Gradient generates A-orthogonal search directions, solving Ax = b in at most n iterations for n×n systems.*

---

## 📐 Mathematical Foundations

### Problem Setting

```
Solve: Ax = b  where A is symmetric positive definite (SPD)

Equivalent to minimizing:
f(x) = ½xᵀAx - bᵀx

Gradient: ∇f(x) = Ax - b = -r  (negative residual)
```

### Conjugate (A-orthogonal) Directions

```
Vectors p₀, p₁, ..., pₙ₋₁ are A-conjugate if:

pᵢᵀ A pⱼ = 0  for i ≠ j

Key insight: With n A-conjugate directions, solve in n steps!

Expansion: x* = Σᵢ αᵢ pᵢ  where αᵢ = (pᵢᵀb)/(pᵢᵀApᵢ)
```

### CG Algorithm Derivation

```
Start with residual r₀ = b - Ax₀ = -∇f(x₀)

Build conjugate directions from residuals:
pₖ = rₖ + βₖ pₖ₋₁

Choose βₖ to ensure pₖᵀ A pₖ₋₁ = 0:
βₖ = (rₖᵀrₖ)/(rₖ₋₁ᵀrₖ₋₁)

Optimal step along pₖ:
αₖ = (rₖᵀrₖ)/(pₖᵀApₖ)
```

### The Complete CG Iteration

```
Given: A (SPD), b, x₀

Initialize:
r₀ = b - Ax₀
p₀ = r₀

For k = 0, 1, 2, ...:
    αₖ = (rₖᵀrₖ) / (pₖᵀApₖ)       # Step size
    xₖ₊₁ = xₖ + αₖpₖ               # Update solution
    rₖ₊₁ = rₖ - αₖApₖ              # Update residual
    
    if ‖rₖ₊₁‖ < tol: STOP
    
    βₖ = (rₖ₊₁ᵀrₖ₊₁) / (rₖᵀrₖ)    # CG coefficient
    pₖ₊₁ = rₖ₊₁ + βₖpₖ             # New search direction

Key properties:
• rᵢᵀrⱼ = 0 for i ≠ j (residuals orthogonal)
• pᵢᵀApⱼ = 0 for i ≠ j (directions A-conjugate)
• Converges in ≤ n iterations (exact arithmetic)
```

### Convergence Rate

```
Error bound:
‖xₖ - x*‖_A ≤ 2 · ((√κ - 1)/(√κ + 1))^k · ‖x₀ - x*‖_A

where κ = λₘₐₓ/λₘᵢₙ (condition number)

Compare to Gradient Descent:
GD: ((κ-1)/(κ+1))^k
CG: ((√κ-1)/(√κ+1))^k ≈ ((√κ-1)/(√κ+1))^k

CG is MUCH faster for ill-conditioned problems!
Example: κ = 100
  GD ratio: 0.98^k
  CG ratio: 0.82^k (sqrt effect!)
```

---

## 💻 Code Implementation

```python
import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Solve Ax = b using Conjugate Gradient
    A must be symmetric positive definite
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    if max_iter is None:
        max_iter = n
    
    r = b - A @ x      # Residual
    p = r.copy()       # Search direction
    rs_old = r @ r     # r'r
    
    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            print(f"Converged in {k+1} iterations")
            break
        
        beta = rs_new / rs_old
        p = r + beta * p
        
        rs_old = rs_new
    
    return x

def preconditioned_cg(A, b, M_inv, x0=None, tol=1e-10, max_iter=None):
    """
    Preconditioned CG: Solve Ax = b with preconditioner M
    M_inv is a function that computes M⁻¹v
    
    Effective condition number: κ(M⁻¹A) << κ(A)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    
    r = b - A @ x
    z = M_inv(r)       # Preconditioned residual
    p = z.copy()
    rz_old = r @ z
    
    for k in range(max_iter or n):
        Ap = A @ p
        alpha = rz_old / (p @ Ap)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        if np.linalg.norm(r) < tol:
            break
        
        z = M_inv(r)
        rz_new = r @ z
        
        beta = rz_new / rz_old
        p = z + beta * p
        
        rz_old = rz_new
    
    return x

# Example usage
n = 100
A = np.random.randn(n, n)
A = A.T @ A + np.eye(n)  # Make SPD
b = np.random.randn(n)

x_cg = conjugate_gradient(A, b)
x_exact = np.linalg.solve(A, b)
print(f"Error: {np.linalg.norm(x_cg - x_exact)}")
```

---

## 🔧 Preconditioning

```
Goal: Transform Ax = b to M⁻¹Ax = M⁻¹b

Good preconditioner M satisfies:
• M⁻¹A has small condition number
• M⁻¹v is cheap to compute

Common preconditioners:
┌────────────────────┬────────────────────────────┐
│ Preconditioner     │ Description                │
├────────────────────┼────────────────────────────┤
│ Jacobi (M = D)     │ Diagonal of A              │
│ SSOR               │ Symmetric SOR              │
│ Incomplete Cholesky│ Approximate L Lᵀ          │
│ Multigrid          │ Hierarchical approach      │
└────────────────────┴────────────────────────────┘
```

---

## 📊 Nonlinear CG (for Optimization)

```
Extend CG to general f(x):

Fletcher-Reeves:
βₖᶠᴿ = (∇fₖ₊₁ᵀ∇fₖ₊₁) / (∇fₖᵀ∇fₖ)

Polak-Ribière:
βₖᴾᴿ = (∇fₖ₊₁ᵀ(∇fₖ₊₁ - ∇fₖ)) / (∇fₖᵀ∇fₖ)

Hestenes-Stiefel:
βₖᴴˢ = (∇fₖ₊₁ᵀ(∇fₖ₊₁ - ∇fₖ)) / (pₖᵀ(∇fₖ₊₁ - ∇fₖ))

Restart: Set β = 0 when directions lose conjugacy
```

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Golub & Van Loan Ch. 10 | Matrix Computations |
| 📖 | Nocedal & Wright Ch. 5 | Numerical Optimization |
| 📄 | Shewchuk (1994) | [An Introduction to CG](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf) |
| 🎥 | MIT 18.086 | [CG Lecture](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/) |
| 🇨🇳 | 共轭梯度法详解 | [知乎](https://zhuanlan.zhihu.com/p/98642663) |
| 🇨🇳 | CG算法原理 | [CSDN](https://blog.csdn.net/lusongno1/article/details/78550803) |

---

---

➡️ [Next: Quasi Newton](./quasi-newton.md)
