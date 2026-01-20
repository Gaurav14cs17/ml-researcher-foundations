<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Advanced%20Optimization%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/advanced-methods.svg" width="100%">

*Caption: Advanced optimization methods include Natural Gradient (invariant updates), Proximal Methods (non-smooth regularizers), and research frontiers like SAM, LAMB, Lion, and Sophia.*

---

## 📂 Overview

Advanced methods go beyond basic gradient descent by:

- Using curvature information (Quasi-Newton)

- Exploiting conjugate directions (CG)

- Handling non-smooth objectives (Proximal)

- Achieving scale invariance (Natural Gradient)

---

## 📊 Method Comparison

| Method | Convergence | Memory | Per-iteration Cost |
|--------|-------------|--------|-------------------|
| **GD** | O(1/t) | O(n) | O(n) |
| **Momentum** | O(1/t²) | O(n) | O(n) |
| **BFGS** | Superlinear | O(n²) | O(n²) |
| **L-BFGS** | Superlinear | O(mn) | O(mn) |
| **CG** | O(√κ) | O(n) | O(n) |
| **Newton** | Quadratic | O(n²) | O(n³) |

```
κ = condition number = λ_max / λ_min
Larger κ = harder optimization

```

---

# Part 1: Quasi-Newton Methods (BFGS, L-BFGS)

## 📐 Mathematical Foundations

### Core Idea

```
Newton: x_{k+1} = x_k - H_k⁻¹ ∇f_k   (requires O(n²) Hessian)

Quasi-Newton: x_{k+1} = x_k - B_k⁻¹ ∇f_k   (B_k approximates H_k)

Key insight: Build B from gradient differences only!

```

### Secant Condition

```
The approximation B_{k+1} must satisfy:

B_{k+1} · s_k = y_k

where:
  s_k = x_{k+1} - x_k        (step taken)
  y_k = ∇f_{k+1} - ∇f_k      (gradient change)

This matches curvature along the direction traveled

```

### BFGS Update Formula

```
B_{k+1} = B_k + (y_k y_kᵀ)/(y_kᵀ s_k) - (B_k s_k s_kᵀ B_k)/(s_kᵀ B_k s_k)

Properties:
• Maintains positive definiteness (if B_0 > 0 and y_kᵀs_k > 0)
• Rank-2 update (efficient)
• Self-correcting behavior

```

### Inverse BFGS (Direct H⁻¹ Update)

```
H_{k+1} = (I - ρ_k s_k y_kᵀ) H_k (I - ρ_k y_k s_kᵀ) + ρ_k s_k s_kᵀ

where ρ_k = 1/(y_kᵀ s_k)

Sherman-Morrison formula avoids matrix inversion!

```

### L-BFGS (Limited Memory)

```
Store only last m pairs: {s_i, y_i} for i = k-m+1, ..., k

Memory: O(mn) instead of O(n²)
Compute H_k · g via two-loop recursion:

Algorithm (Two-Loop Recursion):
1. q = ∇f_k
2. for i = k-1, ..., k-m:
      α_i = ρ_i s_iᵀ q
      q = q - α_i y_i
3. r = H_0 · q   (H_0 = γI, γ = s_{k-1}ᵀy_{k-1}/y_{k-1}ᵀy_{k-1})
4. for i = k-m, ..., k-1:
      β = ρ_i y_iᵀ r
      r = r + (α_i - β) s_i
5. return r = H_k ∇f_k

```

---

## 💻 Code Example

```python
import numpy as np

def bfgs(f, grad_f, x0, tol=1e-6, max_iter=1000):
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  # Initial inverse Hessian approximation
    
    g = grad_f(x)
    
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        
        # Search direction
        d = -H @ g
        
        # Line search
        alpha = backtracking(f, grad_f, x, d)
        
        # Update
        s = alpha * d
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g
        
        # BFGS update (inverse form)
        rho = 1.0 / (y @ s)
        I = np.eye(n)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s))
        H += rho * np.outer(s, s)
        
        x, g = x_new, g_new
    
    return x

def lbfgs(f, grad_f, x0, m=10, tol=1e-6, max_iter=1000):
    """L-BFGS with limited memory m"""
    x = x0.copy()
    g = grad_f(x)
    
    s_list, y_list = [], []
    
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        
        # Two-loop recursion to compute H·g
        q = g.copy()
        alphas = []
        
        for s, y in reversed(list(zip(s_list, y_list))):
            rho = 1.0 / (y @ s)
            alpha = rho * (s @ q)
            alphas.append(alpha)
            q = q - alpha * y
        
        # Initial scaling
        if s_list:
            gamma = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
        else:
            gamma = 1.0
        r = gamma * q
        
        for (s, y), alpha in zip(zip(s_list, y_list), reversed(alphas)):
            rho = 1.0 / (y @ s)
            beta = rho * (y @ r)
            r = r + (alpha - beta) * s
        
        d = -r  # Search direction
        
        # Line search and update
        alpha = backtracking(f, grad_f, x, d)
        s = alpha * d
        x = x + s
        g_new = grad_f(x)
        y = g_new - g
        
        # Update memory
        if len(s_list) >= m:
            s_list.pop(0)
            y_list.pop(0)
        s_list.append(s)
        y_list.append(y)
        
        g = g_new
    
    return x

```

---

## 📊 Convergence Comparison

| Method | Convergence Rate | Per-Iteration Cost | Memory |
|--------|------------------|--------------------| -------|
| Gradient Descent | Linear κ | O(n) | O(n) |
| Newton | Quadratic | O(n³) | O(n²) |
| BFGS | Superlinear | O(n²) | O(n²) |
| L-BFGS | Superlinear | O(mn) | O(mn) |

```
Superlinear means:
‖x_{k+1} - x*‖ / ‖x_k - x*‖ → 0 as k → ∞

Faster than linear, slower than quadratic

```

---

# Part 2: Conjugate Gradient Method

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
+--------------------+----------------------------+

| Preconditioner     | Description                |
+--------------------+----------------------------+

| Jacobi (M = D)     | Diagonal of A              |
| SSOR               | Symmetric SOR              |
| Incomplete Cholesky| Approximate L Lᵀ          |
| Multigrid          | Hierarchical approach      |
+--------------------+----------------------------+

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

## 🚀 Modern Research Frontiers

### SAM (Sharpness-Aware Minimization)

```
Find flat minima for better generalization:

    min_θ max_{||ε||≤ρ} L(θ + ε)

Approximation:
    1. Compute ε = ρ ∇L(θ) / ||∇L(θ)||
    2. Update θ ← θ - α ∇L(θ + ε)

```

### Lion Optimizer

```
Memory-efficient alternative to Adam:
    Uses sign() instead of full gradient moments
    Much lower memory than Adam
    Strong performance on transformers

```

### Sophia (2nd-Order for LLMs)

```
Scalable 2nd-order optimizer for LLMs:
    Uses Hessian diagonal approximation
    Clip updates based on Hessian
    Faster convergence than Adam

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Nocedal & Wright Ch. 6-7 | Numerical Optimization |
| 📄 | Original BFGS (1970) | Broyden, Fletcher, Goldfarb, Shanno |
| 📄 | L-BFGS Paper | Liu & Nocedal, 1989 |
| 📖 | Golub & Van Loan Ch. 10 | Matrix Computations |
| 📄 | Shewchuk (1994) | [An Introduction to CG](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf) |
| 📄 | SAM Paper | [arXiv](https://arxiv.org/abs/2010.01412) |
| 📄 | Lion Paper | [arXiv](https://arxiv.org/abs/2302.06675) |
| 🇨🇳 | 拟牛顿法详解 | [知乎](https://zhuanlan.zhihu.com/p/29672873) |
| 🇨🇳 | 共轭梯度法 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88819089) |

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Basic Methods](../02_basic_methods/) | ➡️ [Next: Convex Optimization](../04_convex_optimization/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
