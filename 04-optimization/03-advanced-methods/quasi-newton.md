<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Quasi-Newton%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/advanced-methods.svg" width="100%">

*Caption: Quasi-Newton methods build curvature approximation B ≈ H from gradient differences, achieving superlinear convergence without computing the actual Hessian.*

---

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

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Nocedal & Wright Ch. 6-7 | Numerical Optimization |
| 📄 | Original BFGS (1970) | Broyden, Fletcher, Goldfarb, Shanno |
| 📄 | L-BFGS Paper | Liu & Nocedal, 1989 |
| 🎥 | BFGS Explained | [YouTube](https://www.youtube.com/watch?v=I39UHs9ux4o) |
| 🇨🇳 | 拟牛顿法详解 | [知乎](https://zhuanlan.zhihu.com/p/29672873) |
| 🇨🇳 | L-BFGS原理 | [CSDN](https://blog.csdn.net/google19890102/article/details/47779147) |

---

---

⬅️ [Back: Conjugate Gradient](./conjugate-gradient.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
