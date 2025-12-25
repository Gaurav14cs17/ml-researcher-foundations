<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Interior%20Point%20Methods&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Interior Point Methods

> **Polynomial-time algorithms that traverse the interior of the feasible region**

---

## 🎯 Visual Overview

<img src="./images/linear-programming.svg" width="100%">

*Caption: Interior point methods follow the central path through the interior, converging to the optimal solution in polynomial time O(√n log(1/ε)).*

---

## 📐 Mathematical Foundations

### Barrier Method

```
Original LP:
    min  cᵀx
    s.t. Ax = b, x ≥ 0

Barrier formulation:
    min  cᵀx - μ Σᵢ log(xᵢ)
    s.t. Ax = b

• Barrier: -log(x) → ∞ as x → 0 (keeps iterates interior)
• μ > 0: barrier parameter
• As μ → 0, solution approaches LP optimum
```

### Central Path

```
For each μ > 0, barrier problem has unique solution x*(μ)

Central path: { x*(μ) : μ > 0 }

Properties:
• x*(μ) strictly in interior (all xᵢ > 0)
• As μ → 0⁺, x*(μ) → x* (LP solution)
• Path is smooth curve through interior
```

### KKT Conditions for Barrier Problem

```
Optimality conditions at x*(μ):

∇ₓL = c - μX⁻¹e + Aᵀλ = 0
Ax = b
x > 0

Where X = diag(x), e = (1,1,...,1)ᵀ

Rearranging: Xλ = μe  (complementarity modified)
```

### Primal-Dual Interior Point

```
Solve primal AND dual simultaneously!

Primal: min cᵀx,  Ax = b, x ≥ 0
Dual:   max bᵀy,  Aᵀy + s = c, s ≥ 0

Modified KKT system:
┌           ┐ ┌    ┐   ┌        ┐
│  A   0  0 │ │ Δx │   │ b - Ax │
│  0  Aᵀ  I │ │ Δy │ = │ c-Aᵀy-s│
│  S   0  X │ │ Δs │   │ μe - XSe│
└           ┘ └    ┘   └        ┘

Where X = diag(x), S = diag(s)
```

### Complexity Analysis

```
Number of iterations: O(√n log(1/ε))

Per iteration: O(n³) for solving linear system

Total: O(n^3.5 log(1/ε))

Compare to Simplex:
• Simplex: O(2ⁿ) worst case, O(m) average
• Interior: O(√n) iterations (polynomial guaranteed)
```

---

## 💻 Algorithm: Path-Following

```python
import numpy as np

def interior_point_lp(c, A, b, tol=1e-8, mu_factor=0.1):
    """
    Primal-Dual Interior Point for LP
    min cᵀx s.t. Ax = b, x ≥ 0
    """
    m, n = A.shape
    
    # Initialize (strictly feasible)
    x = np.ones(n)
    s = np.ones(n)  # Slack for dual
    y = np.zeros(m)  # Dual variables
    
    for iteration in range(100):
        # Duality gap and barrier parameter
        mu = (x @ s) / n
        if mu < tol:
            break
        
        # Target (reduce μ)
        sigma = mu_factor
        mu_target = sigma * mu
        
        # Residuals
        r_primal = b - A @ x
        r_dual = c - A.T @ y - s
        r_cent = mu_target * np.ones(n) - x * s
        
        # Form and solve Newton system
        # [  A    0    0  ] [dx]   [r_primal]
        # [  0   Aᵀ   I  ] [dy] = [r_dual  ]
        # [  S    0    X  ] [ds]   [r_cent  ]
        
        X_inv = np.diag(1/x)
        S = np.diag(s)
        
        # Schur complement: (A X⁻¹ S Aᵀ) dy = ...
        M = A @ X_inv @ S @ A.T
        rhs = r_primal + A @ X_inv @ (r_cent - x * r_dual)
        
        dy = np.linalg.solve(M, rhs)
        ds = r_dual - A.T @ dy
        dx = X_inv @ (r_cent - x * ds)
        
        # Step size (stay interior: x, s > 0)
        alpha_primal = min(1.0, 0.99 * min(-x[dx<0] / dx[dx<0], default=1.0))
        alpha_dual = min(1.0, 0.99 * min(-s[ds<0] / ds[ds<0], default=1.0))
        
        # Update
        x = x + alpha_primal * dx
        y = y + alpha_dual * dy
        s = s + alpha_dual * ds
        
        print(f"Iter {iteration}: μ = {mu:.2e}, obj = {c@x:.6f}")
    
    return x, c @ x

# Example
c = np.array([-1, -2])  # min -x₁ - 2x₂ (maximize x₁ + 2x₂)
A = np.array([[1, 1, 1, 0],
              [2, 1, 0, 1]])  # With slacks
b = np.array([4, 5])

# Note: Need to handle equality constraints properly
```

---

## 📊 Comparison: Simplex vs Interior Point

| Aspect | Simplex | Interior Point |
|--------|---------|----------------|
| **Path** | Along edges (vertices) | Through interior |
| **Complexity** | Exponential worst case | Polynomial O(√n) |
| **Practice** | Fast for sparse/small | Fast for dense/large |
| **Warm start** | Easy | Difficult |
| **Sensitivity** | Direct from basis | From KKT |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Boyd & Vandenberghe Ch. 11 | [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) |
| 📄 | Karmarkar (1984) | Original polynomial-time algorithm |
| 📄 | Mehrotra (1992) | Predictor-corrector method |
| 🎥 | MIT 6.251J | [Interior Point Lecture](https://ocw.mit.edu/) |
| 🇨🇳 | 内点法详解 | [知乎](https://zhuanlan.zhihu.com/p/48476987) |
| 🇨🇳 | 障碍函数法 | [CSDN](https://blog.csdn.net/dymodi/article/details/46570613) |

---

---

⬅️ [Back: Duality](./duality.md) | ➡️ [Next: Simplex](./simplex.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
