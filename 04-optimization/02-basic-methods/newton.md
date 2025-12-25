<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Newtons%20Method&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./newton/images/newton-method.svg" width="100%">

*Caption: Newton's method uses curvature (Hessian) to find the optimal step size, achieving quadratic convergence near the optimum.*

---

## 📐 Mathematical Foundations

### Newton Update Rule

```
Standard Newton Step:
x_{k+1} = x_k - H(x_k)⁻¹∇f(x_k)

Where:
• H(x) = ∇²f(x) is the Hessian matrix
• ∇f(x) is the gradient
• H⁻¹∇f is the Newton direction
```

### Derivation from Taylor Expansion

```
Second-order Taylor approximation:
f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + ½Δxᵀ H Δx

Setting derivative to zero:
∇f(x) + H·Δx = 0

Solving for optimal step:
Δx* = -H⁻¹∇f(x)
```

### Newton Decrement

```
λ² = ∇f(x)ᵀ H(x)⁻¹ ∇f(x)

Interpretation:
• λ² ≈ f(x) - f(x*)  (approximate suboptimality)
• Stopping criterion: λ² < ε
```

### Convergence Analysis

```
Near optimum (local convergence):
‖x_{k+1} - x*‖ ≤ C · ‖x_k - x*‖²

Quadratic convergence means:
• Error squares each iteration
• 10⁻² → 10⁻⁴ → 10⁻⁸ → 10⁻¹⁶
• Very fast once "close enough"

Global convergence (with damping):
x_{k+1} = x_k - α_k · H⁻¹∇f

where α_k found by line search
```

---

## 💻 Algorithm

```python
def newton_method(f, grad_f, hess_f, x0, tol=1e-8, max_iter=100):
    x = x0
    for k in range(max_iter):
        g = grad_f(x)        # Gradient
        H = hess_f(x)        # Hessian
        
        # Newton direction: solve H·d = -g
        d = np.linalg.solve(H, -g)
        
        # Newton decrement (stopping criterion)
        lambda_sq = -g @ d
        if lambda_sq / 2 < tol:
            break
        
        # Damped Newton with backtracking
        alpha = backtracking_line_search(f, x, g, d)
        x = x + alpha * d
    
    return x

def backtracking_line_search(f, x, g, d, alpha=1.0, beta=0.5, c=0.1):
    """Armijo backtracking line search"""
    while f(x + alpha * d) > f(x) + c * alpha * (g @ d):
        alpha *= beta
    return alpha
```

---

## 📊 Comparison with Gradient Descent

| Property | Gradient Descent | Newton's Method |
|----------|------------------|-----------------|
| **Convergence** | Linear O(κ log(1/ε)) | Quadratic O(log log(1/ε)) |
| **Per-iteration cost** | O(n) | O(n³) for Hessian inverse |
| **Memory** | O(n) | O(n²) for Hessian |
| **Condition number** | Sensitive to κ | Affine invariant |
| **Step size** | Needs tuning | Natural scaling |

---

## ⚠️ Challenges

```
1. Hessian computation: O(n²) storage, O(n³) solve
2. Non-positive definite H: Direction may not be descent
3. Far from optimum: May diverge without damping
4. Saddle points: H singular or indefinite

Solutions:
• Regularization: H + λI (Levenberg-Marquardt)
• Modified Newton: Use |H| eigenvalues
• Trust region: Constrain step size
• Line search: Ensure descent
```

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Boyd & Vandenberghe Ch. 9 | [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 | Nocedal & Wright Ch. 3 | Numerical Optimization |
| 🎥 | MIT 18.065 Lecture | [YouTube](https://www.youtube.com/watch?v=AaepZWMGU3w) |
| 🇨🇳 | 牛顿法详解 | [知乎](https://zhuanlan.zhihu.com/p/37524275) |

---

---

⬅️ [Back: Gradient Descent](./gradient-descent.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
