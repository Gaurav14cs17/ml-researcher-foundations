<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Linear%20Algebra%20for%20Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Key Concepts for Optimization

Linear algebra provides the mathematical foundation for understanding optimization algorithms. These concepts appear in every optimization paper and algorithm.

---

## 🎯 Gradients and Directional Derivatives

### Gradient Vector

```
For f: ℝⁿ → ℝ, the gradient is:

∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

Properties:
• Points in direction of steepest ASCENT
• Magnitude = rate of change in that direction
• Perpendicular to level sets of f
```

### Directional Derivative

```
The rate of change of f in direction d:

D_d f(x) = ∇f(x)ᵀ d / ||d||

Key insight:
• Maximum when d = ∇f(x)  (steepest ascent)
• Minimum when d = -∇f(x) (steepest descent)
• Zero when d ⊥ ∇f(x)     (level set direction)
```

---

## 📐 Hessian Matrix

### Definition

```
The Hessian H is the matrix of second derivatives:

H(x) = [∂²f/∂xᵢ∂xⱼ] = 
       +                           +
       | ∂²f/∂x₁² ... ∂²f/∂x₁∂xₙ  |
       |    ⋮      ⋱      ⋮        |
       | ∂²f/∂xₙ∂x₁ ... ∂²f/∂xₙ² |
       +                           +

Properties:
• Symmetric (if f is twice continuously differentiable)
• Size: n × n for f: ℝⁿ → ℝ
• Captures curvature information
```

### Second-Order Taylor Expansion

```
f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + (1/2)ΔxᵀH(x)Δx

This approximation is crucial for:
• Newton's method (uses full Hessian)
• Quasi-Newton methods (approximates Hessian)
• Trust region methods (constrained quadratic)
```

---

## 📐 Positive Definiteness

### Definition

```
A symmetric matrix A is:

Positive definite (PD):      xᵀAx > 0  for all x ≠ 0
Positive semi-definite (PSD): xᵀAx ≥ 0  for all x
Negative definite:           xᵀAx < 0  for all x ≠ 0
Indefinite:                  xᵀAx can be positive or negative
```

### Importance for Optimization

```
At a stationary point x* where ∇f(x*) = 0:

• H(x*) ≻ 0 (positive definite)  ⟹ x* is a LOCAL MINIMUM
• H(x*) ≺ 0 (negative definite)  ⟹ x* is a LOCAL MAXIMUM
• H(x*) indefinite              ⟹ x* is a SADDLE POINT
```

### How to Check Positive Definiteness

```
Method 1: Eigenvalues
  A ≻ 0  ⟺  all eigenvalues λᵢ > 0

Method 2: Cholesky Decomposition
  A ≻ 0  ⟺  Cholesky decomposition A = LLᵀ exists

Method 3: Leading Principal Minors (Sylvester's criterion)
  A ≻ 0  ⟺  all leading principal minors are positive
  
  det([a₁₁]) > 0
  det([a₁₁ a₁₂; a₂₁ a₂₂]) > 0
  det([a₁₁ a₁₂ a₁₃; a₂₁ a₂₂ a₂₃; a₃₁ a₃₂ a₃₃]) > 0
  ...
```

---

## 📐 Eigenvalues and Eigenvectors

### Definition

```
For matrix A, scalar λ and vector v ≠ 0:

Av = λv

• λ is an eigenvalue
• v is the corresponding eigenvector
```

### Properties for Symmetric Matrices

```
For symmetric A = Aᵀ:

1. All eigenvalues are real
2. Eigenvectors are orthogonal
3. A = QΛQᵀ (spectral decomposition)
   where:
   Q = [v₁, v₂, ..., vₙ]  (orthonormal eigenvectors)
   Λ = diag(λ₁, λ₂, ..., λₙ)  (eigenvalues)
```

### Role in Optimization

```
For quadratic f(x) = (1/2)xᵀAx - bᵀx:

• Gradient: ∇f(x) = Ax - b
• Hessian: H = A
• Eigenvalues of A determine convergence rate:

  Condition number κ = λ_max / λ_min
  
  κ ≈ 1:  Well-conditioned, fast convergence
  κ >> 1: Ill-conditioned, slow convergence
```

---

## 📐 Condition Number

### Definition

```
For a matrix A, the condition number is:

κ(A) = ||A|| · ||A⁻¹|| = σ_max / σ_min  (for 2-norm)

For symmetric positive definite A:
κ(A) = λ_max / λ_min
```

### Interpretation

```
κ measures how sensitive Ax = b is to perturbations:

||Δx||/||x|| ≤ κ(A) · ||Δb||/||b||

In optimization:
• Small κ → Fast convergence
• Large κ → Slow convergence (ill-conditioned)

GD convergence rate for quadratic:
  Error decreases by factor (κ-1)/(κ+1) per iteration
  
  κ = 2:    factor = 1/3    (fast)
  κ = 100:  factor = 0.98   (slow)
  κ = 10000: factor = 0.9998 (very slow!)
```

### Improving Conditioning: Preconditioning

```
Instead of solving Ax = b, solve:
  M⁻¹Ax = M⁻¹b

If M ≈ A, then κ(M⁻¹A) ≈ 1 → fast convergence

Common preconditioners:
• Diagonal: M = diag(A)
• Jacobi: M = diag(A)
• Incomplete Cholesky: M = L̃L̃ᵀ
```

---

## 📐 Matrix Norms

### Common Norms

```
Frobenius norm:
||A||_F = √(Σᵢⱼ aᵢⱼ²) = √(trace(AᵀA))

Spectral norm (operator norm):
||A||₂ = σ_max(A) = max||Ax||₂/||x||₂

Nuclear norm (trace norm):
||A||_* = Σᵢ σᵢ(A)  (sum of singular values)
```

### Applications in ML

```
• ||W||₂: Lipschitz constant of linear layer
• ||W||_F: Weight magnitude (L2 regularization)
• ||W||_*: Low-rank regularization
```

---

## 📐 Singular Value Decomposition (SVD)

### Definition

```
Any m × n matrix A can be decomposed as:

A = UΣVᵀ

where:
• U: m × m orthogonal (left singular vectors)
• Σ: m × n diagonal (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
• V: n × n orthogonal (right singular vectors)
```

### Key Properties

```
• rank(A) = number of nonzero σᵢ
• ||A||₂ = σ_max
• ||A||_F = √(Σᵢ σᵢ²)
• A⁺ = VΣ⁺Uᵀ (pseudoinverse)
```

### Applications

```
1. Low-rank approximation:
   A_k = Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ  (best rank-k approximation)

2. PCA: Principal components are right singular vectors
   
3. Least squares: x = A⁺b when A is rank-deficient

4. Conditioning: κ(A) = σ_max/σ_min
```

---

## 💻 Code Examples

### Computing Hessian

```python
import torch

def compute_hessian(f, x):
    """Compute Hessian matrix"""
    return torch.autograd.functional.hessian(f, x)

# Example
def f(x):
    return x[0]**2 + 3*x[1]**2 + x[0]*x[1]

x = torch.tensor([1.0, 1.0])
H = compute_hessian(f, x)
print(f"Hessian:\n{H}")
# [[2., 1.],
#  [1., 6.]]
```

### Check Positive Definiteness

```python
import numpy as np

def is_positive_definite(H):
    """Check if matrix is positive definite"""
    eigenvalues = np.linalg.eigvalsh(H)
    return np.all(eigenvalues > 0)

def check_with_cholesky(H):
    """Check using Cholesky decomposition"""
    try:
        np.linalg.cholesky(H)
        return True
    except np.linalg.LinAlgError:
        return False

# Example
H = np.array([[2, 1], [1, 6]])
print(f"Eigenvalues: {np.linalg.eigvalsh(H)}")
print(f"Is PD: {is_positive_definite(H)}")
print(f"Condition number: {np.linalg.cond(H)}")
```

### Condition Number and Convergence

```python
import numpy as np

def analyze_conditioning(A):
    """Analyze conditioning of optimization problem"""
    eigenvalues = np.linalg.eigvalsh(A)
    lambda_max = np.max(eigenvalues)
    lambda_min = np.min(eigenvalues)
    kappa = lambda_max / lambda_min
    
    # GD convergence rate
    rate = (kappa - 1) / (kappa + 1)
    
    # Iterations to reduce error by 10^-6
    iterations = np.log(1e-6) / np.log(rate)
    
    print(f"λ_max = {lambda_max:.4f}")
    print(f"λ_min = {lambda_min:.4f}")
    print(f"Condition number κ = {kappa:.4f}")
    print(f"GD convergence rate = {rate:.4f}")
    print(f"Iterations for 10⁻⁶ accuracy ≈ {iterations:.0f}")

# Well-conditioned example
A_good = np.array([[2, 0], [0, 3]])
print("Well-conditioned problem:")
analyze_conditioning(A_good)

print("\nIll-conditioned problem:")
A_bad = np.array([[100, 0], [0, 1]])
analyze_conditioning(A_bad)
```

---

## 📐 Key Theorems

### Spectral Theorem

```
For symmetric A ∈ ℝⁿˣⁿ:

A = QΛQᵀ = Σᵢ λᵢqᵢqᵢᵀ

where:
• Q orthogonal: QᵀQ = I
• Λ diagonal with real eigenvalues
• qᵢ orthonormal eigenvectors

Applications:
• Matrix powers: Aᵏ = QΛᵏQᵀ
• Matrix functions: f(A) = Qf(Λ)Qᵀ
• Analysis of quadratics: xᵀAx = Σᵢ λᵢ(qᵢᵀx)²
```

### Courant-Fischer Theorem (Min-Max)

```
λₖ = min_{dim(V)=k} max_{x∈V, ||x||=1} xᵀAx
   = max_{dim(V)=n-k+1} min_{x∈V, ||x||=1} xᵀAx

Applications:
• Understanding eigenvalue sensitivity
• Bounds on condition number
• Analysis of Rayleigh quotient
```

### Weyl's Inequality

```
For symmetric A, B:

λₖ(A+B) ≤ λⱼ(A) + λₖ₋ⱼ₊₁(B)

Useful for:
• Perturbation analysis
• Understanding Hessian changes during optimization
```

---

## 📊 Summary Table

| Concept | Formula | Role in Optimization |
|---------|---------|---------------------|
| **Gradient** | ∇f = [∂f/∂xᵢ] | Direction of steepest descent |
| **Hessian** | H = [∂²f/∂xᵢ∂xⱼ] | Curvature, Newton's method |
| **Eigenvalues** | Av = λv | Convergence rate |
| **Condition number** | κ = λ_max/λ_min | Problem difficulty |
| **Positive definite** | xᵀAx > 0 | Local minimum test |
| **SVD** | A = UΣVᵀ | Low-rank, pseudoinverse |

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Linear Algebra Done Right | Axler |
| 📖 | Matrix Computations | Golub & Van Loan |
| 📖 | Numerical Linear Algebra | Trefethen & Bau |
| 🎥 | 3Blue1Brown Linear Algebra | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) |
| 🎥 | MIT 18.06 Linear Algebra | [MIT OCW](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) |
| 🇨🇳 | 线性代数基础 | [知乎](https://zhuanlan.zhihu.com/p/25385801) |

---

⬅️ [Back: Calculus](../01_calculus/) | ⬆️ [Up: Foundations](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
