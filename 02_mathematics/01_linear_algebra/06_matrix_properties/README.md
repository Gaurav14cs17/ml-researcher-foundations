<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Matrix%20Properties&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Rank%20·%20Determinant%20·%20Trace%20·%20Condition%20Number&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-01.06_Matrix_Properties-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-Rank_Det_Trace_Cond-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Matrix properties reveal essential information for debugging and analysis.** Rank tells you about solvability, condition number about numerical stability.

- 📊 **Rank**: Dimension of column/row space — is the system solvable?
- 🔢 **Determinant**: Volume scaling — is the matrix invertible?
- ➕ **Trace**: Sum of eigenvalues — quick diagnostic
- ⚠️ **Condition Number**: Numerical stability — will my algorithm fail?

---

## 📑 Table of Contents

1. [Rank](#1-rank)
2. [Determinant](#2-determinant)
3. [Trace](#3-trace)
4. [Condition Number](#4-condition-number)
5. [Positive Definiteness](#5-positive-definiteness)
6. [Code Implementation](#6-code-implementation)
7. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/matrix-properties.svg" width="100%">

---

## 1. Rank

### 📌 Definition

$$\text{rank}(A) = \dim(\text{column space}) = \dim(\text{row space})$$

Equivalently: number of non-zero singular values.

### 📐 Key Properties

| Property | Formula | Significance |
|----------|---------|--------------|
| Max rank | $\text{rank}(A) \leq \min(m, n)$ | Bounded by smaller dimension |
| Product | $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$ | Can't increase rank |
| Sum | $\text{rank}(A+B) \leq \text{rank}(A) + \text{rank}(B)$ | Subadditivity |
| Transpose | $\text{rank}(A) = \text{rank}(A^T)$ | Row rank = column rank |

### 🔍 Proof: Row Rank = Column Rank

```
Let r = dim(row space of A) and c = dim(column space of A).

Step 1: Row operations don't change row space
        Reduce A to row echelon form R
        R has r non-zero rows
        These r rows are linearly independent
        So row space of R (= row space of A) has dimension r

Step 2: Column operations don't change column space
        Non-zero rows of R each have leading 1 in different column
        The c corresponding columns of A are linearly independent
        (They reduce to the standard basis vectors)
        So column space has dimension ≥ r

Step 3: Similarly, column space dimension ≤ r
        The r pivot columns span the column space

Therefore: row rank = column rank = r  ∎
```

### 💡 Examples

**Example 1**: Compute Rank
```
A = [1  2  3]
    [2  4  6]
    [0  1  2]

Row reduce:
R₂ - 2R₁: [1  2  3]
          [0  0  0]
          [0  1  2]

Swap R₂, R₃: [1  2  3]
             [0  1  2]
             [0  0  0]

Two non-zero rows → rank(A) = 2
```

**Example 2**: Rank in ML (LoRA)
```
Weight matrix W: (768 × 768) → rank up to 768
LoRA update ΔW = BA where B: (768 × 4), A: (4 × 768)

rank(ΔW) = rank(BA) ≤ min(rank(B), rank(A)) ≤ 4

Only 4 degrees of freedom for update! (vs 589K for full W)
```

---

## 2. Determinant

### 📌 Definition

The determinant is the unique function $\det: \mathbb{R}^{n \times n} \to \mathbb{R}$ satisfying:
1. Multilinear in columns
2. Alternating (swapping columns negates)
3. $\det(I) = 1$

### 📐 Key Properties

| Property | Formula |
|----------|---------|
| Multiplicative | $\det(AB) = \det(A)\det(B)$ |
| Transpose | $\det(A^T) = \det(A)$ |
| Inverse | $\det(A^{-1}) = 1/\det(A)$ |
| Scaling | $\det(cA) = c^n \det(A)$ |
| Eigenvalues | $\det(A) = \prod_i \lambda_i$ |

### 🔍 Geometric Interpretation

```
|det(A)| = volume scaling factor

For 2×2 matrix A = [a b]:
                   [c d]

det(A) = ad - bc

• |det(A)| = area of parallelogram with sides (a,c) and (b,d)
• det(A) > 0: preserves orientation
• det(A) < 0: reverses orientation (reflection)
• det(A) = 0: collapses to lower dimension (singular)
```

### 💡 Examples

**Example 1**: 2×2 Determinant
```
A = [3  1]
    [2  4]

det(A) = 3×4 - 1×2 = 12 - 2 = 10

Geometric: Area of parallelogram = 10 square units
```

**Example 2**: Determinant via Eigenvalues
```
A = [2  1]
    [1  2]

Eigenvalues: λ₁ = 3, λ₂ = 1

det(A) = λ₁ × λ₂ = 3 × 1 = 3

Verify: det(A) = 2×2 - 1×1 = 4 - 1 = 3 ✓
```

---

## 3. Trace

### 📌 Definition

$$\text{tr}(A) = \sum_{i=1}^{n} A_{ii}$$

### 📐 Key Properties

| Property | Formula | Proof Sketch |
|----------|---------|--------------|
| Linearity | $\text{tr}(A+B) = \text{tr}(A) + \text{tr}(B)$ | Sum of sums |
| Cyclic | $\text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)$ | Index manipulation |
| Transpose | $\text{tr}(A) = \text{tr}(A^T)$ | Diagonal unchanged |
| Eigenvalues | $\text{tr}(A) = \sum_i \lambda_i$ | Characteristic poly |
| Frobenius | $\|A\|_F^2 = \text{tr}(A^TA)$ | By definition |

### 🔍 Proof: Cyclic Property

```
tr(AB) = Σᵢ (AB)ᵢᵢ = Σᵢ Σⱼ AᵢⱼBⱼᵢ

tr(BA) = Σⱼ (BA)ⱼⱼ = Σⱼ Σᵢ BⱼᵢAᵢⱼ

These are the same double sum, just indexed differently!

For three matrices:
tr(ABC) = tr((AB)C) = tr(C(AB)) = tr(CAB)  ∎
```

### 💡 Applications

**Efficient Gradient Computation**:
```python
# Goal: Compute tr(ABᵀC)
# Naive: O(n³) to form products, O(n) for trace

# Efficient: tr(ABᵀC) = Σᵢⱼₖ Aᵢⱼ Bₖⱼ Cₖᵢ
# Use: tr(ABᵀC) = tr((ABᵀ)C) = Σᵢ ((ABᵀ)C)ᵢᵢ

# Even better with element-wise:
# tr(ABᵀ) = Σᵢⱼ AᵢⱼBᵢⱼ = sum(A * B)
```

---

## 4. Condition Number

### 📌 Definition

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

### 📐 Interpretation

```
Condition number measures sensitivity to perturbation:

Solving Ax = b with perturbed b + δb:
  A(x + δx) = b + δb
  Aδx = δb
  δx = A⁻¹δb

Relative error:
  ‖δx‖/‖x‖ ≤ κ(A) × ‖δb‖/‖b‖

So:
• κ(A) ≈ 1: Well-conditioned (errors don't amplify)
• κ(A) >> 1: Ill-conditioned (small errors → large errors)
• κ(A) = ∞: Singular (non-invertible)

Rule of thumb: Lose log₁₀(κ) digits of precision
  κ = 10⁶ → lose 6 digits (from 16 to 10 significant digits)
```

### 💡 Examples

**Example 1**: Hilbert Matrix (Notoriously Ill-Conditioned)
```
H_{ij} = 1/(i+j-1)

H₃ = [1    1/2  1/3]
     [1/2  1/3  1/4]
     [1/3  1/4  1/5]

κ(H₃) ≈ 524
κ(H₅) ≈ 476,000
κ(H₁₀) ≈ 10¹³

Almost impossible to invert numerically for n > 10!
```

**Example 2**: Orthogonal Matrix (Perfectly Conditioned)
```
Q is orthogonal ⟹ σᵢ = 1 for all i
⟹ κ(Q) = 1/1 = 1

This is why QR decomposition is numerically stable!
```

---

## 5. Positive Definiteness

### 📌 Definition

A symmetric matrix $A$ is:
- **Positive Definite** ($A \succ 0$) if $\mathbf{x}^TA\mathbf{x} > 0$ for all $\mathbf{x} \neq 0$
- **Positive Semi-Definite** ($A \succeq 0$) if $\mathbf{x}^TA\mathbf{x} \geq 0$ for all $\mathbf{x}$

### 📐 Equivalent Conditions

| Condition | Test |
|-----------|------|
| Definition | $\mathbf{x}^TA\mathbf{x} > 0$ for all $\mathbf{x} \neq 0$ |
| Eigenvalues | All $\lambda_i > 0$ |
| Cholesky | $A = LL^T$ exists (with positive diagonal $L$) |
| Sylvester | All leading principal minors $> 0$ |
| Factorization | $A = B^TB$ for some full-rank $B$ |

### 🔍 Proof: PD ⟺ All Eigenvalues > 0

```
(⟹) Suppose A ≻ 0
For any eigenvalue λ with eigenvector v:
  Av = λv
  vᵀAv = λvᵀv = λ‖v‖²

Since A ≻ 0: vᵀAv > 0
Since v ≠ 0: ‖v‖² > 0
Therefore: λ > 0  ✓

(⟸) Suppose all λᵢ > 0
For any x ≠ 0, write x = Σᵢ αᵢvᵢ (eigenbasis)
  xᵀAx = (Σᵢ αᵢvᵢ)ᵀ A (Σⱼ αⱼvⱼ)
       = Σᵢ Σⱼ αᵢαⱼ vᵢᵀAvⱼ
       = Σᵢ αᵢ² λᵢ  (since vᵢᵀvⱼ = δᵢⱼ and Avⱼ = λⱼvⱼ)
       > 0  (all λᵢ > 0 and not all αᵢ = 0)  ✓
```

### 💡 Applications

**Covariance Matrices are PSD**:
```
Σ = E[(X - μ)(X - μ)ᵀ]

For any vector a:
  aᵀΣa = aᵀE[(X - μ)(X - μ)ᵀ]a
       = E[aᵀ(X - μ)(X - μ)ᵀa]
       = E[(aᵀ(X - μ))²]
       ≥ 0

So covariance matrices are always PSD!
```

**Hessian Analysis for Optimization**:
```
At critical point x* where ∇f(x*) = 0:

• H(x*) ≻ 0 → x* is strict local minimum
• H(x*) ≺ 0 → x* is strict local maximum
• H(x*) indefinite → x* is saddle point
```

---

## 6. Code Implementation

```python
import numpy as np

def matrix_properties(A):
    """Compute comprehensive matrix properties."""
    results = {}
    
    # Basic properties
    results['shape'] = A.shape
    results['rank'] = np.linalg.matrix_rank(A)
    results['trace'] = np.trace(A)
    
    if A.shape[0] == A.shape[1]:  # Square matrix
        results['determinant'] = np.linalg.det(A)
        results['is_singular'] = np.abs(results['determinant']) < 1e-10
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvals(A)
        results['eigenvalues'] = eigenvalues
        results['spectral_radius'] = np.max(np.abs(eigenvalues))
        
        # Condition number
        if not results['is_singular']:
            results['condition_number'] = np.linalg.cond(A)
        else:
            results['condition_number'] = np.inf
        
        # Symmetry check
        results['is_symmetric'] = np.allclose(A, A.T)
        
        if results['is_symmetric']:
            # Positive definiteness
            eigenvalues_real = np.linalg.eigvalsh(A)
            results['is_positive_definite'] = np.all(eigenvalues_real > 0)
            results['is_positive_semidefinite'] = np.all(eigenvalues_real >= -1e-10)
    
    return results

def check_invertibility(A, tol=1e-10):
    """Check various invertibility criteria."""
    if A.shape[0] != A.shape[1]:
        return {"invertible": False, "reason": "Not square"}
    
    det = np.linalg.det(A)
    rank = np.linalg.matrix_rank(A)
    n = A.shape[0]
    
    if np.abs(det) < tol:
        return {"invertible": False, "reason": f"det ≈ 0 ({det:.2e})"}
    if rank < n:
        return {"invertible": False, "reason": f"rank {rank} < {n}"}
    
    cond = np.linalg.cond(A)
    if cond > 1e15:
        return {"invertible": True, "reason": f"⚠️ Very ill-conditioned (κ={cond:.2e})"}
    
    return {"invertible": True, "condition_number": cond}

def stability_analysis(A):
    """Analyze numerical stability of operations with A."""
    cond = np.linalg.cond(A)
    
    print(f"Condition number: {cond:.2e}")
    print(f"Expected precision loss: ~{np.log10(cond):.1f} digits")
    
    if cond < 10:
        print("✓ Excellent conditioning")
    elif cond < 1e4:
        print("✓ Good conditioning")
    elif cond < 1e8:
        print("⚠️ Moderate conditioning - use caution")
    elif cond < 1e15:
        print("⚠️ Poor conditioning - results may be inaccurate")
    else:
        print("⛔ Severe ill-conditioning - consider regularization")
    
    return cond

# Example
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

print("=== Matrix Properties ===")
props = matrix_properties(A)
for key, value in props.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value}")

print("\n=== Stability Analysis ===")
stability_analysis(A)
```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 📖 | [Matrix Computations](https://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm) | Golub & Van Loan |
| 🎥 | [3Blue1Brown](https://www.youtube.com/watch?v=Ip3X9LOh2dk) | Determinant |
| 📖 | [Numerical Recipes](http://numerical.recipes/) | Conditioning |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Matrix Factorization](../05_matrix_factorization/README.md) | [Linear Algebra](../README.md) | [SVD](../07_svd/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
