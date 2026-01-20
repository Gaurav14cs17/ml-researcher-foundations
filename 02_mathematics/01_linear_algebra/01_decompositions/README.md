<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Matrix%20Decompositions&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=SVD%20·%20Eigendecomposition%20·%20QR%20·%20Cholesky%20·%20LU&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-01.01_Decompositions-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-5_Decompositions-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Matrix decompositions break complex matrices into simpler, structured components.** They are the computational backbone of ML - enabling PCA, LoRA, solving linear systems, and much more.

- 🔥 **SVD** (`A = UΣVᵀ`): Works for ANY matrix, enables low-rank approximation (LoRA!)

- 📐 **Eigendecomposition** (`A = QΛQ⁻¹`): Square matrices, powers, PCA

- 🔄 **QR** (`A = QR`): Numerically stable least squares

- 📊 **Cholesky** (`A = LLᵀ`): Fast for positive definite matrices

- 🧮 **LU** (`A = LU`): General linear system solving

---

## 📑 Table of Contents

1. [Visual Overview](#-visual-overview)

2. [Singular Value Decomposition (SVD)](#1-singular-value-decomposition-svd)

3. [Eigendecomposition](#2-eigendecomposition)

4. [QR Decomposition](#3-qr-decomposition)

5. [Cholesky Decomposition](#4-cholesky-decomposition)

6. [LU Decomposition](#5-lu-decomposition)

7. [Comparison Table](#-comparison-which-decomposition-to-use)

8. [Code Examples](#-complete-code-examples)

9. [Common Mistakes](#-common-mistakes)

10. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/svd-decomposition-visual.svg" width="100%">

```
+-----------------------------------------------------------------------------+
|                     MATRIX DECOMPOSITION LANDSCAPE                           |
+-----------------------------------------------------------------------------+
|                                                                              |
|                              ANY MATRIX                                      |
|                                  |                                           |
|                    +-------------+-------------+                            |
|                    ▼             ▼             ▼                            |
|               +--------+   +--------+   +--------+                          |
|               |  SVD   |   |   QR   |   |   LU   |                          |
|               | A=UΣVᵀ |   | A=QR   |   | A=LU   |                          |
|               +--------+   +--------+   +--------+                          |
|                    |                         |                               |
|                    |         SQUARE          |                               |
|                    |           |             |                               |
|                    |     +-----+-----+       |                               |
|                    |     ▼           ▼       |                               |
|                    | +-------+  +--------+   |                               |
|                    | | Eigen |  |Schur   |   |                               |
|                    | |A=QΛQ⁻¹|  |A=QTQ*  |   |                               |
|                    | +-------+  +--------+   |                               |
|                    |     |                   |                               |
|                    |     |    SYMMETRIC PD   |                               |
|                    |     |         |         |                               |
|                    |     |    +----+----+    |                               |
|                    |     |    ▼         |    |                               |
|                    |     | +--------+   |    |                               |
|                    |     | |Cholesky|   |    |                               |
|                    |     | |A=LLᵀ   |   |    |                               |
|                    |     | +--------+   |    |                               |
|                    |     |              |    |                               |
|                    ▼     ▼              ▼    ▼                               |
|               +----------------------------------+                          |
|               |         ML APPLICATIONS          |                          |
|               |  PCA, LoRA, Regression, GPs...   |                          |
|               +----------------------------------+                          |
|                                                                              |
+-----------------------------------------------------------------------------+

```

---

## 1. Singular Value Decomposition (SVD)

### 📌 Theorem

**For ANY matrix** $A \in \mathbb{R}^{m \times n}$:

```math
A = U\Sigma V^T

```

where:

- $U \in \mathbb{R}^{m \times m}$: orthogonal matrix (left singular vectors)

- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal with $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$

- $V \in \mathbb{R}^{n \times n}$: orthogonal matrix (right singular vectors)

```
     A           =        U        ×        Σ        ×        Vᵀ
   (m×n)              (m×m)            (m×n)            (n×n)
   
+---------+       +---------+     +---------+     +---------+

|         |       |         |     |σ₁       |     |         |
|  Input  |   =   | Rotate  |  ×  |  σ₂     |  ×  | Rotate  |
| Matrix  |       |         |     |    ⋱    |     |         |
|         |       |         |     |      σᵣ |     |         |
+---------+       +---------+     +---------+     +---------+

```

### 🔍 Complete Proof of SVD Existence

**Step 1**: Consider $A^TA \in \mathbb{R}^{n \times n}$

$A^TA$ is symmetric positive semi-definite:

```
• Symmetric: (AᵀA)ᵀ = AᵀA  ✓
• PSD: xᵀ(AᵀA)x = (Ax)ᵀ(Ax) = ‖Ax‖² ≥ 0  ✓

```

**Step 2**: Apply Spectral Theorem to $A^TA$

```
Since AᵀA is symmetric:
  AᵀA = VΛVᵀ

where:
  V orthogonal (eigenvectors v₁, ..., vₙ)
  Λ diagonal with λᵢ ≥ 0 (eigenvalues)

Define singular values: σᵢ = √λᵢ

```

**Step 3**: Construct $U$

```
For each non-zero σᵢ, define:
  uᵢ = (1/σᵢ)Avᵢ

Verify orthonormality:
  uᵢᵀuⱼ = (1/σᵢσⱼ)(Avᵢ)ᵀ(Avⱼ)
        = (1/σᵢσⱼ)vᵢᵀ(AᵀA)vⱼ
        = (1/σᵢσⱼ)vᵢᵀ(λⱼvⱼ)
        = (σⱼ/σᵢ)(vᵢᵀvⱼ)
        = δᵢⱼ  ✓

Complete {uᵢ} to orthonormal basis of ℝᵐ using Gram-Schmidt.

```

**Step 4**: Verify $A = U\Sigma V^T$

```
For each vⱼ:
  Avⱼ = σⱼuⱼ  (by construction of uⱼ)

In matrix form:
  AV = UΣ
  AVVᵀ = UΣVᵀ  (multiply right by Vᵀ)
  A = UΣVᵀ     (since VVᵀ = I)  ∎

```

### 📐 Key Properties

| Property | Formula | Significance |
|----------|---------|--------------|
| Rank | $\text{rank}(A) = \#\{\sigma_i > 0\}$ | Non-zero singular values |
| Frobenius norm | $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$ | Matrix "size" |
| Spectral norm | $\|A\|_2 = \sigma_1$ | Largest singular value |
| Condition number | $\kappa(A) = \sigma_1/\sigma_r$ | Numerical stability |
| Pseudoinverse | $A^+ = V\Sigma^+ U^T$ | Generalized inverse |

### 🔍 Eckart-Young Theorem (Low-Rank Approximation)

**Theorem**: The best rank-$k$ approximation to $A$ (in Frobenius norm) is:

```math
A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T

```

**Error**: $\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$

**Proof Sketch**:

```
Step 1: Any rank-k matrix B has null space of dimension n-k

Step 2: span{v₁,...,vₖ₊₁} has dimension k+1

Step 3: By dimension counting, these spaces intersect
        ∃ unit z ∈ span{v₁,...,vₖ₊₁} with Bz = 0

Step 4: Then ‖A-B‖²_F ≥ ‖(A-B)z‖² = ‖Az‖² ≥ σ²ₖ₊₁

Step 5: Aₖ achieves this lower bound  ∎

```

### 💡 Examples

**Example 1**: 2×2 Matrix SVD

```
A = [3  0]
    [0  2]

Already diagonal! So:
  σ₁ = 3, σ₂ = 2
  U = I, V = I
  
A = [1 0] [3 0] [1 0]
    [0 1] [0 2] [0 1]

```

**Example 2**: Rank-1 Matrix

```
A = [2  4]     (rank 1: row 2 = 2 × row 1)
    [1  2]

SVD: A = uσvᵀ where
  σ = √(4+16+1+4) = 5
  v = [2,4]ᵀ/√20 = [1,2]ᵀ/√5
  u = Av/σ = [10,5]ᵀ/(5√5) = [2,1]ᵀ/√5

Verify: A = 5 × [2/√5] × [1/√5, 2/√5]
             [1/√5]

```

**Example 3**: Low-Rank Approximation

```
A = [1  0  0]
    [0  3  0]
    [0  0  0.1]

Singular values: σ₁=3, σ₂=1, σ₃=0.1

Rank-2 approximation (k=2):
A₂ = [1  0  0]    (just zero out smallest singular value)
     [0  3  0]
     [0  0  0]

Error: ‖A-A₂‖_F = σ₃ = 0.1 (only 10% of smallest component lost)

```

### 🤖 ML Application: LoRA (Low-Rank Adaptation)

```python
# LoRA exploits Eckart-Young: weight updates often have low intrinsic rank
#
# Instead of updating full W (d×k parameters):
#   W' = W + ΔW
#
# LoRA uses low-rank factorization (r×(d+k) parameters, r << min(d,k)):
#   W' = W + BA    where B∈ℝᵈˣʳ, A∈ℝʳˣᵏ

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        # Frozen original weights
        self.W = nn.Parameter(torch.randn(out_features, in_features), 
                              requires_grad=False)
        
        # Trainable low-rank matrices
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
    
    def forward(self, x):
        # W' = W + scaling * B @ A
        return x @ self.W.T + self.scaling * (x @ self.A.T @ self.B.T)

# Parameter savings:
# Original: 768 × 768 = 589,824 params
# LoRA (r=4): 768×4 + 4×768 = 6,144 params (96× reduction!)

```

---

## 2. Eigendecomposition

### 📌 Definition

For a **square matrix** $A \in \mathbb{R}^{n \times n}$:

```math
A\mathbf{v} = \lambda\mathbf{v}

```

If $A$ has $n$ linearly independent eigenvectors:

```math
A = Q\Lambda Q^{-1}

```

For **symmetric** $A$: $A = Q\Lambda Q^T$ (orthogonal $Q$)

### 🔍 Proof: Symmetric Matrices Have Real Eigenvalues and Orthogonal Eigenvectors

**Part 1: Eigenvalues are Real**

```
Step 1: Let λ be eigenvalue with eigenvector v (possibly complex)
        Av = λv

Step 2: Take conjugate transpose
        v*A = λ̄v*  (since A is real symmetric: A* = Aᵀ = A)

Step 3: Left-multiply eigenvalue equation by v*
        v*Av = λ(v*v) = λ‖v‖²

Step 4: Right-multiply Step 2 by v
        v*Av = λ̄(v*v) = λ̄‖v‖²

Step 5: Equate: λ‖v‖² = λ̄‖v‖²
        Since ‖v‖² > 0: λ = λ̄  ⟹ λ is real  ∎

```

**Part 2: Eigenvectors of Distinct Eigenvalues are Orthogonal**

```
Step 1: Let Av₁ = λ₁v₁ and Av₂ = λ₂v₂ with λ₁ ≠ λ₂

Step 2: Compute v₁ᵀAv₂ two ways:
        Way 1: v₁ᵀAv₂ = v₁ᵀ(λ₂v₂) = λ₂(v₁ᵀv₂)
        Way 2: v₁ᵀAv₂ = (Av₁)ᵀv₂ = λ₁(v₁ᵀv₂)  (using Aᵀ = A)

Step 3: Equate: λ₂(v₁ᵀv₂) = λ₁(v₁ᵀv₂)
        (λ₂ - λ₁)(v₁ᵀv₂) = 0

Step 4: Since λ₁ ≠ λ₂: v₁ᵀv₂ = 0  (orthogonal)  ∎

```

### 📐 Key Properties

| Property | Formula | Application |
|----------|---------|-------------|
| Sum of eigenvalues | $\sum_i \lambda_i = \text{tr}(A)$ | Quick verification |
| Product | $\prod_i \lambda_i = \det(A)$ | Invertibility check |
| Powers | $A^n = Q\Lambda^n Q^{-1}$ | Fast matrix powers |
| Functions | $f(A) = Qf(\Lambda)Q^{-1}$ | Matrix exponential |
| Inverse | $A^{-1} = Q\Lambda^{-1}Q^{-1}$ | If all $\lambda_i \neq 0$ |

### 💡 Examples

**Example 1**: Finding Eigenvalues (2×2)

```
A = [4  1]
    [2  3]

Step 1: Characteristic polynomial
det(A - λI) = det([4-λ   1 ])
                 ([2   3-λ])
            = (4-λ)(3-λ) - 2
            = λ² - 7λ + 10
            = (λ-5)(λ-2)

Step 2: Eigenvalues: λ₁ = 5, λ₂ = 2

Step 3: Eigenvectors
For λ₁ = 5: (A-5I)v = 0
  [-1  1][v₁]   [0]
  [ 2 -2][v₂] = [0]  ⟹ v₁ = [1, 1]ᵀ

For λ₂ = 2: (A-2I)v = 0
  [2  1][v₁]   [0]
  [2  1][v₂] = [0]  ⟹ v₂ = [1, -2]ᵀ

```

**Example 2**: Symmetric Matrix (Orthogonal Eigenvectors)

```
A = [2  1]
    [1  2]

Eigenvalues: λ₁ = 3, λ₂ = 1

Eigenvectors (normalized):
  v₁ = [1/√2, 1/√2]ᵀ
  v₂ = [1/√2, -1/√2]ᵀ

Check orthogonality: v₁·v₂ = 1/2 - 1/2 = 0 ✓

```

**Example 3**: Matrix Power via Eigendecomposition

```
Compute A¹⁰⁰ where A = [2  1]
                       [1  2]

Using A = QΛQᵀ:
  A¹⁰⁰ = QΛ¹⁰⁰Qᵀ
  
  Λ¹⁰⁰ = [3¹⁰⁰    0  ]
         [  0    1¹⁰⁰]
       = [3¹⁰⁰   0]
         [0      1]

Much faster than 100 matrix multiplications!

```

### ⚠️ When Eigendecomposition Fails

```
Not all matrices are diagonalizable!

Defective matrix example:
A = [1  1]
    [0  1]

Eigenvalue: λ = 1 (multiplicity 2)
But only ONE eigenvector: v = [1, 0]ᵀ

For such matrices, use:
• Jordan Normal Form: A = PJP⁻¹
• Schur Decomposition: A = QTQ* (always exists)
• SVD (always exists for any matrix)

```

---

## 3. QR Decomposition

### 📌 Theorem

For any matrix $A \in \mathbb{R}^{m \times n}$ with $m \geq n$:

```math
A = QR

```

where:

- $Q \in \mathbb{R}^{m \times n}$: orthonormal columns ($Q^TQ = I$)

- $R \in \mathbb{R}^{n \times n}$: upper triangular

### 🔍 Gram-Schmidt Algorithm (Constructive Proof)

```
Input: Matrix A = [a₁ | a₂ | ... | aₙ] (columns)
Output: Q (orthonormal columns), R (upper triangular)

For j = 1, 2, ..., n:
    # Start with original column
    ũⱼ = aⱼ
    
    # Subtract projections onto previous q vectors
    For i = 1 to j-1:
        rᵢⱼ = qᵢᵀ aⱼ           # Projection coefficient
        ũⱼ = ũⱼ - rᵢⱼ qᵢ       # Remove component along qᵢ
    
    # Normalize
    rⱼⱼ = ‖ũⱼ‖
    qⱼ = ũⱼ / rⱼⱼ

Result: A = QR where Q = [q₁|...|qₙ] and R has rᵢⱼ entries

```

**Verification**:

```
We need to show: aⱼ = Σᵢ₌₁ʲ rᵢⱼqᵢ  (j-th column of QR)

By construction:
  ũⱼ = aⱼ - Σᵢ₌₁ʲ⁻¹ rᵢⱼqᵢ
  qⱼ = ũⱼ/rⱼⱼ  ⟹  ũⱼ = rⱼⱼqⱼ

Therefore:
  aⱼ = ũⱼ + Σᵢ₌₁ʲ⁻¹ rᵢⱼqᵢ
     = rⱼⱼqⱼ + Σᵢ₌₁ʲ⁻¹ rᵢⱼqᵢ
     = Σᵢ₌₁ʲ rᵢⱼqᵢ  ✓

```

### 📐 Modified Gram-Schmidt (Numerically Stable)

```
Classical GS: Compute all projections from original vectors
Modified GS:  Update vector after each projection

For j = 1 to n:
    For i = 1 to j-1:
        rᵢⱼ = qᵢᵀ qⱼ          # Use CURRENT qⱼ, not original aⱼ
        qⱼ = qⱼ - rᵢⱼ qᵢ      # Update qⱼ immediately
    rⱼⱼ = ‖qⱼ‖
    qⱼ = qⱼ / rⱼⱼ

Numerical advantage:
• Classical GS error: O(κ²(A)ε) where ε = machine precision
• Modified GS error:  O(κ(A)ε)

```

### 💡 Examples

**Example 1**: 2×2 QR Decomposition

```
A = [1  1]
    [1  0]

Step 1: First column
  a₁ = [1, 1]ᵀ
  r₁₁ = ‖a₁‖ = √2
  q₁ = a₁/r₁₁ = [1/√2, 1/√2]ᵀ

Step 2: Second column
  a₂ = [1, 0]ᵀ
  r₁₂ = q₁ᵀa₂ = 1/√2
  ũ₂ = a₂ - r₁₂q₁ = [1, 0]ᵀ - (1/√2)[1/√2, 1/√2]ᵀ = [1/2, -1/2]ᵀ
  r₂₂ = ‖ũ₂‖ = 1/√2
  q₂ = ũ₂/r₂₂ = [1/√2, -1/√2]ᵀ

Result:
Q = [1/√2   1/√2]    R = [√2    1/√2]
    [1/√2  -1/√2]        [0     1/√2]

```

### 🤖 ML Applications

**1. Least Squares (More Stable than Normal Equations)**

```
Problem: min ‖Ax - b‖²

Normal equations: x = (AᵀA)⁻¹Aᵀb  (can be ill-conditioned!)

QR approach:
  A = QR
  Ax = b  ⟹  QRx = b  ⟹  Rx = Qᵀb
  
  Solve triangular system Rx = Qᵀb (stable back-substitution)

```

**2. QR Algorithm for Eigenvalues**

```
A₀ = A
For k = 1, 2, ...:
    Aₖ₋₁ = QₖRₖ     (QR factorization)
    Aₖ = RₖQₖ       (Reverse multiply)

As k → ∞: Aₖ → upper triangular (eigenvalues on diagonal)

```

---

## 4. Cholesky Decomposition

### 📌 Theorem

For a **symmetric positive definite** matrix $A$:

```math
A = LL^T

```

where $L$ is lower triangular with positive diagonal entries.

### 🔍 Proof of Existence

```
Proof by strong induction on matrix size n:

Base case (n=1): A = [a] where a > 0 (positive definite)
  L = [√a], and L·Lᵀ = a = A ✓

Inductive step: Assume true for (n-1)×(n-1). For n×n:

Partition A:
  A = [a    bᵀ]  where a > 0 (A is PD ⟹ diagonal entries > 0)
      [b    C ]

Let L have the form:
  L = [ℓ     0 ]
      [m    L₂]

Then LLᵀ = [ℓ²           ℓmᵀ        ]
           [ℓm    mmᵀ + L₂L₂ᵀ]

Matching with A:
  ℓ = √a
  m = b/ℓ = b/√a
  L₂L₂ᵀ = C - mmᵀ = C - bbᵀ/a

Need to show: C - bbᵀ/a is positive definite
  For any x ≠ 0:
    xᵀ(C - bbᵀ/a)x = xᵀCx - (bᵀx)²/a
    
  Consider y = [-bᵀx/a, xᵀ]ᵀ (nonzero since x ≠ 0)
    yᵀAy = (bᵀx)²/a - 2(bᵀx)²/a + xᵀCx = xᵀCx - (bᵀx)²/a > 0
    
  So C - bbᵀ/a is PD, apply induction to get L₂.  ∎

```

### 📐 Algorithm

```python
def cholesky(A):
    """
    Compute Cholesky decomposition A = LLᵀ
    Time: O(n³/3) - about 2× faster than LU
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    
    for j in range(n):
        # Diagonal element
        L[j, j] = np.sqrt(A[j, j] - np.sum(L[j, :j]**2))
        
        # Below diagonal
        for i in range(j+1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    
    return L

```

### 💡 Example

```
A = [4  2]
    [2  5]

Step 1: L₁₁ = √4 = 2

Step 2: L₂₁ = A₂₁/L₁₁ = 2/2 = 1

Step 3: L₂₂ = √(A₂₂ - L₂₁²) = √(5 - 1) = 2

Result:
L = [2  0]
    [1  2]

Verify: LLᵀ = [2  0][2  1] = [4  2] = A ✓
              [1  2][0  2]   [2  5]

```

### 🤖 ML Applications

**Gaussian Processes**: Computing $\mathcal{N}(\mu, K)$

```python
# Sampling from multivariate Gaussian
L = np.linalg.cholesky(K)  # K = covariance matrix
z = np.random.randn(n)      # Standard normal
sample = mu + L @ z         # Transform to N(μ, K)

# Computing log-likelihood efficiently
# log|K| = 2 × sum of log(diagonal of L)
log_det = 2 * np.sum(np.log(np.diag(L)))

```

---

## 5. LU Decomposition

### 📌 Theorem

For a square matrix $A$ (with partial pivoting):

```math
PA = LU

```

where:

- $P$: permutation matrix

- $L$: lower triangular with 1s on diagonal

- $U$: upper triangular

### 🔍 Algorithm (Gaussian Elimination)

```
For k = 1 to n-1:
    # Pivot: swap rows if needed for stability
    Find i ≥ k with max |Aᵢₖ|
    Swap rows k and i
    
    # Eliminate below diagonal
    For i = k+1 to n:
        Lᵢₖ = Aᵢₖ / Aₖₖ           # Store multiplier
        Aᵢ,ₖ₊₁:ₙ -= Lᵢₖ × Aₖ,ₖ₊₁:ₙ  # Eliminate

```

### 💡 Application: Solving $Ax = b$

```
Given PA = LU:

1. Apply permutation: Pb = b'

2. Forward substitution: Solve Ly = b' for y

3. Back substitution: Solve Ux = y for x

Total: O(n³/3) for factorization + O(n²) per solve
       (Factorize once, solve many times!)

```

---

## 📊 Comparison: Which Decomposition to Use?

| Decomposition | Matrix Type | Complexity | Best For |
|---------------|-------------|------------|----------|
| **SVD** | Any $m \times n$ | $O(mn \cdot \min(m,n))$ | Low-rank approximation, pseudoinverse |
| **Eigen** | Square $n \times n$ | $O(n^3)$ | PCA, matrix powers, spectral analysis |
| **QR** | Any $m \times n$ | $O(mn^2)$ | Least squares, numerical stability |
| **Cholesky** | Symmetric PD | $O(n^3/3)$ | **Fastest** for PD systems, GPs |
| **LU** | Square $n \times n$ | $O(n^3/3)$ | General linear systems |

### Decision Flowchart

```
Is matrix square?
+- No → Use SVD or QR
+- Yes → Is matrix symmetric?
         +- No → Use LU or SVD
         +- Yes → Is matrix positive definite?
                  +- No → Use Eigen or SVD
                  +- Yes → Use Cholesky (fastest!)

```

---

## 💻 Complete Code Examples

```python
import numpy as np
import torch
from scipy import linalg

def demonstrate_all_decompositions():
    """Complete examples of all matrix decompositions"""
    
    # Create test matrices
    np.random.seed(42)
    A_rect = np.random.randn(5, 3)           # Rectangular
    A_square = np.random.randn(4, 4)         # Square
    A_symmetric = A_square @ A_square.T + np.eye(4)  # Symmetric PD
    
    print("=" * 60)
    print("1. SINGULAR VALUE DECOMPOSITION (Any matrix)")
    print("=" * 60)
    U, S, Vt = np.linalg.svd(A_rect, full_matrices=False)
    print(f"Matrix shape: {A_rect.shape}")
    print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
    print(f"Singular values: {S}")
    print(f"Reconstruction error: {np.linalg.norm(A_rect - U @ np.diag(S) @ Vt):.2e}")
    
    # Low-rank approximation
    k = 2
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    print(f"Rank-{k} approximation error: {np.linalg.norm(A_rect - A_k):.4f}")
    print(f"Theoretical error (σ₃): {S[k]:.4f}")
    
    print("\n" + "=" * 60)
    print("2. EIGENDECOMPOSITION (Square symmetric)")
    print("=" * 60)
    eigenvalues, eigenvectors = np.linalg.eigh(A_symmetric)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Sum of eigenvalues: {eigenvalues.sum():.4f}")
    print(f"Trace of A: {np.trace(A_symmetric):.4f}")
    print(f"Product of eigenvalues: {np.prod(eigenvalues):.4f}")
    print(f"Determinant of A: {np.linalg.det(A_symmetric):.4f}")
    
    # Verify reconstruction
    Lambda = np.diag(eigenvalues)
    A_reconstructed = eigenvectors @ Lambda @ eigenvectors.T
    print(f"Reconstruction error: {np.linalg.norm(A_symmetric - A_reconstructed):.2e}")
    
    print("\n" + "=" * 60)
    print("3. QR DECOMPOSITION (Any matrix)")
    print("=" * 60)
    Q, R = np.linalg.qr(A_rect)
    print(f"Q shape: {Q.shape}, R shape: {R.shape}")
    print(f"Q orthonormal check (QᵀQ = I): {np.allclose(Q.T @ Q, np.eye(Q.shape[1]))}")
    print(f"Reconstruction error: {np.linalg.norm(A_rect - Q @ R):.2e}")
    
    # Solve least squares
    b = np.random.randn(5)
    x_qr = np.linalg.solve(R, Q.T @ b)
    x_lstsq = np.linalg.lstsq(A_rect, b, rcond=None)[0]
    print(f"QR vs lstsq difference: {np.linalg.norm(x_qr - x_lstsq):.2e}")
    
    print("\n" + "=" * 60)
    print("4. CHOLESKY DECOMPOSITION (Symmetric PD)")
    print("=" * 60)
    L = np.linalg.cholesky(A_symmetric)
    print(f"L shape: {L.shape}")
    print(f"L is lower triangular: {np.allclose(L, np.tril(L))}")
    print(f"Reconstruction error: {np.linalg.norm(A_symmetric - L @ L.T):.2e}")
    
    # Solve system using Cholesky
    b = np.random.randn(4)
    y = linalg.solve_triangular(L, b, lower=True)
    x = linalg.solve_triangular(L.T, y, lower=False)
    print(f"Solution error: {np.linalg.norm(A_symmetric @ x - b):.2e}")
    
    print("\n" + "=" * 60)
    print("5. LU DECOMPOSITION (Square)")
    print("=" * 60)
    P, L, U = linalg.lu(A_square)
    print(f"P: {P.shape}, L: {L.shape}, U: {U.shape}")
    print(f"Reconstruction error: {np.linalg.norm(A_square - P @ L @ U):.2e}")
    
    return U, S, Vt, eigenvalues, eigenvectors

# Run demo
demonstrate_all_decompositions()

```

### PyTorch GPU Implementation

```python
import torch

def svd_on_gpu(A):
    """SVD with GPU acceleration"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A_tensor = torch.tensor(A, device=device, dtype=torch.float32)
    
    U, S, Vh = torch.linalg.svd(A_tensor)
    
    return U.cpu().numpy(), S.cpu().numpy(), Vh.cpu().numpy()

def low_rank_approximation_torch(A, k):
    """GPU-accelerated low-rank approximation"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A_tensor = torch.tensor(A, device=device, dtype=torch.float32)
    
    U, S, Vh = torch.linalg.svd(A_tensor, full_matrices=False)
    
    # Keep top-k components
    A_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    
    return A_k.cpu().numpy()

```

---

## ⚠️ Common Mistakes

### ❌ Mistake 1: Using Eigendecomposition on Non-Square Matrix

```python
# WRONG
A = np.random.randn(5, 3)
eigenvalues, eigenvectors = np.linalg.eig(A)  # ERROR!

# CORRECT: Use SVD for rectangular matrices
U, S, Vt = np.linalg.svd(A)

```

### ❌ Mistake 2: Assuming All Matrices are Diagonalizable

```python
# This matrix is defective (not diagonalizable)
A = np.array([[1, 1], [0, 1]])

# Use Schur decomposition instead
T, Q = linalg.schur(A)

```

### ❌ Mistake 3: Using Cholesky on Non-PD Matrix

```python
# WRONG: Matrix must be positive definite
A = np.array([[1, 2], [2, 1]])  # Eigenvalues: 3, -1 (not PD!)
L = np.linalg.cholesky(A)  # LinAlgError!

# Check first
eigenvalues = np.linalg.eigvalsh(A)
if np.all(eigenvalues > 0):
    L = np.linalg.cholesky(A)

```

### ❌ Mistake 4: Ignoring Numerical Stability

```python
# FRAGILE: Normal equations
x = np.linalg.inv(A.T @ A) @ A.T @ b

# STABLE: Use QR
Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ b)

```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 📖 | [Matrix Computations](https://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm) | Golub & Van Loan (Bible of numerical LA) |
| 📖 | [Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html) | Trefethen & Bau |
| 🎥 | [Steve Brunton SVD Series](https://www.youtube.com/watch?v=nbBvuuNVfco) | Visual explanations |
| 📄 | [LoRA Paper](https://arxiv.org/abs/2106.09685) | Low-rank adaptation |

---

## 🗺️ Navigation

<p align="center">

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Linear Algebra](../README.md) | [Mathematics](../../README.md) | [Dimensionality Reduction](../02_dimensionality_reduction/README.md) |

</p>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
