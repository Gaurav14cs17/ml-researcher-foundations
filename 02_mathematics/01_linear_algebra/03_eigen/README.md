<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Eigenvalues%20%26%20Eigenvectors&fontSize=38&fontColor=fff&animation=twinkling&fontAlignY=32&desc=The%20DNA%20of%20Linear%20Transformations&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-01.03_Eigen-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-Eigenvalues_Eigenvectors_Spectral-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Eigenvalues and eigenvectors reveal the intrinsic behavior of linear transformations.** An eigenvector is a direction that only gets scaled (not rotated) when transformed.

- 📐 **Definition**: $A\mathbf{v} = \lambda\mathbf{v}$ — eigenvector $\mathbf{v}$ only scales by factor $\lambda$

- 🔍 **Finding them**: Solve $\det(A - \lambda I) = 0$ (characteristic polynomial)

- 📊 **Key facts**: $\sum \lambda\_i = \text{tr}(A)$, $\prod \lambda\_i = \det(A)$

- 🤖 **ML Uses**: PCA, PageRank, stability analysis, spectral clustering

---

## 📑 Table of Contents

1. [Visual Overview](#-visual-overview)
2. [Definition and Intuition](#1-definition-and-intuition)
3. [Finding Eigenvalues](#2-finding-eigenvalues-characteristic-polynomial)
4. [The Spectral Theorem](#3-the-spectral-theorem-complete-proof)
5. [Power Iteration Algorithm](#4-power-iteration-algorithm)
6. [Properties and Theorems](#5-key-properties-and-theorems)
7. [Examples](#6-worked-examples)
8. [Code Implementation](#7-code-implementation)
9. [ML Applications](#8-ml-applications)
10. [Common Mistakes](#-common-mistakes)
11. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/eigenvalue-visual.svg" width="100%">

```
+-----------------------------------------------------------------------------+

|                     EIGENVALUE / EIGENVECTOR INTUITION                       |
+-----------------------------------------------------------------------------+
|                                                                              |
|   GENERAL VECTOR                      EIGENVECTOR                            |
|   -------------                       -----------                            |
|                                                                              |
|   Before A:        After A:           Before A:        After A:              |
|       ↗               ↑                   →               →→→                |
|      /                |                   v               λv                 |
|     x                Ax                                                      |
|                   (rotated!)              (only scaled by λ, same direction) |
|                                                                              |
|   Most vectors change direction        Eigenvectors ONLY scale              |
|   when multiplied by A                 They are "fixed directions" of A     |
|                                                                              |
+-----------------------------------------------------------------------------+

|                                                                              |
|   EIGENVALUE INTERPRETATION                                                  |
|   -------------------------                                                  |
|                                                                              |
|   λ > 1:  Stretch along eigenvector direction                               |
|   λ = 1:  No change (identity behavior)                                     |
|   0 < λ < 1: Compress along eigenvector direction                           |
|   λ = 0:  Collapse to zero (singular matrix)                                |
|   λ < 0:  Flip and scale (reflection + scaling)                             |
|   λ ∈ ℂ:  Rotation + scaling (complex eigenvalues)                          |
|                                                                              |
+-----------------------------------------------------------------------------+

```

---

## 1. Definition and Intuition

### 📌 Formal Definition

For a square matrix $A \in \mathbb{R}^{n \times n}$:

```math
A\mathbf{v} = \lambda\mathbf{v}

```

where:

- $\lambda \in \mathbb{C}$ is an **eigenvalue**
- $\mathbf{v} \neq \mathbf{0}$ is the corresponding **eigenvector**

### 🔍 Geometric Interpretation

```
When matrix A acts on vector v:
  • Most vectors: Change direction AND magnitude
  • Eigenvector v: ONLY changes magnitude (by factor λ)

Example:
  A = [2  1]    v = [1]    Av = [2  1][1] = [3] = 3[1] = 3v
      [0  3]        [1]         [0  3][1]   [3]    [1]
  
  v is an eigenvector with eigenvalue λ = 3

```

### Why "Eigen"?

The word "eigen" is German for "own" or "characteristic". Eigenvectors are the matrix's "own" special directions — the directions intrinsic to the transformation.

---

## 2. Finding Eigenvalues: Characteristic Polynomial

### 📐 Derivation

```
Step 1: Start with the definition
        Av = λv

Step 2: Rearrange
        Av - λv = 0
        Av - λIv = 0
        (A - λI)v = 0

Step 3: Non-trivial solution exists iff (A - λI) is singular
        det(A - λI) = 0

This is the CHARACTERISTIC POLYNOMIAL

```

### The Characteristic Polynomial

```math
p(\lambda) = \det(A - \lambda I)

```

This is a polynomial of degree $n$ in $\lambda$:

```math
p(\lambda) = (-1)^n \lambda^n + (-1)^{n-1}\text{tr}(A)\lambda^{n-1} + \cdots + \det(A)

```

### 💡 Example: 2×2 Matrix

For $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

```
det(A - λI) = det([a-λ   b  ])
                 ([c    d-λ])
            = (a-λ)(d-λ) - bc
            = λ² - (a+d)λ + (ad-bc)
            = λ² - tr(A)·λ + det(A)

Quadratic formula:
λ = (tr(A) ± √(tr(A)² - 4det(A))) / 2

```

---

## 3. The Spectral Theorem (Complete Proof)

### 📌 Theorem Statement

**Spectral Theorem**: For a real symmetric matrix $A = A^T$:

1. **All eigenvalues are real**
2. **Eigenvectors can be chosen orthonormal**
3. **$A = Q\Lambda Q^T$** where $Q$ is orthogonal

### 🔍 Complete Proof

**Part 1: Eigenvalues are Real**

```
Step 1: Let λ be an eigenvalue with eigenvector v (possibly complex)
        Av = λv

Step 2: Take complex conjugate transpose of both sides
        (Av)* = (λv)*
        v*A* = λ̄v*
        
        Since A is real and symmetric: A* = Aᵀ = A
        So: v*A = λ̄v*

Step 3: Multiply the original equation on the left by v*
        v*Av = v*(λv) = λ(v*v) = λ‖v‖²

Step 4: Multiply Step 2's result on the right by v
        v*Av = (λ̄v*)v = λ̄(v*v) = λ̄‖v‖²

Step 5: From Steps 3 and 4:
        λ‖v‖² = λ̄‖v‖²

Step 6: Since v ≠ 0, we have ‖v‖² > 0, therefore:
        λ = λ̄
        
        This means λ is real!  ∎

```

**Part 2: Eigenvectors of Distinct Eigenvalues are Orthogonal**

```
Step 1: Let Av₁ = λ₁v₁ and Av₂ = λ₂v₂ with λ₁ ≠ λ₂

Step 2: Compute v₁ᵀAv₂ in two ways

        Way 1 (use Av₂ = λ₂v₂):
        v₁ᵀAv₂ = v₁ᵀ(λ₂v₂) = λ₂(v₁ᵀv₂)

        Way 2 (use Aᵀ = A and Av₁ = λ₁v₁):
        v₁ᵀAv₂ = (Aᵀv₁)ᵀv₂ = (Av₁)ᵀv₂ = (λ₁v₁)ᵀv₂ = λ₁(v₁ᵀv₂)

Step 3: Equate the two expressions:
        λ₂(v₁ᵀv₂) = λ₁(v₁ᵀv₂)
        (λ₂ - λ₁)(v₁ᵀv₂) = 0

Step 4: Since λ₁ ≠ λ₂, we must have:
        v₁ᵀv₂ = 0
        
        The eigenvectors are orthogonal!  ∎

```

**Part 3: Eigendecomposition $A = Q\Lambda Q^T$**

```
Step 1: Collect orthonormal eigenvectors as columns of Q
        Q = [v₁ | v₂ | ... | vₙ]

Step 2: By definition of eigenvectors:
        Av₁ = λ₁v₁
        Av₂ = λ₂v₂
        ...
        Avₙ = λₙvₙ

Step 3: In matrix form:
        A[v₁|v₂|...|vₙ] = [λ₁v₁|λ₂v₂|...|λₙvₙ]
        AQ = [v₁|v₂|...|vₙ][λ₁    0  ]
                           [   λ₂    ]
                           [      ⋱  ]
                           [0      λₙ]
        AQ = QΛ

Step 4: Since Q is orthogonal (QᵀQ = I):
        A = QΛQ⁻¹ = QΛQᵀ  ∎

```

### 📐 Corollary: Spectral Decomposition

```math
A = \sum_{i=1}^{n} \lambda_i \mathbf{v}_i \mathbf{v}_i^T

```

Each term $\lambda\_i \mathbf{v}\_i \mathbf{v}\_i^T$ is a rank-1 projection matrix!

---

## 4. Power Iteration Algorithm

### 📌 Algorithm

```python
def power_iteration(A, num_iterations=100):
    """
    Find the dominant eigenvalue and eigenvector.
    
    Convergence rate: |λ₂/λ₁|^k (geometric)
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iterations):
        Av = A @ v
        v = Av / np.linalg.norm(Av)
    
    # Rayleigh quotient gives eigenvalue
    eigenvalue = v @ A @ v
    return eigenvalue, v

```

### 🔍 Proof of Convergence

```
Step 1: Express initial vector in eigenbasis
        v₀ = α₁v₁ + α₂v₂ + ... + αₙvₙ
        (assuming α₁ ≠ 0)

Step 2: After k iterations:
        Aᵏv₀ = α₁λ₁ᵏv₁ + α₂λ₂ᵏv₂ + ... + αₙλₙᵏvₙ

Step 3: Factor out λ₁ᵏ:
        Aᵏv₀ = λ₁ᵏ[α₁v₁ + α₂(λ₂/λ₁)ᵏv₂ + ... + αₙ(λₙ/λ₁)ᵏvₙ]

Step 4: If |λ₁| > |λ₂| ≥ ... ≥ |λₙ| (dominant eigenvalue):
        As k → ∞: (λᵢ/λ₁)ᵏ → 0 for i ≥ 2

Step 5: Therefore:
        Aᵏv₀/‖Aᵏv₀‖ → ±v₁  (dominant eigenvector)
        
        Convergence rate: O(|λ₂/λ₁|ᵏ)  ∎

```

### Inverse Iteration (Find Smallest Eigenvalue)

```python
def inverse_iteration(A, shift=0, num_iterations=100):
    """
    Find eigenvalue closest to 'shift'.
    Uses (A - shift*I)⁻¹ which has largest eigenvalue = 1/(λ - shift)
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    A_shifted = A - shift * np.eye(n)
    
    for _ in range(num_iterations):
        w = np.linalg.solve(A_shifted, v)  # More stable than inverse
        v = w / np.linalg.norm(w)
    
    eigenvalue = v @ A @ v
    return eigenvalue, v

```

---

## 5. Key Properties and Theorems

### 📊 Fundamental Properties

| Property | Formula | Proof |
|----------|---------|-------|
| Sum of eigenvalues | $\sum\_i \lambda\_i = \text{tr}(A)$ | Coefficient of $\lambda^{n-1}$ in char. poly |
| Product of eigenvalues | $\prod\_i \lambda\_i = \det(A)$ | $p(0) = \det(A)$ and $p(\lambda) = \prod\_i(\lambda\_i - \lambda)$ |
| Eigenvalues of $A^k$ | $\lambda\_i^k$ | $Av = \lambda v \Rightarrow A^k v = \lambda^k v$ |
| Eigenvalues of $A^{-1}$ | $1/\lambda\_i$ | $Av = \lambda v \Rightarrow v = \lambda A^{-1}v$ |
| Eigenvalues of $A + cI$ | $\lambda\_i + c$ | $(A+cI)v = Av + cv = (\lambda + c)v$ |

### 🔍 Proof: tr(A) = Sum of Eigenvalues

```
The characteristic polynomial is:
  p(λ) = det(A - λI) = (-1)ⁿλⁿ + (-1)ⁿ⁻¹tr(A)λⁿ⁻¹ + ... + det(A)

Also, by the Fundamental Theorem of Algebra:
  p(λ) = (-1)ⁿ(λ - λ₁)(λ - λ₂)...(λ - λₙ)

Expanding the product:
  = (-1)ⁿ[λⁿ - (λ₁+λ₂+...+λₙ)λⁿ⁻¹ + ...]
  = (-1)ⁿλⁿ + (-1)ⁿ⁻¹(Σᵢλᵢ)λⁿ⁻¹ + ...

Comparing coefficients of λⁿ⁻¹:
  (-1)ⁿ⁻¹tr(A) = (-1)ⁿ⁻¹(Σᵢλᵢ)
  tr(A) = Σᵢλᵢ  ∎

```

### 📐 Cayley-Hamilton Theorem

**Theorem**: Every matrix satisfies its own characteristic polynomial.

```math
p(A) = A^n - \text{tr}(A)A^{n-1} + \cdots + (-1)^n\det(A)I = 0

```

**Application**: Express $A^{-1}$ as polynomial in $A$:

```
For 2×2: A² - tr(A)·A + det(A)·I = 0
         A² = tr(A)·A - det(A)·I
         A⁻¹ = (tr(A)·I - A) / det(A)

```

---

## 6. Worked Examples

### Example 1: Complete 2×2 Eigenanalysis

```
A = [4  2]
    [1  3]

Step 1: Characteristic polynomial
det(A - λI) = det([4-λ   2 ])
                 ([ 1   3-λ])
            = (4-λ)(3-λ) - 2
            = λ² - 7λ + 12 - 2
            = λ² - 7λ + 10
            = (λ - 5)(λ - 2)

Step 2: Eigenvalues
λ₁ = 5,  λ₂ = 2

Check: λ₁ + λ₂ = 7 = tr(A) ✓
       λ₁ × λ₂ = 10 = det(A) ✓

Step 3: Eigenvectors
For λ₁ = 5:
  (A - 5I)v = 0
  [-1  2][v₁]   [0]
  [ 1 -2][v₂] = [0]
  
  -v₁ + 2v₂ = 0  ⟹  v₁ = 2v₂
  v₁ = [2, 1]ᵀ  (or normalized: [2/√5, 1/√5]ᵀ)

For λ₂ = 2:
  (A - 2I)v = 0
  [2  2][v₁]   [0]
  [1  1][v₂] = [0]
  
  v₁ + v₂ = 0  ⟹  v₂ = -v₁
  v₂ = [1, -1]ᵀ  (or normalized: [1/√2, -1/√2]ᵀ)

Step 4: Verify
Av₁ = [4  2][2] = [10] = 5[2] = 5v₁ ✓
      [1  3][1]   [ 5]    [1]

Av₂ = [4  2][ 1] = [2]  = 2[ 1] = 2v₂ ✓
      [1  3][-1]   [-2]    [-1]

```

### Example 2: Symmetric Matrix (Orthogonal Eigenvectors)

```
A = [3  1]
    [1  3]

Characteristic polynomial:
det(A - λI) = (3-λ)² - 1 = λ² - 6λ + 8 = (λ-4)(λ-2)

Eigenvalues: λ₁ = 4, λ₂ = 2

Eigenvectors:
For λ₁ = 4: v₁ = [1, 1]ᵀ/√2
For λ₂ = 2: v₂ = [1, -1]ᵀ/√2

Orthogonality check:
v₁ᵀv₂ = (1)(1) + (1)(-1) = 0 ✓  (as guaranteed by Spectral Theorem)

Eigendecomposition:
A = QΛQᵀ = [1/√2   1/√2][4  0][1/√2   1/√2]
           [1/√2  -1/√2][0  2][1/√2  -1/√2]

```

### Example 3: Complex Eigenvalues (Rotation Matrix)

```
R(θ) = [cos θ  -sin θ]   (rotation by angle θ)
       [sin θ   cos θ]

Characteristic polynomial:
det(R - λI) = (cos θ - λ)² + sin²θ
            = λ² - 2cos(θ)λ + cos²θ + sin²θ
            = λ² - 2cos(θ)λ + 1

Eigenvalues (using quadratic formula):
λ = (2cos θ ± √(4cos²θ - 4)) / 2
  = cos θ ± √(cos²θ - 1)
  = cos θ ± √(-sin²θ)
  = cos θ ± i·sin θ
  = e^{±iθ}

Complex eigenvalues! No real eigenvectors exist (for θ ≠ 0, π).
This makes sense: rotation doesn't preserve any direction.

```

### Example 4: Matrix Power via Eigendecomposition

```
Compute A¹⁰ where A = [2  1]
                      [0  3]

Step 1: Eigenvalues (diagonal, so obvious)
λ₁ = 2, λ₂ = 3

Step 2: Eigenvectors
For λ₁ = 2: [0  1][v₁] = [0]  ⟹  v₁ = [1, 0]ᵀ
            [0  1][v₂]   [0]

For λ₂ = 3: [-1  1][v₁] = [0]  ⟹  v₂ = [1, 1]ᵀ
            [ 0  0][v₂]   [0]

Step 3: Form P and P⁻¹
P = [1  1]    P⁻¹ = [1  -1]
    [0  1]          [0   1]

Step 4: Compute A¹⁰
A¹⁰ = PΛ¹⁰P⁻¹ = [1  1][2¹⁰    0 ][1  -1]
                 [0  1][ 0   3¹⁰][0   1]
     
     = [1  1][1024     0  ][1  -1]
       [0  1][   0  59049][0   1]
     
     = [1024   59049-1024] = [1024  58025]
       [   0        59049]   [   0  59049]

```

---

## 7. Code Implementation

```python
import numpy as np
import torch

def eigenanalysis(A):
    """Complete eigenvalue analysis of a matrix"""
    n = A.shape[0]
    
    # Check if symmetric
    is_symmetric = np.allclose(A, A.T)
    
    if is_symmetric:
        # Use eigh for symmetric (faster, more stable, guaranteed real)
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    else:
        # General case (may have complex eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"Matrix is {'symmetric' if is_symmetric else 'non-symmetric'}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Sum of eigenvalues: {eigenvalues.sum():.6f}")
    print(f"Trace of A: {np.trace(A):.6f}")
    print(f"Product of eigenvalues: {np.prod(eigenvalues):.6f}")
    print(f"Determinant of A: {np.linalg.det(A):.6f}")
    
    return eigenvalues, eigenvectors

def verify_eigen(A, eigenvalues, eigenvectors):
    """Verify eigenvalue equation Av = λv"""
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        
        Av = A @ v
        lam_v = lam * v
        
        error = np.linalg.norm(Av - lam_v)
        print(f"λ_{i+1} = {lam:.4f}, error = {error:.2e}")

def power_iteration(A, num_iterations=100, tol=1e-10):
    """
    Find dominant eigenvalue and eigenvector.
    Returns: (eigenvalue, eigenvector, num_iterations_used)
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for k in range(num_iterations):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)
        
        # Check convergence
        if np.linalg.norm(v_new - v) < tol:
            eigenvalue = v_new @ A @ v_new
            return eigenvalue, v_new, k + 1
        
        v = v_new
    
    eigenvalue = v @ A @ v
    return eigenvalue, v, num_iterations

def spectral_decomposition(A):
    """
    Compute A = Σᵢ λᵢ vᵢvᵢᵀ for symmetric A
    Returns list of (eigenvalue, rank-1 matrix) tuples
    """
    assert np.allclose(A, A.T), "Matrix must be symmetric"
    
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    components = []
    for i, lam in enumerate(eigenvalues):
        v = eigenvectors[:, i:i+1]  # Column vector
        component = lam * (v @ v.T)
        components.append((lam, component))
    
    # Verify reconstruction
    A_reconstructed = sum(comp for _, comp in components)
    error = np.linalg.norm(A - A_reconstructed)
    print(f"Reconstruction error: {error:.2e}")
    
    return components

def matrix_function(A, f):
    """
    Compute f(A) using eigendecomposition.
    f(A) = Q·f(Λ)·Q⁻¹
    
    Examples: f = np.exp (matrix exponential)
              f = np.sqrt (matrix square root)
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Apply f to eigenvalues
    f_eigenvalues = f(eigenvalues)
    
    # Reconstruct
    f_A = eigenvectors @ np.diag(f_eigenvalues) @ np.linalg.inv(eigenvectors)
    
    return f_A

# Example usage
if __name__ == "__main__":
    # Symmetric matrix example
    A = np.array([[3, 1], [1, 3]])
    
    print("=" * 50)
    print("Eigenvalue Analysis")
    print("=" * 50)
    eigenvalues, eigenvectors = eigenanalysis(A)
    
    print("\n" + "=" * 50)
    print("Verification")
    print("=" * 50)
    verify_eigen(A, eigenvalues, eigenvectors)
    
    print("\n" + "=" * 50)
    print("Power Iteration")
    print("=" * 50)
    lam, v, iters = power_iteration(A)
    print(f"Dominant eigenvalue: {lam:.6f} (found in {iters} iterations)")
    print(f"True dominant eigenvalue: {eigenvalues.max():.6f}")
    
    print("\n" + "=" * 50)
    print("Matrix Exponential")
    print("=" * 50)
    exp_A = matrix_function(A, np.exp)
    print(f"exp(A) = \n{exp_A}")

```

### PyTorch GPU Implementation

```python
import torch

def eigen_torch(A_tensor):
    """Eigendecomposition with GPU support"""
    # For symmetric matrices
    if torch.allclose(A_tensor, A_tensor.T):
        eigenvalues, eigenvectors = torch.linalg.eigh(A_tensor)
    else:
        eigenvalues, eigenvectors = torch.linalg.eig(A_tensor)
    
    return eigenvalues, eigenvectors

# GPU example
device = 'cuda' if torch.cuda.is_available() else 'cpu'
A_gpu = torch.randn(1000, 1000, device=device)
A_gpu = A_gpu @ A_gpu.T  # Make symmetric
eigenvalues, eigenvectors = torch.linalg.eigh(A_gpu)

```

---

## 8. ML Applications

### 🤖 Application 1: Principal Component Analysis (PCA)

```python
def pca_via_eigen(X, n_components):
    """
    PCA using eigendecomposition of covariance matrix.
    Principal components = eigenvectors with largest eigenvalues.
    """
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Compute covariance matrix
    n = len(X)
    cov = X_centered.T @ X_centered / (n - 1)
    
    # Eigendecomposition (cov is symmetric PSD)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    components = eigenvectors[:, :n_components]
    
    # Project data
    X_projected = X_centered @ components
    
    # Explained variance ratio
    explained_var_ratio = eigenvalues[:n_components] / eigenvalues.sum()
    
    return X_projected, components, explained_var_ratio

```

### 🤖 Application 2: PageRank

```python
def pagerank(adjacency_matrix, damping=0.85, max_iter=100):
    """
    PageRank: Find dominant eigenvector of Google matrix.
    
    G = d·P + (1-d)/n·J  where P = normalized adjacency
    PageRank = principal eigenvector of G
    """
    n = adjacency_matrix.shape[0]
    
    # Normalize columns (make stochastic)
    out_degree = adjacency_matrix.sum(axis=0)
    out_degree[out_degree == 0] = 1  # Handle dangling nodes
    P = adjacency_matrix / out_degree
    
    # Google matrix
    J = np.ones((n, n)) / n
    G = damping * P + (1 - damping) * J
    
    # Power iteration for dominant eigenvector
    rank = np.ones(n) / n
    for _ in range(max_iter):
        rank = G @ rank
        rank = rank / rank.sum()
    
    return rank

```

### 🤖 Application 3: RNN Gradient Stability

```python
def check_rnn_stability(W_hh):
    """
    Check RNN stability via eigenvalues of hidden-to-hidden weight.
    
    For stable training:
    - All |λᵢ| < 1: Vanishing gradients
    - Any |λᵢ| > 1: Exploding gradients
    - All |λᵢ| ≈ 1: Ideal (hard to achieve)
    """
    eigenvalues = np.linalg.eigvals(W_hh)
    spectral_radius = np.max(np.abs(eigenvalues))
    
    if spectral_radius > 1:
        print(f"⚠️ Spectral radius = {spectral_radius:.4f} > 1")
        print("Risk of EXPLODING gradients!")
    elif spectral_radius < 0.9:
        print(f"⚠️ Spectral radius = {spectral_radius:.4f} < 0.9")
        print("Risk of VANISHING gradients!")
    else:
        print(f"✓ Spectral radius = {spectral_radius:.4f}")
        print("Gradients should be stable.")
    
    return spectral_radius

```

### 🤖 Application 4: Spectral Clustering

```python
def spectral_clustering(adjacency, n_clusters):
    """
    Spectral clustering using graph Laplacian eigenvectors.
    """
    # Degree matrix
    D = np.diag(adjacency.sum(axis=1))
    
    # Laplacian L = D - A
    L = D - adjacency
    
    # Normalized Laplacian (more stable)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Smallest eigenvectors (excluding the trivial one)
    eigenvalues, eigenvectors = np.linalg.eigh(L_normalized)
    
    # Use k smallest non-trivial eigenvectors
    embedding = eigenvectors[:, 1:n_clusters+1]
    
    # Cluster in this embedding space (e.g., k-means)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)
    
    return labels, embedding

```

---

## ⚠️ Common Mistakes

### ❌ Mistake 1: Confusing Eigenvalues with Singular Values

```python
# Eigenvalues: only for SQUARE matrices
# Singular values: for ANY matrix

A = np.random.randn(5, 3)  # Rectangular
# eigenvalues = np.linalg.eig(A)  # ERROR!
U, singular_values, Vt = np.linalg.svd(A)  # Correct

# For square A: singular values = |eigenvalues| only if A is normal (AAᵀ = AᵀA)

```

### ❌ Mistake 2: Expecting Real Eigenvalues for Non-Symmetric Matrices

```python
# Non-symmetric matrices can have complex eigenvalues
A = np.array([[0, -1], [1, 0]])  # 90° rotation
eigenvalues = np.linalg.eigvals(A)
print(eigenvalues)  # [0+1j, 0-1j] - complex!

```

### ❌ Mistake 3: Not Checking for Defective Matrices

```python
# Not all matrices are diagonalizable!
A = np.array([[1, 1], [0, 1]])  # Jordan block
eigenvalues, eigenvectors = np.linalg.eig(A)
# Only ONE eigenvector exists, but we get two (numerically corrupted)

```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 🎥 | [3Blue1Brown: Eigenvectors](https://www.youtube.com/watch?v=PFDu9oVAE-g) | Visual intuition |
| 📖 | [Linear Algebra Done Right](https://linear.axler.net/) | Theoretical treatment |
| 🎥 | [MIT 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | Strang's lectures |

---

## 🗺️ Navigation

<p align="center">

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Dimensionality Reduction](../02_dimensionality_reduction/README.md) | [Linear Algebra](../README.md) | [Eigenvalues Advanced](../04_eigenvalues/README.md) |

</p>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
