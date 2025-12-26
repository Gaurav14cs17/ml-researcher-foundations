<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Linear%20Algebra&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=The%20Language%20of%20Machine%20Learning&descAlignY=52&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-01_Linear_Algebra-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-10_Subtopics-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/LaTeX-008080?style=flat-square&logo=latex&logoColor=white" alt="LaTeX"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Linear Algebra is the backbone of machine learning.** Every neural network layer, every attention mechanism, every optimization step relies on matrix operations.

- 🔢 **Vectors & Matrices**: Data representation, batch processing, linear layers `y = Wx + b`
- 🔄 **Eigendecomposition**: PCA, stability analysis, understanding matrix behavior `Av = λv`
- ✂️ **SVD**: Low-rank approximation, LoRA fine-tuning, compression `A = UΣVᵀ`
- 📐 **Positive Definite**: Covariance matrices, convexity, Hessian analysis

---

## 📑 Table of Contents

<details open>
<summary><strong>Click to expand/collapse</strong></summary>

1. [Visual Overview](#-visual-overview)
2. [Vectors and Basic Operations](#1-vectors-and-basic-operations)
3. [Matrix Operations](#2-matrix-operations)
4. [Vector Norms](#3-vector-norms-complete-theory)
5. [Matrix Norms](#4-matrix-norms)
6. [Eigenvalues and Eigenvectors](#5-eigenvalues-and-eigenvectors)
7. [Spectral Theorem](#6-spectral-theorem-with-proof)
8. [Singular Value Decomposition](#7-singular-value-decomposition-svd)
9. [Low-Rank Approximation](#8-low-rank-approximation-eckart-young-theorem)
10. [Positive Definite Matrices](#9-positive-definite-matrices)
11. [QR Decomposition](#10-qr-decomposition)
12. [Key Formulas Summary](#-key-formulas-summary)
13. [Common Mistakes](#-common-mistakes--pitfalls)
14. [Resources](#-resources)
15. [Subtopics](#-subtopics-in-this-section)

</details>

---

## 🎨 Visual Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LINEAR ALGEBRA IN ML PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input Data          Neural Network Layer           Output                  │
│   ┌─────┐            ┌─────────────────┐            ┌─────┐                 │
│   │  x  │  ───────►  │    y = Wx + b   │  ───────►  │  y  │                 │
│   │(n×1)│            │   (m×n)(n×1)    │            │(m×1)│                 │
│   └─────┘            └─────────────────┘            └─────┘                 │
│                              ▲                                               │
│                              │                                               │
│   ┌──────────────────────────┴──────────────────────────┐                   │
│   │                  MATRIX DECOMPOSITIONS               │                   │
│   ├─────────────────────────────────────────────────────┤                   │
│   │                                                      │                   │
│   │   EIGENDECOMPOSITION          SVD                    │                   │
│   │   A = QΛQ⁻¹                  A = UΣVᵀ               │                   │
│   │   (square matrices)          (any matrix)            │                   │
│   │                                                      │                   │
│   │   Applications:              Applications:           │                   │
│   │   • PCA                      • LoRA                  │                   │
│   │   • Stability Analysis       • Compression           │                   │
│   │   • Graph Laplacian          • Pseudoinverse         │                   │
│   │                                                      │                   │
│   └──────────────────────────────────────────────────────┘                   │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         ATTENTION MECHANISM                           │  │
│   │                                                                       │  │
│   │     Attention(Q,K,V) = softmax(QKᵀ/√d) × V                           │  │
│   │                                                                       │  │
│   │     Q = XWq    K = XWk    V = XWv   (All linear projections!)        │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Vectors and Basic Operations

### 📌 Definition

A **vector** is an ordered list of numbers representing a point or direction in space.

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

### 📐 Vector Operations

| Operation | Notation | Formula | Geometric Meaning |
|-----------|----------|---------|-------------------|
| Addition | $\mathbf{x} + \mathbf{y}$ | $[x_1+y_1, x_2+y_2, \ldots]^T$ | Parallelogram rule |
| Scalar Mult | $\alpha\mathbf{x}$ | $[\alpha x_1, \alpha x_2, \ldots]^T$ | Stretch/shrink |
| Dot Product | $\mathbf{x} \cdot \mathbf{y}$ | $\sum_i x_i y_i$ | Projection |
| Outer Product | $\mathbf{x}\mathbf{y}^T$ | $[x_i y_j]$ matrix | Rank-1 matrix |

### 🔍 Dot Product: Complete Derivation

**Theorem**: $\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$

**Proof**:
```
Step 1: Start with the Law of Cosines
        ‖x - y‖² = ‖x‖² + ‖y‖² - 2‖x‖‖y‖cos(θ)

Step 2: Expand the left side algebraically
        ‖x - y‖² = (x - y)ᵀ(x - y)
                 = xᵀx - xᵀy - yᵀx + yᵀy
                 = ‖x‖² - 2xᵀy + ‖y‖²

Step 3: Equate the two expressions
        ‖x‖² - 2xᵀy + ‖y‖² = ‖x‖² + ‖y‖² - 2‖x‖‖y‖cos(θ)

Step 4: Simplify
        -2xᵀy = -2‖x‖‖y‖cos(θ)
        xᵀy = ‖x‖‖y‖cos(θ)  ∎
```

### 💡 Examples

**Example 1**: Basic Dot Product
```python
x = [1, 2, 3]
y = [4, 5, 6]
x·y = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

**Example 2**: Angle Between Vectors
```python
x = [1, 0]
y = [1, 1]
x·y = 1
‖x‖ = 1, ‖y‖ = √2
cos(θ) = 1/(1×√2) = 1/√2
θ = 45°
```

**Example 3**: Orthogonality Check
```python
x = [1, 1]
y = [1, -1]
x·y = 1×1 + 1×(-1) = 0  ⟹ x ⊥ y (orthogonal)
```

**Example 4**: Projection
```python
# Project x onto y
x = [3, 4]
y = [1, 0]
proj_y(x) = (x·y / ‖y‖²) × y = (3/1) × [1,0] = [3, 0]
```

### 💻 Code Implementation

```python
import numpy as np
import torch

# NumPy implementation
def dot_product_numpy(x, y):
    """Compute dot product with NumPy"""
    return np.dot(x, y)  # or x @ y

def outer_product_numpy(x, y):
    """Compute outer product (rank-1 matrix)"""
    return np.outer(x, y)

def angle_between_vectors(x, y):
    """Compute angle in radians between vectors"""
    cos_theta = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return np.arccos(np.clip(cos_theta, -1, 1))

def project_onto(x, y):
    """Project x onto y"""
    return (np.dot(x, y) / np.dot(y, y)) * y

# PyTorch implementation (GPU-accelerated)
def dot_product_torch(x, y):
    """Compute dot product with PyTorch"""
    return torch.dot(x, y)  # For 1D tensors
    # or torch.matmul(x, y) for general case

# Example usage
x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])

print(f"Dot product: {dot_product_numpy(x, y)}")           # 32.0
print(f"Outer product shape: {outer_product_numpy(x, y).shape}")  # (3, 3)
print(f"Angle (degrees): {np.degrees(angle_between_vectors(x, y)):.2f}")  # 12.93°

# GPU computation with PyTorch
x_gpu = torch.tensor([1.0, 2.0, 3.0], device='cuda' if torch.cuda.is_available() else 'cpu')
y_gpu = torch.tensor([4.0, 5.0, 6.0], device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dot product (GPU): {dot_product_torch(x_gpu, y_gpu)}")
```

### 🤖 ML Application: Attention Scores

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, d_k):
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    The dot product QK^T measures similarity between queries and keys.
    Scaling by √d_k prevents softmax saturation for large d_k.
    """
    # Q, K: (batch, seq_len, d_k)
    # V: (batch, seq_len, d_v)
    
    # Compute attention scores via dot product
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq, seq)
    
    # Scale to prevent vanishing gradients
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Why scale by √d_k?
# If q, k ~ N(0, 1), then q·k ~ N(0, d_k)
# Large variance → softmax saturates → vanishing gradients
# Scaling by √d_k → q·k/√d_k ~ N(0, 1) → stable gradients
```

---

## 2. Matrix Operations

### 📌 Definition

A **matrix** $A \in \mathbb{R}^{m \times n}$ is a rectangular array with $m$ rows and $n$ columns.

### 📐 Matrix Multiplication

**Definition**: $(AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}$

**Dimensions**: $(m \times n) \cdot (n \times p) = (m \times p)$

```
Matrix Multiplication Visualization:

    A (m×n)          B (n×p)          C = AB (m×p)
    ┌───────┐        ┌─────┐          ┌─────┐
    │ row i │   ×    │col j│    =     │ cᵢⱼ │
    │ ──→── │        │  ↓  │          │     │
    └───────┘        └─────┘          └─────┘
    
    cᵢⱼ = (row i of A) · (column j of B)
        = Σₖ Aᵢₖ Bₖⱼ
```

### 🔍 Three Interpretations of Matrix Multiplication

**1. Row-Column Dot Products**
```
(AB)ᵢⱼ = Aᵢ,: · B:,ⱼ  (dot product of row i and column j)
```

**2. Column Combinations**
```
Each column of AB is a linear combination of columns of A:
(AB):,ⱼ = b₁ⱼ·A:,₁ + b₂ⱼ·A:,₂ + ... + bₙⱼ·A:,ₙ
```

**3. Sum of Outer Products**
```
AB = Σₖ A:,ₖ ⊗ Bₖ,:  (sum of rank-1 matrices)
```

### 🔍 Why Matrix Multiplication is NOT Commutative

**Theorem**: In general, $AB \neq BA$

**Counterexample**:
```
A = [1 0]    B = [0 1]
    [0 0]        [0 0]

AB = [0 1]    BA = [0 0]
     [0 0]         [0 0]

AB ≠ BA  ∎
```

**Geometric Intuition**: Rotation then scaling ≠ Scaling then rotation

### 📐 Key Matrix Properties

| Property | Formula | Proof Sketch |
|----------|---------|--------------|
| $(AB)^T = B^T A^T$ | Reverse order | $(AB)^T_{ij} = (AB)_{ji} = \sum_k A_{jk}B_{ki} = \sum_k B^T_{ik}A^T_{kj}$ |
| $(AB)^{-1} = B^{-1}A^{-1}$ | Reverse order | $(AB)(B^{-1}A^{-1}) = A(BB^{-1})A^{-1} = I$ |
| $\text{tr}(AB) = \text{tr}(BA)$ | Cyclic | $\sum_i(AB)_{ii} = \sum_i\sum_j A_{ij}B_{ji}$ (same terms) |

### 💻 Code Implementation

```python
import numpy as np
import torch

# Matrix multiplication
def matmul_examples():
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)
    
    # Three equivalent ways
    C1 = A @ B                    # @ operator (Python 3.5+)
    C2 = np.matmul(A, B)          # np.matmul
    C3 = np.dot(A, B)             # np.dot (for 2D arrays)
    
    # Verify properties
    print(f"(AB)ᵀ = BᵀAᵀ: {np.allclose((A @ B).T, B.T @ A.T)}")
    
    # Batch matrix multiplication
    batch_A = np.random.randn(32, 3, 4)  # 32 matrices of shape 3×4
    batch_B = np.random.randn(32, 4, 5)  # 32 matrices of shape 4×5
    batch_C = np.matmul(batch_A, batch_B)  # Result: (32, 3, 5)
    
    return batch_C

# PyTorch batch operations (essential for ML)
def batch_matmul_torch():
    """Batch matrix multiplication for neural network layers"""
    batch_size = 32
    seq_len = 128
    d_model = 512
    
    # Batch of input sequences
    X = torch.randn(batch_size, seq_len, d_model)
    
    # Weight matrix (shared across batch)
    W = torch.randn(d_model, d_model)
    
    # Apply linear transformation to all sequences
    Y = X @ W  # (32, 128, 512)
    
    # This is exactly what nn.Linear does (plus bias)
    linear = torch.nn.Linear(d_model, d_model, bias=False)
    linear.weight.data = W.T  # Note: PyTorch stores weights transposed
    Y_check = linear(X)
    
    return Y

# Efficient trace computation
def efficient_trace(A, B):
    """Compute tr(AB) without forming AB"""
    # tr(AB) = sum of element-wise product of A and B^T
    return np.sum(A * B.T)  # O(n²) instead of O(n³)
```

---

## 3. Vector Norms: Complete Theory

### 📌 Definition

A **norm** $\|\cdot\|$ is a function that assigns a non-negative length to vectors.

### 📐 Norm Axioms

For any norm $\|\cdot\|$:
1. **Non-negativity**: $\|\mathbf{x}\| \geq 0$, with equality iff $\mathbf{x} = \mathbf{0}$
2. **Homogeneity**: $\|\alpha\mathbf{x}\| = |\alpha| \|\mathbf{x}\|$
3. **Triangle Inequality**: $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

### 📊 Common Norms

| Norm | Formula | Unit Ball Shape | ML Use |
|------|---------|-----------------|--------|
| $L^0$ (pseudo) | $\sum_i \mathbf{1}_{x_i \neq 0}$ | Sparse | Sparsity (not convex!) |
| $L^1$ (Manhattan) | $\sum_i |x_i|$ | Diamond | Lasso, sparse regularization |
| $L^2$ (Euclidean) | $\sqrt{\sum_i x_i^2}$ | Circle | Ridge, weight decay |
| $L^\infty$ (Max) | $\max_i |x_i|$ | Square | Adversarial robustness |
| $L^p$ (General) | $(\sum_i |x_i|^p)^{1/p}$ | Superellipse | General regularization |

### 🔍 Proof: Triangle Inequality for L² Norm

**Theorem (Cauchy-Schwarz)**: $|\mathbf{x} \cdot \mathbf{y}| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$

**Proof**:
```
Step 1: Consider f(t) = ‖x + ty‖² ≥ 0 for all t ∈ ℝ

Step 2: Expand
        f(t) = (x + ty)ᵀ(x + ty)
             = ‖x‖² + 2t(xᵀy) + t²‖y‖²

Step 3: This is a quadratic in t that's always non-negative
        For ax² + bx + c ≥ 0 with a > 0, we need b² - 4ac ≤ 0

Step 4: Apply discriminant condition
        a = ‖y‖², b = 2(xᵀy), c = ‖x‖²
        4(xᵀy)² - 4‖y‖²‖x‖² ≤ 0
        (xᵀy)² ≤ ‖x‖²‖y‖²
        |xᵀy| ≤ ‖x‖‖y‖  ∎
```

**Triangle Inequality Proof**:
```
Step 1: Square both sides (valid since norms are non-negative)
        ‖x + y‖² ≤ (‖x‖ + ‖y‖)²

Step 2: Expand left side
        ‖x + y‖² = xᵀx + 2xᵀy + yᵀy = ‖x‖² + 2xᵀy + ‖y‖²

Step 3: Expand right side
        (‖x‖ + ‖y‖)² = ‖x‖² + 2‖x‖‖y‖ + ‖y‖²

Step 4: Need to show: ‖x‖² + 2xᵀy + ‖y‖² ≤ ‖x‖² + 2‖x‖‖y‖ + ‖y‖²
        ⟺ xᵀy ≤ ‖x‖‖y‖
        
Step 5: This follows from Cauchy-Schwarz  ∎
```

### 💻 Code Implementation

```python
import numpy as np
import torch

def compute_norms(x):
    """Compute various norms of a vector"""
    results = {
        'L0': np.sum(x != 0),           # Count non-zeros
        'L1': np.linalg.norm(x, ord=1),  # Sum of absolute values
        'L2': np.linalg.norm(x, ord=2),  # Euclidean length
        'Linf': np.linalg.norm(x, ord=np.inf),  # Maximum absolute value
    }
    return results

def soft_thresholding(x, threshold):
    """Proximal operator for L1 norm (used in LASSO)"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def normalize_vector(x, p=2):
    """Normalize vector to unit norm"""
    norm = np.linalg.norm(x, ord=p)
    return x / norm if norm > 0 else x

# PyTorch norms for gradients and weights
def regularization_losses(model, lambda_l1=0.01, lambda_l2=0.01):
    """Compute L1 and L2 regularization losses"""
    l1_loss = 0
    l2_loss = 0
    
    for param in model.parameters():
        l1_loss += torch.norm(param, p=1)
        l2_loss += torch.norm(param, p=2) ** 2
    
    return lambda_l1 * l1_loss + lambda_l2 * l2_loss

# Example: Unit ball visualization
x = np.array([3, 4])
print(f"Vector: {x}")
print(f"L1 norm: {np.linalg.norm(x, 1)}")   # 7
print(f"L2 norm: {np.linalg.norm(x, 2)}")   # 5
print(f"Linf norm: {np.linalg.norm(x, np.inf)}")  # 4
```

---

## 4. Matrix Norms

### 📊 Common Matrix Norms

| Norm | Formula | Computation | ML Use |
|------|---------|-------------|--------|
| Frobenius | $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$ | $\sqrt{\text{tr}(A^TA)}$ | Weight decay |
| Spectral | $\|A\|_2 = \sigma_{\max}(A)$ | Largest singular value | Lipschitz constant |
| Nuclear | $\|A\|_* = \sum_i \sigma_i$ | Sum of singular values | Low-rank regularization |

### 🔍 Frobenius Norm: Connection to SVD

**Theorem**: $\|A\|_F^2 = \sum_i \sigma_i^2$ (sum of squared singular values)

**Proof**:
```
Step 1: Express in terms of trace
        ‖A‖²_F = Σᵢⱼ A²ᵢⱼ = tr(AᵀA)

Step 2: Use SVD: A = UΣVᵀ
        AᵀA = (UΣVᵀ)ᵀ(UΣVᵀ) = VΣᵀUᵀUΣVᵀ = VΣ²Vᵀ

Step 3: Apply cyclic property of trace
        tr(AᵀA) = tr(VΣ²Vᵀ) = tr(Σ²VᵀV) = tr(Σ²) = Σᵢ σᵢ²  ∎
```

### 🔍 Submultiplicativity

**Theorem**: $\|AB\|_F \leq \|A\|_F \|B\|_F$

**Proof**:
```
Step 1: Write AB in terms of columns
        (AB)ⱼ = A · Bⱼ  (j-th column of AB is A times j-th column of B)

Step 2: Apply triangle inequality
        ‖(AB)ⱼ‖ = ‖A · Bⱼ‖ ≤ ‖A‖₂ · ‖Bⱼ‖ ≤ ‖A‖_F · ‖Bⱼ‖

Step 3: Sum over columns
        ‖AB‖²_F = Σⱼ ‖(AB)ⱼ‖² ≤ ‖A‖²_F · Σⱼ ‖Bⱼ‖² = ‖A‖²_F · ‖B‖²_F  ∎
```

### 💻 Code Implementation

```python
import numpy as np
import torch

def matrix_norms(A):
    """Compute various matrix norms"""
    U, S, Vh = np.linalg.svd(A)
    
    return {
        'Frobenius': np.linalg.norm(A, 'fro'),      # sqrt(sum of squares)
        'Spectral': np.linalg.norm(A, 2),           # largest singular value
        'Nuclear': np.sum(S),                        # sum of singular values
        'Max': np.max(np.abs(A)),                    # largest element
    }

# Spectral normalization for GANs (stabilizes training)
def spectral_norm(W, num_iters=1):
    """
    Compute spectral norm using power iteration.
    Used in Spectral Normalization for GANs (Miyato et al., 2018)
    """
    u = np.random.randn(W.shape[0])
    u = u / np.linalg.norm(u)
    
    for _ in range(num_iters):
        v = W.T @ u
        v = v / np.linalg.norm(v)
        u = W @ v
        u = u / np.linalg.norm(u)
    
    sigma = u @ W @ v  # Approximation of largest singular value
    return sigma

# PyTorch spectral norm (built-in)
# torch.nn.utils.spectral_norm(layer)
```

---

## 5. Eigenvalues and Eigenvectors

### 📌 Definition

For a square matrix $A \in \mathbb{R}^{n \times n}$:

$$A\mathbf{v} = \lambda\mathbf{v}$$

where:
- $\lambda$ is an **eigenvalue** (scalar)
- $\mathbf{v} \neq \mathbf{0}$ is the corresponding **eigenvector**

### 📐 Geometric Intuition

```
Eigenvector Visualization:

Before transformation (v):          After transformation (Av):
        ↑                                    ↑
        │                                    │
     ───┼───→                            ────┼────→
        │                                    │
        v                                   λv
        
The eigenvector v only gets SCALED by λ, not rotated.
- λ > 1: stretched
- 0 < λ < 1: compressed  
- λ < 0: flipped and scaled
- λ = 0: collapsed to zero (singular)
```

### 🔍 Finding Eigenvalues: Characteristic Polynomial

**Derivation**:
```
Step 1: Start with eigenvalue equation
        Av = λv

Step 2: Rearrange
        Av - λv = 0
        (A - λI)v = 0

Step 3: For non-trivial solution (v ≠ 0), the matrix (A - λI) must be singular
        det(A - λI) = 0  ← Characteristic polynomial

Step 4: This is a polynomial of degree n in λ
        det(A - λI) = (-1)ⁿ(λⁿ - (tr A)λⁿ⁻¹ + ... + det A)
```

### 💡 Examples

**Example 1**: 2×2 Matrix
```
A = [4  1]
    [2  3]

Step 1: Characteristic polynomial
det(A - λI) = det([4-λ   1 ])
                 ([2   3-λ])
            = (4-λ)(3-λ) - 2
            = λ² - 7λ + 10
            = (λ-5)(λ-2)

Step 2: Eigenvalues
λ₁ = 5, λ₂ = 2

Step 3: Find eigenvectors
For λ₁ = 5: (A - 5I)v = 0
[-1  1][v₁]   [0]
[ 2 -2][v₂] = [0]
⟹ v₁ = v₂, so v₁ = [1, 1]ᵀ

For λ₂ = 2: (A - 2I)v = 0
[2  1][v₁]   [0]
[2  1][v₂] = [0]
⟹ 2v₁ + v₂ = 0, so v₂ = [1, -2]ᵀ
```

**Example 2**: Rotation Matrix (Complex Eigenvalues)
```
R(θ) = [cos θ  -sin θ]
       [sin θ   cos θ]

det(R - λI) = (cos θ - λ)² + sin²θ
            = λ² - 2λcos θ + 1

λ = cos θ ± i sin θ = e^{±iθ}

Real interpretation: 2D rotation has no real eigenvectors
(no direction is preserved except at θ = 0 or π)
```

**Example 3**: Symmetric Matrix (Real, Orthogonal Eigenvectors)
```
A = [2  1]
    [1  2]

Eigenvalues: λ₁ = 3, λ₂ = 1
Eigenvectors: v₁ = [1, 1]ᵀ/√2, v₂ = [1, -1]ᵀ/√2

Note: v₁ · v₂ = 0 (orthogonal!)
```

### 📐 Key Properties

| Property | Formula | Significance |
|----------|---------|--------------|
| Sum | $\sum_i \lambda_i = \text{tr}(A)$ | Quick check |
| Product | $\prod_i \lambda_i = \det(A)$ | Invertibility: $\det \neq 0$ |
| Powers | $A^n = Q\Lambda^n Q^{-1}$ | Fast matrix powers |
| Inverse | $A^{-1} = Q\Lambda^{-1}Q^{-1}$ | If all $\lambda_i \neq 0$ |

### 💻 Code Implementation

```python
import numpy as np
import torch

def eigendecomposition(A):
    """
    Compute eigendecomposition A = QΛQ⁻¹
    For symmetric matrices, use eigh (faster, more stable)
    """
    if np.allclose(A, A.T):
        # Symmetric matrix: eigenvalues real, eigenvectors orthogonal
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    else:
        # General matrix: may have complex eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(A)
    
    return eigenvalues, eigenvectors

def power_iteration(A, num_iters=100, tol=1e-10):
    """
    Find dominant eigenvalue and eigenvector using power iteration.
    Converges at rate |λ₂/λ₁| per iteration.
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for i in range(num_iters):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)
        
        # Check convergence
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    
    # Rayleigh quotient gives eigenvalue
    eigenvalue = v @ A @ v
    return eigenvalue, v

def verify_eigendecomposition(A, eigenvalues, eigenvectors):
    """Verify Av = λv for each eigenpair"""
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = A @ v
        lam_v = lam * v
        error = np.linalg.norm(Av - lam_v)
        print(f"Eigenvalue {i}: λ = {lam:.4f}, error = {error:.2e}")

# PCA using eigendecomposition
def pca_eigen(X, n_components):
    """
    PCA via eigendecomposition of covariance matrix.
    Principal components = eigenvectors with largest eigenvalues.
    """
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Compute covariance matrix
    cov = X_centered.T @ X_centered / (len(X) - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project onto top components
    components = eigenvectors[:, :n_components]
    X_projected = X_centered @ components
    
    # Explained variance ratio
    explained_var = eigenvalues[:n_components] / eigenvalues.sum()
    
    return X_projected, explained_var
```

---

## 6. Spectral Theorem (with Proof)

### 📌 Theorem Statement

**Spectral Theorem**: If $A \in \mathbb{R}^{n \times n}$ is symmetric ($A = A^T$), then:
1. All eigenvalues are **real**
2. Eigenvectors can be chosen to be **orthonormal**
3. $A = Q\Lambda Q^T$ where $Q$ is orthogonal

### 🔍 Complete Proof

**Part 1: Eigenvalues are Real**
```
Step 1: Let λ be an eigenvalue with eigenvector v (possibly complex)
        Av = λv

Step 2: Take conjugate transpose of both sides
        (Av)* = (λv)*
        v*A* = λ̄v*  (since A is real, A* = Aᵀ = A)
        v*A = λ̄v*

Step 3: Multiply original equation on left by v*
        v*Av = λv*v = λ‖v‖²

Step 4: Multiply Step 2 result on right by v
        v*Av = λ̄v*v = λ̄‖v‖²

Step 5: Equate
        λ‖v‖² = λ̄‖v‖²
        λ = λ̄  (since ‖v‖² > 0)
        
Therefore λ is real.  ∎
```

**Part 2: Eigenvectors of Distinct Eigenvalues are Orthogonal**
```
Step 1: Let Av₁ = λ₁v₁ and Av₂ = λ₂v₂ with λ₁ ≠ λ₂

Step 2: Compute v₁ᵀAv₂ two ways

        Way 1: v₁ᵀAv₂ = v₁ᵀ(λ₂v₂) = λ₂(v₁ᵀv₂)

        Way 2: v₁ᵀAv₂ = v₁ᵀAᵀv₂  (since A = Aᵀ)
                      = (Av₁)ᵀv₂
                      = (λ₁v₁)ᵀv₂
                      = λ₁(v₁ᵀv₂)

Step 3: Equate
        λ₂(v₁ᵀv₂) = λ₁(v₁ᵀv₂)
        (λ₂ - λ₁)(v₁ᵀv₂) = 0

Step 4: Since λ₁ ≠ λ₂:
        v₁ᵀv₂ = 0  (orthogonal)  ∎
```

**Part 3: Matrix Form**
```
Collect orthonormal eigenvectors as columns of Q:
Q = [v₁ | v₂ | ... | vₙ]

Then:
AQ = A[v₁|...|vₙ] = [λ₁v₁|...|λₙvₙ] = QΛ

Since Q is orthogonal (QᵀQ = I):
A = QΛQᵀ  ∎
```

### 💡 Why This Matters for ML

```
1. PCA: Covariance matrix Σ is symmetric
   → Real eigenvalues = explained variance
   → Orthogonal eigenvectors = uncorrelated components

2. Quadratic Forms: f(x) = xᵀAx
   For symmetric A = QΛQᵀ:
   f(x) = xᵀQΛQᵀx = yᵀΛy = Σᵢ λᵢyᵢ²
   where y = Qᵀx
   
   This diagonalizes the quadratic form!

3. Positive Definiteness:
   A ≻ 0 ⟺ all λᵢ > 0 ⟺ xᵀAx > 0 for all x ≠ 0
```

---

## 7. Singular Value Decomposition (SVD)

### 📌 Theorem

**SVD Existence**: For ANY matrix $A \in \mathbb{R}^{m \times n}$:

$$A = U\Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$: orthogonal (left singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal with $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$: orthogonal (right singular vectors)

```
SVD Visualization:

    A (m×n)     =     U (m×m)    ×    Σ (m×n)    ×    Vᵀ (n×n)
    
   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
   │         │     │         │     │σ₁       │     │         │
   │    A    │  =  │    U    │  ×  │  σ₂     │  ×  │   Vᵀ    │
   │         │     │         │     │    ⋱    │     │         │
   │         │     │         │     │      σᵣ │     │         │
   └─────────┘     └─────────┘     └─────────┘     └─────────┘
   
   Any matrix    Rotation/        Scaling        Rotation/
                 reflection                      reflection
```

### 🔍 Complete Proof of SVD Existence

```
Step 1: Consider AᵀA ∈ ℝⁿˣⁿ

AᵀA is symmetric positive semi-definite:
  • Symmetric: (AᵀA)ᵀ = AᵀA  ✓
  • PSD: xᵀ(AᵀA)x = (Ax)ᵀ(Ax) = ‖Ax‖² ≥ 0  ✓

Step 2: Apply Spectral Theorem to AᵀA

Since AᵀA is symmetric:
  AᵀA = VΛVᵀ

where:
  V orthogonal (eigenvectors vᵢ)
  Λ diagonal with λᵢ ≥ 0 (eigenvalues)

Define: σᵢ = √λᵢ  (singular values)

Step 3: Construct U

For each non-zero σᵢ, define:
  uᵢ = (1/σᵢ)Avᵢ

Verify orthonormality:
  uᵢᵀuⱼ = (1/σᵢσⱼ)(Avᵢ)ᵀ(Avⱼ)
        = (1/σᵢσⱼ)vᵢᵀ(AᵀA)vⱼ
        = (1/σᵢσⱼ)vᵢᵀ(λⱼvⱼ)
        = (1/σᵢσⱼ)σⱼ²(vᵢᵀvⱼ)
        = (σⱼ/σᵢ)δᵢⱼ
        = δᵢⱼ  ✓

Complete {uᵢ} to orthonormal basis of ℝᵐ.

Step 4: Verify A = UΣVᵀ

For each vⱼ:
  Avⱼ = σⱼuⱼ  (by construction of uⱼ)

In matrix form:
  AV = UΣ
  A = UΣVᵀ  ∎
```

### 📐 Key Properties of SVD

| Property | Formula | Significance |
|----------|---------|--------------|
| Rank | $\text{rank}(A) = \#\{\sigma_i > 0\}$ | Count non-zero singular values |
| Frobenius norm | $\|A\|_F^2 = \sum_i \sigma_i^2$ | Sum of squared singular values |
| Spectral norm | $\|A\|_2 = \sigma_1$ | Largest singular value |
| Condition number | $\kappa(A) = \sigma_1/\sigma_r$ | Numerical stability |
| Pseudoinverse | $A^+ = V\Sigma^+U^T$ | Generalized inverse |

### 💻 Code Implementation

```python
import numpy as np
import torch

def svd_decomposition(A, full_matrices=True):
    """
    Compute SVD: A = UΣVᵀ
    
    Parameters:
    - full_matrices: If True, U is m×m, V is n×n (full)
                     If False, U is m×r, V is n×r (economy/thin)
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    
    # S is returned as 1D array of singular values
    # To reconstruct A: U @ np.diag(S) @ Vt (for economy SVD)
    
    return U, S, Vt

def verify_svd(A, U, S, Vt):
    """Verify A = UΣVᵀ"""
    # For economy SVD
    A_reconstructed = U @ np.diag(S) @ Vt
    error = np.linalg.norm(A - A_reconstructed)
    print(f"Reconstruction error: {error:.2e}")

def svd_properties(A):
    """Compute various properties using SVD"""
    U, S, Vt = np.linalg.svd(A)
    
    rank = np.sum(S > 1e-10)
    frobenius = np.sqrt(np.sum(S**2))
    spectral = S[0]
    nuclear = np.sum(S)
    condition = S[0] / S[-1] if S[-1] > 1e-10 else np.inf
    
    return {
        'rank': rank,
        'frobenius_norm': frobenius,
        'spectral_norm': spectral,
        'nuclear_norm': nuclear,
        'condition_number': condition,
    }

# PyTorch SVD (GPU-accelerated)
def svd_torch(A_tensor):
    """PyTorch SVD for GPU computation"""
    U, S, Vh = torch.linalg.svd(A_tensor)
    return U, S, Vh
```

---

## 8. Low-Rank Approximation (Eckart-Young Theorem)

### 📌 Theorem

The best rank-$k$ approximation to $A$ (in Frobenius norm) is:

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T$$

with approximation error:

$$\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$$

### 🔍 Proof (Eckart-Young)

```
Step 1: Setup
Let B be ANY rank-k matrix. Want to show:
  ‖A - Aₖ‖_F ≤ ‖A - B‖_F

Step 2: Compute error for Aₖ
  A - Aₖ = Σᵢ₌ₖ₊₁ʳ σᵢuᵢvᵢᵀ
  
  ‖A - Aₖ‖²_F = Σᵢ₌ₖ₊₁ʳ σᵢ²  (since {uᵢvᵢᵀ} are orthonormal)

Step 3: Key insight - dimension counting
Since rank(B) = k, the null space of B has dimension n - k.
The space span{v₁,...,vₖ₊₁} has dimension k + 1.

By dimension counting, these spaces must intersect:
  ∃ unit vector z ∈ span{v₁,...,vₖ₊₁} ∩ null(B)

Step 4: Lower bound for ‖A - B‖
Write z = Σᵢ₌₁ᵏ⁺¹ αᵢvᵢ with Σᵢαᵢ² = 1

Since Bz = 0:
  ‖A - B‖²_F ≥ ‖(A - B)z‖²
             = ‖Az‖²  (since Bz = 0)
             = ‖Σᵢ₌₁ᵏ⁺¹ αᵢσᵢuᵢ‖²
             = Σᵢ₌₁ᵏ⁺¹ αᵢ²σᵢ²
             ≥ σₖ₊₁² · Σᵢ₌₁ᵏ⁺¹ αᵢ²  (since σᵢ ≥ σₖ₊₁ for i ≤ k+1)
             = σₖ₊₁²

Step 5: Extend to full error
More careful analysis shows:
  ‖A - B‖²_F ≥ Σᵢ₌ₖ₊₁ʳ σᵢ² = ‖A - Aₖ‖²_F

Therefore Aₖ is optimal.  ∎
```

### 💡 Application to LoRA

```
LoRA (Low-Rank Adaptation) exploits this theorem:

Pre-trained weight: W₀ ∈ ℝᵈˣᵏ
Fine-tuning update: ΔW

Key insight: ΔW often has low intrinsic rank!

SVD of ΔW: ΔW = UΣVᵀ

If singular values decay rapidly:
  σ₁ >> σ₂ >> ... >> σᵣ

Then low-rank approximation is accurate:
  ΔW ≈ ΔWᵣ = Σᵢ₌₁ʳ σᵢuᵢvᵢᵀ = U_r Σ_r V_r^T

LoRA parameterization:
  ΔW = BA  where B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵏ

This reduces parameters from d×k to r×(d+k)!
For r << min(d,k), massive compression.
```

### 💻 Code Implementation

```python
import numpy as np
import torch
import torch.nn as nn

def low_rank_approximation(A, rank):
    """
    Compute best rank-k approximation using SVD.
    Error: ||A - A_k||_F² = sum of squared discarded singular values
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep top-k components
    A_k = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    
    # Compute approximation error
    error = np.sqrt(np.sum(S[rank:]**2))
    
    return A_k, error

def compression_ratio(original_shape, rank):
    """Compute compression ratio for low-rank approximation"""
    m, n = original_shape
    original_params = m * n
    compressed_params = rank * (m + n)
    return original_params / compressed_params

# LoRA implementation
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    Original: y = Wx
    LoRA:     y = Wx + BAx  where B: (d_out, r), A: (r, d_in)
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Original weight (frozen)
        self.W = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # Low-rank adaptation
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original output + low-rank update
        return x @ self.W.T + (x @ self.A.T @ self.B.T) * self.scaling

# Image compression example
def compress_image(image, rank):
    """
    Compress grayscale image using SVD.
    For RGB, apply to each channel separately.
    """
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    compressed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    
    # Compute compression metrics
    original_size = image.shape[0] * image.shape[1]
    compressed_size = rank * (image.shape[0] + image.shape[1] + 1)
    compression = original_size / compressed_size
    
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    
    return compressed, compression, error
```

---

## 9. Positive Definite Matrices

### 📌 Definition

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is:

| Type | Condition | Notation |
|------|-----------|----------|
| Positive Definite | $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$ | $A \succ 0$ |
| Positive Semi-Definite | $\mathbf{x}^T A \mathbf{x} \geq 0$ for all $\mathbf{x}$ | $A \succeq 0$ |
| Negative Definite | $\mathbf{x}^T A \mathbf{x} < 0$ for all $\mathbf{x} \neq \mathbf{0}$ | $A \prec 0$ |
| Indefinite | $\mathbf{x}^T A \mathbf{x}$ can be positive or negative | — |

### 🔍 Equivalent Characterizations

**Theorem**: For symmetric $A$, the following are equivalent:

1. $A \succ 0$ (positive definite)
2. All eigenvalues $\lambda_i > 0$
3. All leading principal minors $> 0$ (Sylvester's criterion)
4. $A = B^T B$ for some invertible $B$
5. Cholesky decomposition $A = LL^T$ exists with $L$ having positive diagonal

### 🔍 Proof: (1) ⟺ (2)

```
(⟹) Assume A ≻ 0
Let λ be an eigenvalue with eigenvector v (v ≠ 0):
  Av = λv

Compute:
  vᵀAv = vᵀ(λv) = λ(vᵀv) = λ‖v‖²

Since A ≻ 0: vᵀAv > 0
Since ‖v‖² > 0: λ > 0  ✓

(⟸) Assume all λᵢ > 0
Use spectral theorem: A = QΛQᵀ

For any x ≠ 0, let y = Qᵀx (so x = Qy):
  xᵀAx = xᵀQΛQᵀx = yᵀΛy = Σᵢ λᵢyᵢ²

Since Q is orthogonal, y ≠ 0 when x ≠ 0.
So at least one yᵢ ≠ 0.
Since all λᵢ > 0: Σᵢ λᵢyᵢ² > 0  ✓
```

### 📐 Cholesky Decomposition

For $A \succ 0$: $A = LL^T$ where $L$ is lower triangular with positive diagonal.

**Algorithm**:
```
For j = 1 to n:
    L[j,j] = sqrt(A[j,j] - sum(L[j,1:j-1]²))
    
    For i = j+1 to n:
        L[i,j] = (A[i,j] - sum(L[i,1:j-1] * L[j,1:j-1])) / L[j,j]
```

**Complexity**: $O(n^3/3)$ — faster than LU decomposition!

### 💻 Code Implementation

```python
import numpy as np
import torch

def is_positive_definite(A, tol=1e-10):
    """Check if matrix is positive definite"""
    if not np.allclose(A, A.T):
        return False  # Must be symmetric
    
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > tol)

def make_positive_definite(A, epsilon=1e-6):
    """Make a matrix positive definite by adding to diagonal"""
    # Eigenvalue modification
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def cholesky_decomposition(A):
    """
    Compute Cholesky decomposition A = LLᵀ
    Only works for positive definite matrices!
    """
    try:
        L = np.linalg.cholesky(A)
        return L
    except np.linalg.LinAlgError:
        print("Matrix is not positive definite!")
        return None

def solve_with_cholesky(A, b):
    """
    Solve Ax = b using Cholesky decomposition.
    More stable and faster than direct inversion for PD matrices.
    """
    L = np.linalg.cholesky(A)
    # A = LLᵀ, so solve Ly = b then Lᵀx = y
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x

# Check Hessian for optimization
def analyze_hessian(hessian):
    """Analyze Hessian matrix at critical point"""
    eigenvalues = np.linalg.eigvalsh(hessian)
    
    if np.all(eigenvalues > 0):
        return "Local minimum (Hessian positive definite)"
    elif np.all(eigenvalues < 0):
        return "Local maximum (Hessian negative definite)"
    else:
        return f"Saddle point (eigenvalues: {eigenvalues})"

# Covariance matrix is always PSD
def compute_covariance(X):
    """
    Compute sample covariance matrix.
    Always positive semi-definite by construction.
    """
    X_centered = X - X.mean(axis=0)
    n = len(X)
    cov = X_centered.T @ X_centered / (n - 1)
    
    # Add small regularization for numerical stability
    cov += 1e-6 * np.eye(cov.shape[0])
    
    return cov
```

---

## 10. QR Decomposition

### 📌 Theorem

For any matrix $A \in \mathbb{R}^{m \times n}$ with $m \geq n$:

$$A = QR$$

where:
- $Q \in \mathbb{R}^{m \times n}$: columns are orthonormal
- $R \in \mathbb{R}^{n \times n}$: upper triangular

### 📐 Gram-Schmidt Process

```
Given columns a₁, ..., aₙ of A:

For j = 1 to n:
    # Start with original column
    ũⱼ = aⱼ
    
    # Subtract projections onto previous orthonormal vectors
    For i = 1 to j-1:
        ũⱼ = ũⱼ - (qᵢᵀaⱼ)qᵢ
    
    # Normalize
    qⱼ = ũⱼ / ‖ũⱼ‖
    
    # Record coefficients
    rᵢⱼ = qᵢᵀaⱼ  for i < j
    rⱼⱼ = ‖ũⱼ‖

Result: Q = [q₁|...|qₙ], R = [rᵢⱼ]
```

### 💡 Applications

1. **Least Squares**: Solve $Ax = b$ stably
   ```
   QRx = b  ⟹  Rx = Qᵀb  (triangular system, easy to solve)
   ```

2. **Eigenvalue Computation**: QR algorithm
   ```
   A₀ = A
   For k = 1, 2, ...:
       Aₖ₋₁ = QₖRₖ  (QR factorization)
       Aₖ = RₖQₖ    (reverse multiply)
   
   Aₖ → diagonal (eigenvalues) as k → ∞
   ```

### 💻 Code Implementation

```python
import numpy as np

def gram_schmidt(A):
    """Classical Gram-Schmidt orthogonalization"""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R

def modified_gram_schmidt(A):
    """Modified Gram-Schmidt (numerically stable)"""
    m, n = A.shape
    Q = A.copy().astype(float)
    R = np.zeros((n, n))
    
    for j in range(n):
        for i in range(j):
            R[i, j] = Q[:, i] @ Q[:, j]
            Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]
    
    return Q, R

def solve_least_squares_qr(A, b):
    """Solve min ||Ax - b||² using QR decomposition"""
    Q, R = np.linalg.qr(A)
    
    # Transform: Rx = Qᵀb
    c = Q.T @ b
    
    # Back substitution
    x = np.linalg.solve(R, c)
    
    return x
```

---

## 📋 Key Formulas Summary

| Concept | Formula | When to Use |
|---------|---------|-------------|
| **Dot Product** | $\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$ | Similarity, projection |
| **Matrix Multiply** | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ | Neural network layers |
| **Frobenius Norm** | $\|A\|_F = \sqrt{\sum_{ij}A_{ij}^2} = \sqrt{\sum_i \sigma_i^2}$ | Weight regularization |
| **Eigenvalue** | $A\mathbf{v} = \lambda\mathbf{v}$ | PCA, stability |
| **Spectral Decomp** | $A = Q\Lambda Q^T$ (symmetric) | Quadratic forms |
| **SVD** | $A = U\Sigma V^T$ | Low-rank approx, compression |
| **Low-Rank Approx** | $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i\mathbf{v}_i^T$ | LoRA, denoising |
| **Positive Definite** | $\mathbf{x}^TA\mathbf{x} > 0$ for $\mathbf{x} \neq 0$ | Covariance, Hessian |
| **Cholesky** | $A = LL^T$ | Fast PD solve |
| **QR** | $A = QR$ | Least squares |

---

## ⚠️ Common Mistakes & Pitfalls

### ❌ Mistake 1: Confusing Eigendecomposition and SVD

```python
# WRONG: Using eigen on non-square matrix
A = np.random.randn(3, 5)
eigenvalues, eigenvectors = np.linalg.eig(A)  # ERROR!

# CORRECT: Use SVD for rectangular matrices
U, S, Vt = np.linalg.svd(A)
```

### ❌ Mistake 2: Forgetting Transpose Order Reverses

```python
# WRONG: (AB)ᵀ = AᵀBᵀ
wrong = A.T @ B.T

# CORRECT: (AB)ᵀ = BᵀAᵀ
correct = B.T @ A.T
```

### ❌ Mistake 3: Assuming Matrix Multiplication Commutes

```python
# WRONG: Assuming AB = BA
result1 = A @ B
result2 = B @ A  # Generally different!
```

### ❌ Mistake 4: Numerical Instability with Matrix Inverse

```python
# FRAGILE: Direct inversion
x = np.linalg.inv(A) @ b

# ROBUST: Use solve
x = np.linalg.solve(A, b)

# EVEN BETTER for PD: Use Cholesky
L = np.linalg.cholesky(A)
x = np.linalg.solve(L.T, np.linalg.solve(L, b))
```

### ❌ Mistake 5: Not Checking Positive Definiteness

```python
# WRONG: Assuming covariance is always invertible
cov = X.T @ X / n
inv_cov = np.linalg.inv(cov)  # May fail if singular!

# CORRECT: Add regularization
cov = X.T @ X / n + 1e-6 * np.eye(X.shape[1])
inv_cov = np.linalg.inv(cov)
```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 📖 | [Linear Algebra Done Right](https://linear.axler.net/) - Axler | Theoretical foundation |
| 📖 | [Matrix Computations](https://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm) - Golub & Van Loan | Numerical methods |
| 📖 | [Mathematics for Machine Learning](https://mml-book.github.io/) | ML-focused |
| 🎥 | [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Visual intuition |
| 🎥 | [MIT 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) - Gilbert Strang | Classic course |
| 🎥 | [Steve Brunton](https://www.youtube.com/c/Eigensteve) | SVD applications |

---

## 📂 Subtopics in This Section

| Folder | Topics | Key Concepts | ML Application |
|--------|--------|--------------|----------------|
| [01_decompositions](./01_decompositions/) | SVD, QR, Cholesky | Matrix factorization | LoRA, compression |
| [02_dimensionality_reduction](./02_dimensionality_reduction/) | PCA, t-SNE | Variance maximization | Visualization |
| [03_eigen](./03_eigen/) | Eigenvalues, eigenvectors | Spectral theory | Stability analysis |
| [04_eigenvalues](./04_eigenvalues/) | Characteristic polynomial | Advanced eigen topics | PageRank |
| [05_matrix_factorization](./05_matrix_factorization/) | NMF, matrix completion | Non-negative factorization | Recommender systems |
| [06_matrix_properties](./06_matrix_properties/) | Rank, determinant, trace | Matrix analysis | Debugging |
| [07_svd](./07_svd/) | SVD deep dive | Complete SVD theory | Everything! |
| [08_transformations](./08_transformations/) | Linear maps | Basis change | Layer design |
| [09_vectors_matrices](./09_vectors_matrices/) | Operations, norms | Foundational operations | Data representation |
| [10_vector_spaces](./10_vector_spaces/) | Span, basis, dimension | Abstract algebra | Feature spaces |

---

## 🗺️ Navigation

<p align="center">

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Mathematics Overview](../) | [ML Researcher Foundations](../../) | [02 Calculus](../02_calculus/) |

</p>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>
</p>

<p align="center">
  <em>Linear Algebra: The foundation upon which all of machine learning is built.</em>
</p>
