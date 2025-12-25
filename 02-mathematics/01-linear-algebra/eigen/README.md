<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Eigen%20Theory&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Definition

```
Av = λv

Where:
• A ∈ ℝⁿˣⁿ: square matrix
• v ∈ ℝⁿ:   eigenvector (direction unchanged by A, v ≠ 0)
• λ ∈ ℂ:    eigenvalue (scaling factor)

Intuition:
When A acts on v, it only stretches/shrinks v by factor λ.
The direction of v is preserved (or reversed if λ < 0).
```

---

## 📐 Mathematical Derivation

### Finding Eigenvalues

```
Av = λv
Av - λv = 0
(A - λI)v = 0

For non-trivial solution (v ≠ 0):
det(A - λI) = 0   ← Characteristic polynomial

This gives at most n eigenvalues (counting multiplicity).
```

### Eigendecomposition

```
If A has n linearly independent eigenvectors:

A = VΛV⁻¹

Where:
• V = [v₁ | v₂ | ... | vₙ]  (eigenvectors as columns)
• Λ = diag(λ₁, λ₂, ..., λₙ)  (eigenvalues on diagonal)

For symmetric A:
A = VΛVᵀ  (V is orthogonal, so V⁻¹ = Vᵀ)
```

---

## 🔑 Key Properties

| Property | Formula | Significance |
|----------|---------|--------------|
| Characteristic polynomial | det(A - λI) = 0 | Finds all eigenvalues |
| Sum of eigenvalues | Σλᵢ = tr(A) | Quick check |
| Product of eigenvalues | Πλᵢ = det(A) | Invertibility (det ≠ 0) |
| Symmetric matrices | λᵢ ∈ ℝ, vᵢ ⊥ vⱼ | Real eigenvalues, orthogonal eigenvectors |
| Positive definite | λᵢ > 0 for all i | Covariance matrices |
| Powers | Aⁿ = VΛⁿV⁻¹ | Fast matrix powers |
| Inverse | A⁻¹ = VΛ⁻¹V⁻¹ | If all λᵢ ≠ 0 |

### Spectral Theorem

```
For symmetric A ∈ ℝⁿˣⁿ:

1. All eigenvalues are real
2. Eigenvectors can be chosen orthonormal
3. A = Σᵢ λᵢ vᵢvᵢᵀ  (spectral decomposition)

This is fundamental for:
• PCA (Principal Component Analysis)
• Gaussian distributions
• Quadratic forms
```

---

## 🌍 ML Applications

| Application | How Eigen Is Used | Key Insight |
|-------------|-------------------|-------------|
| **PCA** | Eigenvectors of Xᵀx/n | Directions of max variance |
| **Spectral Clustering** | Eigenvectors of Graph Laplacian | Community detection |
| **PageRank** | Principal eigenvector of link matrix | Stationary distribution |
| **RNN Stability** | |λ| < 1 for all eigenvalues | Prevents exploding gradients |
| **Hessian Analysis** | Eigenvalues of ∇²L | Curvature of loss surface |
| **Covariance** | Eigen of Σ | Feature correlation |
| **SVD** | Eigen of AᵀA and AAᵀ | Building block for SVD |

---

## 💻 Code Examples

```python
import numpy as np
import torch

# Basic eigendecomposition
A = np.random.randn(5, 5)
A = A @ A.T  # Make symmetric positive semi-definite

eigenvalues, eigenvectors = np.linalg.eigh(A)  # For symmetric
# eigenvalues: sorted ascending
# eigenvectors[:, i] corresponds to eigenvalues[i]

# Verify: A @ v = λ * v
v = eigenvectors[:, 0]
lam = eigenvalues[0]
print(f"Av: {A @ v}")
print(f"λv: {lam * v}")

# PCA using eigendecomposition
def pca_eigen(X, n_components):
    """PCA via eigendecomposition of covariance matrix"""
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (len(X) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Select top k eigenvectors (largest eigenvalues)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, idx]
    
    return X_centered @ components

# Power iteration (finds largest eigenvalue)
def power_iteration(A, num_iterations=100):
    """Find dominant eigenvalue and eigenvector"""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iterations):
        Av = A @ v
        v = Av / np.linalg.norm(Av)
    
    eigenvalue = v @ A @ v  # Rayleigh quotient
    return eigenvalue, v

# PyTorch eigendecomposition (for GPU)
A_torch = torch.randn(100, 100, device='cuda')
A_torch = A_torch @ A_torch.T  # Symmetric
eigenvalues, eigenvectors = torch.linalg.eigh(A_torch)
```

---

## 📚 Resources

### 📖 Books

| Title | Author | Focus |
|-------|--------|-------|
| Linear Algebra Done Right | Axler | Theoretical |
| Matrix Computations | Golub & Van Loan | Numerical |
| Mathematics for ML | Deisenroth | ML applications |

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 🎥 | 3Blue1Brown: Eigenvectors | [YouTube](https://www.youtube.com/watch?v=PFDu9oVAE-g) |
| 🎥 | Steve Brunton: SVD/Eigen | [YouTube](https://www.youtube.com/c/Eigensteve) |
| 📖 | MIT 18.06 Linear Algebra | [OCW](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) |
| 🇨🇳 | 特征值与特征向量详解 | [知乎](https://zhuanlan.zhihu.com/p/26980855) |
| 🇨🇳 | 线性代数本质 (3B1B) | [B站](https://www.bilibili.com/video/BV1ys411472E) |
| 🇨🇳 | PCA与特征分解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88785898)

---

## 🔗 Where This Topic Is Used

| Application | How Eigenvalues/Eigenvectors Are Used |
|-------------|---------------------------------------|
| **PCA (Principal Component Analysis)** | Eigenvectors of covariance = principal directions |
| **Spectral Clustering** | Graph Laplacian eigenvectors for community detection |
| **PageRank Algorithm** | Dominant eigenvector of link matrix |
| **Quantum Mechanics** | Eigenvalues = observable measurements |
| **Stability Analysis** | Eigenvalues of Jacobian determine system stability |
| **RNN Gradient Flow** | \|λ\| < 1 prevents exploding gradients |
| **Recommendation Systems** | Matrix factorization via eigendecomposition |
| **Signal Processing** | Fourier basis as eigenvectors of shift operator |

---


⬅️ [Back: Linear Algebra](../)

---

⬅️ [Back: Decompositions](../decompositions/) | ➡️ [Next: Matrix Properties](../matrix-properties/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
