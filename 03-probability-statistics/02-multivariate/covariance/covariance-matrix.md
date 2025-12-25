<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Covariance%20Matrix&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Covariance Matrix

> **Capturing relationships between random variables**

---

## 📐 Definition

```
For random vector X ∈ ℝⁿ:

Σ = Cov(X) = E[(X - μ)(X - μ)ᵀ]

Σ[i,j] = Cov(Xᵢ, Xⱼ) = E[(Xᵢ - μᵢ)(Xⱼ - μⱼ)]
```

---

## 🔑 Properties

| Property | Value |
|----------|-------|
| Symmetric | Σ = Σᵀ |
| Positive semi-definite | Σ ⪰ 0 |
| Diagonal | Variances |
| Off-diagonal | Covariances |

---

## 📊 Sample Covariance

```
Given n samples X₁, ..., Xₙ:

Σ̂ = (1/n) Σᵢ (Xᵢ - X̄)(Xᵢ - X̄)ᵀ

Or unbiased:
Σ̂ = (1/(n-1)) Σᵢ (Xᵢ - X̄)(Xᵢ - X̄)ᵀ
```

---

## 💻 Code

```python
import numpy as np

# Generate correlated data
X = np.random.randn(1000, 3)
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(1000)  # Correlated

# Covariance matrix
cov = np.cov(X.T)
print(cov.shape)  # (3, 3)

# Correlation matrix
corr = np.corrcoef(X.T)
print(corr)  # Diagonal = 1, off-diagonal ∈ [-1, 1]

# PyTorch
import torch
X_torch = torch.tensor(X, dtype=torch.float32)
cov_torch = torch.cov(X_torch.T)
```

---

## 🌍 Applications

| Application | Use |
|-------------|-----|
| PCA | Eigendecomposition of Σ |
| Gaussian models | Multivariate Normal |
| Mahalanobis distance | d² = (x-μ)ᵀΣ⁻¹(x-μ) |
| Whitening | Transform to identity covariance |

---

<- [Back](./README.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
