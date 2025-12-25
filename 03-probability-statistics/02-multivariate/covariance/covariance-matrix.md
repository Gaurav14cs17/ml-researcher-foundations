<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Covariance%20Matrix&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
