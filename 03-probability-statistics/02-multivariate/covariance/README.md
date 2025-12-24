<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Covariance&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Covariance

> **Measuring linear relationships between variables**

---

## 🎯 Visual Overview

<img src="./images/covariance.svg" width="100%">

*Caption: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] measures linear dependence. Positive: variables increase together. Negative: inverse relationship. Zero: no linear relationship (but possible non-linear). Correlation normalizes by std devs.*

---

## 📂 Overview

Covariance matrices are central to ML: they describe feature relationships, enable PCA, and parameterize Gaussian distributions. Understanding covariance is essential for feature analysis.

---

## 📐 Mathematical Definitions

### Covariance
```
Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)]
          = E[XY] - E[X]E[Y]

Properties:
• Cov(X, X) = Var(X)
• Cov(X, Y) = Cov(Y, X)
• Cov(aX, bY) = ab · Cov(X, Y)
• Cov(X + Y, Z) = Cov(X, Z) + Cov(Y, Z)
```

### Correlation
```
ρ(X, Y) = Cov(X, Y) / (σₓ · σᵧ)

Properties:
• -1 ≤ ρ ≤ 1
• ρ = ±1 ⟺ perfect linear relationship
• ρ = 0 ⟺ no linear relationship
```

### Covariance Matrix
```
For random vector X = [X₁, ..., Xₙ]ᵀ:

Σ = E[(X - μ)(X - μ)ᵀ]

Σᵢⱼ = Cov(Xᵢ, Xⱼ)

Properties:
• Σ is symmetric: Σ = Σᵀ
• Σ is positive semi-definite: xᵀΣx ≥ 0
• Diagonal = variances, off-diagonal = covariances
```

---

## 💻 Code Examples

```python
import numpy as np
import torch

# Compute covariance matrix
X = np.random.randn(1000, 3)  # 1000 samples, 3 features
cov_matrix = np.cov(X.T)      # 3x3 covariance matrix

# Correlation matrix
corr_matrix = np.corrcoef(X.T)

# Sample from multivariate Gaussian
mean = np.zeros(3)
cov = np.array([[1, 0.8, 0], [0.8, 1, 0.3], [0, 0.3, 1]])
samples = np.random.multivariate_normal(mean, cov, 1000)

# PyTorch multivariate normal
dist = torch.distributions.MultivariateNormal(
    loc=torch.zeros(3),
    covariance_matrix=torch.eye(3)
)
samples = dist.sample((100,))

# PCA uses covariance matrix
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)  # Projects onto principal components
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 2 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🎥 | 3Blue1Brown: Covariance | [YouTube](https://www.youtube.com/watch?v=PFDu9oVAE-g) |
| 📖 | NumPy cov | [Docs](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) |
| 🇨🇳 | 协方差矩阵详解 | [知乎](https://zhuanlan.zhihu.com/p/37609917) |
| 🇨🇳 | PCA与协方差 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 统计学基础 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: covariance](../)

---

➡️ [Next: Exponential Family](../exponential-family/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
