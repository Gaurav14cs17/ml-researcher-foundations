<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Covariance%20%26%20Covariance%20Matrix&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/covariance.svg" width="100%">

*Caption: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] measures linear dependence. Positive: variables increase together. Negative: inverse relationship. Zero: no linear relationship (but possible non-linear). Correlation normalizes by std devs.*

---

## 📂 Overview

Covariance matrices are central to ML: they describe feature relationships, enable PCA, and parameterize Gaussian distributions. Understanding covariance is essential for feature analysis.

---

## 📐 Covariance: Definition and Properties

### Definition

```math
\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]

```

**Proof of Alternative Form:**

```math
\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]
= E[XY - X\mu_Y - Y\mu_X + \mu_X\mu_Y]
= E[XY] - \mu_Y E[X] - \mu_X E[Y] + \mu_X\mu_Y
= E[XY] - \mu_X\mu_Y - \mu_X\mu_Y + \mu_X\mu_Y
= E[XY] - E[X]E[Y] \quad \blacksquare

```

### Properties

**1. Variance is a Special Case:**

```math
\text{Cov}(X, X) = \text{Var}(X)

```

**2. Symmetry:**

```math
\text{Cov}(X, Y) = \text{Cov}(Y, X)

```

**3. Bilinearity:**

```math
\text{Cov}(aX, bY) = ab \cdot \text{Cov}(X, Y)
\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)

```

**Proof of Bilinearity (Scaling):**

```math
\text{Cov}(aX, bY) = E[(aX - aE[X])(bY - bE[Y])]
= E[ab(X - E[X])(Y - E[Y])]
= ab \cdot E[(X - E[X])(Y - E[Y])]
= ab \cdot \text{Cov}(X, Y) \quad \blacksquare

```

**4. Independence Implies Zero Covariance:**

If X and Y are independent, then $\text{Cov}(X, Y) = 0$.

**Proof:**

If X, Y independent: $E[XY] = E[X]E[Y]$

```math
\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = E[X]E[Y] - E[X]E[Y] = 0 \quad \blacksquare

```

**Warning:** The converse is FALSE! $\text{Cov}(X, Y) = 0$ does NOT imply independence.

**Counter-example:** Let $X \sim \mathcal{N}(0, 1)$ and $Y = X^2$.
- $\text{Cov}(X, Y) = E[X \cdot X^2] - E[X]E[X^2] = E[X^3] - 0 \cdot 1 = 0$ (odd moments of symmetric dist = 0)
- But X and Y are clearly NOT independent!

---

## 📐 Correlation

### Pearson Correlation Coefficient

```math
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}

```

### Properties

**1. Bounded:**

```math
-1 \leq \rho(X, Y) \leq 1

```

**Proof (Cauchy-Schwarz):**

Define $\tilde{X} = X - \mu\_X$ and $\tilde{Y} = Y - \mu\_Y$.

By Cauchy-Schwarz inequality:

```math
|E[\tilde{X}\tilde{Y}]|^2 \leq E[\tilde{X}^2] \cdot E[\tilde{Y}^2]
|\text{Cov}(X, Y)|^2 \leq \text{Var}(X) \cdot \text{Var}(Y) = \sigma_X^2 \sigma_Y^2
|\text{Cov}(X, Y)| \leq \sigma_X \sigma_Y
-1 \leq \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \leq 1 \quad \blacksquare

```

**2. Perfect Correlation:**

```math
|\rho| = 1 \iff Y = aX + b \text{ for some constants } a, b

```

**3. Uncorrelated:**

```math
\rho = 0 \iff \text{Cov}(X, Y) = 0

```

---

## 📐 Covariance Matrix

### Definition

For a random vector $\mathbf{X} = [X\_1, X\_2, \ldots, X\_n]^\top$:

```math
\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}) = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top]

```

**Element-wise:**

```math
\Sigma_{ij} = \text{Cov}(X_i, X_j)

```

**Matrix Form:**

```math
\boldsymbol{\Sigma} = E[\mathbf{X}\mathbf{X}^\top] - \boldsymbol{\mu}\boldsymbol{\mu}^\top

```

### Structure

```math
\boldsymbol{\Sigma} = \begin{bmatrix} 
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \cdots & \text{Var}(X_n)
\end{bmatrix}

```

- **Diagonal entries:** Variances
- **Off-diagonal entries:** Covariances

### Properties

**1. Symmetric:** $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^\top$

**Proof:** $\Sigma\_{ij} = \text{Cov}(X\_i, X\_j) = \text{Cov}(X\_j, X\_i) = \Sigma\_{ji} \quad \blacksquare$

**2. Positive Semi-definite:** $\boldsymbol{\Sigma} \succeq 0$, i.e., $\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} \geq 0$ for all $\mathbf{v}$

**Proof:**

```math
\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = \mathbf{v}^\top E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top] \mathbf{v}
= E[\mathbf{v}^\top (\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top \mathbf{v}]
= E[(\mathbf{v}^\top (\mathbf{X} - \boldsymbol{\mu}))^2]
= E[Z^2] \geq 0 \quad \blacksquare

```

where $Z = \mathbf{v}^\top (\mathbf{X} - \boldsymbol{\mu})$ is a scalar random variable.

**3. Eigenvalue Decomposition:**

Since $\boldsymbol{\Sigma}$ is symmetric positive semi-definite:

```math
\boldsymbol{\Sigma} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top

```

where:
- $\mathbf{U}$: orthogonal matrix of eigenvectors
- $\boldsymbol{\Lambda}$: diagonal matrix of non-negative eigenvalues

---

## 📐 Sample Covariance Matrix

### Biased Estimator

Given n samples $\mathbf{X}\_1, \ldots, \mathbf{X}\_n$:

```math
\hat{\boldsymbol{\Sigma}} = \frac{1}{n} \sum_{i=1}^{n} (\mathbf{X}_i - \bar{\mathbf{X}})(\mathbf{X}_i - \bar{\mathbf{X}})^\top

```

### Unbiased Estimator (Bessel's Correction)

```math
\hat{\boldsymbol{\Sigma}}_{\text{unbiased}} = \frac{1}{n-1} \sum_{i=1}^{n} (\mathbf{X}_i - \bar{\mathbf{X}})(\mathbf{X}_i - \bar{\mathbf{X}})^\top

```

**Why n-1?** We lose one degree of freedom by estimating the mean from the same data.

**Proof that n-1 gives unbiased estimator:**

```math
E\left[\frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2\right] = \sigma^2

```

The key insight is that:

```math
\sum_{i=1}^{n}(X_i - \bar{X})^2 = \sum_{i=1}^{n}(X_i - \mu)^2 - n(\bar{X} - \mu)^2

```

Taking expectations and using $E[(\bar{X} - \mu)^2] = \sigma^2/n$:

```math
E\left[\sum_{i=1}^{n}(X_i - \bar{X})^2\right] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2 \quad \blacksquare

```

---

## 📐 Linear Transformations

**Theorem:** If $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$, then:

```math
\text{Cov}(\mathbf{Y}) = \mathbf{A} \text{Cov}(\mathbf{X}) \mathbf{A}^\top = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top

```

**Proof:**

```math
\text{Cov}(\mathbf{Y}) = E[(\mathbf{Y} - E[\mathbf{Y}])(\mathbf{Y} - E[\mathbf{Y}])^\top]

```

Since $E[\mathbf{Y}] = \mathbf{A}E[\mathbf{X}] + \mathbf{b} = \mathbf{A}\boldsymbol{\mu} + \mathbf{b}$:

```math
\mathbf{Y} - E[\mathbf{Y}] = \mathbf{A}\mathbf{X} + \mathbf{b} - \mathbf{A}\boldsymbol{\mu} - \mathbf{b} = \mathbf{A}(\mathbf{X} - \boldsymbol{\mu})

```

Therefore:

```math
\text{Cov}(\mathbf{Y}) = E[\mathbf{A}(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top\mathbf{A}^\top]
= \mathbf{A} E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top] \mathbf{A}^\top
= \mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^\top \quad \blacksquare

```

---

## 📐 Whitening Transformation

**Goal:** Transform data so that covariance = identity matrix

**Method:** 

```math
\mathbf{Z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{X} - \boldsymbol{\mu})

```

**Result:**

```math
\text{Cov}(\mathbf{Z}) = \boldsymbol{\Sigma}^{-1/2} \boldsymbol{\Sigma} (\boldsymbol{\Sigma}^{-1/2})^\top = \mathbf{I}

```

**Via Eigendecomposition:**

```math
\boldsymbol{\Sigma} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top
\boldsymbol{\Sigma}^{-1/2} = \mathbf{U}\boldsymbol{\Lambda}^{-1/2}\mathbf{U}^\top

```

---

## 📐 Mahalanobis Distance

**Definition:**

```math
d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}

```

**Properties:**
- Accounts for correlations between features
- Scale-invariant
- For spherical Gaussian: reduces to Euclidean distance
- Used in outlier detection

---

## 📐 Connection to PCA

**Principal Component Analysis:**

1. Center data: $\tilde{\mathbf{X}}\_i = \mathbf{X}\_i - \bar{\mathbf{X}}$
2. Compute sample covariance: $\hat{\boldsymbol{\Sigma}} = \frac{1}{n-1}\sum\_i \tilde{\mathbf{X}}\_i \tilde{\mathbf{X}}\_i^\top$
3. Eigendecomposition: $\hat{\boldsymbol{\Sigma}} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$
4. Principal components: columns of $\mathbf{U}$ (eigenvectors)
5. Explained variance: eigenvalues $\lambda\_i$

**Projection onto k principal components:**

```math
\mathbf{Z} = \mathbf{U}_k^\top (\mathbf{X} - \bar{\mathbf{X}})

```

where $\mathbf{U}\_k$ contains the top k eigenvectors.

---

## 💻 Code Examples

```python
import numpy as np
import torch

# Generate correlated data
n_samples = 1000
mean = np.array([0, 0, 0])
true_cov = np.array([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.5],
    [0.3, 0.5, 1.0]
])
X = np.random.multivariate_normal(mean, true_cov, n_samples)

# Compute sample covariance matrix
cov_biased = np.cov(X.T, bias=True)    # Divide by n
cov_unbiased = np.cov(X.T, bias=False)  # Divide by n-1 (default)
print("Sample covariance shape:", cov_unbiased.shape)  # (3, 3)

# Correlation matrix
corr = np.corrcoef(X.T)
print("Correlation matrix:\n", corr)

# Manual covariance computation
X_centered = X - X.mean(axis=0)
cov_manual = X_centered.T @ X_centered / (n_samples - 1)
print("Manual vs NumPy match:", np.allclose(cov_manual, cov_unbiased))

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_unbiased)
print("Eigenvalues:", eigenvalues)

# Whitening transformation
L = np.linalg.cholesky(cov_unbiased)
X_whitened = np.linalg.solve(L, X_centered.T).T
print("Whitened covariance:\n", np.cov(X_whitened.T))  # ≈ Identity

# Mahalanobis distance
def mahalanobis(x, mu, Sigma):
    """Compute Mahalanobis distance"""
    diff = x - mu
    Sigma_inv = np.linalg.inv(Sigma)
    return np.sqrt(diff @ Sigma_inv @ diff)

# PCA using covariance
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# PyTorch multivariate normal
mu_torch = torch.zeros(3)
cov_torch = torch.tensor(true_cov, dtype=torch.float32)
dist = torch.distributions.MultivariateNormal(mu_torch, cov_torch)
samples = dist.sample((100,))
log_prob = dist.log_prob(samples)

# Covariance in batch normalization context
def batch_stats(x):
    """Compute batch statistics for BatchNorm"""
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)  # Uses biased estimator
    return batch_mean, batch_var

```

---

## 🌍 ML Applications

| Application | How Covariance is Used |
|-------------|------------------------|
| **PCA** | Eigendecomposition of covariance matrix |
| **Gaussian Models** | Parameterize multivariate normal |
| **Mahalanobis Distance** | $d² = (\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ |
| **Whitening** | Transform to identity covariance |
| **Gaussian Processes** | Kernel defines covariance function |
| **Batch Normalization** | Uses running variance estimates |
| **Kalman Filter** | Covariance propagation |
| **VAE** | Diagonal covariance in latent space |

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

⬅️ [Back: Multivariate](../) | ➡️ [Next: Exponential Family](../02_exponential_family/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
