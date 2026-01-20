<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Multivariate%20Statistics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/covariance-matrix.svg" width="100%">

*Caption: The covariance matrix Σ captures relationships between variables. Diagonal = variances, off-diagonal = covariances. Central to PCA, VAE, GP, and many other ML methods.*

---

## 📂 Topics in This Folder

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [01_covariance/](./01_covariance/) | Covariance & Correlation | $\Sigma$, $\rho$, PSD |
| [02_exponential_family/](./02_exponential_family/) | Exponential Family | GLMs, sufficient statistics |
| [03_gaussian/](./03_gaussian/) | 🔥 Multivariate Gaussian | Most important! |
| [04_random_vectors/](./04_random_vectors/) | Random Vectors | Joint, marginal, conditional |

---

## 📐 Joint, Marginal, and Conditional Distributions

### Joint Distribution

```math
p(\mathbf{x}, \mathbf{y})
```

Contains all information about the relationship between $\mathbf{x}$ and $\mathbf{y}$.

### Marginal Distribution

```math
p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{y}) \, d\mathbf{y}
```

"Integrate out" other variables to get distribution of $\mathbf{x}$ alone.

### Conditional Distribution

```math
p(\mathbf{x}|\mathbf{y}) = \frac{p(\mathbf{x}, \mathbf{y})}{p(\mathbf{y})}
```

Distribution of $\mathbf{x}$ given that we know $\mathbf{y}$.

### Independence

```math
\mathbf{X} \perp \mathbf{Y} \iff p(\mathbf{x}, \mathbf{y}) = p(\mathbf{x}) \cdot p(\mathbf{y})
```

---

## 📐 Covariance Matrix

### Definition

```math
\boldsymbol{\Sigma} = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top]
\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]
```

### Properties

**1. Symmetric:** $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^\top$

**2. Positive Semi-definite:** $\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} \geq 0$ for all $\mathbf{v}$

**Proof of PSD:**

```math
\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E[(\mathbf{v}^\top(\mathbf{X} - \boldsymbol{\mu}))^2] = E[Z^2] \geq 0 \quad \blacksquare
```

**3. Linear Transformation:**

```math
\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b} \implies \text{Cov}(\mathbf{Y}) = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top
```

### Correlation Matrix

```math
\mathbf{R}_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}} = \frac{\text{Cov}(X_i, X_j)}{\sigma_i \sigma_j}
```

- Diagonal entries = 1
- Off-diagonal $\in [-1, 1]$

---

## 📐 Multivariate Gaussian

```math
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
```

### PDF

```math
p(\mathbf{x}) = (2\pi)^{-d/2} |\boldsymbol{\Sigma}|^{-1/2} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
```

### Key Properties

**Marginal:** Any subset is also Gaussian

```math
\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11})
```

**Conditional:** Conditioning preserves Gaussianity

```math
\mathbf{x}_1 | \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})
```

where:

```math
\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)
\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}
```

**Linear Transform:**

```math
\mathbf{A}\mathbf{X} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)
```

---

## 📐 KL Divergence for Gaussians

```math
D_{KL}(\mathcal{N}_1 \| \mathcal{N}_2) = \frac{1}{2}\left[\text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1) - d + \log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}\right]
```

**VAE Loss (KL to standard normal):**

```math
D_{KL}(\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2)) \| \mathcal{N}(\mathbf{0}, \mathbf{I})) = \frac{1}{2}\sum_{i=1}^{d}\left(\mu_i^2 + \sigma_i^2 - 1 - \log\sigma_i^2\right)
```

---

## 💻 Code Examples

```python
import numpy as np
import torch
from scipy import stats

# Create multivariate Gaussian
mean = np.array([0, 0])
cov = np.array([[1, 0.8], [0.8, 1]])  # Correlated
samples = np.random.multivariate_normal(mean, cov, 1000)

# Compute sample covariance
sample_cov = np.cov(samples.T)
print(f"Sample covariance:\n{sample_cov}")

# Correlation matrix
corr = np.corrcoef(samples.T)
print(f"Correlation: {corr[0,1]:.3f}")

# PyTorch multivariate normal
dist = torch.distributions.MultivariateNormal(
    loc=torch.zeros(2),
    covariance_matrix=torch.tensor([[1., 0.8], [0.8, 1.]])
)
samples = dist.sample((1000,))
log_prob = dist.log_prob(samples)

# Conditional Gaussian
def conditional_gaussian_2d(y_value, mu, Sigma):
    """p(X | Y = y_value)"""
    mu_cond = mu[0] + Sigma[0,1]/Sigma[1,1] * (y_value - mu[1])
    var_cond = Sigma[0,0] - Sigma[0,1]**2/Sigma[1,1]
    return mu_cond, var_cond

mu_cond, var_cond = conditional_gaussian_2d(1.0, mean, cov)
print(f"p(X|Y=1): mean={mu_cond:.3f}, var={var_cond:.3f}")

# VAE KL divergence
def kl_to_standard_normal(mu, log_var):
    """D_KL(N(mu, sigma^2) || N(0, I))"""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

---

## 🔗 Where Multivariate Statistics Is Used

| Application | How It's Used |
|-------------|---------------|
| **PCA** | Eigendecomposition of covariance matrix |
| **VAE** | Latent space is multivariate Gaussian |
| **Diffusion Models** | Forward process adds Gaussian noise |
| **Gaussian Processes** | Multivariate Gaussian over function values |
| **Bayesian Neural Networks** | Posterior over weights |
| **Kalman Filter** | Multivariate Gaussian state estimation |
| **GMM** | Mixture of multivariate Gaussians |
| **Normalizing Flows** | Transform simple → complex distributions |
| **ELBO** | KL between multivariate distributions |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Bishop PRML Ch. 2 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | Murphy MLaPP | [Book](https://probml.github.io/pml-book/book1.html) |
| 🎥 | 3Blue1Brown | [YouTube](https://www.youtube.com/watch?v=IaSGqQa5O-M) |
| 🇨🇳 | 多元统计分析 | [知乎](https://zhuanlan.zhihu.com/p/37609917) |
| 🇨🇳 | 协方差矩阵详解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 多元高斯分布 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

⬅️ [Back: Probability](../01_probability/) | ➡️ [Next: Information Theory](../03_information_theory/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
