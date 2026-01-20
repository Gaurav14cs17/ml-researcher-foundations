<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Multivariate%20Gaussian&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/multivariate-gaussian.svg" width="100%">

*Caption: A 2D Gaussian shows elliptical contours of constant probability. The mean μ determines the center, while the covariance matrix Σ controls the shape and orientation. This distribution is foundational for VAEs, GPs, and diffusion models.*

---

## 📐 Definition

$$
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

**Probability Density Function:**

$$
p(\mathbf{x}) = (2\pi)^{-d/2} |\boldsymbol{\Sigma}|^{-1/2} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

**Parameters:**
- $\boldsymbol{\mu} \in \mathbb{R}^d$: mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$: covariance matrix (symmetric, positive definite)
- $|\boldsymbol{\Sigma}|$: determinant of covariance
- $\boldsymbol{\Sigma}^{-1}$: precision matrix

---

## 📐 Key Properties

| Property | Formula |
|----------|---------|
| Mean | $E[\mathbf{X}] = \boldsymbol{\mu}$ |
| Covariance | $\text{Cov}(\mathbf{X}) = \boldsymbol{\Sigma}$ |
| Entropy | $H(\mathbf{X}) = \frac{d}{2}\log(2\pi e) + \frac{1}{2}\log|\boldsymbol{\Sigma}|$ |
| Mode | $\boldsymbol{\mu}$ (same as mean) |

---

## 📐 Marginal Distribution

**Theorem:** Any subset of variables from a joint Gaussian is also Gaussian.

**Setup:**

$$
\begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{bmatrix}\right)
$$

**Marginal:**

$$
\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11})
$$

**Proof:**

The marginal is obtained by integrating out $\mathbf{x}\_2$:

$$
p(\mathbf{x}_1) = \int p(\mathbf{x}_1, \mathbf{x}_2) \, d\mathbf{x}_2
$$

Using the partitioned form of the Gaussian and completing the square in $\mathbf{x}\_2$, the integral over $\mathbf{x}\_2$ yields a normalizing constant, leaving:

$$
p(\mathbf{x}_1) = (2\pi)^{-d_1/2} |\boldsymbol{\Sigma}_{11}|^{-1/2} \exp\left(-\frac{1}{2}(\mathbf{x}_1-\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_{11}^{-1}(\mathbf{x}_1-\boldsymbol{\mu}_1)\right)
$$

This is $\mathcal{N}(\boldsymbol{\mu}\_1, \boldsymbol{\Sigma}\_{11})$. $\blacksquare$

---

## 📐 Conditional Distribution (Full Derivation)

**Theorem:** The conditional distribution of a Gaussian is also Gaussian.

$$
\mathbf{x}_1 | \mathbf{x}_2 = \mathbf{a} \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})
$$

**Conditional Mean:**

$$
\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{a} - \boldsymbol{\mu}_2)
$$

**Conditional Covariance:**

$$
\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}
$$

### Complete Proof

**Step 1: Write the joint density**

$$
p(\mathbf{x}_1, \mathbf{x}_2) \propto \exp\left(-\frac{1}{2}Q(\mathbf{x}_1, \mathbf{x}_2)\right)
$$

where the quadratic form is:

$$
Q = \begin{bmatrix} \mathbf{x}_1 - \boldsymbol{\mu}_1 \\ \mathbf{x}_2 - \boldsymbol{\mu}_2 \end{bmatrix}^\top \begin{bmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{bmatrix}^{-1} \begin{bmatrix} \mathbf{x}_1 - \boldsymbol{\mu}_1 \\ \mathbf{x}_2 - \boldsymbol{\mu}_2 \end{bmatrix}
$$

**Step 2: Block matrix inversion**

Using the block matrix inversion formula:

$$
\boldsymbol{\Sigma}^{-1} = \begin{bmatrix} \boldsymbol{\Lambda}_{11} & \boldsymbol{\Lambda}_{12} \\ \boldsymbol{\Lambda}_{21} & \boldsymbol{\Lambda}_{22} \end{bmatrix}
$$

where:

$$
\boldsymbol{\Lambda}_{11} = (\boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21})^{-1} = \boldsymbol{\Sigma}_{1|2}^{-1}
$$

**Step 3: Expand the quadratic form**

$$
Q = (\mathbf{x}_1 - \boldsymbol{\mu}_1)^\top \boldsymbol{\Lambda}_{11} (\mathbf{x}_1 - \boldsymbol{\mu}_1) + 2(\mathbf{x}_1 - \boldsymbol{\mu}_1)^\top \boldsymbol{\Lambda}_{12} (\mathbf{x}_2 - \boldsymbol{\mu}_2) + (\mathbf{x}_2 - \boldsymbol{\mu}_2)^\top \boldsymbol{\Lambda}_{22} (\mathbf{x}_2 - \boldsymbol{\mu}_2)
$$

**Step 4: Complete the square in $\mathbf{x}\_1$**

Fixing $\mathbf{x}\_2 = \mathbf{a}$ and completing the square:

$$
Q = (\mathbf{x}_1 - \boldsymbol{\mu}_{1|2})^\top \boldsymbol{\Lambda}_{11} (\mathbf{x}_1 - \boldsymbol{\mu}_{1|2}) + \text{terms not involving } \mathbf{x}_1
$$

where:

$$
\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 - \boldsymbol{\Lambda}_{11}^{-1}\boldsymbol{\Lambda}_{12}(\mathbf{a} - \boldsymbol{\mu}_2)
$$

**Step 5: Use the identity $\boldsymbol{\Lambda}\_{12} = -\boldsymbol{\Sigma}\_{1|2}^{-1}\boldsymbol{\Sigma}\_{12}\boldsymbol{\Sigma}\_{22}^{-1}$**

$$
\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{a} - \boldsymbol{\mu}_2) \quad \blacksquare
$$

### Key Insights

1. **Conditional mean is linear in the condition** - This is the regression formula!
2. **Conditional covariance is independent of the observed value** - It only depends on which variables are observed, not their values
3. **Schur complement** appears naturally: $\boldsymbol{\Sigma}\_{1|2} = \boldsymbol{\Sigma}\_{11} - \boldsymbol{\Sigma}\_{12}\boldsymbol{\Sigma}\_{22}^{-1}\boldsymbol{\Sigma}\_{21}$

---

## 📐 Linear Transformation

**Theorem:** Linear transformations preserve Gaussianity.

If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then:

$$
\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)
$$

**Proof using Characteristic Functions:**

The characteristic function of $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ is:

$$
\phi_{\mathbf{X}}(\mathbf{t}) = \exp\left(i\mathbf{t}^\top\boldsymbol{\mu} - \frac{1}{2}\mathbf{t}^\top\boldsymbol{\Sigma}\mathbf{t}\right)
$$

For $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$:

$$
\phi_{\mathbf{Y}}(\mathbf{t}) = E[e^{i\mathbf{t}^\top\mathbf{Y}}] = E[e^{i\mathbf{t}^\top(\mathbf{A}\mathbf{X}+\mathbf{b})}]
= e^{i\mathbf{t}^\top\mathbf{b}} E[e^{i(\mathbf{A}^\top\mathbf{t})^\top\mathbf{X}}]
= e^{i\mathbf{t}^\top\mathbf{b}} \phi_{\mathbf{X}}(\mathbf{A}^\top\mathbf{t})
= e^{i\mathbf{t}^\top\mathbf{b}} \exp\left(i(\mathbf{A}^\top\mathbf{t})^\top\boldsymbol{\mu} - \frac{1}{2}(\mathbf{A}^\top\mathbf{t})^\top\boldsymbol{\Sigma}(\mathbf{A}^\top\mathbf{t})\right)
= \exp\left(i\mathbf{t}^\top(\mathbf{A}\boldsymbol{\mu}+\mathbf{b}) - \frac{1}{2}\mathbf{t}^\top\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top\mathbf{t}\right)
$$

This is the CF of $\mathcal{N}(\mathbf{A}\boldsymbol{\mu}+\mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$. $\blacksquare$

---

## 📐 Sum of Independent Gaussians

**Theorem:** If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}\_x, \boldsymbol{\Sigma}\_x)$ and $\mathbf{Y} \sim \mathcal{N}(\boldsymbol{\mu}\_y, \boldsymbol{\Sigma}\_y)$ are independent:

$$
\mathbf{X} + \mathbf{Y} \sim \mathcal{N}(\boldsymbol{\mu}_x + \boldsymbol{\mu}_y, \boldsymbol{\Sigma}_x + \boldsymbol{\Sigma}_y)
$$

**Proof:** Follows from linear transformation with $[\mathbf{X}; \mathbf{Y}]$ and $\mathbf{A} = [\mathbf{I}, \mathbf{I}]$.

---

## 📐 Product of Gaussian PDFs

**Theorem:** The product of two Gaussian PDFs is proportional to a Gaussian:

$$
p_1(\mathbf{x}) \cdot p_2(\mathbf{x}) \propto \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

where:

$$
\boldsymbol{\Sigma} = (\boldsymbol{\Sigma}_1^{-1} + \boldsymbol{\Sigma}_2^{-1})^{-1}
\boldsymbol{\mu} = \boldsymbol{\Sigma}(\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_2^{-1}\boldsymbol{\mu}_2)
$$

**Applications:** Kalman filtering, sensor fusion, Bayesian inference

---

## 📐 KL Divergence Between Gaussians

**Theorem:**

$$
D_{KL}(\mathcal{N}(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_1) \| \mathcal{N}(\boldsymbol{\mu}_2,\boldsymbol{\Sigma}_2)) = \frac{1}{2}\left[\text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1) - d + \log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}\right]
$$

**Special Case (VAE Loss - KL to Standard Normal):**

$$
D_{KL}(\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2)) \| \mathcal{N}(\mathbf{0}, \mathbf{I})) = \frac{1}{2}\sum_{i=1}^{d}\left(\mu_i^2 + \sigma_i^2 - 1 - \log\sigma_i^2\right)
$$

**Derivation:**

$$
D_{KL}(q \| p) = E_q[\log q - \log p]
$$

For $q = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $p = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$
= E_q\left[-\frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) + \frac{1}{2}\mathbf{x}^\top\mathbf{x} + \frac{d}{2}\log(2\pi) - \frac{d}{2}\log(2\pi)\right]
= -\frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{d}{2} + \frac{1}{2}E_q[\mathbf{x}^\top\mathbf{x}]
= -\frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{d}{2} + \frac{1}{2}(\text{tr}(\boldsymbol{\Sigma}) + \boldsymbol{\mu}^\top\boldsymbol{\mu})
$$

For diagonal $\boldsymbol{\Sigma} = \text{diag}(\sigma\_1^2, \ldots, \sigma\_d^2)$:

$$
= \frac{1}{2}\sum_{i=1}^{d}\left(\mu_i^2 + \sigma_i^2 - 1 - \log\sigma_i^2\right) \quad \blacksquare
$$

---

## 📐 Sampling Methods

### Cholesky Decomposition

To sample $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

1. Compute Cholesky: $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ (L lower triangular)
2. Sample $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
3. Set $\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$

**Why it works:**

$$
E[\mathbf{X}] = \boldsymbol{\mu} + \mathbf{L}E[\mathbf{Z}] = \boldsymbol{\mu}
\text{Cov}[\mathbf{X}] = \mathbf{L}\text{Cov}[\mathbf{Z}]\mathbf{L}^\top = \mathbf{L}\mathbf{I}\mathbf{L}^\top = \mathbf{L}\mathbf{L}^\top = \boldsymbol{\Sigma} \quad \blacksquare
$$

### Box-Muller Transform (Univariate)

Generate $X \sim \mathcal{N}(0,1)$ from uniform $U\_1, U\_2 \sim \text{Unif}(0,1)$:

$$
X = \sqrt{-2\ln U_1} \cos(2\pi U_2)
Y = \sqrt{-2\ln U_1} \sin(2\pi U_2)
$$

Both X and Y are independent $\mathcal{N}(0,1)$.

---

## 📐 Gaussian Processes Preview

A **Gaussian Process** is a distribution over functions where any finite collection of function values has a joint Gaussian distribution.

$$
f \sim \mathcal{GP}(m, k)
$$

For any set of points $\{x\_1, \ldots, x\_n\}$:

$$
[f(x_1), \ldots, f(x_n)]^\top \sim \mathcal{N}(\mathbf{m}, \mathbf{K})
$$

where $m\_i = m(x\_i)$ and $K\_{ij} = k(x\_i, x\_j)$.

**GP Regression uses the conditional Gaussian formula!**

---

## 💻 Code Examples

```python
import torch
import numpy as np
from torch.distributions import MultivariateNormal

# Create multivariate Gaussian
d = 3
mu = torch.zeros(d)
Sigma = torch.tensor([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.5],
    [0.3, 0.5, 1.0]
])

dist = MultivariateNormal(mu, Sigma)

# Sample
samples = dist.sample((1000,))
print(f"Sample mean: {samples.mean(0)}")
print(f"Sample cov:\n{torch.cov(samples.T)}")

# Log probability (for loss computation)
log_prob = dist.log_prob(samples)

# Conditioning (manual implementation)
def conditional_gaussian(mu, Sigma, obs_indices, obs_values, query_indices):
    """
    Compute p(x_query | x_obs = obs_values) for joint Gaussian
    """
    mu_1 = mu[query_indices]
    mu_2 = mu[obs_indices]
    
    Sigma_11 = Sigma[query_indices][:, query_indices]
    Sigma_12 = Sigma[query_indices][:, obs_indices]
    Sigma_22 = Sigma[obs_indices][:, obs_indices]
    
    # Conditional mean
    Sigma_22_inv = torch.linalg.inv(Sigma_22)
    mu_cond = mu_1 + Sigma_12 @ Sigma_22_inv @ (obs_values - mu_2)
    
    # Conditional covariance
    Sigma_cond = Sigma_11 - Sigma_12 @ Sigma_22_inv @ Sigma_12.T
    
    return mu_cond, Sigma_cond

# Example: Condition on x[2] = 1.0
mu_cond, Sigma_cond = conditional_gaussian(
    mu, Sigma, 
    obs_indices=[2], 
    obs_values=torch.tensor([1.0]), 
    query_indices=[0, 1]
)
print(f"p(x[0,1] | x[2]=1.0) ~ N({mu_cond}, \n{Sigma_cond})")

# Cholesky sampling (numerically stable)
L = torch.linalg.cholesky(Sigma)
z = torch.randn(1000, d)
samples_chol = mu + z @ L.T  # Equivalent to dist.sample()

# VAE KL divergence (diagonal covariance)
def kl_divergence_vae(mu, log_var):
    """D_KL(N(mu, diag(exp(log_var))) || N(0, I))"""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

# Reparameterization trick
def reparameterize(mu, log_var):
    """Sample z ~ N(mu, sigma^2) with gradient flow"""
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

# Gaussian Process prediction
def gp_predict(X_train, y_train, X_test, kernel, noise_var=1e-6):
    """
    GP regression using conditional Gaussian formula
    """
    K_train = kernel(X_train, X_train) + noise_var * torch.eye(len(X_train))
    K_test_train = kernel(X_test, X_train)
    K_test = kernel(X_test, X_test)
    
    K_train_inv = torch.linalg.inv(K_train)
    
    # Conditional mean (prediction)
    mu_pred = K_test_train @ K_train_inv @ y_train
    
    # Conditional covariance (uncertainty)
    Sigma_pred = K_test - K_test_train @ K_train_inv @ K_test_train.T
    
    return mu_pred, Sigma_pred
```

---

## 🌍 ML Applications

| Application | How Multivariate Gaussian is Used |
|-------------|-----------------------------------|
| **VAE** | Latent space prior and posterior |
| **Gaussian Processes** | Prior over functions |
| **Kalman Filter** | State estimation with Gaussian noise |
| **Diffusion Models** | Noise distribution at each step |
| **Bayesian Linear Regression** | Posterior over weights |
| **PCA** | Assumes Gaussian data (implicitly) |
| **GMM** | Mixture of multivariate Gaussians |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Bishop PRML Ch. 2 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | Rasmussen GP Book | [GPML](http://www.gaussianprocess.org/gpml/) |
| 🎥 | 3Blue1Brown | [YouTube](https://www.youtube.com/watch?v=IaSGqQa5O-M) |
| 🇨🇳 | 多元高斯分布 | [知乎](https://zhuanlan.zhihu.com/p/37609917) |
| 🇨🇳 | 高斯过程 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

⬅️ [Back: Exponential Family](../02_exponential_family/) | ➡️ [Next: Random Vectors](../04_random_vectors/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
