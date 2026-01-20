<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Random%20Vectors&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/random-vectors.svg" width="100%">

*Caption: A random vector X = [X₁,...,Xₙ]ᵀ groups multiple RVs. The joint distribution captures all dependencies. Marginals are obtained by integrating out variables.*

---

## 📂 Overview

Random vectors are essential for multivariate ML: each data point is a random vector. Understanding joint, marginal, and conditional distributions is key to probabilistic modeling.

---

## 📐 Definition

### Random Vector

```math
\mathbf{X} = [X_1, X_2, \ldots, X_n]^\top
```

Each $X\_i$ is a random variable. The vector is a measurable function:

```math
\mathbf{X}: \Omega \to \mathbb{R}^n
```

---

## 📐 Joint Distribution

### Joint CDF

```math
F_{\mathbf{X}}(x_1, \ldots, x_n) = P(X_1 \leq x_1, \ldots, X_n \leq x_n)
```

### Joint PMF (Discrete)

```math
p_{\mathbf{X}}(x_1, \ldots, x_n) = P(X_1 = x_1, \ldots, X_n = x_n)
```

### Joint PDF (Continuous)

```math
f_{\mathbf{X}}(x_1, \ldots, x_n) \geq 0
\int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} f_{\mathbf{X}}(x_1, \ldots, x_n) \, dx_1 \cdots dx_n = 1
```

---

## 📐 Marginal Distribution

### Definition

The marginal distribution of $X\_i$ is obtained by "integrating out" other variables.

**Discrete:**

```math
P(X_i = x) = \sum_{x_{-i}} P(X_1 = x_1, \ldots, X_n = x_n)
```

**Continuous:**

```math
f_{X_i}(x) = \int \cdots \int f_{\mathbf{X}}(x_1, \ldots, x_n) \, dx_{-i}
```

where $x\_{-i}$ denotes all variables except $x\_i$.

### Example: Bivariate Normal

```math
f_{X_1}(x) = \int_{-\infty}^{\infty} f_{X_1, X_2}(x, y) \, dy
```

For bivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

```math
X_1 \sim \mathcal{N}(\mu_1, \Sigma_{11})
```

---

## 📐 Conditional Distribution

### Definition

```math
p(X_1 = x_1 | X_2 = x_2) = \frac{p(X_1 = x_1, X_2 = x_2)}{p(X_2 = x_2)}
```

**Continuous:**

```math
f(x|y) = \frac{f(x, y)}{f(y)}
```

### Chain Rule

```math
p(x_1, x_2, \ldots, x_n) = p(x_1) \cdot p(x_2|x_1) \cdot p(x_3|x_1, x_2) \cdots p(x_n|x_1, \ldots, x_{n-1})
```

---

## 📐 Independence

### Definition

Random variables $X$ and $Y$ are **independent** if:

```math
p(x, y) = p(x) \cdot p(y)
```

**Equivalent:**

```math
P(X \in A, Y \in B) = P(X \in A) \cdot P(Y \in B) \quad \forall A, B
```

### Conditional Independence

$X \perp Y | Z$ means:

```math
p(x, y | z) = p(x|z) \cdot p(y|z)
```

**Important:** 
- $X \perp Y | Z$ does NOT imply $X \perp Y$
- $X \perp Y$ does NOT imply $X \perp Y | Z$

---

## 📐 Mean Vector and Covariance Matrix

### Mean Vector

```math
\boldsymbol{\mu} = E[\mathbf{X}] = [E[X_1], \ldots, E[X_n]]^\top
```

### Covariance Matrix

```math
\boldsymbol{\Sigma} = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top]
\Sigma_{ij} = \text{Cov}(X_i, X_j)
```

### Properties

- Diagonal: $\Sigma\_{ii} = \text{Var}(X\_i)$
- Symmetric: $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^\top$
- Positive semi-definite: $\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} \geq 0$

---

## 📐 Transformations

### Linear Transformation

If $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$:

```math
E[\mathbf{Y}] = \mathbf{A}E[\mathbf{X}] + \mathbf{b}
\text{Cov}(\mathbf{Y}) = \mathbf{A} \text{Cov}(\mathbf{X}) \mathbf{A}^\top
```

### Proof

```math
E[\mathbf{Y}] = E[\mathbf{A}\mathbf{X} + \mathbf{b}] = \mathbf{A}E[\mathbf{X}] + \mathbf{b}
\text{Cov}(\mathbf{Y}) = E[(\mathbf{Y} - E[\mathbf{Y}])(\mathbf{Y} - E[\mathbf{Y}])^\top]
= E[\mathbf{A}(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top \mathbf{A}^\top]
= \mathbf{A} E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top] \mathbf{A}^\top = \mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^\top \quad \blacksquare
```

---

## 📐 Correlation Matrix

### Definition

```math
\mathbf{R}_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sqrt{\text{Var}(X_i) \text{Var}(X_j)}} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii} \Sigma_{jj}}}
```

### Properties

- Diagonal entries = 1
- Off-diagonal entries $\in [-1, 1]$
- $\mathbf{R} = \mathbf{D}^{-1/2} \boldsymbol{\Sigma} \mathbf{D}^{-1/2}$ where $\mathbf{D} = \text{diag}(\boldsymbol{\Sigma})$

---

## 📐 Special Case: Bivariate

### Joint Distribution

```math
\begin{bmatrix} X \\ Y \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \mu_X \\ \mu_Y \end{bmatrix}, \begin{bmatrix} \sigma_X^2 & \rho\sigma_X\sigma_Y \\ \rho\sigma_X\sigma_Y & \sigma_Y^2 \end{bmatrix}\right)
```

### Conditional Distribution

```math
X | Y = y \sim \mathcal{N}\left(\mu_X + \rho\frac{\sigma_X}{\sigma_Y}(y - \mu_Y), \sigma_X^2(1 - \rho^2)\right)
```

**Key insights:**
- Conditional mean is linear in $y$
- Conditional variance is reduced by factor $(1 - \rho^2)$
- Higher correlation → more variance reduction

---

## 💻 Code Examples

```python
import numpy as np
import torch
from scipy import stats

# Create correlated random vector
mean = np.array([0, 0])
cov = np.array([[1, 0.8], [0.8, 1]])  # Correlation = 0.8

samples = np.random.multivariate_normal(mean, cov, 1000)
print(f"Sample mean: {samples.mean(axis=0)}")
print(f"Sample cov:\n{np.cov(samples.T)}")

# Marginal distribution
X_samples = samples[:, 0]  # X₁ ~ N(0, 1)
print(f"Marginal X₁ mean: {X_samples.mean():.3f}, std: {X_samples.std():.3f}")

# Conditional distribution
def conditional_gaussian_2d(y_value, mu, Sigma):
    """
    Compute p(X | Y = y_value) for bivariate Gaussian
    """
    mu_X, mu_Y = mu[0], mu[1]
    sigma_X2, sigma_Y2 = Sigma[0, 0], Sigma[1, 1]
    sigma_XY = Sigma[0, 1]
    
    # Conditional mean
    mu_cond = mu_X + sigma_XY / sigma_Y2 * (y_value - mu_Y)
    
    # Conditional variance
    sigma_cond2 = sigma_X2 - sigma_XY**2 / sigma_Y2
    
    return mu_cond, sigma_cond2

mu_cond, var_cond = conditional_gaussian_2d(1.0, mean, cov)
print(f"p(X | Y=1): mean = {mu_cond:.3f}, var = {var_cond:.3f}")

# PyTorch multivariate normal
dist = torch.distributions.MultivariateNormal(
    loc=torch.tensor([0., 0.]),
    covariance_matrix=torch.tensor([[1., 0.8], [0.8, 1.]])
)

samples = dist.sample((1000,))
log_prob = dist.log_prob(samples)
print(f"Average log prob: {log_prob.mean():.3f}")

# Independence test using chi-squared
from scipy.stats import chi2_contingency

# Create contingency table from discretized data
X_discrete = np.digitize(samples[:, 0].numpy(), bins=[-2, -1, 0, 1, 2])
Y_discrete = np.digitize(samples[:, 1].numpy(), bins=[-2, -1, 0, 1, 2])
contingency = np.histogram2d(X_discrete, Y_discrete, bins=5)[0]

chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"Independence test p-value: {p_value:.4f}")  # Low = not independent

# Linear transformation
A = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Rotation
Y = samples @ A.T  # Transform

print(f"Original cov:\n{np.cov(samples.T)}")
print(f"Transformed cov (should be AΣAᵀ):\n{np.cov(Y.T)}")
```

---

## 📊 Key Relationships

| Operation | Formula |
|-----------|---------|
| Joint → Marginal | $p(x) = \int p(x,y) dy$ |
| Joint → Conditional | $p(x\|y) = p(x,y) / p(y)$ |
| Marginal + Conditional → Joint | $p(x,y) = p(x\|y) \cdot p(y)$ |
| Independence | $p(x,y) = p(x) \cdot p(y)$ |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 2 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | Murphy MLaPP Ch. 2 | [Book](https://probml.github.io/pml-book/book1.html) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 🇨🇳 | 多元随机变量 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 联合分布 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |

---

⬅️ [Back: Multivariate Gaussian](../03_gaussian/) | ➡️ [Back: Multivariate](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
