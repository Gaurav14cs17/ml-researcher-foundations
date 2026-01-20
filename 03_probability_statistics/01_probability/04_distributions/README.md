<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Probability%20Distributions&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Discrete Distributions

### Bernoulli Distribution

```math
X \sim \text{Bernoulli}(p)
P(X = 1) = p, \quad P(X = 0) = 1 - p = q

```

**PMF:**

```math
P(X = k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}

```

| Property | Value |
|----------|-------|
| Mean | $E[X] = p$ |
| Variance | $\text{Var}(X) = p(1-p)$ |
| Mode | 1 if p > 0.5, else 0 |
| Entropy | $-p \log p - (1-p) \log(1-p)$ |

**ML Use:** Binary classification output (sigmoid → Bernoulli)

#### Proof: Mean of Bernoulli

```math
E[X] = \sum_{x \in \{0,1\}} x \cdot P(X=x) = 0 \cdot (1-p) + 1 \cdot p = p \quad \blacksquare

```

#### Proof: Variance of Bernoulli

```math
E[X^2] = 0^2 \cdot (1-p) + 1^2 \cdot p = p
\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p) \quad \blacksquare

```

---

### Binomial Distribution

```math
X \sim \text{Binomial}(n, p)
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0,1,\ldots,n

```

| Property | Value |
|----------|-------|
| Mean | $E[X] = np$ |
| Variance | $\text{Var}(X) = np(1-p)$ |

**Interpretation:** Sum of n independent Bernoulli(p) trials

#### Proof: Mean of Binomial

Let $X = \sum_{i=1}^{n} X_i$ where $X_i \sim \text{Bernoulli}(p)$ are independent.

```math
E[X] = E\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} E[X_i] = \sum_{i=1}^{n} p = np \quad \blacksquare

```

---

### Categorical Distribution

```math
X \sim \text{Categorical}(p_1, p_2, \ldots, p_K) \quad \text{where } \sum_i p_i = 1
P(X = i) = p_i

```

**One-hot representation:**

```math
\mathbf{x} = [0, 0, \ldots, 1, \ldots, 0]^\top \quad \text{(1 at position } i\text{)}

```

**ML Use:** Multi-class classification (softmax → Categorical)

---

### Poisson Distribution

```math
X \sim \text{Poisson}(\lambda)
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0,1,2,\ldots

```

| Property | Value |
|----------|-------|
| Mean | $E[X] = \lambda$ |
| Variance | $\text{Var}(X) = \lambda$ |

**ML Use:** Count data, rare events

#### Proof: Poisson Mean

```math
E[X] = \sum_{k=0}^{\infty} k \cdot \frac{\lambda^k e^{-\lambda}}{k!}

```

The k=0 term vanishes. For k ≥ 1:

```math
= e^{-\lambda} \sum_{k=1}^{\infty} k \cdot \frac{\lambda^k}{k!} = e^{-\lambda} \sum_{k=1}^{\infty} \frac{\lambda^k}{(k-1)!}

```

Let j = k - 1:

```math
= e^{-\lambda} \lambda \sum_{j=0}^{\infty} \frac{\lambda^j}{j!} = e^{-\lambda} \lambda \cdot e^{\lambda} = \lambda \quad \blacksquare

```

---

## 📐 Continuous Distributions

### Gaussian (Normal) Distribution

```math
X \sim \mathcal{N}(\mu, \sigma^2)

```

**PDF:**

```math
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

```

| Property | Value |
|----------|-------|
| Mean | $E[X] = \mu$ |
| Variance | $\text{Var}(X) = \sigma^2$ |
| Mode | $\mu$ (same as mean) |
| Entropy | $\frac{1}{2}\log(2\pi e\sigma^2)$ |

**Standard Normal:** $Z \sim \mathcal{N}(0, 1)$

**Standardization:** $Z = \frac{X - \mu}{\sigma}$

**ML Uses:**
- Regression targets (MSE loss)

- Latent space in VAE

- Weight initialization

- Gaussian noise in diffusion

---

### Detailed Gaussian Theory

#### Proof: Gaussian is Maximum Entropy Distribution

**Theorem:** Among all distributions with fixed mean μ and variance σ², the Gaussian has maximum entropy.

**Proof using Lagrange Multipliers:**

We maximize:

```math
H(X) = -\int p(x)\log p(x) \, dx

```

Subject to:

1. $\int p(x) dx = 1$ (normalization)

2. $\int x \cdot p(x) dx = \mu$ (fixed mean)

3. $\int (x-\mu)^2 \cdot p(x) dx = \sigma^2$ (fixed variance)

**Lagrangian:**

```math
\mathcal{L}[p] = -\int p \log p \, dx + \lambda_0\left(\int p \, dx - 1\right) + \lambda_1\left(\int xp \, dx - \mu\right) + \lambda_2\left(\int (x-\mu)^2 p \, dx - \sigma^2\right)

```

**Functional derivative:**

```math
\frac{\delta \mathcal{L}}{\delta p} = -\log p - 1 + \lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2 = 0
\log p = -1 + \lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2
p(x) = \exp\left(-1 + \lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2\right)

```

**Imposing constraints:**
- From fixed mean: $\lambda_1 = 0$

- From fixed variance: $\lambda_2 = -\frac{1}{2\sigma^2}$

- From normalization: $\lambda_0 = 1 - \frac{1}{2}\log(2\pi\sigma^2)$

```math
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

```

This is the Gaussian distribution! $\blacksquare$

**Entropy of Gaussian:**

```math
H(X) = \frac{1}{2}\log(2\pi e\sigma^2) \text{ nats}

```

---

#### Central Limit Theorem

**Theorem:** Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$.

Define sample mean: $\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$

Then as $n \to \infty$:

```math
\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)

```

Or equivalently: $\bar{X}_n \xrightarrow{d} \mathcal{N}(\mu, \sigma^2/n)$

**Proof Sketch (via Characteristic Functions):**

1. Standardize: $Z_i = \frac{X_i - \mu}{\sigma}$, so $E[Z_i]=0$, $\text{Var}(Z_i)=1$

2. Define $S_n = \frac{\sum_{i=1}^{n} Z_i}{\sqrt{n}}$

3. Characteristic function:

```math
\phi_{S_n}(t) = E[e^{itS_n}] = \left[\phi_Z\left(\frac{t}{\sqrt{n}}\right)\right]^n

```

4. Taylor expansion:

```math
\phi_Z\left(\frac{t}{\sqrt{n}}\right) = 1 - \frac{t^2}{2n} + O\left(\frac{t^3}{n^{3/2}}\right)

```

5. Taking limit:

```math
\phi_{S_n}(t) = \left[1 - \frac{t^2}{2n}\right]^n \to e^{-t^2/2}

```

This is the characteristic function of $\mathcal{N}(0,1)$! $\blacksquare$

---

#### Gaussian-Gaussian Conjugacy

**Theorem:** If likelihood is $X|\mu \sim \mathcal{N}(\mu, \sigma^2)$ (σ² known) and prior is $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$, then the posterior is Gaussian:

```math
\mu|X \sim \mathcal{N}(\mu_n, \sigma_n^2)

```

where:

```math
\sigma_n^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}
\mu_n = \sigma_n^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{X}}{\sigma^2}\right)

```

**Proof:**

```math
p(\mu|X) \propto p(X|\mu) \cdot p(\mu)
\propto \exp\left(-\sum_i\frac{(x_i-\mu)^2}{2\sigma^2}\right) \cdot \exp\left(-\frac{(\mu-\mu_0)^2}{2\sigma_0^2}\right)

```

Expanding and completing the square in μ:

```math
\propto \exp\left(-\frac{1}{2}\left[\left(\frac{n}{\sigma^2}+\frac{1}{\sigma_0^2}\right)\mu^2 - 2\left(\frac{n\bar{X}}{\sigma^2}+\frac{\mu_0}{\sigma_0^2}\right)\mu\right]\right)

```

This is the kernel of $\mathcal{N}(\mu_n, \sigma_n^2)$. $\blacksquare$

---

### Multivariate Gaussian

```math
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad \mathbf{X}, \boldsymbol{\mu} \in \mathbb{R}^n, \quad \boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}

```

**PDF:**

```math
p(\mathbf{x}) = (2\pi)^{-n/2} |\boldsymbol{\Sigma}|^{-1/2} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)

```

**Key Properties:**
- Marginals are Gaussian

- Conditionals are Gaussian

- Linear transforms are Gaussian: $A\mathbf{X} + \mathbf{b} \sim \mathcal{N}(A\boldsymbol{\mu}+\mathbf{b}, A\boldsymbol{\Sigma}A^\top)$

---

#### Marginal and Conditional Gaussians

**Joint distribution:**

```math
\begin{bmatrix} \mathbf{x} \\ \mathbf{y} \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \boldsymbol{\mu}_x \\ \boldsymbol{\mu}_y \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\ \boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy} \end{bmatrix}\right)

```

**Marginal:**

```math
\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_x, \boldsymbol{\Sigma}_{xx})

```

**Conditional:**

```math
\mathbf{x}|\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_{x|y}, \boldsymbol{\Sigma}_{x|y})

```

where:

```math
\boldsymbol{\mu}_{x|y} = \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}(\mathbf{y} - \boldsymbol{\mu}_y)
\boldsymbol{\Sigma}_{x|y} = \boldsymbol{\Sigma}_{xx} - \boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx}

```

**Note:** The conditional mean is the **regression formula**! The conditional covariance is the **Schur complement**.

---

#### KL Divergence for Gaussians

**Univariate:**

```math
D_{KL}(\mathcal{N}(\mu_1,\sigma_1^2) \| \mathcal{N}(\mu_2,\sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}

```

**Special case (KL to standard normal - VAE regularization):**

```math
D_{KL}(\mathcal{N}(\mu,\sigma^2) \| \mathcal{N}(0,1)) = \frac{1}{2}\left(\mu^2 + \sigma^2 - 1 - \log(\sigma^2)\right)

```

**Multivariate:**

```math
D_{KL}(\mathcal{N}(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_1) \| \mathcal{N}(\boldsymbol{\mu}_2,\boldsymbol{\Sigma}_2)) = \frac{1}{2}\left[\text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1) - d + \log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}\right]

```

---

### Uniform Distribution

```math
X \sim \text{Uniform}(a, b)

```

**PDF:**

```math
p(x) = \frac{1}{b-a}, \quad x \in [a, b]

```

**CDF:**

```math
F(x) = \frac{x-a}{b-a}

```

| Property | Value |
|----------|-------|
| Mean | $E[X] = \frac{a+b}{2}$ |
| Variance | $\text{Var}(X) = \frac{(b-a)^2}{12}$ |

---

### Exponential Distribution

```math
X \sim \text{Exponential}(\lambda)

```

**PDF:**

```math
p(x) = \lambda e^{-\lambda x}, \quad x \geq 0

```

**CDF:**

```math
F(x) = 1 - e^{-\lambda x}

```

| Property | Value |
|----------|-------|
| Mean | $E[X] = 1/\lambda$ |
| Variance | $\text{Var}(X) = 1/\lambda^2$ |

---

## 📐 The Exponential Family

**General form:**

```math
p(x|\boldsymbol{\theta}) = h(x) \exp\left(\boldsymbol{\eta}(\boldsymbol{\theta})^\top \mathbf{T}(x) - A(\boldsymbol{\theta})\right)

```

| Component | Name | Description |
|-----------|------|-------------|
| $\boldsymbol{\eta}(\boldsymbol{\theta})$ | Natural parameters | Canonical form of parameters |
| $\mathbf{T}(x)$ | Sufficient statistics | All info about θ in data |
| $A(\boldsymbol{\theta})$ | Log-partition function | Normalizer |
| $h(x)$ | Base measure | Carrier measure |

**Members:** Gaussian, Bernoulli, Poisson, Exponential, Gamma, Beta, ...

**Why it matters in ML:**
- Conjugate priors exist

- Maximum entropy distributions

- Natural gradients

- Generalized Linear Models (GLMs)

---

## 💻 Code Examples

```python
import numpy as np
import torch
import torch.distributions as dist

# Bernoulli
p = 0.7
bernoulli = dist.Bernoulli(probs=p)
samples = bernoulli.sample((1000,))
print(f"Bernoulli mean: {samples.mean():.3f} (expected: {p})")

# Gaussian
mu, sigma = 0.0, 1.0
gaussian = dist.Normal(mu, sigma)
samples = gaussian.sample((1000,))
print(f"Gaussian mean: {samples.mean():.3f}, std: {samples.std():.3f}")

# Categorical (softmax output)
logits = torch.tensor([1.0, 2.0, 3.0])
categorical = dist.Categorical(logits=logits)
samples = categorical.sample((1000,))
print(f"Categorical samples: {torch.bincount(samples)}")

# Multivariate Gaussian
mu = torch.zeros(2)
cov = torch.eye(2)
mvn = dist.MultivariateNormal(mu, cov)
samples = mvn.sample((1000,))

# Log probability (for loss computation)
x = torch.tensor([0.5])
log_prob = gaussian.log_prob(x)  # Used in NLL loss

# Sampling via reparameterization (VAE)
def reparameterize(mu, log_var):
    """Sample from N(mu, sigma^2) with gradient flow"""
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)  # N(0, 1)
    return mu + eps * std

# KL divergence to standard normal (VAE loss)
def kl_divergence(mu, log_var):
    """D_KL(N(mu, sigma^2) || N(0, 1))"""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# Cholesky sampling for multivariate Gaussian
def sample_mvn_cholesky(mu, Sigma, n_samples):
    """Numerically stable sampling via Cholesky"""
    L = torch.linalg.cholesky(Sigma)
    z = torch.randn(n_samples, len(mu))
    return mu + z @ L.T

```

---

## 🌍 ML Applications

| Distribution | ML Application |
|--------------|----------------|
| Bernoulli | Binary classification (BCE loss) |
| Categorical | Multi-class classification (CE loss) |
| Gaussian | Regression (MSE loss), VAE |
| Poisson | Count prediction |
| Exponential | Survival analysis |

### Loss Functions as Negative Log-Likelihood

```math
\text{Binary Cross-Entropy} = -\log P(y|x) \text{ for Bernoulli}
\text{Cross-Entropy Loss} = -\log P(y|x) \text{ for Categorical}
\text{MSE Loss} \propto -\log P(y|x) \text{ for Gaussian (fixed } \sigma\text{)}

```

**Training = Maximum Likelihood = Minimize NLL**

---

## 🔗 Where Probability Distributions Are Used

| Application | Distribution Used |
|-------------|-------------------|
| **Binary Classification** | Bernoulli → BCE loss |
| **Multi-class Classification** | Categorical → Cross-entropy loss |
| **Regression** | Gaussian → MSE loss |
| **Language Models** | Categorical over vocabulary |
| **VAE Latent Space** | Gaussian prior and posterior |
| **Diffusion Models** | Gaussian noise at each step |
| **Bayesian Neural Networks** | Gaussian weight priors |
| **Hidden Markov Models** | Multinomial emissions |
| **Dropout** | Mask ~ Bernoulli(1-p) |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 🎥 | 3Blue1Brown: Probability | [YouTube](https://www.youtube.com/watch?v=HZGCoVF3YvM) |
| 🎥 | StatQuest: Distributions | [YouTube](https://www.youtube.com/watch?v=rzFX5NWojp0) |
| 📖 | Bishop PRML Ch. 2 | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🇨🇳 | 概率分布详解 | [知乎](https://zhuanlan.zhihu.com/p/24648612) |
| 🇨🇳 | 概率论基础 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |
| 🇨🇳 | 机器学习概率分布 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88716548)

---

⬅️ [Back: Conditional Probability](../03_conditional/) | ➡️ [Next: Expectation](../05_expectation/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
