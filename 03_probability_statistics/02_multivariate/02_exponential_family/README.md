<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Exponential%20Family&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/exponential-family.svg" width="100%">

*Caption: p(x|θ) = h(x)exp(η(θ)·T(x) - A(θ)). Natural parameters η, sufficient statistics T(x), log-partition A(θ). Includes Gaussian, Bernoulli, Poisson, Gamma.*

---

## 📂 Overview

The exponential family unifies most common distributions under one framework. This makes theoretical analysis cleaner and enables general-purpose algorithms for a wide class of models.

---

## 📐 Exponential Family Form

### Canonical Form

$$p(x|\boldsymbol{\eta}) = h(x) \exp\left(\boldsymbol{\eta}^\top \mathbf{T}(x) - A(\boldsymbol{\eta})\right)$$

| Component | Name | Description |
|-----------|------|-------------|
| $\boldsymbol{\eta}$ | Natural parameters | Canonical parameterization |
| $\mathbf{T}(x)$ | Sufficient statistics | All information about $\eta$ in data |
| $A(\boldsymbol{\eta})$ | Log-partition function | Normalizer (log of partition function) |
| $h(x)$ | Base measure | Carrier density |

### Alternative Form (Standard Parameters)

$$p(x|\boldsymbol{\theta}) = h(x) \exp\left(\boldsymbol{\eta}(\boldsymbol{\theta})^\top \mathbf{T}(x) - A(\boldsymbol{\theta})\right)$$

---

## 📐 Key Properties

### Theorem: Moments from Log-Partition

$$E[\mathbf{T}(x)] = \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta})
\text{Cov}[\mathbf{T}(x)] = \nabla^2_{\boldsymbol{\eta}} A(\boldsymbol{\eta})$$

### Proof

The normalization constraint:

$$\int h(x) \exp\left(\boldsymbol{\eta}^\top \mathbf{T}(x) - A(\boldsymbol{\eta})\right) dx = 1$$

Rearranging:

$$\exp(A(\boldsymbol{\eta})) = \int h(x) \exp\left(\boldsymbol{\eta}^\top \mathbf{T}(x)\right) dx$$

Taking the gradient with respect to $\boldsymbol{\eta}$:

$$\exp(A(\boldsymbol{\eta})) \cdot \nabla A(\boldsymbol{\eta}) = \int h(x) \mathbf{T}(x) \exp\left(\boldsymbol{\eta}^\top \mathbf{T}(x)\right) dx$$

Dividing both sides by $\exp(A(\boldsymbol{\eta}))$:

$$\nabla A(\boldsymbol{\eta}) = \int \mathbf{T}(x) \cdot h(x) \exp\left(\boldsymbol{\eta}^\top \mathbf{T}(x) - A(\boldsymbol{\eta})\right) dx = E[\mathbf{T}(x)] \quad \blacksquare$$

Similarly, the second derivative gives the covariance.

---

## 📐 Common Distributions as Exponential Family

### Bernoulli

$$P(x|\theta) = \theta^x (1-\theta)^{1-x} = \exp\left(x \log\frac{\theta}{1-\theta} + \log(1-\theta)\right)$$

| Component | Value |
|-----------|-------|
| $\eta$ | $\log\frac{\theta}{1-\theta}$ (logit) |
| $T(x)$ | $x$ |
| $A(\eta)$ | $\log(1 + e^\eta)$ |
| $h(x)$ | $1$ |

**Inverse:** $\theta = \sigma(\eta) = \frac{1}{1 + e^{-\eta}}$ (sigmoid)

### Gaussian (Known Variance)

$$p(x|\mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

| Component | Value |
|-----------|-------|
| $\eta$ | $\mu/\sigma^2$ |
| $T(x)$ | $x$ |
| $A(\eta)$ | $\eta^2 \sigma^2/2$ |
| $h(x)$ | $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-x^2/(2\sigma^2)}$ |

### Gaussian (Unknown Mean and Variance)

$$p(x|\mu, \sigma^2) \propto \exp\left(\frac{\mu}{\sigma^2} x - \frac{1}{2\sigma^2} x^2\right)$$

| Component | Value |
|-----------|-------|
| $\boldsymbol{\eta}$ | $[\mu/\sigma^2, -1/(2\sigma^2)]^\top$ |
| $\mathbf{T}(x)$ | $[x, x^2]^\top$ |
| $A(\boldsymbol{\eta})$ | $-\eta_1^2/(4\eta_2) - \frac{1}{2}\log(-2\eta_2)$ |

### Poisson

$$P(x|\lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$$

| Component | Value |
|-----------|-------|
| $\eta$ | $\log \lambda$ |
| $T(x)$ | $x$ |
| $A(\eta)$ | $e^\eta = \lambda$ |
| $h(x)$ | $1/x!$ |

### Categorical/Multinomial

$$P(x|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{x_k}$$

| Component | Value |
|-----------|-------|
| $\boldsymbol{\eta}$ | $[\log\pi_1, \ldots, \log\pi_{K-1}]^\top$ |
| $\mathbf{T}(x)$ | $[x_1, \ldots, x_{K-1}]^\top$ |
| $A(\boldsymbol{\eta})$ | $\log\sum_k e^{\eta_k}$ |

---

## 📐 Maximum Likelihood for Exponential Family

### MLE via Moment Matching

**Theorem:** MLE sets sample moments equal to population moments:

$$\nabla A(\hat{\boldsymbol{\eta}}) = \frac{1}{n} \sum_{i=1}^n \mathbf{T}(x_i)$$

### Proof

Log-likelihood:

$$\ell(\boldsymbol{\eta}) = \sum_{i=1}^n \left[\boldsymbol{\eta}^\top \mathbf{T}(x_i) - A(\boldsymbol{\eta}) + \log h(x_i)\right]$$

Gradient:

$$\nabla_{\boldsymbol{\eta}} \ell = \sum_{i=1}^n \mathbf{T}(x_i) - n \cdot \nabla A(\boldsymbol{\eta})$$

Setting to zero:

$$\nabla A(\hat{\boldsymbol{\eta}}) = \frac{1}{n} \sum_{i=1}^n \mathbf{T}(x_i) \quad \blacksquare$$

---

## 📐 Conjugate Priors

**Theorem:** Exponential family distributions have conjugate priors of the form:

$$p(\boldsymbol{\eta}) \propto \exp\left(\boldsymbol{\eta}^\top \boldsymbol{\chi} - \nu A(\boldsymbol{\eta})\right)$$

where $\boldsymbol{\chi}$ and $\nu$ are hyperparameters.

**Posterior update:**

$$\boldsymbol{\chi}_n = \boldsymbol{\chi}_0 + \sum_{i=1}^n \mathbf{T}(x_i)
\nu_n = \nu_0 + n$$

---

## 📐 Generalized Linear Models (GLMs)

### Framework

**Link function:** $g(E[Y]) = \boldsymbol{X}\boldsymbol{\beta}$

**Canonical link:** $\eta = \boldsymbol{X}\boldsymbol{\beta}$ directly

| Distribution | Canonical Link | Name |
|--------------|----------------|------|
| Gaussian | Identity: $\eta = \mu$ | Linear regression |
| Bernoulli | Logit: $\eta = \log\frac{p}{1-p}$ | Logistic regression |
| Poisson | Log: $\eta = \log\lambda$ | Poisson regression |

### Why Canonical Link?

Using canonical link makes the gradient simple:

$$\nabla_{\boldsymbol{\beta}} \ell = \sum_i (y_i - \mu_i) \mathbf{x}_i$$

This is the same form regardless of distribution!

---

## 📐 Natural Gradients

### Fisher Information Matrix

For exponential families:

$$\mathbf{F}(\boldsymbol{\eta}) = \text{Cov}[\mathbf{T}(x)] = \nabla^2 A(\boldsymbol{\eta})$$

### Natural Gradient

$$\tilde{\nabla} \ell = \mathbf{F}^{-1} \nabla \ell$$

**Advantage:** Invariant to parameterization, faster convergence.

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal, Poisson, Categorical

# Bernoulli from natural parameter (logit)
def bernoulli_from_logit(logit):
    """Convert natural parameter to probability"""
    p = torch.sigmoid(logit)  # Inverse link
    return Bernoulli(probs=p)

# Cross-entropy = negative log-likelihood of Bernoulli
# F.binary_cross_entropy_with_logits uses logits directly
loss = nn.BCEWithLogitsLoss()

# Softmax = categorical in natural parameters
# logits are natural parameters η
# probabilities p_k = exp(η_k) / Σ_j exp(η_j)
logits = torch.tensor([1.0, 2.0, 3.0])
cat = Categorical(logits=logits)
samples = cat.sample((100,))

# Gaussian from natural parameters
def gaussian_from_natural(eta1, eta2):
    """
    Natural params: η₁ = μ/σ², η₂ = -1/(2σ²)
    """
    sigma2 = -1 / (2 * eta2)
    mu = eta1 * sigma2
    return Normal(mu, sigma2.sqrt())

# Log-partition function verification
def log_partition_bernoulli(eta):
    """A(η) = log(1 + exp(η))"""
    return torch.log(1 + torch.exp(eta))

def mean_from_log_partition(eta):
    """E[X] = dA/dη for Bernoulli"""
    # dA/dη = exp(η)/(1+exp(η)) = σ(η)
    return torch.sigmoid(eta)

# Verify
eta = torch.tensor(1.0, requires_grad=True)
A = log_partition_bernoulli(eta)
A.backward()
print(f"E[X] from A'(η): {eta.grad:.4f}")
print(f"E[X] from σ(η): {torch.sigmoid(eta):.4f}")

# GLM: Logistic regression
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # Returns logits (natural parameters)
        return self.linear(x)
    
    def predict_proba(self, x):
        # Convert to probabilities
        return torch.sigmoid(self.forward(x))

# MLE for exponential family = moment matching
def mle_bernoulli(data):
    """
    MLE: E[T(x)] = sample mean of T(x)
    For Bernoulli: T(x) = x
    """
    sample_mean = data.mean()
    # Convert to natural parameter
    eta_mle = torch.log(sample_mean / (1 - sample_mean))
    return eta_mle

```

---

## 📊 Why Exponential Family Matters for ML

| Property | Implication for ML |
|----------|-------------------|
| Conjugate priors exist | Efficient Bayesian updates |
| MLE = moment matching | Simple estimation |
| Natural gradients | Better optimization |
| GLM framework | Unified regression |
| Sufficient statistics | Data compression |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 2 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | Murphy MLaPP Ch. 9 | [Book](https://probml.github.io/pml-book/book1.html) |
| 📄 | Jordan Notes | [PDF](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf) |
| 🇨🇳 | 指数族分布详解 | [知乎](https://zhuanlan.zhihu.com/p/62954109) |
| 🇨🇳 | GLM原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |

---

⬅️ [Back: Covariance](../01_covariance/) | ➡️ [Next: Multivariate Gaussian](../03_gaussian/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
