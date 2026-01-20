<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Bayesian%20Inference&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Core Concept

Bayesian inference is the process of updating beliefs about parameters based on observed data using Bayes' theorem.

---

## 📐 Mathematical Framework

### Bayes' Theorem

$$
P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}
$$

| Term | Name | Description |
|------|------|-------------|
| $P(\theta\|D)$ | Posterior | Updated belief after seeing data |
| $P(D\|\theta)$ | Likelihood | Probability of data given parameters |
| $P(\theta)$ | Prior | Initial belief before data |
| $P(D)$ | Evidence | Normalizing constant (marginal likelihood) |

### Posterior Computation

$$
\text{Posterior} \propto \text{Likelihood} \times \text{Prior}
P(\theta|D) \propto P(D|\theta) P(\theta)
$$

**Evidence (marginal likelihood):**

$$
P(D) = \int P(D|\theta) P(\theta) \, d\theta
$$

---

## 📐 Detailed Derivation

### Proof of Bayes' Rule

**Starting from conditional probability:**

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
P(B|A) = \frac{P(A \cap B)}{P(A)}
$$

**From the second equation:**

$$
P(A \cap B) = P(B|A) \cdot P(A)
$$

**Substituting into the first:**

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

This is **Bayes' Theorem**. $\quad \blacksquare$

---

## 📐 Conjugate Prior Analysis

### Beta-Bernoulli Conjugacy

**Prior:**

$$
\theta \sim \text{Beta}(\alpha, \beta)
p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}
$$

**Likelihood:**

$$
X|\theta \sim \text{Bernoulli}(\theta)
P(D|\theta) = \theta^k (1-\theta)^{n-k}
$$

where $k = \sum\_{i=1}^n x\_i$ is the number of successes.

**Posterior:**

$$
p(\theta|D) \propto \theta^k (1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}
= \theta^{\alpha + k - 1}(1-\theta)^{\beta + n - k - 1}
\theta|D \sim \text{Beta}(\alpha + k, \beta + n - k) \quad \blacksquare
$$

**Interpretation:**
- $\alpha$: "prior successes" (pseudo-counts)
- $\beta$: "prior failures" (pseudo-counts)
- After observing $k$ successes in $n$ trials: add $k$ to $\alpha$, add $(n-k)$ to $\beta$

---

### Gaussian-Gaussian Conjugacy

**Prior:** $\mu \sim \mathcal{N}(\mu\_0, \sigma\_0^2)$

**Likelihood:** $X|\mu \sim \mathcal{N}(\mu, \sigma^2)$ (known variance)

**Posterior:** $\mu|D \sim \mathcal{N}(\mu\_n, \sigma\_n^2)$

where:

$$
\sigma_n^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}
\mu_n = \sigma_n^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}\right)
$$

**Proof:**

$$
p(\mu|D) \propto \exp\left(-\frac{(\mu-\mu_0)^2}{2\sigma_0^2}\right) \cdot \prod_{i=1}^n \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)
$$

Expanding and completing the square in $\mu$:

$$
\propto \exp\left(-\frac{1}{2}\left[\left(\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}\right)\mu^2 - 2\left(\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}\right)\mu\right]\right)
$$

This is the kernel of $\mathcal{N}(\mu\_n, \sigma\_n^2)$. $\quad \blacksquare$

---

## 📐 Predictive Distribution

**Posterior Predictive:**

$$
p(x^*|D) = \int p(x^*|\theta) p(\theta|D) \, d\theta
$$

This integrates out uncertainty about parameters!

**For Gaussian:**

$$
p(x^*|D) = \mathcal{N}(\mu_n, \sigma^2 + \sigma_n^2)
$$

The predictive variance includes both observation noise ($\sigma^2$) and parameter uncertainty ($\sigma\_n^2$).

---

## 📐 Evidence (Marginal Likelihood)

$$
P(D|M) = \int P(D|\theta, M) P(\theta|M) \, d\theta
$$

**Bayesian Model Comparison:**

$$
\frac{P(M_1|D)}{P(M_2|D)} = \frac{P(D|M_1)}{P(D|M_2)} \cdot \frac{P(M_1)}{P(M_2)}
$$

The ratio $\frac{P(D|M\_1)}{P(D|M\_2)}$ is called the **Bayes Factor**.

---

## 📐 Common Conjugate Pairs

| Likelihood | Conjugate Prior | Posterior |
|------------|-----------------|-----------|
| Bernoulli($\theta$) | Beta($\alpha, \beta$) | Beta($\alpha + k, \beta + n - k$) |
| Poisson($\lambda$) | Gamma($\alpha, \beta$) | Gamma($\alpha + \sum x\_i, \beta + n$) |
| Normal($\mu$, known $\sigma^2$) | Normal($\mu\_0, \sigma\_0^2$) | Normal($\mu\_n, \sigma\_n^2$) |
| Multinomial($\boldsymbol{\theta}$) | Dirichlet($\boldsymbol{\alpha}$) | Dirichlet($\boldsymbol{\alpha} + \text{counts}$) |

---

## 📐 Approximate Inference Methods

### 1. Markov Chain Monte Carlo (MCMC)

When posterior is intractable, sample from it:

- Metropolis-Hastings
- Gibbs Sampling  
- Hamiltonian Monte Carlo (HMC)

### 2. Variational Inference

Approximate $p(\theta|D)$ with simpler $q(\theta)$:

$$
q^* = \arg\min_q D_{KL}(q(\theta) \| p(\theta|D))
$$

**ELBO:**

$$
\mathcal{L}(q) = \mathbb{E}_q[\log p(D|\theta)] - D_{KL}(q(\theta) \| p(\theta))
$$

### 3. Laplace Approximation

Approximate posterior as Gaussian centered at MAP:

$$
p(\theta|D) \approx \mathcal{N}(\theta_{MAP}, H^{-1})
$$

where $H$ is the Hessian of the negative log-posterior at $\theta\_{MAP}$.

---

## 💻 Code Examples

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Bayesian coin flip: Beta-Bernoulli conjugate
def bayesian_coin_inference(data, prior_alpha=1, prior_beta=1):
    """
    Conjugate Bayesian update for coin flip
    """
    n_heads = sum(data)
    n_tails = len(data) - n_heads
    
    # Prior
    prior = stats.beta(prior_alpha, prior_beta)
    
    # Posterior (conjugate update)
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_tails
    posterior = stats.beta(post_alpha, post_beta)
    
    # Credible interval
    ci_low, ci_high = posterior.interval(0.95)
    
    # Posterior mean (point estimate)
    mean = post_alpha / (post_alpha + post_beta)
    
    return {
        'posterior': posterior,
        'mean': mean,
        'ci': (ci_low, ci_high),
        'params': (post_alpha, post_beta)
    }

# Example
data = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]  # 7 heads, 3 tails
result = bayesian_coin_inference(data)
print(f"Posterior mean: {result['mean']:.3f}")
print(f"95% CI: [{result['ci'][0]:.3f}, {result['ci'][1]:.3f}]")

# Bayesian Linear Regression
def bayesian_linear_regression(X, y, alpha=1.0, beta=1.0):
    """
    Bayesian linear regression with Gaussian prior
    Prior: w ~ N(0, alpha^-1 I)
    Likelihood: y|X,w ~ N(Xw, beta^-1 I)
    """
    n, d = X.shape
    
    # Posterior covariance
    Sigma_n = np.linalg.inv(alpha * np.eye(d) + beta * X.T @ X)
    
    # Posterior mean
    mu_n = beta * Sigma_n @ X.T @ y
    
    return mu_n, Sigma_n

def predict_with_uncertainty(X_new, mu_n, Sigma_n, beta):
    """Predictive distribution"""
    y_mean = X_new @ mu_n
    y_var = 1/beta + np.diag(X_new @ Sigma_n @ X_new.T)
    return y_mean, np.sqrt(y_var)

# Monte Carlo estimation of posterior expectation
def monte_carlo_posterior_expectation(samples, f):
    """
    E[f(θ)|D] ≈ (1/N) Σ f(θ^(i))
    """
    return np.mean([f(s) for s in samples])
```

---

## 🎯 Bayesian vs Frequentist

| Aspect | Bayesian | Frequentist |
|--------|----------|-------------|
| **Parameters** | Random variables | Fixed unknowns |
| **Inference** | $P(\theta\|D)$ | Point estimate $\hat{\theta}$ |
| **Uncertainty** | Posterior distribution | Confidence intervals |
| **Prior** | Required (explicit) | Not used |
| **Interpretation** | Probability = degree of belief | Probability = long-run frequency |
| **Advantages** | Uncertainty quantification, prior knowledge | Simpler, well-established |

---

## 🌍 Modern Applications

| Application | How Bayesian Inference is Used |
|-------------|-------------------------------|
| **Bayesian Optimization** | GP surrogate with posterior uncertainty |
| **VAE** | Variational inference for latent variables |
| **Bayesian Neural Networks** | Posterior over weights for uncertainty |
| **Thompson Sampling** | Sample from posterior for exploration |
| **A/B Testing** | Posterior probability of improvement |
| **Active Learning** | Use uncertainty to guide data collection |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Pattern Recognition and ML (Bishop) | Ch. 1-3 |
| 📖 | Bayesian Data Analysis (Gelman) | [Book](http://www.stat.columbia.edu/~gelman/book/) |
| 📄 | Practical Bayesian Optimization | [arXiv](https://arxiv.org/abs/1807.02811) |
| 🎥 | Bayesian Methods for ML | [Coursera](https://www.coursera.org/learn/bayesian-methods-in-machine-learning) |
| 🇨🇳 | 贝叶斯推断详解 | [知乎](https://zhuanlan.zhihu.com/p/37976562) |
| 🇨🇳 | 贝叶斯机器学习 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

⬅️ [Back: Probability](../) | ➡️ [Next: Bayes Theorem](../02_bayes_theorem/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
