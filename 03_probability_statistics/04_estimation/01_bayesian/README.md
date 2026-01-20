<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Bayesian%20Estimation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/bayesian-estimation.svg" width="100%">

*Caption: Bayesian inference computes the full posterior P(θ|D), providing uncertainty quantification unlike point estimates (MLE/MAP).*

---

## 📐 Mathematical Foundation

### Bayes' Theorem

```math
P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}
```

| Term | Name | Role |
|------|------|------|
| $P(\theta\|D)$ | **Posterior** | What we want |
| $P(D\|\theta)$ | **Likelihood** | How data explains θ |
| $P(\theta)$ | **Prior** | Beliefs before data |
| $P(D)$ | **Evidence** | Normalizing constant |

### Log-Posterior Decomposition

```math
\log P(\theta|D) = \underbrace{\log P(D|\theta)}_{\text{log-likelihood}} + \underbrace{\log P(\theta)}_{\text{log-prior}} - \underbrace{\log P(D)}_{\text{constant}}
```

**Key insight:** Posterior ∝ Likelihood × Prior

---

## 📐 Predictive Distribution

For new observation $x^*$:

```math
P(x^*|D) = \int P(x^*|\theta) \cdot P(\theta|D) \, d\theta
```

**This is the key advantage of Bayesian inference:**
- Averages predictions over all possible θ
- Automatically includes parameter uncertainty
- More robust than point estimates

---

## 📐 Conjugate Priors

### Definition

Prior $P(\theta)$ is **conjugate** to likelihood $P(D|\theta)$ if:

```math
P(\theta|D) \in \text{same family as } P(\theta)
```

### Proof: Beta-Bernoulli Conjugacy

**Prior:** $\theta \sim \text{Beta}(\alpha, \beta)$

```math
P(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}
```

**Likelihood:** $D = \{x\_1, \ldots, x\_n\}$ where $x\_i \sim \text{Bernoulli}(\theta)$

```math
P(D|\theta) = \theta^k (1-\theta)^{n-k}
```

where $k = \sum\_{i=1}^n x\_i$ (number of successes).

**Posterior:**

```math
P(\theta|D) \propto P(D|\theta) \cdot P(\theta)
\propto \theta^k (1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}
= \theta^{(\alpha+k)-1}(1-\theta)^{(\beta+n-k)-1}
\therefore \theta|D \sim \text{Beta}(\alpha + k, \beta + n - k) \quad \blacksquare
```

**Interpretation:**
- $\alpha$ = "pseudo-observations" of success
- $\beta$ = "pseudo-observations" of failure
- Data simply adds to these counts!

---

## 📐 Common Conjugate Pairs

| Likelihood | Prior | Posterior |
|------------|-------|-----------|
| Bernoulli($\theta$) | Beta($\alpha, \beta$) | Beta($\alpha+k, \beta+n-k$) |
| Binomial($n, \theta$) | Beta($\alpha, \beta$) | Beta($\alpha+k, \beta+n-k$) |
| Poisson($\lambda$) | Gamma($\alpha, \beta$) | Gamma($\alpha+\sum x\_i, \beta+n$) |
| Gaussian($\mu$, known $\sigma^2$) | Gaussian($\mu\_0, \sigma\_0^2$) | Gaussian($\mu\_n, \sigma\_n^2$) |
| Gaussian(known $\mu$, $\sigma^2$) | Inverse-Gamma($\alpha, \beta$) | Inverse-Gamma($\alpha', \beta'$) |
| Multinomial | Dirichlet | Dirichlet |

### Gaussian-Gaussian Conjugacy (Proof)

**Prior:** $\mu \sim \mathcal{N}(\mu\_0, \sigma\_0^2)$

**Likelihood:** $x\_1, \ldots, x\_n \sim \mathcal{N}(\mu, \sigma^2)$ (known $\sigma^2$)

**Posterior:**

```math
\mu | D \sim \mathcal{N}(\mu_n, \sigma_n^2)
```

where:

```math
\frac{1}{\sigma_n^2} = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}
\frac{\mu_n}{\sigma_n^2} = \frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}
```

**Precision form (more intuitive):**

```math
\text{Posterior precision} = \text{Prior precision} + \text{Data precision}
\text{Posterior mean} = \text{Weighted average of prior mean and sample mean}
```

---

## 📐 Evidence Lower Bound (ELBO)

When $P(\theta|D)$ is intractable, use **Variational Inference**.

### Derivation

Start with log-evidence:

```math
\log P(D) = \log \int P(D, \theta) \, d\theta
```

Introduce approximate posterior $q(\theta)$:

```math
\log P(D) = \log \int \frac{P(D, \theta)}{q(\theta)} q(\theta) \, d\theta
```

Apply Jensen's inequality ($\log$ is concave):

```math
\log P(D) \geq \int q(\theta) \log \frac{P(D, \theta)}{q(\theta)} d\theta
= \int q(\theta) \log \frac{P(D|\theta)P(\theta)}{q(\theta)} d\theta
= \underbrace{E_q[\log P(D|\theta)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q(\theta) \| P(\theta))}_{\text{regularization}}
```

This is the **ELBO** (Evidence Lower Bound).

### ELBO = log P(D) - KL(q || posterior)

```math
\text{ELBO} = \log P(D) - D_{KL}(q(\theta) \| P(\theta|D))
```

Since $D\_{KL} \geq 0$, ELBO ≤ log P(D). Maximizing ELBO:
1. Maximizes lower bound on evidence
2. Minimizes KL to true posterior

---

## 📐 Credible Intervals vs Confidence Intervals

| Bayesian (Credible) | Frequentist (Confidence) |
|---------------------|-------------------------|
| P(θ ∈ CI \| D) = 0.95 | "95% of such intervals contain θ" |
| Direct probability statement | Long-run frequency |
| More intuitive | Technically correct |

### Highest Posterior Density (HPD)

The smallest interval containing 95% posterior probability:

```math
\text{HPD}_{0.95} = \{\theta : P(\theta|D) \geq k\}
```

where $k$ is chosen so $\int\_{\text{HPD}} P(\theta|D) d\theta = 0.95$.

---

## 💻 Code Examples

### Conjugate Bayesian Update

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def bayesian_coin_flip(data, prior_alpha=1, prior_beta=1):
    """
    Bayesian inference for coin flip (Beta-Bernoulli conjugacy)
    
    Prior: Beta(α, β)
    Likelihood: Bernoulli(θ)
    Posterior: Beta(α + k, β + n - k)
    """
    n = len(data)
    k = sum(data)  # Number of heads
    
    # Prior
    prior = stats.beta(prior_alpha, prior_beta)
    
    # Posterior (conjugate update - just add counts!)
    post_alpha = prior_alpha + k
    post_beta = prior_beta + (n - k)
    posterior = stats.beta(post_alpha, post_beta)
    
    # Posterior statistics
    posterior_mean = post_alpha / (post_alpha + post_beta)
    posterior_var = posterior.var()
    ci_95 = posterior.interval(0.95)
    
    print(f"Data: {k} heads out of {n} flips")
    print(f"Prior: Beta({prior_alpha}, {prior_beta})")
    print(f"Posterior: Beta({post_alpha}, {post_beta})")
    print(f"Posterior mean: {posterior_mean:.4f}")
    print(f"95% Credible Interval: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print(f"MLE would be: {k/n:.4f}")
    
    # Visualization
    theta = np.linspace(0, 1, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(theta, prior.pdf(theta), 'b--', label=f'Prior: Beta({prior_alpha},{prior_beta})')
    plt.plot(theta, posterior.pdf(theta), 'r-', label=f'Posterior: Beta({post_alpha},{post_beta})')
    plt.axvline(k/n, color='g', linestyle=':', label=f'MLE: {k/n:.3f}')
    plt.fill_between(theta, posterior.pdf(theta), alpha=0.3, color='red')
    plt.xlabel('θ (probability of heads)')
    plt.ylabel('Density')
    plt.title('Bayesian Update: Prior → Posterior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return posterior

# Example: 7 heads, 3 tails with uniform prior
data = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
posterior = bayesian_coin_flip(data, prior_alpha=1, prior_beta=1)

# With informative prior (believing coin is fair)
posterior_informative = bayesian_coin_flip(data, prior_alpha=10, prior_beta=10)
```

### Gaussian-Gaussian Conjugate Update

```python
import numpy as np
from scipy import stats

def gaussian_posterior(data, prior_mean, prior_var, likelihood_var):
    """
    Bayesian inference for Gaussian mean with Gaussian prior
    
    Prior: μ ~ N(μ₀, σ₀²)
    Likelihood: xᵢ ~ N(μ, σ²) (known σ²)
    Posterior: μ|D ~ N(μₙ, σₙ²)
    """
    n = len(data)
    sample_mean = np.mean(data)
    
    # Precision = 1/variance
    prior_precision = 1 / prior_var
    data_precision = n / likelihood_var
    
    # Posterior precision = prior precision + data precision
    post_precision = prior_precision + data_precision
    post_var = 1 / post_precision
    
    # Posterior mean = weighted average
    post_mean = post_var * (prior_precision * prior_mean + 
                            data_precision * sample_mean)
    
    print(f"Prior: N({prior_mean}, {prior_var})")
    print(f"Data: n={n}, sample mean={sample_mean:.4f}")
    print(f"Posterior: N({post_mean:.4f}, {post_var:.4f})")
    print(f"95% Credible Interval: [{post_mean - 1.96*np.sqrt(post_var):.4f}, "
          f"{post_mean + 1.96*np.sqrt(post_var):.4f}]")
    
    return post_mean, post_var

# Example
data = np.random.normal(5, 2, 50)  # True mean = 5
gaussian_posterior(data, prior_mean=0, prior_var=10, likelihood_var=4)
```

### Variational Inference

```python
import torch
import torch.nn as nn
import torch.distributions as dist

class VariationalInference(nn.Module):
    """
    Mean-field Variational Inference
    
    Approximate posterior q(θ) = N(μ, σ²)
    Maximize ELBO = E_q[log p(D|θ)] - KL(q||prior)
    """
    def __init__(self, dim, prior_std=1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_var = nn.Parameter(torch.zeros(dim))
        self.prior = dist.Normal(0, prior_std)
    
    def q_dist(self):
        """Approximate posterior q(θ)"""
        std = torch.exp(0.5 * self.log_var)
        return dist.Normal(self.mu, std)
    
    def elbo(self, log_likelihood_fn, n_samples=10):
        """
        ELBO = E_q[log p(D|θ)] - KL(q || prior)
        """
        q = self.q_dist()
        
        # Monte Carlo estimate of expected log-likelihood
        log_lik = 0
        for _ in range(n_samples):
            theta = q.rsample()  # Reparameterization trick
            log_lik += log_likelihood_fn(theta)
        log_lik /= n_samples
        
        # KL divergence (analytical for Gaussians)
        kl = dist.kl_divergence(q, self.prior).sum()
        
        return log_lik - kl
    
    def fit(self, log_likelihood_fn, n_iter=1000, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = -self.elbo(log_likelihood_fn)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 200 == 0:
                print(f"Iter {i+1}: ELBO = {-loss.item():.4f}")
        
        return self.mu.detach(), torch.exp(0.5 * self.log_var).detach()

# Example usage
def log_likelihood(theta):

    # Example: data likelihood
    data = torch.tensor([1.0, 2.0, 1.5, 2.5, 1.8])
    return dist.Normal(theta[0], 1.0).log_prob(data).sum()

vi = VariationalInference(dim=1)
mu, std = vi.fit(log_likelihood)
print(f"\nApproximate posterior: N({mu.item():.4f}, {std.item():.4f})")
```

### MCMC with PyMC

```python
import pymc as pm
import numpy as np
import arviz as az

def bayesian_linear_regression(X, y):
    """
    Bayesian Linear Regression with MCMC
    
    Priors:
        α ~ N(0, 10)
        β ~ N(0, 10)  
        σ ~ HalfNormal(1)
    
    Likelihood:
        y ~ N(α + Xβ, σ²)
    """
    with pm.Model() as model:

        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=X.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Expected value
        mu = alpha + pm.math.dot(X, beta)
        
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        # Sample from posterior using NUTS
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    
    # Summary statistics
    print(az.summary(trace, var_names=['alpha', 'beta', 'sigma']))
    
    # Plot posteriors
    az.plot_posterior(trace, var_names=['alpha', 'beta'])
    
    return trace, model

# Example
np.random.seed(42)
X = np.random.randn(100, 2)
true_beta = np.array([2.0, -1.0])
y = 1.0 + X @ true_beta + 0.5 * np.random.randn(100)

trace, model = bayesian_linear_regression(X, y)
```

---

## 📊 MLE vs MAP vs Full Bayesian

| Aspect | MLE | MAP | Bayesian |
|--------|-----|-----|----------|
| **Output** | Point $\hat{\theta}$ | Point $\hat{\theta}$ | Distribution $P(\theta\|D)$ |
| **Formula** | $\arg\max P(D\|\theta)$ | $\arg\max P(D\|\theta)P(\theta)$ | $P(\theta\|D)$ |
| **Prior** | No | Yes | Yes |
| **Uncertainty** | No | No | ✅ Yes |
| **Predictions** | $P(x^*\|\hat{\theta})$ | $P(x^*\|\hat{\theta})$ | $\int P(x^*\|\theta)P(\theta\|D)d\theta$ |
| **Computation** | Easy | Easy | Often intractable |
| **Approximations** | N/A | N/A | VI, MCMC, Laplace |

---

## 🔗 Where Bayesian Methods Are Used

| Application | How Bayesian Is Used |
|-------------|---------------------|
| **Gaussian Processes** | Prior over functions → posterior predictions |
| **Bayesian Optimization** | Uncertainty-guided exploration |
| **Bayesian Neural Networks** | Uncertainty in predictions |
| **A/B Testing** | Posterior probability of improvement |
| **Recommendation Systems** | Thompson sampling |
| **Medical Diagnosis** | Updating disease probability with tests |
| **Kalman Filter** | Sequential Bayesian update for state estimation |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Pattern Recognition (Bishop) | [PRML Ch. 1-2](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | Bayesian Data Analysis | Gelman et al. |
| 🎥 | Bayesian Statistics | [StatQuest](https://www.youtube.com/watch?v=3OJEae7Qb_o) |
| 📄 | Variational Inference | [arXiv](https://arxiv.org/abs/1601.00670) |
| 🇨🇳 | 贝叶斯推断详解 | [知乎](https://zhuanlan.zhihu.com/p/26262151) |
| 🇨🇳 | 共轭先验完全指南 | [CSDN](https://blog.csdn.net/qq_39388410/article/details/78606882) |
| 🇨🇳 | 变分推断教程 | [机器之心](https://www.jiqizhixin.com/articles/2018-01-25-5) |

---

⬅️ [Back: Estimation](../) | ➡️ [Next: MAP](../02_map/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
