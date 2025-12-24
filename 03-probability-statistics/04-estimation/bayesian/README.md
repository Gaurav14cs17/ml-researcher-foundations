<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Bayesian&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Bayesian Estimation

> **Full posterior inference over parameters**

---

## 🎯 Visual Overview

<img src="./images/bayesian-estimation.svg" width="100%">

*Caption: Compute full posterior P(θ|D) ∝ P(D|θ)P(θ). Predictions integrate over θ: P(x*|D) = ∫P(x*|θ)P(θ|D)dθ. Provides uncertainty estimates. Often intractable → approximations (VI, MCMC).*

---

## 📂 Overview

Bayesian inference provides principled uncertainty quantification - crucial for safety-critical applications. While exact inference is often intractable, variational and sampling methods make it practical.

---

## 📐 Mathematical Foundation

### Bayes' Theorem

```
P(θ|D) = P(D|θ) P(θ) / P(D)

Where:
    P(θ|D):  Posterior - what we want to know
    P(D|θ):  Likelihood - how data explains parameters
    P(θ):    Prior - our beliefs before seeing data
    P(D):    Evidence (marginal likelihood) - normalizing constant
```

### The Posterior Decomposition

```
log P(θ|D) = log P(D|θ) + log P(θ) - log P(D)
             +- Likelihood  +- Prior  +- Constant
             
Posterior ∝ Likelihood × Prior
```

### Predictive Distribution

```
For new data point x*:

P(x*|D) = ∫ P(x*|θ) P(θ|D) dθ
          +-----------------------------+
          Average over all possible θ!

This naturally includes uncertainty about θ
```

---

## 📊 MLE vs MAP vs Full Bayesian

| Method | Formula | Output | Uncertainty? |
|--------|---------|--------|--------------|
| **MLE** | argmax P(D\|θ) | Point θ̂ | No |
| **MAP** | argmax P(θ\|D) | Point θ̂ | No |
| **Bayesian** | P(θ\|D) | Distribution | Yes! |

```
              +----------------------------------+
              |         Full Posterior           |
              |                                  |
P(θ|D)        |         ╱╲    ← Uncertainty     |
              |        ╱  ╲                      |
              |       ╱    ╲                     |
              |      ╱      ╲                    |
              +------+-------+------------------+
                     θ_MAP   θ_MLE
                     
MLE/MAP: Single point estimate
Bayesian: Full distribution → confidence intervals!
```

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **Prior** | Encodes domain knowledge before data |
| **Posterior** | Updated beliefs after seeing data |
| **Conjugate Prior** | Prior & posterior same family |
| **Credible Interval** | Bayesian confidence interval |
| **Posterior Predictive** | Predictions with uncertainty |

---

## 📐 Common Conjugate Pairs

| Likelihood | Conjugate Prior | Posterior |
|------------|-----------------|-----------|
| Gaussian (mean) | Gaussian | Gaussian |
| Gaussian (variance) | Inverse-Gamma | Inverse-Gamma |
| Bernoulli | Beta | Beta |
| Poisson | Gamma | Gamma |
| Multinomial | Dirichlet | Dirichlet |

### Example: Beta-Bernoulli

```
Prior:     θ ~ Beta(α, β)
Likelihood: x₁,...,xₙ ~ Bernoulli(θ)

Let k = Σxᵢ (number of successes)

Posterior: θ|D ~ Beta(α + k, β + n - k)

Prior parameters act as "pseudo-observations":
    α = prior successes
    β = prior failures
```

---

## 💻 Code Examples

### Conjugate Bayesian Update

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def bayesian_coin_flip(data, prior_alpha=1, prior_beta=1):
    """
    Bayesian inference for coin flip (Beta-Bernoulli)
    """
    # Prior
    prior = stats.beta(prior_alpha, prior_beta)
    
    # Count successes
    n_heads = sum(data)
    n_tails = len(data) - n_heads
    
    # Posterior (conjugate update)
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_tails
    posterior = stats.beta(post_alpha, post_beta)
    
    # Plot
    theta = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 6))
    plt.plot(theta, prior.pdf(theta), 'b--', label='Prior')
    plt.plot(theta, posterior.pdf(theta), 'r-', label='Posterior')
    plt.axvline(n_heads/len(data), color='g', linestyle=':', label='MLE')
    plt.xlabel('θ (probability of heads)')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'After {len(data)} flips: {n_heads} heads')
    
    # Credible interval
    ci = posterior.interval(0.95)
    print(f"95% Credible Interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    return posterior

# Example usage
data = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]  # 7 heads, 3 tails
posterior = bayesian_coin_flip(data)
```

### Variational Inference (Approximation)

```python
import torch
import torch.nn as nn
import torch.distributions as dist

class VariationalBayes(nn.Module):
    """Simple mean-field variational inference"""
    
    def __init__(self, dim):
        super().__init__()
        # Variational parameters (mean, log-variance)
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_var = nn.Parameter(torch.zeros(dim))
    
    def q_theta(self):
        """Approximate posterior q(θ)"""
        std = torch.exp(0.5 * self.log_var)
        return dist.Normal(self.mu, std)
    
    def elbo(self, x, log_likelihood_fn, prior):
        """Evidence Lower Bound"""
        q = self.q_theta()
        
        # Sample from q(θ)
        theta = q.rsample()
        
        # ELBO = E_q[log p(x|θ)] - KL(q(θ) || p(θ))
        log_lik = log_likelihood_fn(x, theta)
        kl = dist.kl_divergence(q, prior).sum()
        
        return log_lik - kl
    
    def fit(self, x, log_likelihood_fn, prior, n_iter=1000, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = -self.elbo(x, log_likelihood_fn, prior)
            loss.backward()
            optimizer.step()
        
        return self.mu.detach(), torch.exp(0.5 * self.log_var).detach()
```

### MCMC Sampling (PyMC3)

```python
import pymc3 as pm
import numpy as np

# Example: Bayesian linear regression
def bayesian_linear_regression(X, y):
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10, shape=X.shape[1])
        sigma = pm.HalfNormal('sigma', sd=1)
        
        # Likelihood
        mu = alpha + pm.math.dot(X, beta)
        y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)
        
        # Sample from posterior
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    
    return trace

# Predictions with uncertainty
def predict_with_uncertainty(trace, X_new):
    alpha = trace.posterior['alpha'].values.flatten()
    beta = trace.posterior['beta'].values
    
    # Generate predictions for each sample
    predictions = []
    for i in range(len(alpha)):
        pred = alpha[i] + np.dot(X_new, beta[i])
        predictions.append(pred)
    
    predictions = np.array(predictions)
    return predictions.mean(axis=0), predictions.std(axis=0)
```

---

## 📊 Bayesian Neural Networks

```
Standard NN:
    y = f(x; θ)         ← Point estimate θ

Bayesian NN:
    P(y|x,D) = ∫ P(y|x,θ) P(θ|D) dθ    ← Integrate over θ

Approximations:
    1. Monte Carlo Dropout (simple!)
    2. Variational Inference
    3. MCMC / HMC
    4. Laplace Approximation
```

---

## 🔗 Connection to Other Topics

```
Bayesian Inference
    +-- Conjugate Priors (exact inference)
    +-- MCMC (sampling)
    |   +-- Metropolis-Hastings
    |   +-- Hamiltonian Monte Carlo
    +-- Variational Inference (optimization)
    |   +-- Mean-field VI
    |   +-- VAE (amortized)
    +-- Applications
        +-- Gaussian Processes
        +-- Bayesian Optimization
        +-- Bayesian Neural Networks
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | MLE | [../mle/](../mle/) |
| 📖 | MAP Estimation | [../map/](../map/) |
| 📖 | Information Theory | [../../information-theory/](../../information-theory/) |
| 📄 | Bishop PRML Ch. 1-2 | [PRML Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🇨🇳 | 贝叶斯推断详解 | [知乎](https://zhuanlan.zhihu.com/p/26262151) |
| 🇨🇳 | 共轭先验完全指南 | [CSDN](https://blog.csdn.net/qq_39388410/article/details/78606882) |
| 🇨🇳 | 贝叶斯统计入门 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |
| 🇨🇳 | 变分推断教程 | [机器之心](https://www.jiqizhixin.com/articles/2018-01-25-5) |
| 🇨🇳 | 贝叶斯神经网络 | [PaperWeekly](https://www.paperweekly.site/papers/notes/386)


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Estimation](../)

---

➡️ [Next: Map](../map/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
