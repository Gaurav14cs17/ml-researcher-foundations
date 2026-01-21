<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Statistical%20Estimation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/estimation.svg" width="100%">

*Caption: Statistical estimation infers unknown parameters θ from data. MLE maximizes likelihood, MAP adds prior (= regularization), Bayesian computes full posterior for uncertainty quantification.*

---

## 📂 Topics in This Folder

| Folder | Topic | Key Idea |
|--------|-------|----------|
| [01_bayesian/](./01_bayesian/) | Bayesian Estimation | Full $P(\theta\|D)$ |
| [02_map/](./02_map/) | Maximum A Posteriori | $\arg\max P(\theta\|D)$ |
| [03_mle/](./03_mle/) | Maximum Likelihood | $\arg\max P(D\|\theta)$ |

---

## 📐 Mathematical Framework

### Maximum Likelihood Estimation (MLE)

$$\theta_{MLE} = \arg\max_\theta P(D|\theta) = \arg\max_\theta \prod_{i=1}^{n} P(x_i|\theta)$$

**Log-likelihood (for numerical stability):**

$$\theta_{MLE} = \arg\max_\theta \ell(\theta) = \arg\max_\theta \sum_{i=1}^{n} \log P(x_i|\theta)$$

### Maximum A Posteriori (MAP)

$$\theta_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta P(D|\theta) P(\theta)$$

**Log form:**

$$\theta_{MAP} = \arg\max_\theta \left[\sum_{i=1}^{n} \log P(x_i|\theta) + \log P(\theta)\right]$$

### Full Bayesian

$$P(\theta|D) = \frac{P(D|\theta) P(\theta)}{\int P(D|\theta') P(\theta') d\theta'}$$

**Predictive distribution:**

$$P(x_{new}|D) = \int P(x_{new}|\theta) P(\theta|D) d\theta$$

---

## 📐 Common MLE Examples

### Gaussian (Unknown Mean, Known Variance)

$$X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$$

**Log-likelihood:**

$$\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**MLE:**

$$\mu_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

### Bernoulli

$$X_1, \ldots, X_n \sim \text{Bernoulli}(p)$$

**Log-likelihood:**

$$\ell(p) = k \log p + (n-k) \log(1-p)$$

where $k = \sum_{i=1}^{n} x_i$.

**MLE:**

$$p_{MLE} = \frac{k}{n} = \frac{\sum_i x_i}{n}$$

---

## 📐 Regularization = Prior

### L2 Regularization = Gaussian Prior

$$P(\theta) = \mathcal{N}(0, \sigma^2 I)
\log P(\theta) = -\frac{\|\theta\|^2}{2\sigma^2} + \text{const}$$

**MAP objective:**

$$\arg\max_\theta \left[\sum_i \log P(x_i|\theta) - \lambda\|\theta\|^2\right]$$

where $\lambda = \frac{1}{2\sigma^2}$. This is **Ridge Regression / Weight Decay**.

### L1 Regularization = Laplace Prior

$$P(\theta) \propto \exp(-|\theta|/b)$$

**MAP objective:**

$$\arg\max_\theta \left[\sum_i \log P(x_i|\theta) - \lambda\|\theta\|_1\right]$$

This is **Lasso Regression** → promotes sparsity.

---

## 📊 Comparison: MLE vs MAP vs Bayesian

| Aspect | MLE | MAP | Bayesian |
|--------|-----|-----|----------|
| **Output** | Point $\hat{\theta}$ | Point $\hat{\theta}$ | Distribution $P(\theta\|D)$ |
| **Formula** | $\arg\max P(D\|\theta)$ | $\arg\max P(D\|\theta)P(\theta)$ | $P(\theta\|D)$ |
| **Prior** | No | Yes | Yes |
| **Uncertainty** | No | No | Yes |
| **Computation** | Easy | Easy | Often hard |
| **Regularization** | None | Implicit | Implicit |
| **Overfitting** | Prone | Reduced | Most robust |

---

## 📐 MLE = Neural Network Training

**Cross-entropy loss = Negative log-likelihood:**

$$\mathcal{L} = -\sum_{i=1}^{n} \log P(y_i|x_i; \theta)$$

**Minimizing cross-entropy = Maximizing likelihood = MLE!**

```
Classification:
L = -Σ yᵢ log p(yᵢ|xᵢ;θ)  = Cross-entropy
θ* = argmin L = argmax Σ log p(yᵢ|xᵢ;θ) = MLE

Regression (MSE):
L = Σ (yᵢ - f(xᵢ;θ))²  ∝ -log P(D|θ) for Gaussian noise

```

---

## 💻 Code Examples

```python
import numpy as np
import torch
from scipy.optimize import minimize

# MLE for Gaussian
def gaussian_mle(data):
    """Closed form MLE for Gaussian"""
    mu_mle = np.mean(data)
    sigma_mle = np.std(data, ddof=0)  # Biased estimator
    return mu_mle, sigma_mle

# MAP with Gaussian prior (Ridge)
def map_ridge(X, y, lambda_reg=1.0):
    """MAP = Ridge regression (closed form)"""
    n, d = X.shape
    return np.linalg.solve(X.T @ X + lambda_reg * np.eye(d), X.T @ y)

# Weight decay in PyTorch = MAP with Gaussian prior
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=0.01  # λ for L2 regularization
)

# Bayesian with conjugate prior (Beta-Bernoulli)
def bayesian_coin_flip(data, prior_alpha=1, prior_beta=1):
    """
    Conjugate Bayesian update
    Prior: Beta(α, β)
    Posterior: Beta(α + heads, β + tails)
    """
    heads = sum(data)
    tails = len(data) - heads
    
    post_alpha = prior_alpha + heads
    post_beta = prior_beta + tails
    
    # Posterior mean
    mean = post_alpha / (post_alpha + post_beta)
    
    # 95% credible interval
    from scipy import stats
    ci = stats.beta(post_alpha, post_beta).interval(0.95)
    
    return mean, ci

```

---

## 📐 Fisher Information

$$I(\theta) = -E\left[\frac{\partial^2 \log P(X|\theta)}{\partial \theta^2}\right]$$

**Properties:**
- Measures "curvature" of log-likelihood

- Higher $I(\theta)$ → more information about $\theta$

- $\text{Var}(\hat{\theta}_{MLE}) \approx \frac{1}{I(\theta)}$ (asymptotically)

**Cramér-Rao Lower Bound:**

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

MLE achieves this bound asymptotically → **efficient estimator**.

---

## 🔗 Where Estimation Is Used

| Application | Estimation Method |
|-------------|-------------------|
| **Neural Network Training** | MLE (cross-entropy, MSE) |
| **Ridge Regression** | MAP with Gaussian prior |
| **Lasso Regression** | MAP with Laplace prior |
| **Bayesian Neural Networks** | Full Bayesian |
| **Gaussian Processes** | Full Bayesian |
| **Variational Autoencoders** | Approximate Bayesian (VI) |
| **Dropout** | Approximate Bayesian |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Statistical Inference | Casella & Berger |
| 📖 | Pattern Recognition (Bishop) | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🎥 | StatQuest: MLE/MAP | [YouTube](https://www.youtube.com/watch?v=XepXtl9YKwc) |
| 🇨🇳 | 极大似然估计 | [知乎](https://zhuanlan.zhihu.com/p/26614750) |
| 🇨🇳 | MLE与MAP对比 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 贝叶斯估计入门 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

⬅️ [Back: Information Theory](../03_information_theory/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
