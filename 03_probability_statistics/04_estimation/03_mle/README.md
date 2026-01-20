<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Maximum%20Likelihood%20Estimation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/mle.svg" width="100%">

*Caption: MLE finds the parameter θ that makes observed data most probable. Training neural networks with cross-entropy = MLE!*

---

## 📐 Mathematical Definition

### The Likelihood Function

Given data $D = \{x\_1, x\_2, \ldots, x\_n\}$ and model $P(x|\theta)$:

$$
L(\theta) = P(D|\theta) = \prod_{i=1}^{n} P(x_i|\theta)
$$

### Maximum Likelihood Estimator

$$
\theta_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta P(D|\theta)
$$

### Log-Likelihood (More Practical)

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log P(x_i|\theta)
\theta_{MLE} = \arg\max_\theta \ell(\theta)
$$

**Why log?**
1. Products → Sums (numerical stability)
2. Avoids underflow for large n
3. Same argmax (log is monotonic)
4. Simpler gradients

---

## 📐 MLE Derivations

### Example 1: Gaussian Distribution

**Setup:** $x\_1, \ldots, x\_n \sim \mathcal{N}(\mu, \sigma^2)$

**Log-likelihood:**

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
$$

**Finding $\mu\_{MLE}$:**

$$
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0
\sum_{i=1}^n x_i = n\mu
\boxed{\mu_{MLE} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}} \quad \blacksquare
$$

**Finding $\sigma^2\_{MLE}$:**

$$
\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \mu)^2 = 0
\boxed{\sigma^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2} \quad \blacksquare
$$

**Note:** This is the biased estimator. Unbiased: divide by $(n-1)$.

---

### Example 2: Bernoulli Distribution

**Setup:** $x\_1, \ldots, x\_n \sim \text{Bernoulli}(p)$, where $x\_i \in \{0, 1\}$

Let $k = \sum\_{i=1}^n x\_i$ (number of successes)

**Likelihood:**

$$
L(p) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i} = p^k(1-p)^{n-k}
$$

**Log-likelihood:**

$$
\ell(p) = k\log p + (n-k)\log(1-p)
$$

**Finding $p\_{MLE}$:**

$$
\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0
k(1-p) = (n-k)p
k = np
\boxed{p_{MLE} = \frac{k}{n} = \frac{\sum_i x_i}{n}} \quad \blacksquare
$$

---

### Example 3: Linear Regression = MLE

**Model:** $y\_i = \mathbf{w}^T\mathbf{x}\_i + \epsilon\_i$, where $\epsilon\_i \sim \mathcal{N}(0, \sigma^2)$

**Likelihood:**

$$
P(y_i|\mathbf{x}_i, \mathbf{w}, \sigma^2) = \mathcal{N}(y_i; \mathbf{w}^T\mathbf{x}_i, \sigma^2)
$$

**Log-likelihood:**

$$
\ell(\mathbf{w}) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mathbf{w}^T\mathbf{x}_i)^2
$$

**MLE:**

$$
\mathbf{w}_{MLE} = \arg\max_\mathbf{w} \ell(\mathbf{w}) = \arg\min_\mathbf{w} \sum_{i=1}^n (y_i - \mathbf{w}^T\mathbf{x}_i)^2
\boxed{\text{Maximum Likelihood} = \text{Least Squares!}} \quad \blacksquare
$$

**Closed form:**

$$
\mathbf{w}_{MLE} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

---

### Example 4: Classification = MLE

**Model:** $P(y=1|x) = \sigma(\mathbf{w}^T\mathbf{x})$ where $\sigma(z) = \frac{1}{1+e^{-z}}$

**Log-likelihood:**

$$
\ell(\mathbf{w}) = \sum_{i=1}^n \left[y_i \log \sigma(\mathbf{w}^T\mathbf{x}_i) + (1-y_i)\log(1-\sigma(\mathbf{w}^T\mathbf{x}_i))\right]
= -\sum_{i=1}^n \text{BCE}(y_i, \hat{y}_i)
\boxed{\text{Maximizing Log-Likelihood} = \text{Minimizing Cross-Entropy!}} \quad \blacksquare
$$

---

## 📐 Properties of MLE

### 1. Consistency

$$
\hat{\theta}_{MLE} \xrightarrow{p} \theta_{true} \quad \text{as } n \to \infty
$$

**MLE converges to the true parameter as we get more data.**

### 2. Asymptotic Normality

$$
\sqrt{n}(\hat{\theta}_{MLE} - \theta_{true}) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})
$$

**MLE is approximately Gaussian for large n, with variance determined by Fisher Information.**

### 3. Efficiency (Cramér-Rao Bound)

For any unbiased estimator $\hat{\theta}$:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

**MLE achieves this lower bound asymptotically → most efficient!**

### 4. Invariance

If $\hat{\theta}$ is MLE of $\theta$, then $g(\hat{\theta})$ is MLE of $g(\theta)$.

**Example:** If $\hat{\mu}, \hat{\sigma}$ are MLEs, then $\hat{\sigma}^2$ is MLE of variance.

---

## 📐 Fisher Information

### Definition

$$
I(\theta) = -E\left[\frac{\partial^2 \log P(X|\theta)}{\partial\theta^2}\right] = E\left[\left(\frac{\partial \log P(X|\theta)}{\partial\theta}\right)^2\right]
$$

### Interpretation

- **Curvature** of log-likelihood at maximum
- High $I(\theta)$ → sharp peak → low variance of MLE
- Low $I(\theta)$ → flat peak → high variance of MLE

### Fisher Information for Common Distributions

| Distribution | Fisher Information |
|--------------|-------------------|
| Bernoulli($p$) | $\frac{1}{p(1-p)}$ |
| Gaussian($\mu$, known $\sigma^2$) | $\frac{n}{\sigma^2}$ |
| Poisson($\lambda$) | $\frac{n}{\lambda}$ |
| Exponential($\lambda$) | $\frac{n}{\lambda^2}$ |

### Example: Bernoulli Fisher Information (Proof)

$$
\log P(x|\theta) = x\log\theta + (1-x)\log(1-\theta)
\frac{\partial \log P}{\partial\theta} = \frac{x}{\theta} - \frac{1-x}{1-\theta}
\frac{\partial^2 \log P}{\partial\theta^2} = -\frac{x}{\theta^2} - \frac{1-x}{(1-\theta)^2}
I(\theta) = -E\left[-\frac{x}{\theta^2} - \frac{1-x}{(1-\theta)^2}\right] = \frac{\theta}{\theta^2} + \frac{1-\theta}{(1-\theta)^2}
= \frac{1}{\theta} + \frac{1}{1-\theta} = \frac{1}{\theta(1-\theta)} \quad \blacksquare
$$

---

## 📐 Connection to Neural Network Training

### Cross-Entropy Loss = Negative Log-Likelihood

$$
\mathcal{L}_{CE} = -\sum_{i=1}^n \log P(y_i|x_i; \theta)
$$

### Training = MLE

$$
\theta^* = \arg\min_\theta \mathcal{L}_{CE} = \arg\max_\theta \sum_i \log P(y_i|x_i; \theta)
$$

| Loss Function | Assumed Distribution | MLE Connection |
|---------------|---------------------|----------------|
| MSE | Gaussian $\mathcal{N}(f(x), \sigma^2)$ | MLE for mean |
| Cross-Entropy | Categorical | MLE for class probs |
| Binary CE | Bernoulli | MLE for probability |

---

## 💻 Code Examples

### MLE for Gaussian (Analytical)

```python
import numpy as np

def gaussian_mle(data):
    """
    Closed-form MLE for Gaussian distribution
    
    μ_MLE = sample mean
    σ²_MLE = sample variance (biased)
    """
    n = len(data)
    mu_mle = np.mean(data)
    sigma2_mle = np.sum((data - mu_mle)**2) / n  # Biased
    sigma2_unbiased = np.sum((data - mu_mle)**2) / (n - 1)  # Unbiased
    
    return mu_mle, np.sqrt(sigma2_mle), np.sqrt(sigma2_unbiased)

# Generate data
np.random.seed(42)
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, 1000)

mu_mle, sigma_mle, sigma_unbiased = gaussian_mle(data)
print(f"True: μ={true_mu}, σ={true_sigma}")
print(f"MLE:  μ={mu_mle:.4f}, σ={sigma_mle:.4f}")
print(f"Unbiased σ: {sigma_unbiased:.4f}")
```

### MLE via Optimization

```python
import numpy as np
from scipy.optimize import minimize
from scipy import stats

def negative_log_likelihood(params, data, distribution):
    """General NLL for optimization"""
    if distribution == 'gaussian':
        mu, sigma = params
        if sigma <= 0:
            return 1e10
        return -np.sum(stats.norm.logpdf(data, mu, sigma))
    
    elif distribution == 'exponential':
        lam = params[0]
        if lam <= 0:
            return 1e10
        return -np.sum(stats.expon.logpdf(data, scale=1/lam))
    
    elif distribution == 'poisson':
        lam = params[0]
        if lam <= 0:
            return 1e10
        return -np.sum(stats.poisson.logpmf(data.astype(int), lam))

# Example: Exponential MLE
data_exp = np.random.exponential(scale=2.0, size=500)  # True λ = 0.5

result = minimize(
    negative_log_likelihood, 
    x0=[1.0], 
    args=(data_exp, 'exponential'),
    method='Nelder-Mead'
)
lambda_mle = result.x[0]
print(f"True λ: 0.5, MLE λ: {lambda_mle:.4f}")
```

### MLE = Neural Network Training

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def mle_training(model, X, y, n_epochs=1000, lr=0.01):
    """
    Training = MLE
    
    Minimize NLL = Minimize Cross-Entropy = Maximize Likelihood
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        logits = model(X).squeeze()
        
        # Negative log-likelihood (Binary Cross-Entropy)
        nll = F.binary_cross_entropy_with_logits(logits, y)
        
        nll.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            prob = torch.sigmoid(logits)
            acc = ((prob > 0.5) == y).float().mean()
            print(f"Epoch {epoch+1}: NLL = {nll.item():.4f}, Acc = {acc:.4f}")
    
    return model

# Example
torch.manual_seed(42)
n, d = 200, 5
X = torch.randn(n, d)
true_w = torch.tensor([1.0, -2.0, 0.5, 0.0, 1.5])
y = (X @ true_w > 0).float()

model = LogisticRegression(d)
model = mle_training(model, X, y)
print(f"\nTrue weights: {true_w}")
print(f"MLE weights: {model.linear.weight.data.squeeze()}")
```

### Fisher Information & Confidence Intervals

```python
import numpy as np
from scipy import stats

def mle_with_confidence(data):
    """
    MLE with asymptotic confidence intervals
    
    Using Fisher Information: Var(θ̂) ≈ 1/I(θ)
    """
    n = len(data)
    
    # MLE
    mu_mle = np.mean(data)
    sigma_mle = np.std(data, ddof=0)
    
    # Fisher Information for Gaussian mean
    # I(μ) = n/σ²
    fisher_info = n / sigma_mle**2
    
    # Standard error of MLE
    se = 1 / np.sqrt(fisher_info)
    
    # 95% CI using asymptotic normality
    z_95 = 1.96
    ci_lower = mu_mle - z_95 * se
    ci_upper = mu_mle + z_95 * se
    
    print(f"MLE: μ = {mu_mle:.4f}")
    print(f"Fisher Information: I(μ) = {fisher_info:.4f}")
    print(f"Standard Error: {se:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return mu_mle, (ci_lower, ci_upper)

# Example
data = np.random.normal(5, 2, 100)
mle_with_confidence(data)
```

---

## ⚠️ Limitations of MLE

| Issue | Description | Solution |
|-------|-------------|----------|
| **Overfitting** | Perfect fit to training data | Regularization (→ MAP) |
| **Unbounded likelihood** | e.g., σ → 0 when fitting single point | Priors |
| **Local optima** | Non-convex likelihood | Multiple restarts |
| **Small sample bias** | MLE of σ² is biased | Bias correction |
| **Model misspecification** | Wrong model family | Model selection |

---

## 📊 MLE vs MAP vs Bayesian

| Aspect | MLE | MAP | Bayesian |
|--------|-----|-----|----------|
| **Formula** | $\arg\max P(D\|\theta)$ | $\arg\max P(\theta\|D)$ | $P(\theta\|D)$ |
| **Prior** | ❌ | ✅ | ✅ |
| **Output** | Point $\hat{\theta}$ | Point $\hat{\theta}$ | Distribution |
| **Regularization** | None | Implicit | Implicit |
| **Uncertainty** | Via Fisher Info | ❌ | ✅ Natural |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 2 | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | Statistical Inference | Casella & Berger |
| 🎥 | StatQuest: MLE | [YouTube](https://www.youtube.com/watch?v=XepXtl9YKwc) |
| 📄 | MLE Theory | All of Statistics (Wasserman) |
| 🇨🇳 | 最大似然估计详解 | [知乎](https://zhuanlan.zhihu.com/p/26614750) |
| 🇨🇳 | MLE与交叉熵的关系 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88393322) |
| 🇨🇳 | Fisher信息量 | [知乎](https://zhuanlan.zhihu.com/p/85454091) |

---

⬅️ [Back: MAP](../02_map/) | ➡️ [Back: Estimation](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
