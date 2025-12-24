# Maximum A Posteriori Estimation

> **MLE with prior beliefs**

---

## 🎯 Visual Overview

<img src="./images/map.svg" width="100%">

*Caption: θ_MAP = argmax_θ P(θ|D) ∝ P(D|θ)P(θ). Prior P(θ) encodes beliefs before seeing data. L2 regularization = Gaussian prior; L1 = Laplace prior. More robust than MLE with limited data.*

---

## 📂 Overview

MAP bridges MLE and full Bayesian inference. It shows that regularization is equivalent to placing a prior on parameters - L2 assumes parameters are normally distributed around zero.

---

## 📐 Mathematical Definition

### MAP Objective

```
θ_MAP = argmax_θ P(θ|D)
      = argmax_θ P(D|θ) P(θ) / P(D)
      = argmax_θ P(D|θ) P(θ)        (P(D) is constant)
      = argmax_θ [log P(D|θ) + log P(θ)]
```

### Comparison with MLE

```
MLE: θ_MLE = argmax_θ P(D|θ)           ← Likelihood only
MAP: θ_MAP = argmax_θ P(D|θ) P(θ)      ← Likelihood × Prior

When P(θ) is uniform (uninformative): MAP = MLE
When data is abundant: MAP → MLE (data dominates)
When data is limited: Prior matters more
```

---

## 🔑 Regularization as Prior

### L2 Regularization = Gaussian Prior

```
Prior: P(θ) = N(0, σ²I)
log P(θ) = -||θ||² / (2σ²) + const

MAP objective:
    argmax [log P(D|θ) - λ||θ||²]
    
Where λ = 1/(2σ²)

This is Ridge Regression / Weight Decay!
```

### L1 Regularization = Laplace Prior

```
Prior: P(θ) ∝ exp(-|θ|/b)
log P(θ) = -|θ|/b + const

MAP objective:
    argmax [log P(D|θ) - λ||θ||₁]
    
This is Lasso Regression!
Promotes sparsity (some θᵢ = 0)
```

---

## 📊 Prior Effects

| Prior Type | Formula | Effect | Use Case |
|------------|---------|--------|----------|
| **Gaussian** | N(0, σ²) | Shrinks towards 0 | Ridge, weight decay |
| **Laplace** | Laplace(0, b) | Sparse solutions | Lasso, feature selection |
| **Spike-and-slab** | π·δ(0) + (1-π)·N | Exact sparsity | Variable selection |
| **Horseshoe** | Half-Cauchy | Heavy-tailed, sparse | Bayesian sparse |

---

## 💻 Code Examples

### MAP with Gaussian Prior (L2)

```python
import numpy as np
from scipy.optimize import minimize

def map_linear_regression(X, y, lambda_reg=1.0):
    """
    MAP estimation for linear regression with Gaussian prior
    Equivalent to Ridge Regression
    """
    n, d = X.shape
    
    def neg_log_posterior(w):
        # Negative log-likelihood
        residuals = y - X @ w
        nll = 0.5 * np.sum(residuals ** 2)
        
        # Negative log-prior (Gaussian)
        neg_log_prior = 0.5 * lambda_reg * np.sum(w ** 2)
        
        return nll + neg_log_prior
    
    def gradient(w):
        grad_nll = -X.T @ (y - X @ w)
        grad_prior = lambda_reg * w
        return grad_nll + grad_prior
    
    w_init = np.zeros(d)
    result = minimize(neg_log_posterior, w_init, jac=gradient, method='L-BFGS-B')
    
    return result.x

# Closed form solution (same as Ridge)
def ridge_closed_form(X, y, lambda_reg):
    n, d = X.shape
    return np.linalg.solve(X.T @ X + lambda_reg * np.eye(d), X.T @ y)
```

### MAP with Laplace Prior (L1)

```python
from sklearn.linear_model import Lasso

def map_lasso(X, y, alpha=1.0):
    """
    MAP estimation with Laplace prior = Lasso
    """
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model.coef_

# From scratch with proximal gradient
def proximal_l1(w, alpha):
    """Soft thresholding (proximal operator for L1)"""
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)

def map_lasso_ista(X, y, lambda_reg, lr=0.01, n_iter=1000):
    """ISTA algorithm for Lasso"""
    n, d = X.shape
    w = np.zeros(d)
    
    for _ in range(n_iter):
        # Gradient of log-likelihood
        grad = -X.T @ (y - X @ w)
        
        # Gradient step
        w = w - lr * grad
        
        # Proximal step (soft thresholding)
        w = proximal_l1(w, lr * lambda_reg)
    
    return w
```

### PyTorch with L2 Prior (Weight Decay)

```python
import torch
import torch.nn as nn

# Weight decay in optimizer = L2 prior
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=0.01  # λ for L2 regularization
)

# This is equivalent to:
def training_step_explicit_l2(model, X, y, lambda_reg=0.01):
    pred = model(X)
    loss = F.mse_loss(pred, y)
    
    # Add L2 penalty (Gaussian prior)
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    
    total_loss = loss + lambda_reg * l2_penalty
    return total_loss
```

---

## 🔗 Comparison: MLE vs MAP vs Bayesian

```
              Data Only          + Prior             Full Posterior
                 ↓                  ↓                      ↓
MLE ------------------> MAP ------------------> Bayesian
argmax P(D|θ)        argmax P(θ|D)           P(θ|D) = ∫...

Point estimate       Point estimate          Distribution
No uncertainty       No uncertainty          Full uncertainty
Can overfit          Regularized             Most robust
```

| Aspect | MLE | MAP | Bayesian |
|--------|-----|-----|----------|
| Output | Point θ̂ | Point θ̂ | Distribution P(θ\|D) |
| Prior | No | Yes | Yes |
| Uncertainty | No | No | Yes |
| Computation | Easy | Easy | Hard (usually) |
| Regularization | None | Implicit | Implicit |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | MLE | [../mle/](../mle/) |
| 📖 | Bayesian Estimation | [../bayesian/](../bayesian/) |
| 📖 | Regularization | [../../../ml-theory/generalization/regularization/](../../../ml-theory/generalization/regularization/) |
| 📄 | Bishop PRML Ch. 3 | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🇨🇳 | MAP与正则化的关系 | [知乎](https://zhuanlan.zhihu.com/p/32480810) |
| 🇨🇳 | MLE、MAP与贝叶斯对比 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88716548) |
| 🇨🇳 | 最大后验估计 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |
| 🇨🇳 | L1/L2正则化的先验解释 | [机器之心](https://www.jiqizhixin.com/articles/2018-11-20-6)


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Estimation](../)

---

⬅️ [Back: Bayesian](../bayesian/) | ➡️ [Next: Mle](../mle/)
