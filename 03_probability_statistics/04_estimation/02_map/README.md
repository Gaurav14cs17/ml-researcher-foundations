<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Maximum%20A%20Posteriori%20(MAP)&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/map.svg" width="100%">

*Caption: MAP finds the mode of the posterior. Prior acts as regularization: L2 = Gaussian prior, L1 = Laplace prior.*

---

## 📐 Mathematical Definition

### MAP Objective

```math
\theta_{MAP} = \arg\max_\theta P(\theta|D)
```

Using Bayes' theorem:

```math
\theta_{MAP} = \arg\max_\theta \frac{P(D|\theta) \cdot P(\theta)}{P(D)}
```

Since $P(D)$ doesn't depend on $\theta$:

```math
\theta_{MAP} = \arg\max_\theta P(D|\theta) \cdot P(\theta)
```

**Log form (more practical):**

```math
\theta_{MAP} = \arg\max_\theta \left[\log P(D|\theta) + \log P(\theta)\right]
= \arg\max_\theta \left[\underbrace{\sum_{i=1}^n \log P(x_i|\theta)}_{\text{log-likelihood}} + \underbrace{\log P(\theta)}_{\text{log-prior}}\right]
```

---

## 📐 MAP vs MLE

```math
\theta_{MLE} = \arg\max_\theta P(D|\theta)
\theta_{MAP} = \arg\max_\theta P(D|\theta) \cdot P(\theta)
```

| Condition | MAP Behavior |
|-----------|--------------|
| Uniform prior $P(\theta) = c$ | MAP = MLE |
| Abundant data | MAP → MLE (likelihood dominates) |
| Limited data | Prior matters more |
| Strong prior | MAP ≈ Prior mean |

---

## 📐 Regularization = Prior (Proof)

### L2 Regularization = Gaussian Prior

**Prior:** $\theta \sim \mathcal{N}(0, \sigma\_p^2 I)$

```math
P(\theta) = \frac{1}{(2\pi\sigma_p^2)^{d/2}} \exp\left(-\frac{\|\theta\|^2}{2\sigma_p^2}\right)
\log P(\theta) = -\frac{\|\theta\|^2}{2\sigma_p^2} + \text{const}
```

**MAP objective:**

```math
\theta_{MAP} = \arg\max_\theta \left[\sum_i \log P(x_i|\theta) - \frac{\|\theta\|^2}{2\sigma_p^2}\right]
= \arg\min_\theta \left[\underbrace{-\sum_i \log P(x_i|\theta)}_{\text{NLL}} + \underbrace{\frac{1}{2\sigma_p^2}\|\theta\|^2}_{\lambda\|\theta\|^2}\right]
```

**where $\lambda = \frac{1}{2\sigma\_p^2}$**

**This is exactly Ridge Regression / Weight Decay!** $\quad \blacksquare$

---

### L1 Regularization = Laplace Prior

**Prior:** $\theta \sim \text{Laplace}(0, b)$

```math
P(\theta) = \frac{1}{2b} \exp\left(-\frac{|\theta|}{b}\right)
\log P(\theta) = -\frac{|\theta|}{b} + \text{const}
```

**MAP objective:**

```math
\theta_{MAP} = \arg\min_\theta \left[-\sum_i \log P(x_i|\theta) + \frac{1}{b}\|\theta\|_1\right]
```

**This is exactly Lasso Regression!** $\quad \blacksquare$

---

## 📐 Why L1 Promotes Sparsity (Proof)

For scalar $\theta$, consider MAP gradient at $\theta = 0$:

**L2 (Ridge):**

```math
\frac{\partial}{\partial\theta}(\lambda\theta^2) = 2\lambda\theta
```

At $\theta = 0$: gradient = 0. No force pushing to exactly 0.

**L1 (Lasso):**

```math
\frac{\partial}{\partial\theta}(\lambda|\theta|) = \lambda \cdot \text{sign}(\theta)
```

At $\theta = 0^+$: gradient = $\lambda > 0$ (pushes toward 0)  
At $\theta = 0^-$: gradient = $-\lambda < 0$ (pushes toward 0)

**Conclusion:** L1 has non-zero gradient at origin → pushes parameters to exactly zero → sparsity! $\quad \blacksquare$

---

## 📐 Prior Effects Comparison

| Prior | Distribution | log P(θ) | Effect |
|-------|-------------|----------|--------|
| **Gaussian** | $\mathcal{N}(0, \sigma^2)$ | $-\frac{\theta^2}{2\sigma^2}$ | Shrinks to 0, smooth |
| **Laplace** | $\text{Lap}(0, b)$ | $-\frac{\|\theta\|}{b}$ | Sparse solutions |
| **Student-t** | $t\_\nu(0, \sigma)$ | Heavy-tailed | Robust to outliers |
| **Horseshoe** | Half-Cauchy × N | Strong sparsity | Bayesian Lasso++ |
| **Spike-and-Slab** | $\pi \delta\_0 + (1-\pi)N$ | Exact sparsity | Variable selection |

### Visualization

```
Prior comparison (log scale):

           Laplace                    Gaussian
              |                          ╱╲
              |╲                        ╱  ╲
              | ╲                      ╱    ╲
              |  ╲                    ╱      ╲
              |   ╲                  ╱        ╲
              |    ╲                ╱          ╲
        ------+-----╲------    ---╱------------╲---
              0      θ           0              θ

Laplace: Sharp peak at 0 → promotes θ = 0
Gaussian: Smooth curve → shrinks but rarely exactly 0
```

---

## 📐 Closed-Form Solutions

### Ridge Regression (L2)

```math
\theta_{MAP} = \arg\min_\theta \|y - X\theta\|^2 + \lambda\|\theta\|^2
```

**Solution:**

```math
\theta_{MAP} = (X^TX + \lambda I)^{-1}X^Ty
```

**Proof:**

```math
\nabla_\theta\left[\|y - X\theta\|^2 + \lambda\|\theta\|^2\right] = -2X^T(y - X\theta) + 2\lambda\theta = 0
X^TX\theta + \lambda\theta = X^Ty
(X^TX + \lambda I)\theta = X^Ty
\theta = (X^TX + \lambda I)^{-1}X^Ty \quad \blacksquare
```

**Key insight:** Adding $\lambda I$ makes the matrix always invertible (regularization!).

---

### Bayesian Interpretation of Ridge Solution

The Ridge solution is the **posterior mean** for:
- Prior: $\theta \sim \mathcal{N}(0, \sigma\_p^2 I)$
- Likelihood: $y|X,\theta \sim \mathcal{N}(X\theta, \sigma^2 I)$

where $\lambda = \sigma^2/\sigma\_p^2$.

---

## 💻 Code Examples

### Ridge Regression (MAP with Gaussian Prior)

```python
import numpy as np
from scipy.optimize import minimize

def map_ridge(X, y, lambda_reg=1.0):
    """
    MAP estimation = Ridge Regression
    
    Prior: θ ~ N(0, σ²I) where λ = 1/(2σ²)
    Likelihood: y ~ N(Xθ, σ²I)
    
    Objective: min ||y - Xθ||² + λ||θ||²
    """
    n, d = X.shape
    
    # Closed-form solution
    theta_map = np.linalg.solve(
        X.T @ X + lambda_reg * np.eye(d), 
        X.T @ y
    )
    
    return theta_map

def map_ridge_gradient(X, y, lambda_reg=1.0):
    """Ridge via gradient descent (for understanding)"""
    n, d = X.shape
    theta = np.zeros(d)
    lr = 0.01
    
    for _ in range(1000):
        # Gradient of negative log-posterior
        grad_nll = -X.T @ (y - X @ theta)  # ∂(-log likelihood)/∂θ
        grad_prior = lambda_reg * theta     # ∂(-log prior)/∂θ = λθ
        grad = grad_nll + grad_prior
        
        theta = theta - lr * grad
    
    return theta

# Example
np.random.seed(42)
n, d = 100, 10
X = np.random.randn(n, d)
true_theta = np.random.randn(d)
y = X @ true_theta + 0.1 * np.random.randn(n)

theta_map = map_ridge(X, y, lambda_reg=0.1)
print(f"MAP estimate: {theta_map[:3]}")
print(f"True theta: {true_theta[:3]}")
```

### Lasso Regression (MAP with Laplace Prior)

```python
import numpy as np

def soft_threshold(x, alpha):
    """Proximal operator for L1 norm"""
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def map_lasso_ista(X, y, lambda_reg=1.0, lr=0.01, n_iter=1000):
    """
    MAP estimation = Lasso via ISTA
    
    Prior: θ ~ Laplace(0, b) where λ = 1/b
    
    ISTA = Iterative Shrinkage-Thresholding Algorithm
    """
    n, d = X.shape
    theta = np.zeros(d)
    
    for _ in range(n_iter):
        # Gradient step (likelihood)
        grad = -X.T @ (y - X @ theta) / n
        theta = theta - lr * grad
        
        # Proximal step (prior = soft thresholding)
        theta = soft_threshold(theta, lr * lambda_reg)
    
    return theta

def map_lasso_cd(X, y, lambda_reg=1.0, n_iter=100):
    """
    Lasso via Coordinate Descent (faster)
    """
    n, d = X.shape
    theta = np.zeros(d)
    
    for _ in range(n_iter):
        for j in range(d):
            # Compute residual without feature j
            residual = y - X @ theta + X[:, j] * theta[j]
            
            # Coordinate update
            rho = X[:, j] @ residual
            z = X[:, j] @ X[:, j]
            
            # Soft threshold
            theta[j] = soft_threshold(rho, lambda_reg * n) / z
    
    return theta

# Example: Sparse recovery
np.random.seed(42)
n, d = 100, 50
X = np.random.randn(n, d)
true_theta = np.zeros(d)
true_theta[:5] = [1, -2, 1.5, -0.5, 2]  # Only 5 non-zero
y = X @ true_theta + 0.1 * np.random.randn(n)

theta_lasso = map_lasso_cd(X, y, lambda_reg=0.1)
print(f"Non-zero coefficients: {np.sum(np.abs(theta_lasso) > 0.01)}")
print(f"True non-zero: 5")
```

### PyTorch Weight Decay = L2 Prior

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Weight decay in optimizer (= L2 prior)
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=0.01  # λ for L2 regularization
)

# This is equivalent to:
def train_step_explicit_l2(model, X, y, lambda_reg=0.01):
    """Explicit L2 regularization = MAP with Gaussian prior"""
    pred = model(X)
    
    # Negative log-likelihood
    nll = F.mse_loss(pred, y)
    
    # Negative log-prior (Gaussian)
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    
    # MAP objective
    loss = nll + lambda_reg * l2_penalty
    
    return loss

# Method 2: Explicit L1 regularization
def train_step_l1(model, X, y, lambda_reg=0.01):
    """L1 regularization = MAP with Laplace prior"""
    pred = model(X)
    nll = F.mse_loss(pred, y)
    
    l1_penalty = 0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    
    loss = nll + lambda_reg * l1_penalty
    return loss
```

### Visualizing Prior Effect

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Compare Ridge vs Lasso on correlated features
np.random.seed(42)
n = 100

# Create correlated features
x1 = np.random.randn(n)
x2 = x1 + 0.1 * np.random.randn(n)  # Highly correlated with x1
X = np.column_stack([x1, x2])

# True relationship uses only x1
y = 2 * x1 + 0.1 * np.random.randn(n)

# Ridge solution
lambda_vals = np.logspace(-2, 2, 50)
ridge_paths = []
lasso_paths = []

for lam in lambda_vals:
    # Ridge
    theta_ridge = np.linalg.solve(X.T @ X + lam * np.eye(2), X.T @ y)
    ridge_paths.append(theta_ridge)
    
    # Lasso (simplified)
    theta_lasso = map_lasso_cd(X, y, lambda_reg=lam, n_iter=100)
    lasso_paths.append(theta_lasso)

ridge_paths = np.array(ridge_paths)
lasso_paths = np.array(lasso_paths)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(lambda_vals, ridge_paths[:, 0], label='θ₁')
axes[0].plot(lambda_vals, ridge_paths[:, 1], label='θ₂')
axes[0].set_xscale('log')
axes[0].set_xlabel('λ')
axes[0].set_ylabel('Coefficient')
axes[0].set_title('Ridge: Coefficients shrink together')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(lambda_vals, lasso_paths[:, 0], label='θ₁')
axes[1].plot(lambda_vals, lasso_paths[:, 1], label='θ₂')
axes[1].set_xscale('log')
axes[1].set_xlabel('λ')
axes[1].set_ylabel('Coefficient')
axes[1].set_title('Lasso: θ₂ becomes exactly 0 (sparse!)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 📊 MLE vs MAP vs Bayesian

| Aspect | MLE | MAP | Bayesian |
|--------|-----|-----|----------|
| **Formula** | $\arg\max P(D\|\theta)$ | $\arg\max P(D\|\theta)P(\theta)$ | $P(\theta\|D)$ |
| **Output** | Point estimate | Point estimate | Distribution |
| **Prior** | ❌ | ✅ | ✅ |
| **Regularization** | None | Implicit | Implicit |
| **Uncertainty** | ❌ | ❌ | ✅ |
| **Computation** | Easy | Easy | Hard |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 3 | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | Elements of Statistical Learning | Hastie, Tibshirani, Friedman |
| 🎥 | StatQuest: Regularization | [YouTube](https://www.youtube.com/watch?v=Q81RR3yKn30) |
| 🇨🇳 | MAP与正则化的关系 | [知乎](https://zhuanlan.zhihu.com/p/32480810) |
| 🇨🇳 | L1/L2正则化的先验解释 | [机器之心](https://www.jiqizhixin.com/articles/2018-11-20-6) |

---

⬅️ [Back: Bayesian](../01_bayesian/) | ➡️ [Next: MLE](../03_mle/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
