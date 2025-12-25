<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Gaussian%20Processes&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/gaussian-processes.svg" width="100%">

*Caption: GPs define distributions over functions: f(x) ~ GP(m(x), k(x,x')). They provide uncertainty estimates automatically. GP regression fits data while showing confidence bands that grow away from observations.*

---

## 📂 Overview

Gaussian Processes are powerful non-parametric models for regression with built-in uncertainty quantification. Used for Bayesian optimization, scientific modeling, and as priors for neural networks.

---

## 📐 Mathematical Definitions

### Gaussian Process Definition
```
f(x) ~ GP(m(x), k(x, x'))

Where:
• m(x) = E[f(x)]  (mean function, often 0)
• k(x, x') = Cov(f(x), f(x'))  (kernel/covariance function)

Any finite collection [f(x₁),...,f(xₙ)] is jointly Gaussian
```

### Common Kernels
```
RBF (Squared Exponential):
k(x, x') = σ² exp(-||x-x'||²/(2ℓ²))

Matérn:
k(x, x') = σ²(2^(1-ν)/Γ(ν))(√(2ν)r/ℓ)^ν K_ν(√(2ν)r/ℓ)

Linear:
k(x, x') = σ² xᵀx'
```

### GP Regression
```
Given: Training data (X, y), test points X*
Prior: f ~ GP(0, K)
Likelihood: y = f(X) + ε, where ε ~ N(0, σ²I)

Posterior:
f* | X*, X, y ~ N(μ*, Σ*)

μ* = K(X*, X)[K(X, X) + σ²I]⁻¹y
Σ* = K(X*, X*) - K(X*, X)[K(X, X) + σ²I]⁻¹K(X, X*)
```

### Marginal Likelihood
```
log p(y|X) = -½yᵀ(K + σ²I)⁻¹y - ½log|K + σ²I| - (n/2)log(2π)

Used to optimize hyperparameters (ℓ, σ²)
```

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Define kernel
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

# GP Regression
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_train, y_train)

# Predict with uncertainty
y_mean, y_std = gp.predict(X_test, return_std=True)

# Bayesian Optimization (hyperparameter tuning)
from scipy.stats import norm

def expected_improvement(X, gp, y_best):
    mu, sigma = gp.predict(X, return_std=True)
    z = (y_best - mu) / sigma
    ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei
```

---

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Rasmussen: GP for ML | [Book](https://gaussianprocess.org/gpml/) |
| 📖 | Bishop PRML Ch. 6 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | GPyTorch | [Docs](https://gpytorch.ai/) |
| 🇨🇳 | 高斯过程详解 | [知乎](https://zhuanlan.zhihu.com/p/75589452) |
| 🇨🇳 | 贝叶斯优化 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | GP回归 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

<- [Back](../)

---

⬅️ [Back: gaussian-processes](../)

---

➡️ [Next: Kernel Trick](../kernel-trick/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
