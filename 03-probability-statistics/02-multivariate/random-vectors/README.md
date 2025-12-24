<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Random Vectors&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Random Vectors

> **Multiple random variables as a single entity**

---

## 🎯 Visual Overview

<img src="./images/random-vectors.svg" width="100%">

*Caption: A random vector X = [X₁,...,Xₙ]ᵀ groups multiple RVs. The joint distribution captures all dependencies. Marginals are obtained by integrating out variables. Conditionals and Bayes rule extend naturally.*

---

## 📂 Overview

Random vectors are essential for multivariate ML: each data point is a random vector. Understanding joint, marginal, and conditional distributions is key to probabilistic modeling.

---

## 📐 Mathematical Definitions

### Random Vector
```
X = [X₁, X₂, ..., Xₙ]ᵀ

Each Xᵢ is a random variable
X: Ω → ℝⁿ
```

### Joint Distribution
```
Joint CDF: F(x₁,...,xₙ) = P(X₁ ≤ x₁, ..., Xₙ ≤ xₙ)

Discrete: P(X₁ = x₁, ..., Xₙ = xₙ)
Continuous: f(x₁,...,xₙ) where ∫...∫ f dx₁...dxₙ = 1
```

### Marginal Distribution
```
Discrete: P(Xᵢ = x) = Σ_{x₋ᵢ} P(X₁,...,Xₙ)
Continuous: fₓᵢ(x) = ∫...∫ f(x₁,...,xₙ) dx₋ᵢ

"Integrate out" or "sum out" other variables
```

### Conditional Distribution
```
P(X | Y = y) = P(X, Y = y) / P(Y = y)

f(x | y) = f(x, y) / f(y)
```

### Independence
```
X, Y independent ⟺ f(x, y) = f(x)f(y)
                 ⟺ P(X ∈ A, Y ∈ B) = P(X ∈ A)P(Y ∈ B)

Conditional independence:
X ⊥ Y | Z ⟺ f(x, y | z) = f(x | z)f(y | z)
```

### Mean Vector and Covariance Matrix
```
μ = E[X] = [E[X₁], ..., E[Xₙ]]ᵀ

Σ = E[(X - μ)(X - μ)ᵀ]
Σᵢⱼ = Cov(Xᵢ, Xⱼ)
```

---

## 💻 Code Examples

```python
import numpy as np
import torch
from scipy import stats

# Multivariate Gaussian
mean = np.array([0, 0])
cov = np.array([[1, 0.8], [0.8, 1]])  # Correlated
samples = np.random.multivariate_normal(mean, cov, 1000)

# PyTorch multivariate normal
dist = torch.distributions.MultivariateNormal(
    loc=torch.zeros(2),
    covariance_matrix=torch.tensor([[1., 0.8], [0.8, 1.]])
)
samples = dist.sample((1000,))
log_prob = dist.log_prob(samples)

# Marginal: just take component
marginal_X1 = samples[:, 0]  # X₁ ~ N(0, 1)

# Conditional Gaussian
# X₂ | X₁ = x₁ ~ N(μ₂ + Σ₂₁Σ₁₁⁻¹(x₁-μ₁), Σ₂₂ - Σ₂₁Σ₁₁⁻¹Σ₁₂)
def conditional_gaussian(x1, mu, Sigma):
    mu1, mu2 = mu[0], mu[1]
    S11, S12, S21, S22 = Sigma[0,0], Sigma[0,1], Sigma[1,0], Sigma[1,1]
    cond_mean = mu2 + S21/S11 * (x1 - mu1)
    cond_var = S22 - S21*S12/S11
    return cond_mean, cond_var

# Independence test
from scipy.stats import chi2_contingency
# chi2, p, dof, expected = chi2_contingency(contingency_table)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 2 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | Murphy MLaPP Ch. 2 | [Book](https://probml.github.io/pml-book/book1.html) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 🇨🇳 | 多元随机变量 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 联合分布 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 概率论课程 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: random-vectors](../)

---

⬅️ [Back: Gaussian](../gaussian/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
