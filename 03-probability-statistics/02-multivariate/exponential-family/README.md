# Exponential Family

> **A unified framework for distributions**

---

## 🎯 Visual Overview

<img src="./images/exponential-family.svg" width="100%">

*Caption: p(x|θ) = h(x)exp(η(θ)·T(x) - A(θ)). Natural parameters η, sufficient statistics T(x), log-partition A(θ). Includes Gaussian, Bernoulli, Poisson, Gamma. Enables efficient MLE via moment matching.*

---

## 📂 Overview

The exponential family unifies most common distributions under one framework. This makes theoretical analysis cleaner and enables general-purpose algorithms for a wide class of models.

---

## 📐 Mathematical Definitions

### Exponential Family Form
```
p(x | θ) = h(x) exp(η(θ)ᵀT(x) - A(θ))

Where:
• h(x): base measure
• η(θ): natural parameters
• T(x): sufficient statistics
• A(θ): log-partition function (normalizer)
```

### Key Properties
```
Moments from log-partition:
E[T(x)] = ∇A(η)
Var[T(x)] = ∇²A(η)

MLE: Set sample moments = population moments
η̂: ∇A(η̂) = (1/n)Σᵢ T(xᵢ)
```

### Common Distributions
```
Bernoulli(p):
η = log(p/(1-p))  (logit)
T(x) = x
A(η) = log(1 + eⁿ)

Gaussian(μ, σ²):
η = [μ/σ², -1/(2σ²)]
T(x) = [x, x²]
A(η) = -η₁²/(4η₂) - ½log(-2η₂)

Poisson(λ):
η = log(λ)
T(x) = x
A(η) = eⁿ
```

### GLMs (Generalized Linear Models)
```
Link function: g(E[Y]) = Xβ

Canonical link: η = Xβ directly
• Logistic regression: η = log(p/(1-p))
• Poisson regression: η = log(λ)
• Linear regression: η = μ
```

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal, Poisson

# Bernoulli from natural parameter
def bernoulli_from_logit(logit):
    p = torch.sigmoid(logit)  # Inverse link
    return Bernoulli(probs=p)

# Cross-entropy = negative log-likelihood of Bernoulli
loss = nn.BCEWithLogitsLoss()  # Directly uses logits (natural param)

# Gaussian GLM = linear regression
# MSE loss = -log p(y|x) for Gaussian

# Softmax = multiclass extension
# Output: natural parameters η = logits
# P(y=k) = exp(ηₖ) / Σⱼ exp(ηⱼ)

# Exponential family in JAX/PyTorch
class ExponentialFamily:
    def log_prob(self, x):
        return (self.natural_params @ self.sufficient_stats(x) 
                - self.log_partition())
    
    def mean(self):
        # E[T(x)] = ∇A(η)
        return torch.autograd.grad(self.log_partition(), self.natural_params)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 2 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | Murphy MLaPP Ch. 9 | [Book](https://probml.github.io/pml-book/book1.html) |
| 📄 | Jordan Notes | [PDF](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf) |
| 🇨🇳 | 指数族分布详解 | [知乎](https://zhuanlan.zhihu.com/p/62954109) |
| 🇨🇳 | GLM原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 统计学习 | [B站](https://www.bilibili.com/video/BV164411b7dx) |

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

⬅️ [Back: exponential-family](../)

---

⬅️ [Back: Covariance](../covariance/) | ➡️ [Next: Gaussian](../gaussian/)
