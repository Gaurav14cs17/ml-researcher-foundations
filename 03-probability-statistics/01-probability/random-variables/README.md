<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Random%20Variables&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/random-variables.svg" width="100%">

*Caption: A random variable X: Ω→ℝ assigns numbers to outcomes. Discrete RVs have probability mass functions (PMF); continuous RVs have probability density functions (PDF). Key statistics: E[X], Var[X], CDF.*

---

## 📂 Overview

Random variables transform abstract outcomes into numerical values we can work with. They're the mathematical objects we manipulate in probabilistic machine learning.

---

## 📐 Mathematical Definitions

### Random Variable
```
X: Ω → ℝ  (maps outcomes to real numbers)

Example: Coin flip
Ω = {heads, tails}
X(heads) = 1, X(tails) = 0
```

### Discrete RV (PMF)
```
P(X = x) = probability mass function

Requirements:
• P(X = x) ≥ 0 for all x
• Σₓ P(X = x) = 1
```

### Continuous RV (PDF)
```
f(x) = probability density function

Requirements:
• f(x) ≥ 0 for all x
• ∫_{-∞}^{∞} f(x) dx = 1

P(a ≤ X ≤ b) = ∫_a^b f(x) dx
```

### Cumulative Distribution Function (CDF)
```
F(x) = P(X ≤ x)

Properties:
• F(-∞) = 0, F(∞) = 1
• Non-decreasing
• P(a < X ≤ b) = F(b) - F(a)
```

---

## 💻 Code Examples

```python
import numpy as np
import torch
from scipy import stats

# Discrete: Bernoulli
p = 0.7
X = np.random.binomial(1, p, 1000)  # 0 or 1
print(f"P(X=1) ≈ {X.mean():.3f}")    # ≈ 0.7

# Continuous: Gaussian
mu, sigma = 0, 1
X = np.random.normal(mu, sigma, 10000)
print(f"E[X] ≈ {X.mean():.3f}")      # ≈ 0
print(f"Var[X] ≈ {X.var():.3f}")     # ≈ 1

# CDF: P(X ≤ 0) for standard normal
cdf_0 = stats.norm.cdf(0)  # = 0.5

# PyTorch distributions
dist = torch.distributions.Normal(0, 1)
samples = dist.sample((1000,))
log_prob = dist.log_prob(samples)  # For likelihood
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Ross: Probability | [Book](https://www.elsevier.com/books/a-first-course-in-probability/ross/978-0-321-79477-2) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 📖 | PyTorch Distributions | [Docs](https://pytorch.org/docs/stable/distributions.html) |
| 🇨🇳 | 随机变量详解 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 概率分布 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
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

⬅️ [Back: Spaces](../spaces/) | ➡️ [Next: Distributions](../distributions/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
