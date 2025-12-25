<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Expected%20Value&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/expectation.svg" width="100%">

*Caption: E[X] is the weighted average of outcomes. Key properties: linearity (E[aX+b]=aE[X]+b), addition (E[X+Y]=E[X]+E[Y]). LOTUS allows computing E[g(X)] without knowing g(X)'s distribution.*

---

## 📂 Overview

Expected value is the most important summary statistic of a random variable. In ML, loss functions are expected values: L = E[loss(ŷ,y)].

---

## 📐 Mathematical Definitions

### Discrete Random Variable
```
E[X] = Σₓ x · P(X = x)
```

### Continuous Random Variable
```
E[X] = ∫_{-∞}^{∞} x · f(x) dx
```

### Key Properties
```
Linearity:     E[aX + b] = a·E[X] + b
Addition:      E[X + Y] = E[X] + E[Y]  (always!)
Multiplication: E[XY] = E[X]·E[Y]     (only if X,Y independent)

LOTUS (Law of the Unconscious Statistician):
E[g(X)] = Σₓ g(x)·P(X=x)   (discrete)
E[g(X)] = ∫ g(x)·f(x)dx    (continuous)
```

### Variance
```
Var(X) = E[(X - μ)²] = E[X²] - E[X]²

Var(aX + b) = a² Var(X)
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
           = Var(X) + Var(Y)  (if independent)
```

---

## 💻 Code Examples

```python
import numpy as np
import torch

# Empirical expectation (Monte Carlo)
samples = np.random.normal(0, 1, 10000)
E_X = samples.mean()      # ≈ 0
E_X2 = (samples**2).mean() # ≈ 1 (variance)

# Loss function is an expectation
def cross_entropy_loss(logits, targets):
    """E[-log p(y|x)] over data"""
    return -torch.log_softmax(logits, dim=-1)[range(len(targets)), targets].mean()

# Reparameterization trick (VAE)
def sample_gaussian(mu, log_var):
    """E_z[f(z)] where z ~ N(mu, sigma²)"""
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)  # N(0,1)
    return mu + eps * std  # z ~ N(mu, sigma²)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 1 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 📖 | Ross: Probability | [Book](https://www.elsevier.com/books/a-first-course-in-probability/ross/978-0-321-79477-2) |
| 🇨🇳 | 期望与方差 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 概率论基础 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
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

⬅️ [Back: Distributions](../distributions/) | ➡️ [Next: Bayes](../bayes/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
