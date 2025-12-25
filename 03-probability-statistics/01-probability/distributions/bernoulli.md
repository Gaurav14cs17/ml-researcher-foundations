<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Bernoulli%20Distribution&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Definition

```
X ~ Bernoulli(p)

P(X = 1) = p
P(X = 0) = 1 - p

Single binary outcome (coin flip, success/failure)
```

---

## 🔑 Properties

| Property | Value |
|----------|-------|
| Mean | p |
| Variance | p(1-p) |
| Mode | 1 if p > 0.5, else 0 |
| Entropy | -p log p - (1-p) log(1-p) |

---

## 📊 Related Distributions

| Distribution | Relation |
|--------------|----------|
| Binomial(n, p) | Sum of n Bernoulli(p) |
| Categorical | Multi-class Bernoulli |
| Geometric | # trials until success |

---

## 💻 Code

```python
import numpy as np
import torch

# NumPy
samples = np.random.binomial(1, p=0.3, size=1000)
mean = samples.mean()  # ≈ 0.3

# PyTorch
dist = torch.distributions.Bernoulli(probs=0.3)
samples = dist.sample((1000,))
log_prob = dist.log_prob(torch.tensor([1.0]))  # log(0.3)

# In neural networks: Binary cross-entropy
# p(y=1|x) = σ(f(x))
probs = torch.sigmoid(logits)
bce = -y * torch.log(probs) - (1-y) * torch.log(1-probs)
```

---

## 🌍 ML Applications

| Application | Use |
|-------------|-----|
| Binary classification | P(y=1\|x) |
| Dropout | Mask ~ Bernoulli(1-p) |
| VAE | Binary latents |

---

---

➡️ [Next: Gaussian](./gaussian.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
