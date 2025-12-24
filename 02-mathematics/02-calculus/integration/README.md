# Integration

> **Computing areas, averages, and expectations**

---

## 🎯 Visual Overview

<img src="./images/integration.svg" width="100%">

*Caption: Integration computes areas under curves. In ML, this appears in probability (normalization, expectations, marginalization) and is often approximated via Monte Carlo sampling when intractable.*

---

## 📂 Overview

Integration is the inverse of differentiation. In ML, we rarely compute integrals analytically - instead we use Monte Carlo approximation or variational methods.

---

## 🔑 Key Concepts

| Concept | Formula | ML Usage |
|---------|---------|----------|
| **Expectation** | 𝔼[X] = ∫xp(x)dx | Mean predictions |
| **Variance** | Var[X] = ∫(x-μ)²p(x)dx | Uncertainty |
| **Normalization** | ∫p(x)dx = 1 | Valid probability |
| **Marginalization** | p(x) = ∫p(x,z)dz | Latent variables |

---

## 📐 Monte Carlo Integration

```
Problem: ∫f(x)p(x)dx is hard to compute analytically

Solution: Sample xᵢ ~ p(x), then approximate:
𝔼[f(X)] ≈ (1/N) Σᵢ f(xᵢ)

Error ∝ 1/√N (converges, but slowly)

Used in: VAEs, diffusion models, RL policy gradients
```

---

## 💻 Code

```python
import numpy as np
import torch

# Monte Carlo estimate of E[f(X)] where X ~ N(0,1)
def monte_carlo_expectation(f, n_samples=10000):
    samples = np.random.normal(0, 1, n_samples)
    return np.mean(f(samples))

# Example: E[X²] where X ~ N(0,1) should be 1 (variance)
print(monte_carlo_expectation(lambda x: x**2))  # ≈ 1.0

# PyTorch: reparameterization trick for VAE
mu, logvar = encoder(x)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std  # Sample from q(z|x)
```


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: Calculus](../)

---

⬅️ [Back: Gradients](../gradients/) | ➡️ [Next: Limits Continuity](../limits-continuity/)
