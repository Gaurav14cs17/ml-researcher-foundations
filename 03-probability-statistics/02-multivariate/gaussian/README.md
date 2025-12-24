<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Gaussian&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🔥 Multivariate Gaussian

> **The most important distribution in ML**

---

## 🎯 Visual Overview

<img src="./images/multivariate-gaussian.svg" width="100%">

*Caption: A 2D Gaussian shows elliptical contours of constant probability. The mean μ determines the center, while the covariance matrix Σ controls the shape and orientation. This distribution is foundational for VAEs, GPs, and diffusion models.*

---

## 📐 Definition

```
p(x) = (2π)^(-d/2) |Σ|^(-1/2) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

Parameters:
• μ ∈ ℝᵈ: mean vector
• Σ ∈ ℝᵈˣᵈ: covariance matrix (positive definite)
```

---

## 🔑 Key Properties

| Property | Formula |
|----------|---------|
| Marginal | p(x₁) = N(μ₁, Σ₁₁) |
| Conditional | p(x₁\|x₂) = N(μ₁\|₂, Σ₁\|₂) |
| Linear transform | Ax ~ N(Aμ, AΣAᵀ) |
| Sum | X + Y ~ N(μₓ+μᵧ, Σₓ+Σᵧ) if independent |

---

## 🌍 ML Applications

| Application | How |
|-------------|-----|
| **VAE** | Latent space is Gaussian |
| **GP** | Prior over functions |
| **Kalman Filter** | State estimation |
| **Diffusion** | Noise distribution |

---

## 💻 Code

```python
import torch
from torch.distributions import MultivariateNormal

# Create distribution
mu = torch.zeros(2)
cov = torch.eye(2)
dist = MultivariateNormal(mu, cov)

# Sample
samples = dist.sample((1000,))

# Log probability
log_prob = dist.log_prob(samples)
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

⬅️ [Back: Multivariate](../)

---

⬅️ [Back: Exponential Family](../exponential-family/) | ➡️ [Next: Random Vectors](../random-vectors/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
