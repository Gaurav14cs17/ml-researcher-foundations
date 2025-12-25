<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Bayes Theorem&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🎯 Bayes' Theorem

> **Updating beliefs with evidence**

---

## 📐 The Formula

```
P(A|B) = P(B|A) × P(A) / P(B)

Posterior = Likelihood × Prior / Evidence

Bayesian ML:
P(θ|D) = P(D|θ) × P(θ) / P(D)
  ↓         ↓       ↓      ↓
Posterior  Like-  Prior  Marginal
           lihood        likelihood
```

---

## 💻 Code Example

```python
import numpy as np

# Prior: P(disease) = 0.01
prior = 0.01

# Likelihood: P(positive|disease) = 0.95
likelihood = 0.95

# P(positive|no disease) = 0.05 (false positive)
false_positive = 0.05

# Evidence: P(positive)
evidence = likelihood * prior + false_positive * (1 - prior)

# Posterior: P(disease|positive)
posterior = (likelihood * prior) / evidence
print(f"P(disease|positive) = {posterior:.4f}")
```

---

## 🔗 Applications

| Application | Usage |
|-------------|-------|
| **Naive Bayes** | Text classification |
| **Bayesian Neural Networks** | Uncertainty estimation |
| **Gaussian Processes** | Posterior over functions |
| **MCMC** | Posterior sampling |

---

⬅️ [Back: Probability](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

