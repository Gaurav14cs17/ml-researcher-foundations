<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Bayes%20Theorem&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Bayes' Theorem

> **Updating beliefs with evidence**

---

## 📐 Formula

```
P(A|B) = P(B|A) · P(A) / P(B)

Or with θ (parameters) and D (data):

P(θ|D) = P(D|θ) · P(θ) / P(D)

posterior = likelihood × prior / evidence
```

---

## 🔑 Components

| Term | Meaning | In ML |
|------|---------|-------|
| P(θ) | Prior | Initial beliefs |
| P(D\|θ) | Likelihood | How likely is data given params |
| P(θ\|D) | Posterior | Updated beliefs |
| P(D) | Evidence | Normalizing constant |

---

## 🌍 Applications

| Application | How |
|-------------|-----|
| Naive Bayes | P(class\|features) |
| Bayesian optimization | Prior over functions |
| Bayesian neural nets | Prior over weights |
| Kalman filter | Update state beliefs |

---

## 💻 Code

```python
import numpy as np

# Coin flip example: Is coin fair?
def posterior_beta(prior_a, prior_b, heads, tails):
    """
    Beta prior: Beta(a, b)
    Likelihood: Binomial
    Posterior: Beta(a + heads, b + tails)
    """
    return prior_a + heads, prior_b + tails

# Prior: Beta(1, 1) = Uniform
a, b = 1, 1

# Observe 7 heads, 3 tails
a, b = posterior_beta(a, b, heads=7, tails=3)

# Posterior mean = a / (a + b) = 8/12 = 0.67
print(f"Posterior mean: {a / (a + b):.2f}")
```

---

## 📊 Bayesian vs Frequentist

| Aspect | Bayesian | Frequentist |
|--------|----------|-------------|
| Parameters | Random variables | Fixed |
| Uncertainty | Posterior distribution | Confidence interval |
| Prior | Explicit | Implicit |

---

<- [Back](./README.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
