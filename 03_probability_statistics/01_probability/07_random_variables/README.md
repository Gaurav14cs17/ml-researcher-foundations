<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Random%20Variables&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/random-variables.svg" width="100%">

*Caption: A random variable X: Ω→ℝ assigns numbers to outcomes. Discrete RVs have probability mass functions (PMF); continuous RVs have probability density functions (PDF).*

---

## 📂 Overview

Random variables transform abstract outcomes into numerical values we can work with. They're the mathematical objects we manipulate in probabilistic machine learning.

---

## 📐 Formal Definition

### Random Variable

**Definition:** A random variable $X$ is a measurable function from a sample space to the real numbers:

```math
X: \Omega \to \mathbb{R}
```

**Example: Coin Flip**
- Sample space: $\Omega = \{\text{heads}, \text{tails}\}$
- Random variable: $X(\text{heads}) = 1, \quad X(\text{tails}) = 0$

---

## 📐 Discrete Random Variables

### Probability Mass Function (PMF)

```math
p_X(x) = P(X = x)
```

**Requirements:**
1. $p\_X(x) \geq 0$ for all $x$
2. $\sum\_x p\_X(x) = 1$

### Examples

**Bernoulli:**
```math
P(X = 1) = p, \quad P(X = 0) = 1 - p
```

**Binomial:**
```math
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
```

**Poisson:**
```math
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
```

---

## 📐 Continuous Random Variables

### Probability Density Function (PDF)

```math
f_X(x) \geq 0, \quad \int_{-\infty}^{\infty} f_X(x) \, dx = 1
```

**Note:** $f(x)$ is NOT a probability! It can be > 1.

```math
P(a \leq X \leq b) = \int_a^b f_X(x) \, dx
```

### Examples

**Uniform:**
```math
f(x) = \frac{1}{b-a}, \quad x \in [a, b]
```

**Exponential:**
```math
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
```

**Gaussian:**
```math
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
```

---

## 📐 Cumulative Distribution Function (CDF)

### Definition

```math
F_X(x) = P(X \leq x)
```

### Properties

1. $F(-\infty) = 0, \quad F(\infty) = 1$
2. $F$ is non-decreasing
3. $F$ is right-continuous
4. $P(a < X \leq b) = F(b) - F(a)$

### Relation to PMF/PDF

**Discrete:**
```math
F(x) = \sum_{x_i \leq x} p(x_i)
```

**Continuous:**
```math
F(x) = \int_{-\infty}^{x} f(t) \, dt
f(x) = \frac{dF}{dx}
```

---

## 📐 Transformations of Random Variables

### Change of Variables (Univariate)

If $Y = g(X)$ where $g$ is monotonic:

```math
f_Y(y) = f_X(g^{-1}(y)) \cdot \left|\frac{d}{dy}g^{-1}(y)\right|
```

**Proof:**

For increasing $g$:
```math
F_Y(y) = P(Y \leq y) = P(g(X) \leq y) = P(X \leq g^{-1}(y)) = F_X(g^{-1}(y))
```

Differentiating:
```math
f_Y(y) = f_X(g^{-1}(y)) \cdot \frac{d}{dy}g^{-1}(y) \quad \blacksquare
```

### Example: $Y = X^2$ for $X \sim \mathcal{N}(0, 1)$

```math
f_Y(y) = \frac{1}{2\sqrt{y}} \cdot \frac{1}{\sqrt{2\pi}} e^{-y/2}, \quad y > 0
```

This is the chi-squared distribution with 1 degree of freedom!

---

## 📐 Expectation and Variance

### Expectation (Mean)

**Discrete:**
```math
E[X] = \sum_x x \cdot P(X = x)
```

**Continuous:**
```math
E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx
```

### LOTUS (Law of the Unconscious Statistician)

```math
E[g(X)] = \sum_x g(x) \cdot P(X = x)
E[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f(x) \, dx
```

**Key:** No need to find distribution of $g(X)$!

### Variance

```math
\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2
```

**Proof of alternative form:**
```math
\text{Var}(X) = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu^2 + \mu^2 = E[X^2] - \mu^2 \quad \blacksquare
```

---

## 📐 Moment Generating Functions

### Definition

```math
M_X(t) = E[e^{tX}]
```

### Properties

1. $M\_X(0) = 1$
2. $\frac{d^n M\_X}{dt^n}\bigg|\_{t=0} = E[X^n]$
3. If $M\_X(t) = M\_Y(t)$ for all $t$, then $X \stackrel{d}{=} Y$

### Example: Gaussian MGF

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

```math
M_X(t) = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)
```

**Proof:**
```math
M_X(t) = E[e^{tX}] = \int \frac{1}{\sqrt{2\pi\sigma^2}} e^{tx} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx
```

Complete the square and integrate. $\quad \blacksquare$

---

## 📐 Characteristic Function

### Definition

```math
\phi_X(t) = E[e^{itX}]
```

**Advantages over MGF:**
- Always exists (bounded)
- Uniquely determines distribution

### Gaussian Characteristic Function

```math
\phi_X(t) = \exp\left(i\mu t - \frac{\sigma^2 t^2}{2}\right)
```

---

## 💻 Code Examples

```python
import numpy as np
import torch
from scipy import stats

# Discrete random variable: Bernoulli
p = 0.7
bernoulli = stats.bernoulli(p)
samples = bernoulli.rvs(1000)
print(f"Bernoulli mean: {samples.mean():.3f} (theoretical: {p})")
print(f"Bernoulli PMF at 1: {bernoulli.pmf(1):.3f}")

# Continuous random variable: Gaussian
mu, sigma = 0, 1
normal = stats.norm(mu, sigma)
samples = normal.rvs(10000)
print(f"Gaussian mean: {samples.mean():.3f}, std: {samples.std():.3f}")

# CDF
print(f"P(X ≤ 0) = {normal.cdf(0):.3f}")  # = 0.5
print(f"P(X ≤ 1.96) = {normal.cdf(1.96):.3f}")  # ≈ 0.975

# Quantile function (inverse CDF)
print(f"95th percentile: {normal.ppf(0.95):.3f}")  # ≈ 1.645

# PyTorch distributions
dist = torch.distributions.Normal(0, 1)
samples = dist.sample((1000,))
log_prob = dist.log_prob(samples)  # For likelihood computation
print(f"Log prob at 0: {dist.log_prob(torch.tensor(0.0)):.3f}")

# Transformation: Y = X^2
X = np.random.normal(0, 1, 100000)
Y = X**2
print(f"Y = X² mean: {Y.mean():.3f} (theoretical: 1)")  # Chi-squared(1)

# Moment generating function verification
def estimate_moments(samples, k):
    """Estimate E[X^k] from samples"""
    return np.mean(samples**k)

X = np.random.normal(2, 1, 100000)  # N(2, 1)
print(f"E[X] ≈ {estimate_moments(X, 1):.3f}")  # Should be 2
print(f"E[X²] ≈ {estimate_moments(X, 2):.3f}")  # Should be μ² + σ² = 5
```

---

## 📊 Common Distributions Summary

| Distribution | PMF/PDF | Mean | Variance |
|--------------|---------|------|----------|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| Binomial($n,p$) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Poisson($\lambda$) | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ |
| Uniform($a,b$) | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| Gaussian($\mu,\sigma^2$) | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Ross: Probability | [Book](https://www.elsevier.com/books/a-first-course-in-probability/ross/978-0-321-79477-2) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 📖 | PyTorch Distributions | [Docs](https://pytorch.org/docs/stable/distributions.html) |
| 🇨🇳 | 随机变量详解 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 概率分布 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |

---

⬅️ [Back: Limit Theorems](../06_limit_theorems/) | ➡️ [Next: Probability Spaces](../08_spaces/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
