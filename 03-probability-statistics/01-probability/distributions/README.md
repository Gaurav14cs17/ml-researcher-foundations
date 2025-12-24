<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Distributions&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Probability Distributions

> **The building blocks of probabilistic ML**

<img src="./images/distributions.svg" width="100%">

---

## 📂 Topics in This Folder

| File | Distribution | ML Application |
|------|--------------|----------------|
| [bernoulli.md](./bernoulli.md) | Bernoulli & Binomial | Binary classification |
| [gaussian.md](./gaussian.md) | Gaussian/Normal | Everywhere! |

---

## 📐 Discrete Distributions

### Bernoulli Distribution

```
X ~ Bernoulli(p)

P(X = 1) = p
P(X = 0) = 1 - p = q

PMF:     P(X = k) = p^k (1-p)^(1-k),  k ∈ {0, 1}
Mean:    E[X] = p
Variance: Var(X) = p(1-p)

ML Use: Binary classification output (sigmoid → Bernoulli)
```

### Binomial Distribution

```
X ~ Binomial(n, p)

P(X = k) = C(n,k) p^k (1-p)^(n-k),  k = 0,1,...,n

Mean:    E[X] = np
Variance: Var(X) = np(1-p)

ML Use: n independent Bernoulli trials
```

### Categorical Distribution

```
X ~ Categorical(p₁, p₂, ..., pₖ)  where Σpᵢ = 1

P(X = i) = pᵢ

One-hot representation:
X = [0, 0, ..., 1, ..., 0]  (1 at position i)

ML Use: Multi-class classification (softmax → Categorical)
```

### Poisson Distribution

```
X ~ Poisson(λ)

P(X = k) = (λ^k e^(-λ)) / k!,  k = 0,1,2,...

Mean:    E[X] = λ
Variance: Var(X) = λ

ML Use: Count data, rare events
```

---

## 📐 Continuous Distributions

### Gaussian (Normal) Distribution

```
X ~ N(μ, σ²)

PDF:  p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

Mean:    E[X] = μ
Variance: Var(X) = σ²
Mode:    μ (same as mean)

Standard Normal: Z ~ N(0, 1)
Standardization: Z = (X - μ) / σ

ML Use: 
• Regression targets (MSE loss)
• Latent space in VAE
• Weight initialization
• Gaussian noise in diffusion
```

### Multivariate Gaussian

```
X ~ N(μ, Σ)  where X, μ ∈ ℝⁿ, Σ ∈ ℝⁿˣⁿ

PDF: p(x) = (2π)^(-n/2) |Σ|^(-1/2) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

Mean:       E[X] = μ
Covariance: Cov(X) = Σ

Properties:
• Marginals are Gaussian
• Conditionals are Gaussian
• Linear transforms are Gaussian: AX + b ~ N(Aμ+b, AΣAᵀ)

ML Use: 
• Gaussian Processes
• Multivariate regression
• VAE latent space
```

### Uniform Distribution

```
X ~ Uniform(a, b)

PDF:  p(x) = 1/(b-a),  x ∈ [a, b]
CDF:  F(x) = (x-a)/(b-a)

Mean:    E[X] = (a+b)/2
Variance: Var(X) = (b-a)²/12

ML Use: Weight initialization, random sampling
```

### Exponential Distribution

```
X ~ Exponential(λ)

PDF:  p(x) = λ exp(-λx),  x ≥ 0
CDF:  F(x) = 1 - exp(-λx)

Mean:    E[X] = 1/λ
Variance: Var(X) = 1/λ²

ML Use: Time-to-event modeling
```

---

## 📐 The Exponential Family

```
General form:
p(x|θ) = h(x) exp(η(θ)ᵀT(x) - A(θ))

Where:
• η(θ): natural parameters
• T(x): sufficient statistics
• A(θ): log-partition function (normalizer)
• h(x): base measure

Members: Gaussian, Bernoulli, Poisson, Exponential, Gamma, Beta, ...

Why it matters:
• Conjugate priors exist
• Maximum entropy distributions
• Natural gradients
• Generalized Linear Models (GLMs)
```

---

## 💻 Code Examples

```python
import numpy as np
import torch
import torch.distributions as dist

# Bernoulli
p = 0.7
bernoulli = dist.Bernoulli(probs=p)
samples = bernoulli.sample((1000,))
print(f"Bernoulli mean: {samples.mean():.3f} (expected: {p})")

# Gaussian
mu, sigma = 0.0, 1.0
gaussian = dist.Normal(mu, sigma)
samples = gaussian.sample((1000,))
print(f"Gaussian mean: {samples.mean():.3f}, std: {samples.std():.3f}")

# Categorical (softmax output)
logits = torch.tensor([1.0, 2.0, 3.0])
categorical = dist.Categorical(logits=logits)
samples = categorical.sample((1000,))
print(f"Categorical samples: {torch.bincount(samples)}")

# Multivariate Gaussian
mu = torch.zeros(2)
cov = torch.eye(2)
mvn = dist.MultivariateNormal(mu, cov)
samples = mvn.sample((1000,))

# Log probability (for loss computation)
x = torch.tensor([0.5])
log_prob = gaussian.log_prob(x)  # Used in NLL loss
```

---

## 🌍 ML Applications

| Distribution | ML Application |
|--------------|----------------|
| Bernoulli | Binary classification (BCE loss) |
| Categorical | Multi-class classification (CE loss) |
| Gaussian | Regression (MSE loss), VAE |
| Poisson | Count prediction |
| Exponential | Survival analysis |

### Loss Functions as NLL

```
Binary Cross-Entropy = -log P(y|x) for Bernoulli
Cross-Entropy Loss = -log P(y|x) for Categorical
MSE Loss ∝ -log P(y|x) for Gaussian (fixed σ)

Training = Maximum Likelihood = Minimize NLL
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 🎥 | 3Blue1Brown: Probability | [YouTube](https://www.youtube.com/watch?v=HZGCoVF3YvM) |
| 🎥 | StatQuest: Distributions | [YouTube](https://www.youtube.com/watch?v=rzFX5NWojp0) |
| 📖 | Bishop PRML Ch. 2 | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🇨🇳 | 概率分布详解 | [知乎](https://zhuanlan.zhihu.com/p/24648612) |
| 🇨🇳 | 概率论基础 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |
| 🇨🇳 | 机器学习概率分布 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88716548)

---

## 🔗 Where Probability Distributions Are Used

| Application | Distribution Used |
|-------------|-------------------|
| **Binary Classification** | Bernoulli → BCE loss |
| **Multi-class Classification** | Categorical → Cross-entropy loss |
| **Regression** | Gaussian → MSE loss |
| **Language Models** | Categorical over vocabulary |
| **VAE Latent Space** | Gaussian prior and posterior |
| **Diffusion Models** | Gaussian noise at each step |
| **Bayesian Neural Networks** | Gaussian weight priors |
| **Hidden Markov Models** | Multinomial emissions |

---


⬅️ [Back: Random Variables](../random-variables/) | ➡️ [Next: Expectation](../expectation/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
