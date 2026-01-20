<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Probability%20Theory&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/probability-ml-view.svg" width="100%">

*Caption: Probability provides the mathematical framework for ML. P(Y|X,θ) defines predictions, P(D|θ) is the likelihood we maximize in training (MLE), and P(θ|D) is the posterior in Bayesian learning.*

---

## 📂 Topics in This Folder

| Folder | Topics | ML Application |
|--------|--------|----------------|
| [01_bayes/](./01_bayes/) | Bayesian Inference | Posterior computation |
| [02_bayes_theorem/](./02_bayes_theorem/) | Bayes' Theorem | Prior → Posterior |
| [03_conditional/](./03_conditional/) | Conditional probability | 🔥 Bayesian ML |
| [04_distributions/](./04_distributions/) | PMF, PDF, common distributions | Likelihood |
| [05_expectation/](./05_expectation/) | Mean, variance, moments | Loss functions |
| [06_limit_theorems/](./06_limit_theorems/) | LLN, CLT, concentration | Generalization |
| [07_random_variables/](./07_random_variables/) | Discrete, continuous RVs | Data modeling |
| [08_spaces/](./08_spaces/) | Sample space, σ-algebra | Rigorous foundations |

---

## 📐 Mathematical Foundations

### Kolmogorov Axioms

For a probability measure $P$ on sample space $\Omega$:

```math
\textbf{Axiom 1: } P(A) \geq 0 \text{ for all events } A
\textbf{Axiom 2: } P(\Omega) = 1
\textbf{Axiom 3: } P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i) \text{ for disjoint } A_i
```

### Derived Properties

**Complement:**
```math
P(A^c) = 1 - P(A)
```

**Inclusion-Exclusion:**
```math
P(A \cup B) = P(A) + P(B) - P(A \cap B)
```

**Union Bound:**
```math
P\left(\bigcup_{i=1}^{n} A_i\right) \leq \sum_{i=1}^{n} P(A_i)
```

---

## 📐 Bayes' Theorem

```math
P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}
```

| Term | Name | Interpretation |
|------|------|----------------|
| $P(\theta\|D)$ | **Posterior** | Updated belief after data |
| $P(D\|\theta)$ | **Likelihood** | How likely data given θ |
| $P(\theta)$ | **Prior** | Initial belief |
| $P(D)$ | **Evidence** | Normalizing constant |

**Simplified:**
```math
\text{Posterior} \propto \text{Likelihood} \times \text{Prior}
```

---

## 📐 Expectation and Variance

### Expectation

```math
E[X] = \begin{cases} \sum_x x \cdot P(X=x) & \text{discrete} \\ \int x \cdot f(x) \, dx & \text{continuous} \end{cases}
```

**Properties:**
- Linearity: $E[aX + bY] = aE[X] + bE[Y]$ (always!)
- LOTUS: $E[g(X)] = \sum\_x g(x) P(X=x)$

### Variance

```math
\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2
```

**Properties:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$

---

## 📐 Key Distributions

### Discrete

| Distribution | PMF | Mean | Variance |
|--------------|-----|------|----------|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| Binomial($n,p$) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Poisson($\lambda$) | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ |

### Continuous

| Distribution | PDF | Mean | Variance |
|--------------|-----|------|----------|
| Uniform($a,b$) | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| Gaussian($\mu,\sigma^2$) | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |

---

## 📐 Limit Theorems

### Law of Large Numbers

```math
\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{p} \mu \quad \text{as } n \to \infty
```

### Central Limit Theorem

```math
\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty
```

**Key insight:** Sum of ANY i.i.d. random variables → Gaussian!

---

## 🎯 The Probabilistic View of ML

```
Machine Learning as Probability:

Data:    D = {(x₁,y₁), ..., (xₙ,yₙ)}
Model:   p(y|x,θ) - probabilistic prediction
Goal:    Find θ that maximizes p(D|θ)  ← MLE!

This perspective gives us:
• Loss functions = negative log-likelihood
• Regularization = priors on θ
• Uncertainty quantification
• Generalization theory
```

---

## 💻 Code Examples

```python
import numpy as np
from scipy import stats
import torch

# Bayes' theorem: Medical diagnosis
def posterior_probability(prior, sensitivity, specificity):
    """P(disease | positive test)"""
    p_positive = sensitivity * prior + (1 - specificity) * (1 - prior)
    return (sensitivity * prior) / p_positive

# Example: 1% base rate, 95% sensitivity, 99% specificity
posterior = posterior_probability(0.01, 0.95, 0.99)
print(f"P(disease | positive) = {posterior:.3f}")  # ≈ 0.49

# Expectation and variance
samples = np.random.normal(0, 1, 10000)
print(f"E[X] ≈ {np.mean(samples):.4f}")
print(f"Var[X] ≈ {np.var(samples):.4f}")

# CLT demonstration
n_samples = 50
n_trials = 1000
sample_means = [np.random.uniform(0, 1, n_samples).mean() for _ in range(n_trials)]
print(f"Sample means are Gaussian: {stats.normaltest(sample_means).pvalue:.4f}")

# PyTorch distributions
dist = torch.distributions.Normal(0, 1)
samples = dist.sample((1000,))
log_prob = dist.log_prob(samples)  # For likelihood computation
```

---

## 🌍 Why Probability for ML?

| Concept | ML Translation |
|---------|----------------|
| $P(Y\|X)$ | Prediction |
| $P(D\|\theta)$ | Likelihood (training objective) |
| $P(\theta\|D)$ | Posterior (Bayesian learning) |
| $E[L]$ | Expected loss (what we minimize) |
| $\text{Var}(\nabla L)$ | Gradient variance (SGD noise) |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Probability Theory (Jaynes) | [Book](https://bayes.wustl.edu/etj/prob/book.pdf) |
| 📖 | Pattern Recognition (Bishop) | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🎥 | Stats 110 (Harvard) | [YouTube](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 🇨🇳 | 概率论基础 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 机器学习概率视角 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 概率论讲解 | [B站](https://www.bilibili.com/video/BV164411b7dx) |

---

## 🔗 Where Probability Is Used in ML

| Application | How Probability Is Used |
|-------------|------------------------|
| **Cross-Entropy Loss** | $-\log P(y\|x)$ = negative log-likelihood |
| **Softmax** | Converts logits to probability distribution |
| **VAE** | KL divergence between distributions |
| **Bayesian NN** | Posterior over weights |
| **Gaussian Processes** | Prior over functions |
| **Diffusion Models** | Forward/reverse probability flow |
| **Dropout** | Bernoulli mask at training time |

---

⬅️ [Back: Probability & Statistics](../) | ➡️ [Next: Multivariate](../02_multivariate/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
