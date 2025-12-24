<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Bayes&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🎯 Bayesian Inference

> **The foundation of probabilistic machine learning**

<img src="./images/bayes-inference.svg" width="100%">

---

## 🎯 Core Concept

Bayesian inference is the process of updating beliefs about parameters based on observed data using Bayes' theorem.

---

## 📐 Mathematical Framework

### Bayes' Theorem

```
P(θ|D) = P(D|θ)P(θ) / P(D)

Where:
• P(θ|D) = Posterior (updated belief)
• P(D|θ) = Likelihood (data given parameters)
• P(θ)   = Prior (initial belief)
• P(D)   = Evidence (normalization constant)
```

### Posterior Computation

```
Posterior ∝ Likelihood × Prior

P(θ|D) ∝ P(D|θ) P(θ)

Evidence (marginal likelihood):
P(D) = ∫ P(D|θ) P(θ) dθ
```

---

## 🔑 Key Concepts

### Prior Distribution P(θ)

Represents initial beliefs before seeing data:
- **Informative prior**: Strong beliefs (e.g., θ ~ N(0, 0.1²))
- **Non-informative prior**: Weak beliefs (e.g., θ ~ N(0, 100²))
- **Conjugate prior**: Posterior has same form as prior

### Likelihood P(D|θ)

Probability of observing data given parameters:
```
For i.i.d. data D = {x₁, ..., xₙ}:
P(D|θ) = ∏ᵢ P(xᵢ|θ)

Log-likelihood:
log P(D|θ) = Σᵢ log P(xᵢ|θ)
```

### Posterior Distribution P(θ|D)

Updated beliefs after seeing data:
- Combines prior knowledge with data
- Quantifies uncertainty about parameters
- Used for prediction and decision-making

---

## 💻 Bayesian Inference Methods

### 1. Conjugate Analysis (Analytical)

```
If prior and likelihood are conjugate:
→ Posterior has closed form

Example (Beta-Bernoulli):
Prior:      θ ~ Beta(α, β)
Likelihood: X|θ ~ Bernoulli(θ)
Posterior:  θ|D ~ Beta(α + Σxᵢ, β + n - Σxᵢ)
```

### 2. Markov Chain Monte Carlo (MCMC)

```
When posterior is intractable:
→ Sample from P(θ|D) using MCMC

Methods:
• Metropolis-Hastings
• Gibbs Sampling
• Hamiltonian Monte Carlo (HMC)
```

### 3. Variational Inference

```
Approximate P(θ|D) with simpler q(θ):
→ Minimize KL(q||p)

ELBO = E_q[log P(D|θ)] - KL(q(θ)||P(θ))
```

---

## 🔗 Applications in Machine Learning

### Bayesian Linear Regression

```
Prior:      w ~ N(0, α⁻¹I)
Likelihood: y|X,w ~ N(Xw, β⁻¹I)
Posterior:  w|D ~ N(μₙ, Σₙ)

μₙ = βΣₙXᵀy
Σₙ = (αI + βXᵀX)⁻¹
```

### Bayesian Neural Networks

```
Place prior over weights:
w ~ P(w)

Posterior:
P(w|D) = P(D|w)P(w) / P(D)

Prediction (Bayesian Model Averaging):
P(y*|x*, D) = ∫ P(y*|x*, w) P(w|D) dw
```

### Gaussian Processes

```
Prior over functions:
f ~ GP(m, k)

Posterior (given data):
f*|D ~ N(μ*, Σ*)

Closed-form predictive distribution!
```

---

## 🎯 Bayesian vs Frequentist

| Aspect | Bayesian | Frequentist |
|--------|----------|-------------|
| **Parameters** | Random variables | Fixed unknowns |
| **Inference** | P(θ\|D) | Point estimate θ̂ |
| **Uncertainty** | Posterior distribution | Confidence intervals |
| **Prior** | Required | Not used |
| **Interpretation** | Probability = belief | Probability = frequency |

---

## 🔥 Modern Applications

| Application | How Bayesian Inference is Used |
|-------------|-------------------------------|
| **Bayesian Optimization** | Model P(f\|D) for hyperparameter tuning |
| **Variational Autoencoders** | Variational inference for latent variables |
| **Bayesian Deep Learning** | Uncertainty quantification in NNs |
| **Thompson Sampling** | Bayesian bandits for exploration |
| **Probabilistic Programming** | PyMC, Stan, Pyro for modeling |
| **Active Learning** | Posterior uncertainty guides data collection |

---

## 💡 Advantages of Bayesian Approach

✅ **Uncertainty Quantification**
- Full posterior distribution, not just point estimate
- Know when model is uncertain

✅ **Principled Model Comparison**
- Marginal likelihood P(D|M) for model selection
- Bayes factors for hypothesis testing

✅ **Incorporate Prior Knowledge**
- Use domain expertise
- Transfer learning via informative priors

✅ **Natural Regularization**
- Prior acts as regularizer
- Prevents overfitting

✅ **Sequential Learning**
- Today's posterior = tomorrow's prior
- Online learning naturally

---

## 📊 Common Conjugate Pairs

| Likelihood | Prior | Posterior |
|------------|-------|-----------|
| Bernoulli(θ) | Beta(α, β) | Beta(α + Σx, β + n - Σx) |
| Poisson(λ) | Gamma(α, β) | Gamma(α + Σx, β + n) |
| Normal(μ, σ²) | Normal(μ₀, σ₀²) | Normal(μₙ, σₙ²) |
| Multinomial(θ) | Dirichlet(α) | Dirichlet(α + counts) |

---

## 🛠️ Tools and Libraries

| Tool | Description | Link |
|------|-------------|------|
| **PyMC** | Probabilistic programming in Python | [pymc.io](https://www.pymc.io/) |
| **Stan** | Bayesian inference with HMC | [mc-stan.org](https://mc-stan.org/) |
| **Pyro** | Deep probabilistic programming | [pyro.ai](https://pyro.ai/) |
| **TensorFlow Probability** | Bayesian layers, MCMC | [tfp](https://www.tensorflow.org/probability) |
| **Edward2** | Probabilistic programming | [Edward](https://github.com/google/edward2) |

---

## 💻 Example Code

```python
import numpy as np
from scipy import stats

# Bayesian coin flip example
# Prior: Beta(2, 2) (slightly biased toward fair)
alpha_prior, beta_prior = 2, 2

# Observe data: 7 heads out of 10 flips
heads, tails = 7, 3

# Posterior: Beta(alpha + heads, beta + tails)
alpha_post = alpha_prior + heads
beta_post = beta_prior + tails

# Posterior distribution
posterior = stats.beta(alpha_post, beta_post)

# Point estimate (posterior mean)
theta_mean = alpha_post / (alpha_post + beta_post)
print(f"Posterior mean: {theta_mean:.3f}")

# 95% credible interval
ci_low, ci_high = posterior.interval(0.95)
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

# Posterior predictive: P(next flip is heads | data)
prob_heads = alpha_post / (alpha_post + beta_post)
print(f"P(next heads): {prob_heads:.3f}")
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Pattern Recognition and ML (Bishop) | Ch. 1-3 |
| 📖 | Bayesian Data Analysis (Gelman) | [Book](http://www.stat.columbia.edu/~gelman/book/) |
| 📄 | Practical Bayesian Optimization | [arXiv](https://arxiv.org/abs/1807.02811) |
| 📄 | Weight Uncertainty in Neural Networks | [arXiv](https://arxiv.org/abs/1505.05424) |
| 🎥 | Bayesian Methods for ML | [Coursera](https://www.coursera.org/learn/bayesian-methods-in-machine-learning) |
| 🇨🇳 | 贝叶斯推断详解 | [知乎](https://zhuanlan.zhihu.com/p/37976562) |
| 🇨🇳 | 贝叶斯机器学习 | [CSDN](https://blog.csdn.net/qq_40027052/article/details/78735444) |

---

⬅️ [Back: Expectation](../expectation/) | ➡️ [Next: Conditional Probability](../conditional/)


---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
