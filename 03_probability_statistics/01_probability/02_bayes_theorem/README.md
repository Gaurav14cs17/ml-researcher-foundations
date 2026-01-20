<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Bayes%20Theorem&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/bayes-mle-map-complete.svg" width="100%">

*Caption: Bayes' theorem relates prior beliefs to posterior beliefs after observing data. It is the foundation of Bayesian ML.*

---

## 📐 The Formula

### Bayes' Theorem

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

**In Machine Learning notation:**

$$
P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}
$$

| Term | Name | Description |
|------|------|-------------|
| $P(\theta\|D)$ | **Posterior** | Updated belief about θ after seeing data |
| $P(D\|\theta)$ | **Likelihood** | How likely is the data given θ |
| $P(\theta)$ | **Prior** | Initial belief about θ |
| $P(D)$ | **Evidence** | Marginal likelihood (normalizing constant) |

**Simplified (unnormalized) form:**

$$
\text{Posterior} \propto \text{Likelihood} \times \text{Prior}
$$

---

## 📐 Derivation and Proof

### Step 1: Conditional Probability Definition

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
P(B|A) = \frac{P(A \cap B)}{P(A)}
$$

### Step 2: Express Joint Probability Two Ways

From the above equations:

$$
P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
$$

### Step 3: Rearrange to Get Bayes' Theorem

$$
P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \quad \blacksquare
$$

---

## 📐 Evidence (Marginal Likelihood)

The evidence $P(D)$ is computed using the **Law of Total Probability**:

**Discrete:**

$$
P(D) = \sum_{\theta} P(D|\theta) P(\theta)
$$

**Continuous:**

$$
P(D) = \int P(D|\theta) P(\theta) \, d\theta
$$

This integral is often **intractable** → need approximations:
- MCMC (Markov Chain Monte Carlo)
- Variational Inference
- Laplace Approximation

---

## 📐 Sequential Bayesian Updates

**Key Property:** Posterior becomes prior for next observation.

After observing $D\_1$:

$$
P(\theta|D_1) \propto P(D_1|\theta) P(\theta)
$$

After observing $D\_2$:

$$
P(\theta|D_1, D_2) \propto P(D_2|\theta) P(\theta|D_1)
$$

**For i.i.d. data:**

$$
P(\theta|D_1, \ldots, D_n) \propto P(\theta) \prod_{i=1}^{n} P(D_i|\theta)
$$

---

## 📐 Bayesian vs Frequentist Inference

| Aspect | Bayesian | Frequentist |
|--------|----------|-------------|
| Parameters | Random variables with distributions | Fixed but unknown constants |
| Inference | Compute $P(\theta\|D)$ | Point estimate $\hat{\theta}$ |
| Uncertainty | Full posterior distribution | Confidence intervals |
| Prior | Explicit and required | Not used (or implicit) |
| Interpretation | Degree of belief | Long-run frequency |

---

## 📐 MLE vs MAP

### Maximum Likelihood Estimation (MLE)

$$
\theta_{MLE} = \arg\max_\theta P(D|\theta)
$$

Ignores prior → can overfit with limited data.

### Maximum A Posteriori (MAP)

$$
\theta_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta P(D|\theta) P(\theta)
$$

**Equivalence to Regularization:**
- Gaussian prior $\mathcal{N}(0, \sigma^2)$ → L2 regularization
- Laplace prior → L1 regularization

$$
\log P(\theta|D) = \log P(D|\theta) + \log P(\theta)
$$

If $P(\theta) = \mathcal{N}(0, \sigma^2)$:

$$
\log P(\theta) = -\frac{\theta^2}{2\sigma^2} + \text{const} = -\lambda\|\theta\|^2
$$

---

## 💻 Code Examples

```python
import numpy as np
from scipy import stats

# Medical diagnosis example
def bayesian_diagnosis(prior_disease, sensitivity, specificity):
    """
    Compute P(disease | positive test) using Bayes theorem
    
    sensitivity = P(positive | disease) = True Positive Rate
    specificity = P(negative | no disease) = True Negative Rate
    """

    # P(positive | no disease) = 1 - specificity (False Positive Rate)
    false_positive_rate = 1 - specificity
    
    # P(positive) = P(positive|disease)P(disease) + P(positive|no disease)P(no disease)
    p_positive = sensitivity * prior_disease + false_positive_rate * (1 - prior_disease)
    
    # Bayes theorem
    posterior = (sensitivity * prior_disease) / p_positive
    
    return posterior

# Example: COVID test
prior = 0.01  # 1% base rate
sensitivity = 0.95  # 95% true positive rate
specificity = 0.99  # 99% true negative rate

posterior = bayesian_diagnosis(prior, sensitivity, specificity)
print(f"P(COVID | positive test) = {posterior:.4f}")  # ≈ 0.49, not 0.95!

# Sequential Bayesian update with Beta-Bernoulli
def beta_bernoulli_update(alpha, beta, data):
    """
    Conjugate update: Beta(α, β) + Bernoulli observations
    Posterior: Beta(α + successes, β + failures)
    """
    successes = sum(data)
    failures = len(data) - successes
    return alpha + successes, beta + failures

# Start with uniform prior
alpha, beta = 1, 1

# Observe coin flips
data = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0]  # 7 heads, 3 tails

# Update
alpha_post, beta_post = beta_bernoulli_update(alpha, beta, data)
posterior = stats.beta(alpha_post, beta_post)

print(f"Posterior: Beta({alpha_post}, {beta_post})")
print(f"Posterior mean: {alpha_post / (alpha_post + beta_post):.3f}")
print(f"95% Credible interval: {posterior.interval(0.95)}")

# Compare MLE vs MAP
mle = sum(data) / len(data)  # 0.7
map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)  # Mode of Beta

print(f"MLE: {mle:.3f}")
print(f"MAP: {map_estimate:.3f}")
```

---

## 🌍 Applications

| Application | Usage |
|-------------|-------|
| **Naive Bayes** | Text classification using Bayes theorem |
| **Bayesian Neural Networks** | Posterior over weights for uncertainty |
| **Gaussian Processes** | Posterior distribution over functions |
| **MCMC** | Sampling from posterior distribution |
| **A/B Testing** | Bayesian hypothesis testing |
| **Spam Filtering** | Naive Bayes on word frequencies |
| **Medical Diagnosis** | Prior disease rate × test sensitivity |
| **VAE** | Approximate posterior via variational inference |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Jaynes: Probability Theory | [Cambridge](https://www.cambridge.org/core/books/probability-theory/9CA08E224FF30123304E6D8935CF1A99) |
| 📖 | Bishop PRML | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🎥 | 3Blue1Brown: Bayes | [YouTube](https://www.youtube.com/watch?v=HZGCoVF3YvM) |
| 🎥 | StatQuest: Bayes | [YouTube](https://www.youtube.com/watch?v=9wCnvr7Xw4E) |
| 🇨🇳 | 贝叶斯定理详解 | [知乎](https://zhuanlan.zhihu.com/p/26262151) |
| 🇨🇳 | 贝叶斯统计 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

⬅️ [Back: Bayesian Inference](../01_bayes/) | ➡️ [Next: Conditional Probability](../03_conditional/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
