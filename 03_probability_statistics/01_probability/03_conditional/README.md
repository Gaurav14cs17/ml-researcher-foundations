<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Conditional%20Probability%20%26%20Bayes&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Conditional Probability

### Definition

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad \text{for } P(B) > 0$$

**Interpretation:** "Probability of A given that B occurred"

**Example:**
- A = "patient has disease"

- B = "test is positive"

- P(A|B) = "probability of disease given positive test"

### Properties

1. $P(A|B) \in [0, 1]$

2. $P(\Omega|B) = 1$

3. $P(A \cup C|B) = P(A|B) + P(C|B)$ if $A \cap C = \emptyset$

### Chain Rule

$$P(A \cap B) = P(A|B) \cdot P(B)
P(A \cap B \cap C) = P(A|B,C) \cdot P(B|C) \cdot P(C)$$

**General Form:**

$$P(A_1 \cap \ldots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1,A_2) \cdots P(A_n|A_1,\ldots,A_{n-1})$$

---

## 📐 Bayes' Theorem

### The Formula

$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

| Term | Name | Interpretation |
|------|------|----------------|
| $P(\theta\|D)$ | **Posterior** | What we want (updated belief) |
| $P(D\|\theta)$ | **Likelihood** | How likely is data given θ |
| $P(\theta)$ | **Prior** | Initial belief before data |
| $P(D)$ | **Evidence** | Normalizing constant |

**Simplified form:**

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}
P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$$

### Derivation and Proof

**Step 1:** Start from conditional probability definition:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}
P(B|A) = \frac{P(A \cap B)}{P(A)}$$

**Step 2:** Express joint probability two ways:

$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

**Step 3:** Rearrange to get Bayes' Theorem:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \quad \blacksquare$$

### Law of Total Probability

For a partition $\{A_i\}$ of the sample space:

$$P(B) = \sum_i P(B|A_i) \cdot P(A_i)$$

**Continuous case:**

$$P(D) = \int P(D|\theta) \cdot P(\theta) \, d\theta$$

*Note: This integral is often intractable → need approximations (MCMC, Variational Inference)*

---

## 📐 Independence

### Definition

Events A and B are **independent** if:

$$P(A \cap B) = P(A) \cdot P(B)$$

**Equivalently:**
- $P(A|B) = P(A)$ → B doesn't change A's probability

- $P(B|A) = P(B)$ → A doesn't change B's probability

### Conditional Independence

$A \perp B \mid C$ means:

$$P(A \cap B|C) = P(A|C) \cdot P(B|C)$$

"A and B are independent **GIVEN** C"

**Important:** 

- $A \perp B \mid C$ does **NOT** imply $A \perp B$

- $A \perp B$ does **NOT** imply $A \perp B \mid C$

---

## 📐 Detailed Mathematical Theory

### Proof: Bayes' Rule for Continuous Random Variables

For continuous random variables X (prior) and Y (observation):

**Given:**
- Prior density: $p(x)$

- Likelihood: $p(y|x)$

- Marginal: $p(y) = \int p(y|x)p(x)dx$

**Posterior density:**

$$p(x|y) = \frac{p(y|x)p(x)}{p(y)} = \frac{p(y|x)p(x)}{\int p(y|x')p(x')dx'}$$

**Proof:**

Starting from the definition of conditional density:

$$p(x|y) = \frac{p(x,y)}{p(y)}$$

By the chain rule for densities:

$$p(x,y) = p(y|x)p(x)$$

Substituting:

$$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$$

And $p(y)$ by marginalization:

$$p(y) = \int_{-\infty}^{\infty} p(y|x)p(x)dx \quad \blacksquare$$

### Proof: Chain Rule for Probability

**Theorem:** For events $A_1, A_2, \ldots, A_n$:

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \prod_{k=2}^{n} P(A_k | A_1 \cap \cdots \cap A_{k-1})$$

**Proof by Induction:**

*Base case (n=2):*

$$P(A_1 \cap A_2) = P(A_2|A_1)P(A_1) \quad \checkmark$$

*Inductive step:*
Assume true for n-1. Define $B = A_1 \cap \cdots \cap A_{n-1}$.

$$P(A_1 \cap \cdots \cap A_n) = P(B \cap A_n) = P(A_n|B) \cdot P(B)$$

By induction hypothesis:

$$P(B) = P(A_1) \prod_{k=2}^{n-1} P(A_k | A_1 \cap \cdots \cap A_{k-1})$$

Therefore:

$$P(A_1 \cap \cdots \cap A_n) = P(A_1) \prod_{k=2}^{n} P(A_k | A_1 \cap \cdots \cap A_{k-1}) \quad \blacksquare$$

### Proof: Independence Implies Zero Covariance

**Theorem:** If X and Y are independent, then $\text{Cov}(X,Y) = 0$.

**Proof:**

$$\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$$

For independent random variables:

$$E[XY] = \int\int xy \cdot p(x,y) \, dx\,dy$$

Since $p(x,y) = p(x)p(y)$ for independent variables:

$$E[XY] = \int\int xy \cdot p(x)p(y) \, dx\,dy = \left(\int x \cdot p(x)dx\right)\left(\int y \cdot p(y)dy\right) = E[X]E[Y]$$

Therefore:

$$\text{Cov}(X,Y) = E[X]E[Y] - E[X]E[Y] = 0 \quad \blacksquare$$

*Note: The converse is **NOT** true! Zero covariance does not imply independence.*

---

## 📐 Beta-Binomial Conjugacy (Proof)

**Setup:**
- Prior: $\theta \sim \text{Beta}(\alpha, \beta)$

- Likelihood: $X|\theta \sim \text{Binomial}(n, \theta)$

**Theorem:** Posterior is $\theta|X \sim \text{Beta}(\alpha + x, \beta + n - x)$

**Proof:**

Prior density:

$$p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}$$

Likelihood:

$$P(X=x|\theta) = \binom{n}{x}\theta^x(1-\theta)^{n-x}$$

Posterior (using Bayes):

$$p(\theta|X=x) \propto p(X=x|\theta) \cdot p(\theta)
\propto \theta^x(1-\theta)^{n-x} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}
= \theta^{\alpha+x-1}(1-\theta)^{\beta+n-x-1}$$

This is the kernel of $\text{Beta}(\alpha+x, \beta+n-x)$. $\quad \blacksquare$

**Interpretation:**
- $\alpha$: "prior successes"

- $\beta$: "prior failures"  

- After observing x successes in n trials: add x to α, add (n-x) to β

---

## 🌍 ML Applications

### Bayesian Machine Learning

**Training:**
- Prior: $P(\theta)$ ← Initial belief about parameters

- Likelihood: $P(D|\theta)$ ← How well does θ explain data

- Posterior: $P(\theta|D) \propto P(D|\theta)P(\theta)$

**Prediction:**

$$P(y^*|x^*, D) = \int P(y^*|x^*, \theta) P(\theta|D) d\theta$$

*We integrate out the uncertainty in parameters!*

### Naive Bayes Classifier

**Assumption:** Features are conditionally independent given class

$$P(C|x_1,\ldots,x_n) \propto P(C) \cdot \prod_i P(x_i|C)$$

**Why "naive"?** Features are rarely truly independent, but it works well in practice!

### Maximum Likelihood vs MAP

**Maximum Likelihood Estimation (MLE):**

$$\theta_{\text{MLE}} = \arg\max_\theta P(D|\theta) = \arg\max_\theta \log P(D|\theta)$$

**Maximum A Posteriori (MAP):**

$$\theta_{\text{MAP}} = \arg\max_\theta P(\theta|D) = \arg\max_\theta \left[\log P(D|\theta) + \log P(\theta)\right]$$

**Connection to Regularization:**
- **L2 regularization** = Gaussian prior on θ

- **L1 regularization** = Laplace prior on θ

---

## 📊 Bayesian vs Frequentist

| Aspect | Bayesian | Frequentist |
|--------|----------|-------------|
| Parameters | Random variables | Fixed |
| Uncertainty | Posterior distribution | Confidence interval |
| Prior | Explicit | Implicit |
| Interpretation | Degree of belief | Long-run frequency |

---

## 💻 Code Examples

```python
import numpy as np
import torch

# Bayes' theorem example: Medical diagnosis
def bayesian_diagnosis(prior_disease, sensitivity, specificity, test_positive):
    """
    prior_disease: P(disease) - base rate
    sensitivity: P(positive | disease) - true positive rate
    specificity: P(negative | no disease) - true negative rate
    """
    # P(positive)
    p_positive = sensitivity * prior_disease + (1-specificity) * (1-prior_disease)
    
    if test_positive:
        # P(disease | positive) = P(positive | disease) * P(disease) / P(positive)
        posterior = (sensitivity * prior_disease) / p_positive
    else:
        # P(disease | negative)
        p_negative = 1 - p_positive
        posterior = ((1-sensitivity) * prior_disease) / p_negative
    
    return posterior

# Example: COVID test
prior = 0.01  # 1% base rate
sensitivity = 0.95  # 95% true positive
specificity = 0.99  # 99% true negative

posterior = bayesian_diagnosis(prior, sensitivity, specificity, test_positive=True)
print(f"P(COVID | positive test) = {posterior:.3f}")  # Much lower than you'd think!

# Beta-Binomial conjugate update
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

# Naive Bayes in PyTorch style
class NaiveBayes:
    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def fit(self, X, y):
        # Compute class priors P(C)
        self.class_prior = np.bincount(y) / len(y)
        
        # Compute feature likelihoods P(x|C) assuming Gaussian
        self.means = np.array([X[y==c].mean(axis=0) for c in range(self.n_classes)])
        self.stds = np.array([X[y==c].std(axis=0) for c in range(self.n_classes)])
    
    def predict_proba(self, X):
        # log P(C|x) ∝ log P(C) + Σ log P(xᵢ|C)
        log_probs = np.zeros((len(X), self.n_classes))
        for c in range(self.n_classes):
            log_prior = np.log(self.class_prior[c])
            log_likelihood = -0.5 * np.sum(((X - self.means[c]) / self.stds[c])**2, axis=1)
            log_probs[:, c] = log_prior + log_likelihood
        
        # Softmax to get probabilities
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        return probs / probs.sum(axis=1, keepdims=True)

# Bayesian neural network inference (simplified)
def bayesian_predict(models, x):
    """
    Approximate P(y|x,D) by sampling from posterior
    models: list of models sampled from P(θ|D)
    """
    predictions = [model(x) for model in models]
    mean = torch.stack(predictions).mean(dim=0)
    uncertainty = torch.stack(predictions).std(dim=0)
    return mean, uncertainty

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Bishop PRML | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📄 | Jaynes: Probability Theory | [Cambridge](https://www.cambridge.org/core/books/probability-theory/9CA08E224FF30123304E6D8935CF1A99) |
| 📄 | Gelman: Bayesian Data Analysis | [Book](http://www.stat.columbia.edu/~gelman/book/) |
| 🎥 | 3Blue1Brown: Bayes | [YouTube](https://www.youtube.com/watch?v=HZGCoVF3YvM) |
| 🎥 | StatQuest: Bayes | [YouTube](https://www.youtube.com/watch?v=9wCnvr7Xw4E) |
| 🇨🇳 | 贝叶斯定理详解 | [知乎](https://zhuanlan.zhihu.com/p/26262151) |
| 🇨🇳 | 贝叶斯统计 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |
| 🇨🇳 | 朴素贝叶斯分类器 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88819427)

---

## 🔗 Where Bayesian Methods Are Used

| Application | How It's Applied |
|-------------|------------------|
| **Naive Bayes Classifier** | P(class\|features) via Bayes theorem |
| **Bayesian Neural Networks** | Posterior distribution over weights |
| **Gaussian Processes** | Posterior over functions for regression |
| **VAE** | Approximate posterior q(z\|x) via variational inference |
| **Hidden Markov Models** | Forward-backward for P(state\|observations) |
| **A/B Testing** | Bayesian hypothesis testing |
| **Spam Filtering** | Naive Bayes on word frequencies |
| **Medical Diagnosis** | Prior disease rate × test sensitivity |

---

⬅️ [Back: Bayes' Theorem](../02_bayes_theorem/) | ➡️ [Next: Distributions](../04_distributions/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
