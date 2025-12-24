<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Conditional&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Conditional Probability & Bayes Theorem

> **The foundation of Bayesian ML and inference**

<img src="./images/bayes-theorem.svg" width="100%">

---

## 📂 Topics in This Folder

| File | Topic | ML Application |
|------|-------|----------------|
| [bayes-theorem.md](./bayes-theorem.md) | Bayes' theorem | Bayesian inference |

---

## 📐 Conditional Probability

### Definition

```
P(A|B) = P(A ∩ B) / P(B),  for P(B) > 0

"Probability of A given that B occurred"

Example:
• A = "patient has disease"
• B = "test is positive"
• P(A|B) = "probability of disease given positive test"
```

### Properties

```
1. P(A|B) ∈ [0, 1]
2. P(Ω|B) = 1
3. P(A ∪ C|B) = P(A|B) + P(C|B) if A ∩ C = ∅

Chain Rule:
P(A ∩ B) = P(A|B) · P(B)
P(A ∩ B ∩ C) = P(A|B,C) · P(B|C) · P(C)

General:
P(A₁ ∩ ... ∩ Aₙ) = P(A₁) · P(A₂|A₁) · P(A₃|A₁,A₂) · ... · P(Aₙ|A₁,...,Aₙ₋₁)
```

---

## 📐 Bayes' Theorem

### The Formula

```
P(θ|D) = P(D|θ) · P(θ) / P(D)

Where:
• P(θ|D):  Posterior     - what we want (updated belief)
• P(D|θ):  Likelihood    - how likely is data given θ
• P(θ):    Prior         - initial belief before data
• P(D):    Evidence      - normalizing constant

Simplified form:
Posterior ∝ Likelihood × Prior
P(θ|D) ∝ P(D|θ) · P(θ)
```

### Derivation

```
From conditional probability definition:
P(A|B) = P(A ∩ B) / P(B)
P(B|A) = P(A ∩ B) / P(A)

Therefore:
P(A ∩ B) = P(A|B) · P(B) = P(B|A) · P(A)

Rearranging:
P(A|B) = P(B|A) · P(A) / P(B)  ← Bayes' Theorem!
```

### Law of Total Probability

```
P(B) = Σᵢ P(B|Aᵢ) · P(Aᵢ)

Where {Aᵢ} is a partition of sample space.

For continuous case:
P(D) = ∫ P(D|θ) · P(θ) dθ

This is often intractable → need approximations (MCMC, VI)
```

---

## 📐 Independence

### Definition

```
Events A and B are independent if:
P(A ∩ B) = P(A) · P(B)

Equivalently:
P(A|B) = P(A)  (B doesn't change A's probability)
P(B|A) = P(B)  (A doesn't change B's probability)
```

### Conditional Independence

```
A ⊥ B | C  means:
P(A ∩ B|C) = P(A|C) · P(B|C)

"A and B are independent GIVEN C"

Important: A ⊥ B | C does NOT imply A ⊥ B
           A ⊥ B does NOT imply A ⊥ B | C
```

---

## 🌍 ML Applications

### Bayesian Machine Learning

```
Training:
Prior:      P(θ)         ← Initial belief about parameters
Likelihood: P(D|θ)       ← How well does θ explain data
Posterior:  P(θ|D) ∝ P(D|θ)P(θ)

Prediction:
P(y*|x*, D) = ∫ P(y*|x*, θ) P(θ|D) dθ
              -------------------------
              Integrate out uncertainty!
```

### Naive Bayes Classifier

```
Assumption: Features are conditionally independent given class

P(C|x₁,...,xₙ) ∝ P(C) · ∏ᵢ P(xᵢ|C)

Why "naive"? 
Features are rarely truly independent, but it works well!
```

### Maximum Likelihood vs MAP

```
Maximum Likelihood Estimation (MLE):
θ_MLE = argmax P(D|θ)
      = argmax log P(D|θ)

Maximum A Posteriori (MAP):
θ_MAP = argmax P(θ|D)
      = argmax [log P(D|θ) + log P(θ)]
                ----------   ---------
                likelihood   prior = regularization!

L2 regularization = Gaussian prior on θ
L1 regularization = Laplace prior on θ
```

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


⬅️ [Back: Bayes](../bayes/) | ➡️ [Next: Limit Theorems](../limit-theorems/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
