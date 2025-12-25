<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Limit%20Theorems&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/limit-theorems.svg" width="100%">

*Caption: LLN: sample mean → population mean. CLT: sum of any iid RVs → Gaussian. These explain why sampling works and why Gaussians appear everywhere. Foundation of confidence intervals and hypothesis testing.*

---

## 📂 Overview

Limit theorems explain why statistics and machine learning work. They guarantee that averaging many samples gives accurate estimates, and that sums become approximately Gaussian.

---

## 📐 Mathematical Definitions

### Law of Large Numbers (LLN)
```
Weak LLN:
For iid X₁, X₂, ..., Xₙ with E[Xᵢ] = μ:

X̄ₙ = (1/n)Σᵢ Xᵢ →ᵖ μ  as n → ∞

Sample mean converges in probability to true mean.
```

### Central Limit Theorem (CLT)
```
For iid X₁, ..., Xₙ with E[Xᵢ]=μ, Var[Xᵢ]=σ²:

√n(X̄ₙ - μ)/σ →ᵈ N(0, 1)  as n → ∞

Equivalently: X̄ₙ ~ N(μ, σ²/n) approximately

The sum of any iid RVs → Gaussian!
```

### Convergence Rates
```
LLN: |X̄ₙ - μ| = O(1/√n) typically

Concentration inequalities (tighter):
Hoeffding: P(|X̄ₙ - μ| > ε) ≤ 2exp(-2nε²/(b-a)²)
Chernoff:  P(X̄ₙ > (1+δ)μ) ≤ exp(-δ²μn/3)
```

### Berry-Esseen Theorem
```
|P(√n(X̄ₙ-μ)/σ ≤ z) - Φ(z)| ≤ C·E[|X|³]/(σ³√n)

Rate of convergence to Gaussian: O(1/√n)
```

### Why It Matters for ML
```
1. SGD uses mini-batch averages
   ∇̂f ≈ E[∇f] by LLN
   
2. Test accuracy converges to true accuracy
   Acc_test →ᵖ Acc_true
   
3. Confidence intervals
   μ̂ ± z·σ̂/√n
```

---

## 💻 Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Demonstrate LLN
np.random.seed(42)
true_mean = 5
samples = np.random.exponential(true_mean, 10000)

# Running average converges to true mean
running_avg = np.cumsum(samples) / np.arange(1, len(samples)+1)
# running_avg[-1] ≈ 5

# Demonstrate CLT
n_trials = 1000
sample_size = 50
sample_means = [np.random.uniform(0, 1, sample_size).mean() for _ in range(n_trials)]
# Distribution of means is approximately Normal!
stats.normaltest(sample_means)  # Should not reject null

# Confidence interval
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error = σ/√n
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

# Monte Carlo integration (uses LLN)
def monte_carlo_integral(f, a, b, n_samples=10000):
    """∫f(x)dx ≈ (b-a) × (1/n)Σf(xᵢ) for xᵢ ~ Uniform(a,b)"""
    x = np.random.uniform(a, b, n_samples)
    return (b - a) * np.mean(f(x))
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Ross: Probability | [Book](https://www.elsevier.com/books/a-first-course-in-probability/ross/978-0-321-79477-2) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 📖 | Wasserman: All of Statistics | [Book](https://www.stat.cmu.edu/~larry/all-of-statistics/) |
| 🇨🇳 | 大数定律与中心极限 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 概率收敛 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 概率论课程 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: Conditional](../conditional/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
