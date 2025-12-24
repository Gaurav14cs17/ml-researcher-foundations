# Sample Spaces and Events

> **The foundation of probability theory**

---

## 🎯 Visual Overview

<img src="./images/sample-spaces.svg" width="100%">

*Caption: Sample space Ω contains all possible outcomes. Events are subsets of Ω. Kolmogorov axioms define probability: non-negativity, normalization (P(Ω)=1), and additivity.*

---

## 📂 Overview

Sample spaces and events are the building blocks of probability theory. Understanding these concepts is essential for modeling uncertainty in machine learning.

---

## 📐 Mathematical Definitions

### Sample Space (Ω)
```
Ω = set of all possible outcomes

Examples:
• Coin flip: Ω = {H, T}
• Die roll: Ω = {1, 2, 3, 4, 5, 6}
• Real number: Ω = ℝ
• Image: Ω = [0, 255]^{H×W×3}
```

### Events
```
Event A ⊆ Ω (subset of sample space)

Examples:
• "Heads": A = {H}
• "Even number": A = {2, 4, 6}
• "Pixel > 128": A = {x ∈ [0,255] : x > 128}
```

### Kolmogorov Axioms
```
For probability measure P:

1. Non-negativity: P(A) ≥ 0 for all events A
2. Normalization: P(Ω) = 1
3. Additivity: P(A ∪ B) = P(A) + P(B) if A ∩ B = ∅
```

### Derived Properties
```
P(∅) = 0
P(Aᶜ) = 1 - P(A)
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(A) ≤ P(B) if A ⊆ B
```

### σ-Algebra (Measurable Events)
```
F is a σ-algebra on Ω if:
1. Ω ∈ F
2. A ∈ F ⟹ Aᶜ ∈ F
3. A₁, A₂, ... ∈ F ⟹ ∪ᵢAᵢ ∈ F

Probability space: (Ω, F, P)
```

---

## 💻 Code Examples

```python
import numpy as np
from scipy import stats

# Discrete sample space: die roll
omega = [1, 2, 3, 4, 5, 6]
p = np.ones(6) / 6  # Uniform

# Events as masks
event_even = np.array([x % 2 == 0 for x in omega])
P_even = p[event_even].sum()  # = 0.5

# Continuous: Gaussian
dist = stats.norm(0, 1)
P_positive = 1 - dist.cdf(0)  # P(X > 0) = 0.5

# Monte Carlo probability estimation
def estimate_probability(sample_fn, event_fn, n_samples=10000):
    """P(A) ≈ (1/n)Σ 1[xᵢ ∈ A]"""
    samples = sample_fn(n_samples)
    return event_fn(samples).mean()

# Example: P(X² + Y² < 1) for X,Y ~ N(0,1)
def sample_fn(n): return np.random.randn(n, 2)
def event_fn(xy): return (xy**2).sum(axis=1) < 1
print(estimate_probability(sample_fn, event_fn))
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Ross: Probability | [Book](https://www.elsevier.com/books/a-first-course-in-probability/ross/978-0-321-79477-2) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 📖 | Billingsley: Probability | [Book](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) |
| 🇨🇳 | 概率空间详解 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 概率论基础 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
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

⬅️ [Back: Probability](../) | ➡️ [Next: Random Variables](../random-variables/)
