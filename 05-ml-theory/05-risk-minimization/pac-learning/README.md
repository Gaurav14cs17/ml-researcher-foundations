# PAC Learning

> **Probably Approximately Correct learning framework**

---

## 🎯 Visual Overview

<img src="./images/pac-learning.svg" width="100%">

*Caption: PAC learning provides theoretical guarantees: with probability (1-δ), the learned hypothesis has error at most ε. The sample complexity depends on the hypothesis class size and desired accuracy/confidence.*

---

## 📂 Overview

PAC learning is a foundational framework for understanding when and how machine learning is possible. It gives bounds on sample complexity and generalization.

---

## 🔑 Key Concepts

| Concept | Definition |
|---------|------------|
| **ε (epsilon)** | Accuracy parameter - max allowed error |
| **δ (delta)** | Confidence parameter - failure probability |
| **Sample Complexity** | Number of samples needed |
| **Hypothesis Class H** | Set of possible learners |

---

## 📐 PAC Guarantee

```
A concept class C is PAC-learnable if there exists an algorithm A such that:

For any ε > 0, δ > 0, and distribution D:
With m ≥ poly(1/ε, 1/δ, n) samples,
A outputs h with:

P[error(h) ≤ ε] ≥ 1 - δ

Sample complexity for finite H:
m ≥ (1/ε)[ln|H| + ln(1/δ)]
```

---

## 🌍 Implications for ML

| Insight | Practical Meaning |
|---------|------------------|
| More data → better | Error decreases with samples |
| Simpler models need less data | Smaller H = easier learning |
| No free lunch | Must assume structure |
| VC dimension | For infinite H |

---

## 💻 Code

```python
import numpy as np

def pac_sample_complexity(hypothesis_class_size, epsilon, delta):
    """
    Minimum samples for PAC guarantee with finite hypothesis class.
    
    Args:
        hypothesis_class_size: |H|
        epsilon: accuracy parameter
        delta: confidence parameter
    
    Returns:
        Minimum number of samples needed
    """
    return (1/epsilon) * (np.log(hypothesis_class_size) + np.log(1/delta))

# Example: Learning boolean functions with 10 features
# H = 2^(2^10) possible functions, want ε=0.1, δ=0.05
m = pac_sample_complexity(2**(2**10), 0.1, 0.05)
print(f"Need at least {m:.0f} samples")
```


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: Risk Minimization](../)

---

⬅️ [Back: Erm](../erm/) | ➡️ [Next: Structural Risk](../structural-risk/)
