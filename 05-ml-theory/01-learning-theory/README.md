<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=01 Learning Theory&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📚 Learning Theory

> **Mathematical foundations of machine learning**

---

## 🎯 Visual Overview

<img src="./pac-learning/images/pac-vc-dimension-complete.svg" width="100%">

*Caption: Learning theory provides mathematical guarantees for when and why machine learning works. Key concepts include PAC learning, VC dimension, and sample complexity.*

---

## 📐 Mathematical Foundations

### PAC Learning

```
Probably Approximately Correct (PAC):

A concept class C is PAC-learnable if:
∃ algorithm A, polynomial p(·,·,·,·) such that:

For all:
• c ∈ C (target concept)
• D (distribution)
• ε > 0 (accuracy)
• δ > 0 (confidence)

Using m ≥ p(1/ε, 1/δ, n, size(c)) samples:
P[error(h) ≤ ε] ≥ 1 - δ
```

### VC Dimension

```
VC Dimension = Maximum number of points that can be shattered

Definition of shattering:
A set S is shattered by H if:
∀ labeling of S, ∃ h ∈ H that achieves it

Examples:
• Linear classifiers in ℝ²: VC = 3
• Linear classifiers in ℝᵈ: VC = d + 1
• Decision stumps: VC = 1
• k-NN (k=1): VC = ∞
```

### Generalization Bound

```
With probability ≥ 1 - δ:

R(h) ≤ R̂(h) + √((d·log(2n/d) + log(1/δ)) / n)

Where:
• R(h) = true risk (test error)
• R̂(h) = empirical risk (training error)
• d = VC dimension
• n = sample size
```

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [pac-learning/](./pac-learning/) | PAC framework | Sample complexity, efficiency |

---

## 🎯 Key Concepts

| Concept | Definition | Significance |
|---------|------------|--------------|
| **PAC Learnable** | Can learn with polynomial samples | Efficient learning is possible |
| **VC Dimension** | Capacity of hypothesis class | Controls generalization |
| **Sample Complexity** | Samples needed for accuracy ε | How much data needed? |
| **Rademacher Complexity** | Data-dependent capacity | Tighter bounds |

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

# VC Dimension intuition
# Linear classifier in d dimensions has VC = d + 1

def can_shatter_points(X, y_all_labelings, classifier):
    """Check if classifier can shatter points X"""
    for y in y_all_labelings:
        clf = classifier()
        try:
            clf.fit(X, y)
            if not np.allclose(clf.predict(X), y):
                return False
        except:
            return False
    return True

# Example: 3 points in 2D
X = np.array([[0, 0], [1, 0], [0, 1]])

# All 2^3 = 8 labelings
from itertools import product
all_labelings = list(product([0, 1], repeat=3))

# Linear classifier can shatter 3 points in 2D (VC = 3)
print("Testing linear classifier on 3 points in 2D")
for labeling in all_labelings:
    y = np.array(labeling)
    clf = Perceptron()
    clf.fit(X, y)
    pred = clf.predict(X)
    print(f"Labeling {labeling}: {'✓' if np.allclose(pred, y) else '✗'}")

# Sample complexity bound
def sample_complexity(vc_dim, epsilon, delta):
    """PAC sample complexity bound"""
    # Fundamental theorem of learning
    m = (4 / epsilon) * (vc_dim * np.log(12 / epsilon) + np.log(2 / delta))
    return int(np.ceil(m))

vc = 10  # Hypothesis class VC dimension
eps = 0.1  # 10% error tolerance
delta = 0.05  # 95% confidence

m = sample_complexity(vc, eps, delta)
print(f"Need at least {m} samples for VC={vc}, ε={eps}, δ={delta}")
```

---

## 🌍 ML Applications

| Application | How Learning Theory Is Used |
|-------------|----------------------------|
| **Model Selection** | VC dimension guides capacity |
| **Sample Size** | Theory tells us how much data |
| **Generalization** | Bounds on test error |
| **Algorithm Design** | Efficient learning guarantees |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Understanding ML (Shalev-Shwartz) | [Book](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📖 | Foundations of ML (Mohri) | [Book](https://cs.nyu.edu/~mohri/mlbook/) |
| 📄 | VC Dimension Original | [Paper](https://link.springer.com/article/10.1007/BF00116037) |
| 🇨🇳 | PAC学习理论 | [知乎](https://zhuanlan.zhihu.com/p/35261164) |
| 🇨🇳 | VC维详解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88882888) |

---

## 🔗 Where This Topic Is Used

| Application | How Learning Theory Is Used |
|-------------|----------------------------|
| **Generalization Analysis** | Bounds on test error |
| **Model Complexity** | VC dimension as capacity measure |
| **Sample Efficiency** | How much data is enough? |
| **Algorithm Correctness** | Provable learning guarantees |
| **Neural Network Theory** | Understanding deep learning |

---

⬅️ [Back: ML Theory](../) | ➡️ [Next: 02-Generalization](../02-generalization/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

