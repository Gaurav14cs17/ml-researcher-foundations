# Model Complexity

> **Measuring the expressiveness of hypothesis classes**

---

## 🎯 Visual Overview

<img src="./images/complexity.svg" width="100%">

*Caption: Model complexity measures include VC dimension (combinatorial), Rademacher complexity (data-dependent), and parameter count (practical). Generalization bounds relate complexity to the gap between training and test error.*

---

## 📂 Overview

Model complexity determines how flexible a model is. Higher complexity allows fitting more patterns but risks overfitting. Understanding complexity is key to generalization theory.

---

## 📐 Mathematical Definitions

### VC Dimension
```
VC(F) = largest n such that F can shatter n points

Shatter: ∀ labelings of n points, ∃f∈F achieving that labeling

Examples:
• Linear classifiers in ℝᵈ: VC = d + 1
• Decision stumps: VC = 2
• Neural nets: VC ∝ number of parameters
```

### Rademacher Complexity
```
R̂ₙ(F) = Eσ[sup_{f∈F} (2/n)Σᵢ σᵢf(xᵢ)]

Where σᵢ ∈ {-1, +1} are random signs (Rademacher RVs)

Measures: How well can F fit random noise?
Data-dependent (unlike VC dimension)
```

### Generalization Bounds
```
VC bound:
R(f) ≤ R̂(f) + O(√(VC(F)/n))

Rademacher bound:
R(f) ≤ R̂(f) + 2Rₙ(F) + O(√(log(1/δ)/n))

Both show: more data → better generalization
          simpler F → better generalization
```

### Neural Network Complexity
```
Parameter counting: Complexity ∝ # parameters

But modern DNNs contradict this!
• Overparameterized: params >> data points
• Yet still generalize well

Implicit regularization:
• SGD biases toward simpler solutions
• Architecture constraints (CNNs, Transformers)
• Lottery ticket hypothesis
```

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

# Estimate effective complexity via learning curves
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5
    )
    # Gap between train and val indicates complexity
    complexity_proxy = train_scores.mean(axis=1) - val_scores.mean(axis=1)
    return complexity_proxy

# VC dimension of linear classifier: d + 1
# For d=2, can shatter 3 points but not 4
def can_shatter_points(X, model_class):
    """Check if model can achieve any labeling"""
    n = len(X)
    for labels in range(2**n):
        y = np.array([(labels >> i) & 1 for i in range(n)])
        model = model_class()
        model.fit(X, y)
        if not np.allclose(model.predict(X), y):
            return False
    return True

# Neural network: count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model complexity heuristics
model = MyTransformer()
print(f"Parameters: {count_parameters(model):,}")
# GPT-3: 175B parameters
# But effective complexity << parameter count
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Shalev-Shwartz Ch. 6 | [Book](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📄 | Rademacher Complexity | [Paper](https://www.jmlr.org/papers/v3/bartlett02a.html) |
| 📄 | Double Descent | [arXiv](https://arxiv.org/abs/1912.02292) |
| 🇨🇳 | VC维详解 | [知乎](https://zhuanlan.zhihu.com/p/38853908) |
| 🇨🇳 | 模型复杂度 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 泛化理论 | [B站](https://www.bilibili.com/video/BV164411b7dx) |

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

⬅️ [Back: complexity](../)

---

⬅️ [Back: Bias Variance](../bias-variance/) | ➡️ [Next: Overfitting](../overfitting/)
