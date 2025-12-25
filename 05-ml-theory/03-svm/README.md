<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Support%20Vector%20Machines&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/svm-kernel-complete.svg" width="100%">

*Caption: SVM finds the hyperplane that maximizes margin between classes. Kernel trick enables non-linear decision boundaries by mapping to higher dimensions.*

---

## 📐 Mathematical Foundations

### Hard-Margin SVM

```
Objective: Find hyperplane w·x + b = 0 maximizing margin

Primal Problem:
min_{w,b} (1/2)||w||²
s.t. yᵢ(w·xᵢ + b) ≥ 1  ∀i

Margin = 2/||w||

Dual Problem:
max_α Σᵢαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)
s.t. αᵢ ≥ 0, Σᵢαᵢyᵢ = 0
```

### Soft-Margin SVM

```
Allow some misclassification with slack variables ξᵢ:

min_{w,b,ξ} (1/2)||w||² + C Σᵢξᵢ
s.t. yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
     ξᵢ ≥ 0

C controls trade-off:
• Large C: Less slack, harder margin
• Small C: More slack, softer margin
```

### Kernel Trick

```
Replace dot product with kernel:
K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)

Common Kernels:
• Linear:     K(x,y) = x·y
• Polynomial: K(x,y) = (x·y + c)^d
• RBF:        K(x,y) = exp(-γ||x-y||²)
• Sigmoid:    K(x,y) = tanh(αx·y + c)

Decision function:
f(x) = sign(Σᵢ αᵢyᵢK(xᵢ, x) + b)
```

---

## 🎯 Key Concepts

| Concept | Description | Impact |
|---------|-------------|--------|
| **Support Vectors** | Points on margin boundary | Define the classifier |
| **Margin** | Distance to hyperplane | Larger = better generalization |
| **Kernel** | Implicit feature mapping | Non-linear boundaries |
| **Slack Variables** | Allow violations | Handle non-separable data |

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Create dataset
X, y = make_classification(n_samples=200, n_features=2, 
                           n_redundant=0, n_clusters_per_class=1)

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X, y)
print(f"Linear SVM - Support vectors: {len(svm_linear.support_vectors_)}")

# RBF Kernel SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X, y)
print(f"RBF SVM - Support vectors: {len(svm_rbf.support_vectors_)}")

# Polynomial Kernel SVM
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X, y)

# Custom kernel
def custom_kernel(X1, X2):
    """Custom RBF kernel"""
    gamma = 0.5
    dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
            np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    return np.exp(-gamma * dists)

svm_custom = SVC(kernel=custom_kernel)
svm_custom.fit(X, y)

# Get margin width for linear SVM
w = svm_linear.coef_[0]
margin = 2 / np.linalg.norm(w)
print(f"Margin width: {margin:.4f}")

# Decision function values
decision_values = svm_linear.decision_function(X[:5])
print(f"Decision values: {decision_values}")
```

---

## 🌍 ML Applications

| Application | How SVM Is Used |
|-------------|-----------------|
| **Text Classification** | Linear SVM on TF-IDF |
| **Image Recognition** | RBF kernel on features |
| **Bioinformatics** | Protein classification |
| **Anomaly Detection** | One-class SVM |
| **Handwriting Recognition** | Multi-class SVM |

---

## 📊 Kernel Comparison

| Kernel | Formula | Use Case | Pros | Cons |
|--------|---------|----------|------|------|
| **Linear** | x·y | High-dim, text | Fast, interpretable | Linear only |
| **RBF** | exp(-γ\|\|x-y\|\|²) | General purpose | Flexible | Slower, tune γ |
| **Polynomial** | (x·y + c)^d | Images | Captures interactions | Tune d |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Vapnik's Original | [Book](https://www.springer.com/gp/book/9780387987804) |
| 📄 | Platt's SMO | [Paper](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) |
| 🎥 | SVM Explained | [YouTube](https://www.youtube.com/watch?v=efR1C6CvhmE) |
| 🇨🇳 | SVM详解 | [知乎](https://zhuanlan.zhihu.com/p/31886934) |
| 🇨🇳 | 核方法入门 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88889999) |

---

## 🔗 Where This Topic Is Used

| Application | How SVM Is Used |
|-------------|-----------------|
| **Kernel Methods** | Foundation for kernel learning |
| **Maximum Margin** | Principle used in many algorithms |
| **Feature Mapping** | Kernel trick widely applied |
| **Ensemble Methods** | SVM as base learner |
| **Neural Networks** | Hinge loss from SVM |

---

⬅️ [Back: 02-Generalization](../02-generalization/) | ➡️ [Next: 04-Representation](../04-representation/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
