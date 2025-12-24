<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Erm&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Empirical Risk Minimization (ERM)

> **Minimize training error to find the best model**

---

## 🎯 Visual Overview

<img src="./images/erm.svg" width="100%">

*Caption: ERM finds the function f̂ that minimizes average loss on training data. It's consistent (converges to true minimizer as n→∞) but can overfit without regularization.*

---

## 📂 Overview

Empirical Risk Minimization is the most fundamental principle in machine learning. Find the model that minimizes the empirical (training) loss.

---

## 📐 Mathematical Definitions

### True Risk
```
R(f) = E_{(x,y)~D}[L(f(x), y)]

The expected loss over the true data distribution D.
We can't compute this (D is unknown)!
```

### Empirical Risk
```
R̂(f) = (1/n) Σᵢ L(f(xᵢ), yᵢ)

Average loss over training data.
This is what we actually minimize!
```

### ERM Principle
```
f̂ = argmin_{f∈F} R̂(f)

Find f that minimizes training loss.

Key question: Does R̂(f̂) ≈ R(f̂)?
Answer: Yes, if |F| is controlled (complexity)
```

### Generalization Bound
```
With probability 1-δ:

R(f̂) ≤ R̂(f̂) + O(√(log|F|/n))

• n = sample size
• |F| = hypothesis class size
• More data ⟹ better generalization
• Simpler F ⟹ better generalization
```

### Loss Functions
```
Classification: L(ŷ, y) = 1[ŷ ≠ y]  (0-1 loss)
              Cross-entropy: -log p(y|x)

Regression:   L(ŷ, y) = (ŷ - y)²   (MSE)
              L(ŷ, y) = |ŷ - y|    (MAE)
```

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn

# ERM: minimize empirical risk
model = nn.Linear(10, 1)
criterion = nn.MSELoss()  # Loss function L
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training = ERM
for epoch in range(100):
    # Compute empirical risk R̂(f)
    predictions = model(X_train)
    empirical_risk = criterion(predictions, y_train)  # (1/n)ΣL
    
    # Minimize it
    optimizer.zero_grad()
    empirical_risk.backward()
    optimizer.step()

# Evaluate true risk approximation (test set)
with torch.no_grad():
    test_risk = criterion(model(X_test), y_test)
    print(f"Train risk: {empirical_risk:.4f}, Test risk: {test_risk:.4f}")
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Vapnik: Statistical Learning | [Book](https://link.springer.com/book/10.1007/978-1-4757-3264-1) |
| 📖 | Shalev-Shwartz: Understanding ML | [Book](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📄 | ESL Ch. 7 | [Book](https://hastie.su.domains/ElemStatLearn/) |
| 🇨🇳 | 经验风险最小化 | [知乎](https://zhuanlan.zhihu.com/p/38853908) |
| 🇨🇳 | 泛化理论 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 统计学习方法 | [B站](https://www.bilibili.com/video/BV164411b7dx) |

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

⬅️ [Back: erm](../)

---

➡️ [Next: Pac Learning](../pac-learning/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
