<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Structural Risk&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Structural Risk Minimization (SRM)

> **Balance training error with model complexity**

---

## 🎯 Visual Overview

<img src="./images/srm.svg" width="100%">

*Caption: SRM adds a complexity penalty to ERM: minimize R̂(f) + Ω(f). This prevents overfitting by penalizing complex models. Examples: L1/L2 regularization, weight decay, early stopping.*

---

## 📂 Overview

Structural Risk Minimization extends ERM by adding model complexity control. This is the theoretical foundation for regularization in machine learning.

---

## 📐 Mathematical Definitions

### SRM Principle
```
f̂ = argmin_{f∈Fₖ} [R̂(f) + Ω(k)]

Where:
• F₁ ⊂ F₂ ⊂ F₃ ⊂ ... (nested hypothesis classes)
• R̂(f): empirical risk
• Ω(k): complexity penalty for class Fₖ
```

### Generalization Bound
```
With high probability:
R(f) ≤ R̂(f) + O(√(complexity(F)/n))

SRM finds the sweet spot:
• Too simple F → high training error (underfitting)
• Too complex F → large complexity term (overfitting)
```

### Regularization Connection
```
SRM ≡ Regularized ERM

f̂ = argmin [R̂(f) + λ ||f||²]

Where ||f||² measures complexity in function space
• L2: ||θ||² promotes small weights
• L1: ||θ||₁ promotes sparse solutions
• RKHS norm: promotes smooth functions
```

### Vapnik's Bound
```
R(f) ≤ R̂(f) + √((h(log(2n/h) + 1) - log(δ/4))/n)

Where h = VC dimension of hypothesis class F
```

### Bias-Variance Tradeoff
```
E[(f̂(x) - y)²] = Bias² + Variance + Noise

SRM trades off:
• Complex model: low bias, high variance
• Simple model: high bias, low variance
• λ controls this tradeoff
```

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

# SRM via regularization strength selection
lambdas = [0.001, 0.01, 0.1, 1, 10, 100]
best_lambda, best_score = None, -float('inf')

for lam in lambdas:
    model = Ridge(alpha=lam)
    score = cross_val_score(model, X, y, cv=5).mean()
    if score > best_score:
        best_score = score
        best_lambda = lam

# SRM via early stopping (implicit regularization)
model = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 1))
optimizer = torch.optim.Adam(model.parameters())

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1000):
    # Train
    train_loss = train_step(model, train_data)
    val_loss = evaluate(model, val_data)
    
    # SRM: stop when validation starts increasing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= 10:
            break  # Early stopping
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Vapnik: Statistical Learning | [Book](https://link.springer.com/book/10.1007/978-1-4757-3264-1) |
| 📖 | Shalev-Shwartz: Understanding ML | [Book](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📖 | ESL Ch. 7 | [Book](https://hastie.su.domains/ElemStatLearn/) |
| 🇨🇳 | 结构风险最小化 | [知乎](https://zhuanlan.zhihu.com/p/38853908) |
| 🇨🇳 | 正则化理论 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
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

⬅️ [Back: structural-risk](../)

---

⬅️ [Back: Pac Learning](../pac-learning/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
