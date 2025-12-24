# Regularization

> **Preventing overfitting through constraints**

---

## 🎯 Visual Overview

<img src="./images/regularization.svg" width="100%">

*Caption: Regularization constrains models to improve generalization. Explicit (L1/L2 penalties), implicit (SGD, early stopping), and data-based (augmentation, dropout) regularization all reduce overfitting.*

---

## 📂 Overview

Regularization is crucial for preventing overfitting. It adds constraints or penalties that prefer simpler models, improving generalization to unseen data.

---

## 📐 Mathematical Definitions

### Regularized Loss
```
L_reg(θ) = L_data(θ) + λR(θ)

Where:
• L_data: empirical risk (e.g., cross-entropy)
• R(θ): regularization term
• λ: regularization strength
```

### L2 Regularization (Ridge / Weight Decay)
```
R(θ) = ||θ||₂² = Σᵢ θᵢ²

∂R/∂θ = 2θ  →  Shrinks weights proportionally

Bayesian view: Gaussian prior θ ~ N(0, 1/λ)
```

### L1 Regularization (Lasso)
```
R(θ) = ||θ||₁ = Σᵢ |θᵢ|

∂R/∂θ = sign(θ)  →  Constant push toward 0

Bayesian view: Laplace prior
Result: Sparse solutions (feature selection)
```

### Elastic Net
```
R(θ) = α||θ||₁ + (1-α)||θ||₂²

Best of both worlds:
• Sparsity from L1
• Grouped selection from L2
```

### Dropout
```
During training: h̃ᵢ = hᵢ · mᵢ where mᵢ ~ Bernoulli(1-p)
During inference: h̃ᵢ = (1-p) · hᵢ

Effect: Ensemble of 2^n networks
        Prevents co-adaptation
```

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn

# L2 via weight_decay (most common)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Manual L1 regularization
l1_lambda = 0.001
l1_loss = sum(p.abs().sum() for p in model.parameters())
total_loss = data_loss + l1_lambda * l1_loss

# Elastic Net
def elastic_net_loss(model, alpha=0.5, l1_lambda=0.001, l2_lambda=0.01):
    l1 = sum(p.abs().sum() for p in model.parameters())
    l2 = sum(p.pow(2).sum() for p in model.parameters())
    return alpha * l1_lambda * l1 + (1-alpha) * l2_lambda * l2

# Dropout
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # 50% dropout
    nn.Linear(256, 10)
)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Dropout Paper | [JMLR](https://jmlr.org/papers/v15/srivastava14a.html) |
| 📖 | ESL Ch. 3 | [Book](https://hastie.su.domains/ElemStatLearn/) |
| 📖 | Bishop PRML Ch. 3 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🇨🇳 | 正则化详解 | [知乎](https://zhuanlan.zhihu.com/p/29360425) |
| 🇨🇳 | L1与L2对比 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | Dropout原理 | [B站](https://www.bilibili.com/video/BV164411b7dx) |

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

⬅️ [Back: regularization](../)

---

⬅️ [Back: Overfitting](../overfitting/) | ➡️ [Next: Vc Dimension](../vc-dimension/)
