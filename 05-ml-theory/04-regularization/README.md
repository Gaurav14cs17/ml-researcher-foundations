<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Regularization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/l1-l2-regularization-complete.svg" width="100%">

*Caption: Regularization adds penalties to prevent overfitting. L1 (Lasso) promotes sparsity, L2 (Ridge) shrinks weights uniformly. The choice depends on whether feature selection is desired.*

---

## 📐 Mathematical Foundations

### L2 Regularization (Ridge)

```
Loss = L(θ) + λ||θ||₂²

Gradient:
∇(L + λ||θ||²) = ∇L + 2λθ

Update rule:
θ ← θ - η(∇L + 2λθ)
θ ← (1 - 2ηλ)θ - η∇L  (weight decay!)

Effect: Shrinks all weights toward zero
```

### L1 Regularization (Lasso)

```
Loss = L(θ) + λ||θ||₁

Subgradient:
∂||θ||₁/∂θᵢ = sign(θᵢ)

Effect: 
• Drives small weights to exactly zero
• Automatic feature selection
• Sparse solutions
```

### Elastic Net

```
Loss = L(θ) + λ₁||θ||₁ + λ₂||θ||₂²

Combines benefits of L1 and L2:
• Sparsity from L1
• Stability from L2
• Handles correlated features
```

---

## 🎯 Comparison

| Regularization | Formula | Effect | Use Case |
|----------------|---------|--------|----------|
| **L2 (Ridge)** | λ\|\|θ\|\|₂² | Shrinks all weights | General, many features |
| **L1 (Lasso)** | λ\|\|θ\|\|₁ | Sparse weights | Feature selection |
| **Elastic Net** | λ₁\|\|θ\|\|₁ + λ₂\|\|θ\|\|₂² | Combined | Correlated features |
| **Dropout** | Random zeros | Ensemble effect | Neural networks |
| **Early Stopping** | Stop training | Implicit regularization | All models |

---

## 💻 Code Examples

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# L2 Regularization (Ridge)
X = np.random.randn(100, 10)
y = np.random.randn(100)

ridge = Ridge(alpha=1.0)  # alpha = λ
ridge.fit(X, y)
print(f"Ridge coefficients: {ridge.coef_}")

# L1 Regularization (Lasso)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print(f"Lasso coefficients: {lasso.coef_}")
print(f"Non-zero coefficients: {np.sum(lasso.coef_ != 0)}")

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X, y)

# PyTorch weight decay (L2)
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

# Manual L1 regularization in PyTorch
def l1_loss(model, lambda_l1=0.01):
    l1_reg = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_reg

# Training with L1
x = torch.randn(32, 10)
y = torch.randn(32, 1)

loss = nn.MSELoss()(model(x), y) + l1_loss(model)
loss.backward()
optimizer.step()

# Dropout (neural network regularization)
class RegularizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Applied during training
        return self.fc2(x)
```

---

## 🌍 ML Applications

| Application | Regularization | Why |
|-------------|----------------|-----|
| **Linear Regression** | Ridge, Lasso | Prevent overfitting |
| **Neural Networks** | Weight decay, Dropout | Control capacity |
| **Feature Selection** | L1 (Lasso) | Sparse features |
| **LLM Training** | AdamW (decoupled weight decay) | Stable training |
| **Image Models** | Dropout, Data augmentation | Generalization |

---

## 📊 L1 vs L2 Geometry

```
L2: Circular constraint
    Weight space: ||w||₂ ≤ r (sphere)
    Solution: Smooth shrinkage
    
L1: Diamond constraint  
    Weight space: ||w||₁ ≤ r (diamond)
    Solution: Corners → sparse (some wᵢ = 0)

Why L1 gives sparsity:
    Optimal point often at corners of diamond
    Corners have coordinate(s) = 0
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Elements of Statistical Learning | [Book](https://hastie.su.domains/ElemStatLearn/) |
| 📄 | Dropout Paper | [arXiv](https://arxiv.org/abs/1207.0580) |
| 📄 | AdamW Paper | [arXiv](https://arxiv.org/abs/1711.05101) |
| 🇨🇳 | 正则化详解 | [知乎](https://zhuanlan.zhihu.com/p/29360425) |
| 🇨🇳 | L1与L2正则化 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88777777) |

---

## 🔗 Where This Topic Is Used

| Application | How Regularization Is Used |
|-------------|---------------------------|
| **Training Neural Networks** | Weight decay prevents overfitting |
| **Feature Selection** | L1 selects important features |
| **Bayesian Learning** | Priors as regularization |
| **LLM Fine-tuning** | LoRA + weight decay |
| **Compressed Sensing** | L1 for sparse recovery |

---

⬅️ [Back: 03-Kernel Methods](../03-kernel-methods/) | ➡️ [Next: 05-Risk Minimization](../05-risk-minimization/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
