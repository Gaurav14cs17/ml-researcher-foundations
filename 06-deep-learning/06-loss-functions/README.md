<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=06 Loss Functions&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📉 Loss Functions

> **Objective functions that guide neural network learning**

---

## 🎯 Visual Overview

<img src="./images/loss-functions-complete.svg" width="100%">

*Caption: Loss functions measure prediction error. Cross-entropy for classification, MSE for regression, contrastive for embeddings.*

---

## 📐 Mathematical Foundations

### Classification Losses

```
Binary Cross-Entropy:
L = -[y log(p) + (1-y) log(1-p)]

Multi-class Cross-Entropy:
L = -Σᵢ yᵢ log(pᵢ)

Focal Loss (for imbalanced):
L = -αₜ(1-pₜ)^γ log(pₜ)
```

### Regression Losses

```
MSE (L2):
L = (1/n) Σᵢ(yᵢ - ŷᵢ)²

MAE (L1):
L = (1/n) Σᵢ|yᵢ - ŷᵢ|

Huber Loss:
L = { 0.5(y-ŷ)²     if |y-ŷ| < δ
    { δ|y-ŷ| - 0.5δ² otherwise
```

### Contrastive Losses

```
Triplet Loss:
L = max(0, d(a,p) - d(a,n) + margin)

InfoNCE (Contrastive):
L = -log(exp(sim(q,k⁺)/τ) / Σᵢexp(sim(q,kᵢ)/τ))
```

---

## 🎯 Loss Selection Guide

| Task | Loss Function | When to Use |
|------|---------------|-------------|
| **Binary Classification** | BCE | Two classes |
| **Multi-class** | Cross-Entropy | Mutually exclusive classes |
| **Multi-label** | BCE per label | Multiple labels per sample |
| **Regression** | MSE/MAE | Continuous targets |
| **Embedding** | Triplet/Contrastive | Similarity learning |
| **Generation** | Reconstruction + KL | VAE |

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Classification losses
bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()

# With label smoothing
ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

# Focal loss (for imbalanced data)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Regression losses
mse = nn.MSELoss()
mae = nn.L1Loss()
huber = nn.SmoothL1Loss()

# Contrastive loss
def contrastive_loss(z1, z2, temperature=0.5):
    """Simple contrastive loss"""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    similarity = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    
    loss = F.cross_entropy(similarity, labels)
    return loss
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Focal Loss | [arXiv](https://arxiv.org/abs/1708.02002) |
| 📄 | InfoNCE | [arXiv](https://arxiv.org/abs/1807.03748) |
| 🇨🇳 | 损失函数详解 | [知乎](https://zhuanlan.zhihu.com/p/35709485) |

---

## 🔗 Where Loss Functions Are Used

| Application | Loss Function |
|-------------|---------------|
| **Image Classification** | Cross-Entropy |
| **Object Detection** | Focal Loss + IoU |
| **Language Modeling** | Cross-Entropy |
| **Contrastive Learning** | InfoNCE |
| **VAE** | Reconstruction + KL |
| **GAN** | Adversarial losses |

---

⬅️ [Back: 05-Training-Techniques](../05-training-techniques/) | ➡️ [Next: 07-Transfer Learning](../07-transfer-learning/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

