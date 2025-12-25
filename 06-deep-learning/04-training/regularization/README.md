<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Regularization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/regularization.svg" width="100%">

*Caption: Regularization techniques prevent overfitting. L2 (Ridge) shrinks weights toward zero; L1 (Lasso) promotes sparsity; Dropout randomly zeros neurons during training. The goal: models that generalize to unseen data.*

---

## 📂 Overview

Regularization adds constraints or penalties to prevent models from fitting noise in training data. It's essential for achieving good generalization.

---

## 🔑 Main Techniques

| Technique | Method | Effect |
|-----------|--------|--------|
| **L2 (Ridge)** | Add λΣw² to loss | Shrink all weights |
| **L1 (Lasso)** | Add λΣ\|w\| to loss | Promote sparsity |
| **Dropout** | Zero random neurons | Ensemble effect |
| **Early Stopping** | Stop when val loss ↑ | Limit training time |
| **Weight Decay** | Decay weights each step | Similar to L2 |

---

## 📐 L1 vs L2

```
L2 Regularization:
L = L_data + λ Σ wᵢ²
∂L/∂w = ∂L_data/∂w + 2λw  → shrinks proportionally

L1 Regularization:
L = L_data + λ Σ |wᵢ|
∂L/∂w = ∂L_data/∂w + λ·sign(w)  → constant push to 0

Result: L1 creates exact zeros (sparse), L2 creates small values
```

---

## 💻 Code

```python
import torch
import torch.nn as nn

# L2 via weight_decay in optimizer (most common)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# L1 regularization (manual)
l1_lambda = 0.001
l1_loss = sum(p.abs().sum() for p in model.parameters())
total_loss = loss + l1_lambda * l1_loss

# Dropout layer
dropout = nn.Dropout(p=0.1)  # 10% dropout
x = dropout(x)  # Only during training!

# Label smoothing in cross-entropy
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Dropout Paper | [JMLR](https://jmlr.org/papers/v15/srivastava14a.html) |
| 📄 | L1/L2 Regularization | [ESL Ch. 3](https://hastie.su.domains/ElemStatLearn/) |
| 📖 | PyTorch Weight Decay | [Docs](https://pytorch.org/docs/stable/optim.html) |
| 🇨🇳 | 正则化详解 | [知乎](https://zhuanlan.zhihu.com/p/29360425) |
| 🇨🇳 | Dropout原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 防止过拟合方法 | [B站](https://www.bilibili.com/video/BV164411b7dx) |


## 🔗 Where This Topic Is Used

| Technique | Application |
|-----------|------------|
| **Dropout** | Prevent overfitting |
| **Weight Decay** | L2 regularization |
| **Data Augmentation** | CNNs, vision |
| **Early Stopping** | Validation-based |

---

⬅️ [Back: Training](../)

---

⬅️ [Back: Optimizers](../optimizers/) | ➡️ [Next: Scheduling](../scheduling/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
