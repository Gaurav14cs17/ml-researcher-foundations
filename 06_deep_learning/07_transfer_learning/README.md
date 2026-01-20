<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Transfer%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/transfer-learning-finetuning.svg" width="100%">

*Caption: Transfer learning uses knowledge from pretrained models. Feature extraction freezes pretrained weights, fine-tuning updates them on new data.*

---

## 📐 Mathematical Foundations

### Transfer Learning Paradigm

```
Pretrained model: f(x; θ_pretrained)
New task: Learn f(x; θ_new) using θ_pretrained

Strategies:

1. Feature Extraction: Freeze θ_pretrained, train new head

2. Fine-tuning: Update all or part of θ_pretrained

3. Adapter methods: Add small trainable modules

```

### Domain Adaptation

```
Source domain: D_s = {(x_s, y_s)}
Target domain: D_t = {x_t} (often unlabeled)

Goal: Minimize target risk using source knowledge

Methods:
• Domain adversarial training
• Maximum Mean Discrepancy (MMD)
• Self-training

```

---

## 🎯 Transfer Learning Strategies

| Strategy | Freeze Pretrained | Train New Layers | Use Case |
|----------|-------------------|------------------|----------|
| **Feature Extraction** | Yes | Head only | Small dataset |
| **Fine-tuning (full)** | No | All layers | Large dataset |
| **Fine-tuning (partial)** | Lower layers | Upper layers | Medium dataset |
| **Adapter** | Most weights | Small adapters | Efficient |
| **LoRA** | All weights | Low-rank matrices | LLM tuning |

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
from torchvision import models

# Feature Extraction
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(2048, num_classes)

# Only train the new layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Fine-tuning (all layers)
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_classes)

# Different learning rates
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
])

# Gradual unfreezing
def unfreeze_layers(model, epoch):
    if epoch >= 5:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if epoch >= 10:
        for param in model.layer3.parameters():
            param.requires_grad = True

# HuggingFace Transformers
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

```

---

## 🌍 Applications

| Domain | Pretrained Model | Task |
|--------|------------------|------|
| **Vision** | ImageNet ResNet | Medical imaging |
| **NLP** | BERT, GPT | Text classification |
| **Speech** | Wav2Vec | Speech recognition |
| **Multimodal** | CLIP | Zero-shot classification |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | How transferable are features? | [arXiv](https://arxiv.org/abs/1411.1792) |
| 📄 | BERT | [arXiv](https://arxiv.org/abs/1810.04805) |
| 🇨🇳 | 迁移学习详解 | [知乎](https://zhuanlan.zhihu.com/p/27657264) |

---

⬅️ [Back: Loss Functions](../06_loss_functions/README.md) | ➡️ [Next: Data Augmentation](../08_data_augmentation/README.md)

---

⬅️ [Back: Main](../README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
