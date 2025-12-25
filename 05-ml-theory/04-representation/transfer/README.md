<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Transfer%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/transfer.svg" width="100%">

*Caption: Transfer learning: pre-train on large dataset, transfer to target task via feature extraction or fine-tuning. Foundation models (GPT, CLIP, BERT) enable transfer to many downstream tasks with minimal data.*

---

## 📂 Overview

Transfer learning reuses learned representations from source tasks for target tasks. This dramatically reduces data requirements and is the foundation of modern AI with pre-trained foundation models.

---

## 📐 Key Concepts

### Transfer Learning Methods
```
Feature Extraction: Freeze pretrained, train new head
Fine-tuning: Update all/some pretrained weights
Adapter: Add small trainable modules
LoRA: Low-rank updates to frozen weights
```

### Domain Shift
```
Source domain: Dₛ with distribution Pₛ(x,y)
Target domain: Dₜ with distribution Pₜ(x,y)

Domain adaptation: Handle Pₛ ≠ Pₜ
Covariate shift: Pₛ(x) ≠ Pₜ(x), P(y|x) same
```

### Foundation Model Transfer
```
Pre-training: Learn general representations
    GPT: Next token prediction
    BERT: Masked language modeling  
    CLIP: Image-text contrastive

Fine-tuning: Adapt to specific task
    Full: Update all parameters
    PEFT: Update few parameters (LoRA, adapters)
```

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Feature extraction (freeze backbone)
backbone = AutoModel.from_pretrained("bert-base-uncased")
for param in backbone.parameters():
    param.requires_grad = False

classifier = nn.Linear(768, num_classes)

# Fine-tuning (update all)
model = AutoModel.from_pretrained("bert-base-uncased")
model.train()  # All parameters trainable

# LoRA fine-tuning (efficient)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # ~0.1% of params!

# Vision transfer
import torchvision.models as models
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(2048, num_classes)  # Replace head
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | LoRA Paper | [arXiv](https://arxiv.org/abs/2106.09685) |
| 📄 | Foundation Models | [arXiv](https://arxiv.org/abs/2108.07258) |
| 📖 | Transfer Learning Survey | [arXiv](https://arxiv.org/abs/1911.02685) |
| 🇨🇳 | 迁移学习详解 | [知乎](https://zhuanlan.zhihu.com/p/27657264) |
| 🇨🇳 | Fine-tuning技巧 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 预训练模型 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

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

⬅️ [Back: transfer](../)

---

⬅️ [Back: Feature Learning](../feature-learning/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
