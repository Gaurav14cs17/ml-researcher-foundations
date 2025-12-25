<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Representation%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/representation.svg" width="100%">

*Caption: Representation learning transforms raw data into useful features. Deep networks learn hierarchical representations automatically, unlike hand-crafted features. Modern foundation models (BERT, CLIP) learn powerful representations via self-supervised learning.*

---

## 📂 Overview

Representation learning is about learning useful features from data. Good representations make downstream tasks easier by capturing relevant structure while discarding noise.

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [feature-learning/](./feature-learning/) | Hierarchical features | CNNs, Transformers |
| [embeddings/](./embeddings/) | Dense vectors | Word2Vec, CLIP |
| [transfer/](./transfer/) | Transfer learning | Pre-training, fine-tuning |

---

## 📐 Key Concepts

### What is a Good Representation?
```
A good representation z = f(x) should:
1. Capture task-relevant information
2. Be invariant to irrelevant variations
3. Disentangle factors of variation
4. Enable easy downstream learning
```

### Representation Learning Paradigms
```
Supervised: Learn features jointly with task
Self-supervised: Learn from data structure
    • Contrastive (SimCLR, CLIP)
    • Masked prediction (BERT, MAE)
    • Generative (VAE, flow)
```

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn

# Extract representations from pretrained model
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
model.fc = nn.Identity()  # Remove classification head

with torch.no_grad():
    features = model(images)  # [batch, 2048]

# Use representations for downstream task
classifier = nn.Linear(2048, num_classes)
logits = classifier(features)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Representation Learning Survey | [arXiv](https://arxiv.org/abs/1206.5538) |
| 📄 | SimCLR Paper | [arXiv](https://arxiv.org/abs/2002.05709) |
| 📖 | Deep Learning Book Ch. 15 | [Book](https://www.deeplearningbook.org/contents/representation.html) |
| 🇨🇳 | 表示学习详解 | [知乎](https://zhuanlan.zhihu.com/p/27657264) |
| 🇨🇳 | 特征学习 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 深度学习特征 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

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

⬅️ [Back: 03-Kernel Methods](../03-kernel-methods/) | ➡️ [Next: 05-Risk Minimization](../05-risk-minimization/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
