<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=09 Self-Supervised Learning&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🔄 Self-Supervised Learning

> **Learning representations without labels**

---

## 🎯 Visual Overview

<img src="./images/contrastive-learning-complete.svg" width="100%">

*Caption: Self-supervised learning creates supervisory signals from the data itself. Contrastive methods pull similar pairs together and push dissimilar pairs apart.*

---

## 📐 Key Methods

### Contrastive Learning
```
InfoNCE Loss:
L = -log(exp(sim(z,z⁺)/τ) / Σᵢexp(sim(z,zᵢ)/τ))

Methods:
• SimCLR: Augmentation-based positive pairs
• MoCo: Momentum-updated encoder
• CLIP: Image-text pairs
```

### Masked Prediction
```
BERT: Mask tokens, predict them
MAE: Mask image patches, reconstruct
```

---

## 💻 Code Examples

```python
import torch
import torch.nn.functional as F

def simclr_loss(z1, z2, temperature=0.5):
    """SimCLR contrastive loss"""
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    
    sim = torch.mm(z, z.t()) / temperature
    n = z1.size(0)
    
    # Mask self-similarities
    mask = torch.eye(2*n, device=z.device).bool()
    sim = sim.masked_fill(mask, -float('inf'))
    
    # Positive pairs
    labels = torch.cat([torch.arange(n, 2*n), torch.arange(n)], dim=0).to(z.device)
    
    return F.cross_entropy(sim, labels)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | SimCLR | [arXiv](https://arxiv.org/abs/2002.05709) |
| 📄 | CLIP | [arXiv](https://arxiv.org/abs/2103.00020) |
| 🇨🇳 | 自监督学习详解 | [知乎](https://zhuanlan.zhihu.com/p/108906502) |

---

⬅️ [Back: 08-Data Augmentation](../08-data-augmentation/) | ➡️ [Next: 10-NAS](../10-neural-architecture-search/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

