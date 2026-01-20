<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Multi-Task%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/multi-task-learning-complete.svg" width="100%">

*Caption: MTL shares representations across tasks. Hard parameter sharing uses common backbone, soft sharing connects task-specific networks.*

---

## 📐 Key Concepts

```
Total Loss: L = Σᵢ wᵢ Lᵢ

Challenges:
• Task balancing: Different loss scales
• Negative transfer: Tasks interfere
• Architecture design: What to share

Solutions:
• Uncertainty weighting
• Gradient normalization
• Task-specific heads
```

---

## 💻 Code Example

```python
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, shared_dim=512, num_classes=[10, 5]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, shared_dim)
        )
        self.heads = nn.ModuleList([
            nn.Linear(shared_dim, n) for n in num_classes
        ])
    
    def forward(self, x):
        shared = self.shared(x)
        return [head(shared) for head in self.heads]
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | MTL Survey | [arXiv](https://arxiv.org/abs/1706.05098) |
| 🇨🇳 | 多任务学习详解 | [知乎](https://zhuanlan.zhihu.com/p/59413549) |

---

⬅️ [Back: NAS](../10_neural_architecture_search/README.md) | ➡️ [Next: Meta-Learning](../12_meta_learning/README.md)

---

⬅️ [Back: Main](../README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
