<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Neural%20Architecture%20Search&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/nas-complete.svg" width="100%">

*Caption: NAS automates architecture design. Search space defines possible architectures, search strategy explores it, evaluation measures performance.*

---

## 📐 Key Components

```
1. Search Space: Possible operations and connections

2. Search Strategy: How to explore the space
   • Random search
   • Evolutionary algorithms
   • Reinforcement learning
   • Differentiable (DARTS)

3. Performance Estimation: Evaluate candidate architectures

```

### DARTS (Differentiable)

```
Mixed operation: ō(x) = Σᵢ αᵢ oᵢ(x)

where αᵢ = exp(aᵢ)/Σⱼexp(aⱼ) (softmax)

Jointly optimize:
• Architecture weights α
• Network weights w

```

---

## 💻 Code Example

```python
# DARTS-style mixed operation
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    def __init__(self, C, ops):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.alphas = nn.Parameter(torch.zeros(len(ops)))
    
    def forward(self, x):
        weights = F.softmax(self.alphas, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | DARTS | [arXiv](https://arxiv.org/abs/1806.09055) |
| 📄 | EfficientNet | [arXiv](https://arxiv.org/abs/1905.11946) |
| 🇨🇳 | NAS详解 | [知乎](https://zhuanlan.zhihu.com/p/68188970) |

---

⬅️ [Back: Self-Supervised](../09_self_supervised/README.md) | ➡️ [Next: Multi-Task Learning](../11_multi_task_learning/README.md)

---

⬅️ [Back: Main](../README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
