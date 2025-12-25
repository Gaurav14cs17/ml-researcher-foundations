<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=10 NAS&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🔍 Neural Architecture Search

> **Automatically finding optimal network architectures**

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

⬅️ [Back: 09-Self-Supervised](../09-self-supervised/) | ➡️ [Next: 11-Multi-Task](../11-multi-task-learning/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

