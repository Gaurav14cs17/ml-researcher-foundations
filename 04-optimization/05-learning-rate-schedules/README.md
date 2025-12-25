<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=05 Learning Rate Schedules&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 📉 Learning Rate Schedules

> **Adapting learning rate during training**

---

## 📐 Common Schedules

```
Step Decay:
  η(t) = η₀ × γ^⌊t/s⌋

Exponential Decay:
  η(t) = η₀ × e^(-λt)

Cosine Annealing:
  η(t) = η_min + (η₀ - η_min) × (1 + cos(πt/T))/2

Warmup + Decay (Transformers):
  η(t) = d^(-0.5) × min(t^(-0.5), t × warmup^(-1.5))
```

---

## 💻 Code Examples

```python
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, OneCycleLR
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# One cycle (fast.ai style)
scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=100, 
                       steps_per_epoch=len(dataloader))
```

---

## 🔗 Best Practices

| Scenario | Recommended Schedule |
|----------|---------------------|
| **Vision (ResNet)** | Step decay |
| **Transformers** | Warmup + linear/cosine decay |
| **Fast training** | One Cycle |
| **Fine-tuning** | Lower LR + linear decay |

---

⬅️ [Back: Convex Optimization](../04-convex-optimization/) | ➡️ [Next: Linear Programming](../06-linear-programming/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

